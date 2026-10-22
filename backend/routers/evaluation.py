import json
import logging
import math
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from backend import config
from backend.limits import RateLimiter, acquire_slot, exclusive, heavy_slots
from backend.schemas import EvaluationResponse, ReportInfo

router = APIRouter()
log = logging.getLogger("hybrid_rag.evaluation")

run_limiter = RateLimiter(3, 300)
read_limiter = RateLimiter(60, 60)

REPORT_NAME = re.compile(r"^report_\d{8}_\d{6}_[0-9a-fA-F-]{36}\.json$")


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _flat(mapping: Any) -> dict:
    return {str(k): _num(v) for k, v in dict(mapping or {}).items()}


def _user_reports(uid: str) -> list:
    if not config.REPORT_DIR.exists():
        return []
    return sorted(config.REPORT_DIR.glob(f"report_*_{uid}.json"), reverse=True)


@router.post("/run", response_model=EvaluationResponse)
def run_evaluation(
    request: Request,
    file: UploadFile = File(...),
    sample: int = Form(50),
    no_llm: bool = Form(False),
    save_json: bool = Form(True),
    fusion_method: str = Form("rrf"),
    generate: bool = Form(True),
    user=Depends(run_limiter.by_user()),
):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file")
    if fusion_method not in ("rrf", "linear"):
        raise HTTPException(status_code=400, detail="Invalid fusion method")
    sample = max(5, min(sample, 500))
    uid = str(user.id)
    limit = config.MAX_UPLOAD_MB * 1024 * 1024

    with exclusive(f"eval:{uid}", "An evaluation is already running for your account."):
        with acquire_slot(
            heavy_slots, "The server is busy running other evaluations. Try again shortly."
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                csv_path = Path(tmpdir) / "dataset.csv"
                written = 0
                with open(csv_path, "wb") as out:
                    while True:
                        block = file.file.read(1024 * 1024)
                        if not block:
                            break
                        written += len(block)
                        if written > limit:
                            raise HTTPException(
                                status_code=413,
                                detail=f"CSV exceeds the {config.MAX_UPLOAD_MB} MB limit",
                            )
                        out.write(block)
                if written == 0:
                    raise HTTPException(status_code=400, detail="The CSV is empty")

                try:
                    from evaluation.metrics import RAGEvaluator
                    from hybrid.fusion import HybridPipeline

                    pipeline = HybridPipeline(
                        fusion_method=fusion_method,
                        generate=generate,
                        user_id=uid,
                        supabase_client=request.app.state.supabase,
                    )
                    evaluator = RAGEvaluator(pipeline, use_llm_metrics=not no_llm)
                    report = evaluator.run(str(csv_path), sample=sample)
                except Exception as exc:
                    log.exception("Evaluation failed for user %s", uid)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Evaluation failed: {type(exc).__name__}: {exc}",
                    )

            report_name = None
            if save_json:
                config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                report_name = f"report_{stamp}_{uid}.json"
                try:
                    report.save(config.REPORT_DIR / report_name)
                except Exception:
                    log.exception("Could not save report for user %s", uid)
                    report_name = None

            overall = (getattr(report, "latency_stats", None) or {}).get("overall", {})
            per_route = {
                str(route): _flat(values)
                for route, values in dict(getattr(report, "per_route", None) or {}).items()
            }
            return EvaluationResponse(
                metrics_summary=_flat(getattr(report, "metrics_summary", None)),
                routing_accuracy=_num(getattr(report, "routing_accuracy", None)),
                per_route=per_route,
                latency=_flat(overall),
                report_name=report_name,
            )


@router.get("/reports", response_model=list[ReportInfo])
def list_reports(user=Depends(read_limiter.by_user())):
    items = []
    for path in _user_reports(str(user.id))[:5]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append(
            ReportInfo(name=path.name, metrics_summary=_flat(data.get("metrics_summary")))
        )
    return items


@router.get("/reports/{name}")
def download_report(name: str, user=Depends(read_limiter.by_user())):
    uid = str(user.id)
    if not REPORT_NAME.match(name) or not name.endswith(f"_{uid}.json"):
        raise HTTPException(status_code=404, detail="Report not found")
    path = config.REPORT_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="application/json", filename=name)
