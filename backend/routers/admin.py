import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend import config
from backend.deps import current_user, is_admin, require_admin
from backend.limits import acquire_slot, exclusive, heavy_slots
from backend.schemas import ClassifierEvalResponse, ConfigResponse, TrainResponse

router = APIRouter()
log = logging.getLogger("hybrid_rag.admin")


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _pick(obj: Any, names: list) -> dict:
    return {name: _plain(getattr(obj, name, None)) for name in names}


@router.get("/config", response_model=ConfigResponse)
def get_config(user=Depends(current_user)):
    admin = is_admin(user)
    sections: dict = {}
    error = None
    try:
        from src.config import get_settings

        cfg = get_settings()
        sections = {
            "llm": _pick(cfg.llm, ["provider", "model", "max_tokens", "temperature"]),
            "retrieval": _pick(
                cfg.retrieval,
                ["fusion_method", "rrf_k", "reranker_top_k", "reranker_model"],
            ),
            "classifier": _pick(
                cfg.classifier,
                ["mode", "confidence_threshold", "fallback", "model_path"],
            ),
        }
    except Exception as exc:
        error = f"Could not load config: {exc}"

    env = (
        {
            "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        }
        if admin
        else {}
    )
    return ConfigResponse(sections=sections, is_admin=admin, env=env, error=error)


@router.post("/classifier/train", response_model=TrainResponse)
def train_classifier(admin=Depends(require_admin)):
    with exclusive("classifier-train", "Training is already in progress."):
        with acquire_slot(heavy_slots, "The server is busy. Try again shortly."):
            try:
                from classifier.train import train
                from src.config import get_settings

                train()
                model_path = str(get_settings().classifier.model_path)
            except Exception as exc:
                log.exception("Classifier training failed")
                raise HTTPException(
                    status_code=500, detail=f"Training failed: {exc}"
                )
    return TrainResponse(message="Classifier trained", model_path=model_path)


@router.get("/classifier/evaluate", response_model=ClassifierEvalResponse)
def evaluate_classifier(admin=Depends(require_admin)):
    if not config.CLASSIFIER_TEST_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Test CSV not found at {config.CLASSIFIER_TEST_CSV}",
        )
    with acquire_slot(heavy_slots, "The server is busy. Try again shortly."):
        try:
            from classifier.train import evaluate

            result = evaluate(str(config.CLASSIFIER_TEST_CSV))
        except Exception as exc:
            log.exception("Classifier evaluation failed")
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")
    return ClassifierEvalResponse(
        macro_f1=float(result.get("macro_f1", 0.0) or 0.0),
        report=str(result.get("report", "")),
    )
