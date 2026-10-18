from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.dependencies import get_current_user_id, get_runtime
from api.schemas import DocumentListOut, DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm", ".csv", ".xlsx"}


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    runtime=Depends(get_runtime),
):
    filename = Path((file.filename or "").strip()).name
    if not filename or filename == ".":
        raise HTTPException(400, "A filename is required")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported extension. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(400, "Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File exceeds the 20 MiB upload limit")
    try:
        result = await runtime.ingest_document(user_id=user_id, filename=filename, content=content)
        return DocumentOut(**result)
    except NotImplementedError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("", response_model=DocumentListOut)
async def list_documents(
    user_id: str = Depends(get_current_user_id),
    runtime=Depends(get_runtime),
):
    rows = await runtime.list_documents(user_id=user_id)
    return DocumentListOut(items=[DocumentOut(**row) for row in rows], total=len(rows))


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
    runtime=Depends(get_runtime),
):
    row = await runtime.get_document(user_id=user_id, doc_id=doc_id)
    if row is None:
        raise HTTPException(404, "Document not found")
    return DocumentOut(**row)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
    runtime=Depends(get_runtime),
):
    if not await runtime.delete_document(user_id=user_id, doc_id=doc_id):
        raise HTTPException(404, "Document not found")
    return None
