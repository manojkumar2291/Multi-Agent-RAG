"""HTTP routes: receive requests, call the RAG service, and return JSON.

Start reading at upload_files() and ask_question() to follow a request.
The actual document processing lives in service.py.
"""

import logging
from functools import lru_cache
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from .config import get_settings
from .models import AnswerResponse, QuestionRequest, UploadResponse
from .service import RAGService

logger = logging.getLogger(__name__)
app = FastAPI(title="Multi-agent RAG API", version="1.0.0")


# Keep one service in memory so every request reuses the same models and database.
# It is created on the first upload or question, rather than when this file loads.
@lru_cache(maxsize=1)
def get_service() -> RAGService:
    settings = get_settings()
    service = RAGService(settings)
    return service


def service_dependency() -> RAGService:
    """FastAPI calls this function to get the service needed by a route."""
    try:
        return get_service()
    except Exception as exc:
        # Put technical details in the terminal and a short message in the response.
        logger.exception("RAG initialization failed")
        raise HTTPException(503, "Cannot initialize RAG storage or embedding model") from exc


@app.get("/")
def root():
    return {"message": "Multi-agent RAG API", "docs": "/docs"}


@app.get("/health")
def health():
    """Liveness only; does not download models or contact Ollama."""
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
def upload_files(
    files: list[UploadFile] = File(...),
    service: RAGService = Depends(service_dependency),
):
    # File(...) means files must be supplied as a multipart upload.
    # Depends(...) tells FastAPI to call service_dependency() for the service.
    # A normal def lets FastAPI run slow work in a background worker thread.
    try:
        if len(files) > 10:
            raise HTTPException(400, "Upload at most 10 files per request")
        results = []
        # Convert megabytes to bytes, because read() works with bytes.
        max_file_bytes = service.settings.max_upload_mb * 1024 * 1024
        for uploaded_file in files:
            # Read one extra byte so we can detect an oversized file.
            data = uploaded_file.file.read(max_file_bytes + 1)
            if len(data) > max_file_bytes:
                raise HTTPException(413, f"File exceeds {service.settings.max_upload_mb} MB")
            try:
                filename = uploaded_file.filename or "upload"
                result = service.ingest(filename, data)
                results.append(result)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return {"files": results}
    except HTTPException:
        # Keep the 400 or 413 error that we already chose above.
        raise
    except Exception as exc:
        logger.exception("File indexing failed")
        raise HTTPException(503, "File indexing failed; check the server logs") from exc
    finally:
        # finally runs whether the upload succeeded or raised an error.
        for file in files:
            file.file.close()


@app.post("/ask", response_model=AnswerResponse)
def ask_question(
    request: QuestionRequest,
    service: RAGService = Depends(service_dependency),
):
    # request contains the validated JSON: question, k and optional document_ids.
    try:
        return service.answer(request.question, request.k, request.document_ids)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(503, "Query failed; check Ollama and the server logs") from exc


@app.get("/answer", response_model=AnswerResponse)
def legacy_answer(
    Question: str = Query(min_length=1, max_length=4000),
    service: RAGService = Depends(service_dependency),
):
    # Keep the original GET endpoint working by sending it through the same flow.
    try:
        request = QuestionRequest(question=Question)
    except ValueError as exc:
        raise HTTPException(422, "Question cannot be blank") from exc
    return ask_question(request=request, service=service)
