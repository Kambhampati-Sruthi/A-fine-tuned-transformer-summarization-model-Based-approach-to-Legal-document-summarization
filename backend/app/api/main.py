"""FastAPI application: upload → background analysis job → summary / clauses / risks / Q&A / export."""
from __future__ import annotations
import logging, shutil, uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..agents.graph import Pipeline
from ..agents.llm import LLM
from ..agents.qa import answer_question
from ..agents.verify import Verifier
from ..config import Settings, get_settings
from ..database.store import Store
from ..ingestion.extract import IngestionError, validate_upload
from ..rag.retriever import HybridRetriever, make_embedder, make_reranker
from ..report import to_docx, to_markdown, to_pdf
from ..schemas.models import Report

log = logging.getLogger("legal_ai")


@asynccontextmanager
async def lifespan(_: FastAPI):
    purge_expired()
    yield


app = FastAPI(title="Legal Document Intelligence API", version="1.0.0", lifespan=lifespan)
_cfg = get_settings()
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in _cfg.cors_origins.split(",")],
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])


@lru_cache
def store() -> Store:
    return Store(get_settings().database_url)


def auth(x_api_key: str = Header(default=""), cfg: Settings = Depends(get_settings)) -> str:
    """Optional API-key gate; the key doubles as the document-owner namespace (per-user isolation)."""
    if cfg.app_api_key and x_api_key != cfg.app_api_key:
        raise HTTPException(401, "Invalid or missing API key.")
    return (x_api_key[:16] or "anon")


_retrievers: dict[str, HybridRetriever] = {}


def get_retriever(doc_id: str, cfg: Settings) -> HybridRetriever:
    if doc_id not in _retrievers:
        path = cfg.index_dir / doc_id
        if not (path / "chunks.json").exists():
            raise HTTPException(409, "Document analysis is not finished yet.")
        chunks = HybridRetriever.load_chunks(path)
        _retrievers[doc_id] = HybridRetriever(chunks, make_embedder("" if cfg.embedding_model == "none" else cfg.embedding_model),
                                              make_reranker(cfg.reranker_model))
        if len(_retrievers) > 20:
            _retrievers.pop(next(iter(_retrievers)))
    return _retrievers[doc_id]


def run_job(job_id: str, doc_id: str, filename: str, path: str, cfg: Settings):
    st = store()
    try:
        st.update_job(job_id, status="running", stage="starting")
        pipe = Pipeline(cfg, on_stage=lambda s: st.update_job(job_id, stage=s))
        out = pipe.run(filename, Path(path).read_bytes(), doc_id)
        rep: Report = out["report"]
        st.save_report(rep.report_id, doc_id, rep.model_dump_json())
        st.update_job(job_id, status="completed", stage="done")
    except IngestionError as exc:
        st.update_job(job_id, status="failed", stage="ingest", error=str(exc))
    except Exception as exc:  # never leak internals or document text
        log.exception("job %s failed", job_id)
        st.update_job(job_id, status="failed", error=f"Analysis failed ({type(exc).__name__}).")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/documents", status_code=202)
async def upload(bg: BackgroundTasks, file: UploadFile = File(...), owner: str = Depends(auth),
                 cfg: Settings = Depends(get_settings)):
    data = await file.read(cfg.max_upload_mb * 1024 * 1024 + 1)
    try:
        ext = validate_upload(file.filename or "", data, cfg.max_upload_mb)
    except IngestionError as exc:
        raise HTTPException(422, str(exc))
    doc_id, job_id = uuid.uuid4().hex, uuid.uuid4().hex   # random ids; never user filenames as paths
    dest = cfg.upload_dir / f"{doc_id}{ext}"
    dest.write_bytes(data)
    store().add_document(doc_id, owner, Path(file.filename or "upload").name[:120], dest)
    store().add_job(job_id, doc_id)
    bg.add_task(run_job, job_id, doc_id, (file.filename or "upload" + ext), str(dest), cfg)
    return {"document_id": doc_id, "job_id": job_id}


@app.get("/api/documents")
def list_docs(owner: str = Depends(auth)):
    return store().list_documents(owner)


@app.get("/api/jobs/{job_id}")
def job(job_id: str, owner: str = Depends(auth)):
    j = store().get_job(job_id)
    if not j or not store().get_document(j["doc_id"], owner):
        raise HTTPException(404, "Job not found.")
    return j


def _report(doc_id: str, owner: str) -> Report:
    if not store().get_document(doc_id, owner):
        raise HTTPException(404, "Document not found.")
    row = store().get_report(doc_id=doc_id)
    if not row:
        raise HTTPException(409, "Report not ready.")
    return Report.model_validate_json(row["json"])


@app.get("/api/documents/{doc_id}/summary")
def summary(doc_id: str, owner: str = Depends(auth)):
    r = _report(doc_id, owner)
    return {"summary": r.summary, "key_facts": r.key_facts, "warnings": r.warnings, "disclaimer": r.disclaimer}


@app.get("/api/documents/{doc_id}/clauses")
def clauses(doc_id: str, owner: str = Depends(auth)):
    return _report(doc_id, owner).clauses


@app.get("/api/documents/{doc_id}/risks")
def risks(doc_id: str, owner: str = Depends(auth)):
    r = _report(doc_id, owner)
    return {"risks": r.risks, "unresolved": r.unresolved, "disclaimer": r.disclaimer}


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.post("/api/documents/{doc_id}/questions")
def ask(doc_id: str, q: Question, owner: str = Depends(auth), cfg: Settings = Depends(get_settings)):
    _report(doc_id, owner)
    return answer_question(q.question, get_retriever(doc_id, cfg), cfg, Verifier(cfg.nli_model), LLM(cfg))


@app.get("/api/documents/{doc_id}/chunks/{chunk_id}")
def chunk(doc_id: str, chunk_id: str, owner: str = Depends(auth), cfg: Settings = Depends(get_settings)):
    _report(doc_id, owner)
    c = get_retriever(doc_id, cfg).by_id.get(chunk_id)
    if not c:
        raise HTTPException(404, "Chunk not found.")
    return {**c.model_dump(), "ref": c.ref()}


class Review(BaseModel):
    status: str = Field(pattern="^(approved|edited|rejected)$")
    notes: str = Field(default="", max_length=2000)


@app.post("/api/documents/{doc_id}/review")
def review(doc_id: str, body: Review, owner: str = Depends(auth)):
    r = _report(doc_id, owner)
    r.review_status, r.reviewer_notes = body.status, body.notes
    store().save_report(r.report_id, doc_id, r.model_dump_json())
    return {"review_status": r.review_status}


def _render(rep: Report, format: str, cfg: Settings):
    if format == "json":
        return rep
    refs = {c.chunk_id: c.ref() for c in get_retriever(rep.doc_id, cfg).chunks}
    if format == "md":
        return Response(to_markdown(rep, refs), media_type="text/markdown")
    if format == "docx":
        return Response(to_docx(rep, refs), headers={"Content-Disposition": "attachment; filename=report.docx"},
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    if format == "pdf":
        return Response(to_pdf(rep, refs), headers={"Content-Disposition": "attachment; filename=report.pdf"},
                        media_type="application/pdf")
    raise HTTPException(400, "format must be json|md|docx|pdf")


@app.get("/api/reports/{report_id}")
def report_meta(report_id: str, format: str = "json", owner: str = Depends(auth), cfg: Settings = Depends(get_settings)):
    row = store().get_report(report_id=report_id)
    if not row or not store().get_document(row["doc_id"], owner):
        raise HTTPException(404, "Report not found.")
    return _render(Report.model_validate_json(row["json"]), format, cfg)


@app.get("/api/documents/{doc_id}/export")
def export(doc_id: str, format: str = "pdf", owner: str = Depends(auth), cfg: Settings = Depends(get_settings)):
    """Document-scoped export so the UI never needs to know report ids."""
    return _render(_report(doc_id, owner), format, cfg)


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete(doc_id: str, owner: str = Depends(auth), cfg: Settings = Depends(get_settings)):
    d = store().get_document(doc_id, owner)
    if not d:
        raise HTTPException(404, "Document not found.")
    Path(d["stored_path"]).unlink(missing_ok=True)
    shutil.rmtree(cfg.index_dir / doc_id, ignore_errors=True)
    _retrievers.pop(doc_id, None)
    store().delete_document(doc_id)


def purge_expired():
    """Retention policy: delete files/indexes/reports older than RETENTION_DAYS."""
    cfg = get_settings()
    for d in store().expired_documents(cfg.retention_days * 86400):
        Path(d["stored_path"]).unlink(missing_ok=True)
        shutil.rmtree(cfg.index_dir / d["id"], ignore_errors=True)
        store().delete_document(d["id"])
