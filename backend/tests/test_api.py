import time
from fastapi.testclient import TestClient
from app.api.main import app
from app.evaluation.synthetic import make_contract, to_docx_bytes, to_pdf_bytes

c = TestClient(app)


def _wait(job_id):
    for _ in range(100):
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("completed", "failed"):
            return j
        time.sleep(0.1)
    raise AssertionError("job timeout")


def test_health():
    assert c.get("/health").json() == {"status": "ok"}


def test_rejects_bad_file():
    r = c.post("/api/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 422


def test_full_flow_and_exports():
    r = c.post("/api/documents", files={"file": ("a.pdf", to_pdf_bytes(make_contract(1)["text"]), "application/pdf")})
    assert r.status_code == 202
    doc, job = r.json()["document_id"], r.json()["job_id"]
    assert _wait(job)["status"] == "completed"
    s = c.get(f"/api/documents/{doc}/summary").json()
    assert s["summary"]["executive_summary"] and "Not legal advice" in s["disclaimer"]
    assert c.get(f"/api/documents/{doc}/clauses").json()
    rk = c.get(f"/api/documents/{doc}/risks").json()
    assert "risks" in rk
    qa = c.post(f"/api/documents/{doc}/questions", json={"question": "Which law governs the agreement?"}).json()
    assert qa["evidence"] and qa["answer_type"] != "absent"
    assert c.get(f"/api/documents/{doc}/chunks/{qa['evidence'][0]['chunk_id']}").status_code == 200
    rep_id = __import__("app.api.main", fromlist=["store"]).store().get_report(doc_id=doc)["id"]
    assert c.get(f"/api/reports/{rep_id}?format=pdf").content[:5] == b"%PDF-"
    assert c.get(f"/api/reports/{rep_id}?format=docx").content[:2] == b"PK"
    assert c.get(f"/api/documents/{doc}/export?format=md").text.startswith("# Contract Analysis")
    assert c.get(f"/api/documents/{doc}/export?format=exe").status_code == 400
    assert c.post(f"/api/documents/{doc}/review", json={"status": "approved", "notes": "ok"}).json()["review_status"] == "approved"
    assert c.post(f"/api/documents/{doc}/review", json={"status": "hack"}).status_code == 422
    assert c.delete(f"/api/documents/{doc}").status_code == 204
    assert c.get(f"/api/documents/{doc}/summary").status_code == 404


def test_scanned_style_empty_pdf_fails_cleanly():
    import pymupdf
    d = pymupdf.open(); d.new_page(); b = d.tobytes()
    r = c.post("/api/documents", files={"file": ("blank.pdf", b, "application/pdf")})
    j = _wait(r.json()["job_id"])
    assert j["status"] == "failed" and "No extractable text" in j["error"]


def test_question_validation_and_unready_doc():
    assert c.post("/api/documents/nope/questions", json={"question": "hi"}).status_code == 422
    assert c.post("/api/documents/nope/questions", json={"question": "what is the term?"}).status_code == 404


def test_api_key_isolation(monkeypatch):
    from app.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("APP_API_KEY", "secret")
    assert c.get("/api/documents").status_code == 401
    assert c.get("/api/documents", headers={"X-API-Key": "secret"}).status_code == 200
    get_settings.cache_clear()
