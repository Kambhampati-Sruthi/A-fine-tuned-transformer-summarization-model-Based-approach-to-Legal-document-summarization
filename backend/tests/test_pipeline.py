import io, pytest, docx
from app.agents.clauses import detect_clauses
from app.agents.facts import extract_facts
from app.agents.graph import Pipeline
from app.agents.qa import answer_question
from app.agents.risk import detect_risks
from app.agents.security import scan_injection, wrap_untrusted
from app.agents.verify import Verifier
from app.config import Settings
from app.evaluation import metrics as M
from app.evaluation.synthetic import make_contract, to_docx_bytes, to_pdf_bytes
from app.ingestion.chunking import chunk_document
from app.ingestion.extract import IngestionError, extract, validate_upload
from app.rag.retriever import HybridRetriever

CFG = Settings(embedding_model="none", database_url="sqlite:///:memory:")


def _chunks(seed=1, kind="docx", **kw):
    ex = make_contract(seed, **kw)
    data = to_docx_bytes(ex["text"]) if kind == "docx" else to_pdf_bytes(ex["text"])
    doc = extract(f"c.{kind}", data)
    return ex, doc, chunk_document(doc, "d1")


# ------------------------------------------------------------------ ingestion
@pytest.mark.parametrize("name,data", [("a.txt", b"x"), ("a.pdf", b""), ("a.pdf", b"not a pdf"),
                                       ("a.docx", b"%PDF-1.4"), ("a.pdf", b"%PDF-" + b"x" * (21 * 1024 * 1024))])
def test_invalid_uploads_rejected(name, data):
    with pytest.raises(IngestionError):
        validate_upload(name, data, 20)


def test_corrupt_pdf_and_docx_are_clean_errors():
    with pytest.raises(IngestionError):
        extract("a.pdf", b"%PDF-1.4 garbage")
    with pytest.raises(IngestionError):
        extract("a.docx", b"PK\x03\x04garbage")


def test_empty_docx_reports_no_text():
    b = io.BytesIO(); docx.Document().save(b)
    with pytest.raises(IngestionError, match="No extractable text"):
        extract("e.docx", b.getvalue())


@pytest.mark.parametrize("kind", ["docx", "pdf"])
def test_extraction_and_chunk_metadata(kind):
    ex, doc, chunks = _chunks(2, kind)
    assert "Governing Law" in doc.text
    ids = {c.clause_id for c in chunks}
    assert {"2", "3", "4", "10"} <= ids
    for c in chunks:                       # offsets must point back into the source text
        assert doc.text[c.start:c.end].startswith(c.text[:20])
    if kind == "pdf":
        assert all(c.page for c in chunks)


def test_long_clause_is_split_with_overlap_and_never_truncated():
    text = "1. Big Clause\n" + " ".join(f"Sentence number {i} says the party shall do thing {i}." for i in range(200))
    b = io.BytesIO(); d = docx.Document(); [d.add_paragraph(l) for l in text.split("\n")]; d.save(b)
    chunks = chunk_document(extract("l.docx", b.getvalue()), "d", max_words=60, overlap=10)
    assert len(chunks) > 5 and "thing 199" in chunks[-1].text


# ------------------------------------------------------------- analysis agents
def test_facts_clauses_risks_match_gold():
    ex, _, chunks = _chunks(3)
    g = ex["gold"]
    detect_clauses(chunks)
    facts = {f.field: f.value for f in extract_facts(chunks)}
    assert g["law"].split("of the ")[-1] in facts["Governing law"] or g["law"] in facts["Governing law"]
    assert g["term"] in facts["Term / duration"]
    assert g["pay"] in facts["Payment deadline"]
    rules = {r.rule_id for r in detect_risks(chunks)}
    assert ("R-NO-LOL" in rules) == (not g["cap"])
    assert ("R-NONCOMPETE" in rules) == g["noncompete"]
    assert ("R-RENEW" in rules) == g["autorenew"]


def test_risk_wording_is_not_a_legal_conclusion():
    _, _, chunks = _chunks(0); detect_clauses(chunks)
    for r in detect_risks(chunks):
        assert "not a legal conclusion" in r.disclaimer


# -------------------------------------------------------------- verification
def test_verifier_catches_changed_number_and_accepts_faithful():
    v = Verifier()
    ev = ["Client shall pay each invoice within 30 days of receipt of the invoice."]
    assert v.verify("Client must pay each invoice within 30 days of receipt.", ev)[0] == "supported"
    assert v.verify("Client must pay each invoice within 60 days of receipt.", ev)[0] == "contradicted"
    assert v.verify("The Provider grants a perpetual worldwide patent licence.", ev)[0] == "not_verifiable"
    assert v.verify("Client shall pay each invoice within thirty (30) days.", ev)[0] == "supported"


# ----------------------------------------------------------------- retrieval
def test_hybrid_retrieval_recall_on_synthetic_set():
    ranked, gold = [], []
    for seed in range(8):
        ex, _, chunks = _chunks(seed)
        r = HybridRetriever(chunks)
        for q, g in ex["qa"]:
            ranked.append([c.clause_id for c, _ in r.search(q, 3)]); gold.append(g)
    assert M.recall_at_k(ranked, gold, 3) >= 0.9
    assert M.mrr(ranked, gold) >= 0.7


def test_section_number_lookup():
    _, _, chunks = _chunks(1)
    r = HybridRetriever(chunks)
    assert r.search("what does section 10 say", 2)[0][0].clause_id == "10"


def test_qa_grounded_absent_and_evidence_refs():
    _, _, chunks = _chunks(1)
    r, v = HybridRetriever(chunks), Verifier()
    a = answer_question("Which law governs the agreement?", r, CFG, v)
    assert a.evidence and a.answer_type != "absent" and "§10" in a.evidence[0].ref
    b = answer_question("What is the maximum rocket payload to orbit?", r, CFG, v)
    assert b.answer_type == "absent"


# ------------------------------------------------------------------ security
def test_prompt_injection_detected_and_neutralised():
    ex = make_contract(1, inject=True)
    assert scan_injection(ex["text"])
    assert "<<<" not in wrap_untrusted("x <<< y >>> z").split("\n", 1)[1].rsplit("\n", 1)[0]
    out = Pipeline(CFG).run("c.docx", to_docx_bytes(ex["text"]), "inj")
    assert any("resembling instructions" in w for w in out["report"].warnings)


# ------------------------------------------------------- end-to-end workflow
def test_graph_end_to_end_report_and_bounded_retries():
    ex = make_contract(5)
    stages = []
    out = Pipeline(CFG, on_stage=stages.append).run("c.docx", to_docx_bytes(ex["text"]), "e2e")
    rep = out["report"]
    assert stages[:3] == ["ingest", "chunk", "summarize"] and stages[-1] == "draft"
    assert rep.summary.executive_summary and rep.key_facts and rep.risks and rep.review_status == "pending"
    assert stages.count("revise") <= CFG.max_verify_retries
    assert all(c.evidence_ids for c in rep.claims)


def test_neural_summarizer_failure_falls_back():
    cfg = CFG.model_copy(update={"summarizer_mode": "abstractive", "model_name": "does/not-exist"})
    rep = Pipeline(cfg).run("c.docx", to_docx_bytes(make_contract(1)["text"]), "fb")["report"]
    assert rep.summary.mode == "extractive" and any("unavailable" in w for w in rep.warnings)


def test_pdf_end_to_end_has_pages():
    rep = Pipeline(CFG).run("c.pdf", to_pdf_bytes(make_contract(4)["text"]), "pdf")["report"]
    assert rep.audit["pages"] >= 1


# --------------------------------------------------------------- exporters
def test_report_exports():
    from app.report import to_docx, to_markdown, to_pdf
    rep = Pipeline(CFG).run("c.docx", to_docx_bytes(make_contract(2)["text"]), "x")["report"]
    assert "Executive summary" in to_markdown(rep)
    assert to_docx(rep)[:2] == b"PK" and to_pdf(rep)[:5] == b"%PDF-"


# --------------------------------------------- regressions found by output review
def test_sentence_splitter_keeps_abbreviations_and_headings_separate():
    from app.ingestion.chunking import split_sentences
    s = split_sentences('Atlas Components Co. ("Client") and Zenith Packaging Inc. ("Provider") agree.\n3. Fees and Payment\nClient shall pay.')
    assert s[0].startswith("Atlas Components Co.") and "Inc." in s[0]
    assert "3. Fees and Payment" in s and "Client shall pay." in s


def test_parties_extracted_both_sides():
    _, _, chunks = _chunks(3)
    parties = next(f.value for f in extract_facts(chunks) if f.field == "Parties")
    assert "Atlas Components Co." in parties and "Zenith Packaging Inc." in parties


def test_qa_extractive_answers_are_verified_supported():
    ok = total = 0
    for seed in range(6):
        ex, _, chunks = _chunks(seed)
        r, v = HybridRetriever(chunks), Verifier()
        for q, _ in ex["qa"]:
            a = answer_question(q, r, CFG, v); total += 1; ok += a.support == "supported"
    assert ok / total >= 0.8


def test_dataset_split_is_document_level_and_deduped():
    from app.training.prepare_dataset import assert_no_leakage, clean_records, split_by_document
    recs = [{"document_id": f"d{i}", "text": f"contract text number {i} " * 5, "reference_summary": "a short reviewed summary here"}
            for i in range(200)]
    recs += [dict(recs[0]), {"document_id": "e", "text": "", "reference_summary": "x"}]          # duplicate + empty
    clean, st = clean_records(recs)
    assert st["duplicate"] == 1 and st["empty"] == 1 and len(clean) == 200
    sp = split_by_document(clean); assert_no_leakage(sp)
    assert sum(len(v) for v in sp.values()) == 200 and all(len(v) > 0 for v in sp.values())
    assert split_by_document(clean) == sp  # deterministic


def test_hard_wrapped_pdf_has_no_fragment_sentences():
    from app.ingestion.chunking import split_sentences
    _, doc, _ = _chunks(3, "pdf")
    frags = [s for s in split_sentences(doc.text) if s[0].islower()]
    assert not frags
