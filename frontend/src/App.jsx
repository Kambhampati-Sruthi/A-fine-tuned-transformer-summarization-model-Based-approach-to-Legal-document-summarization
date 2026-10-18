import React, { useEffect, useState } from "react";
import { api } from "./api";
import UploadPanel from "./components/UploadPanel";
import JobProgress from "./components/JobProgress";
import SummaryView from "./components/SummaryView";
import ClauseTable from "./components/ClauseTable";
import RiskReview from "./components/RiskReview";
import ContractQA from "./components/ContractQA";
import { Card } from "./components/ui";

const TABS = ["Summary", "Clauses", "Risks", "Q&A"];

export default function App() {
  const [docs, setDocs] = useState([]);
  const [active, setActive] = useState(null);   // {docId, jobId, name, ready}
  const [tab, setTab] = useState("Summary");
  const [data, setData] = useState({});
  const [review, setReview] = useState("pending");
  const [error, setError] = useState("");

  const refresh = () => api.list().then(setDocs).catch((e) => setError(e.message));
  useEffect(() => { refresh(); }, []);

  async function load(docId) {
    setError("");
    try {
      const [summary, clauses, risks] = await Promise.all([api.summary(docId), api.clauses(docId), api.risks(docId)]);
      setData({ summary, clauses, risks });
      setReview("pending");
      setActive((a) => ({ ...a, ready: true }));
      refresh();
    } catch (e) { setError(e.message); }
  }

  const open = (d) => {
    setActive({ docId: d.id, jobId: d.job_id, name: d.filename, ready: false });
    setTab("Summary");
    if (d.status === "completed") load(d.id);
  };

  async function remove(d) {
    if (!confirm(`Delete "${d.filename}" and its analysis? This cannot be undone.`)) return;
    await api.remove(d.id);
    if (active?.docId === d.id) setActive(null);
    refresh();
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold">Legal Document Intelligence</h1>
        <p className="text-sm text-slate-600">Source-grounded summaries, clause detection, potential-issue review and contract Q&A.</p>
      </header>
      {error && <p role="alert" className="mb-4 rounded bg-red-50 p-3 text-sm text-red-700">{error}</p>}
      <div className="grid gap-6 md:grid-cols-[18rem_1fr]">
        <aside className="space-y-4">
          <UploadPanel onStarted={(r, name) => { setActive({ docId: r.document_id, jobId: r.job_id, name, ready: false }); setTab("Summary"); refresh(); }} />
          <Card title="Recent">
            <ul className="space-y-1 text-sm">
              {docs.map((d) => (
                <li key={d.id} className="flex items-center justify-between gap-2">
                  <button onClick={() => open(d)} className="truncate text-left text-indigo-700 hover:underline" title={d.filename}>{d.filename}</button>
                  <span className="flex items-center gap-2 text-xs text-slate-500">{d.status}
                    <button onClick={() => remove(d)} aria-label={`Delete ${d.filename}`} className="text-red-600">x</button></span>
                </li>
              ))}
              {!docs.length && <li className="text-slate-500">No analyses yet.</li>}
            </ul>
          </Card>
        </aside>
        <main>
          {!active && <Card><p className="text-slate-600">Upload a contract or open a recent analysis.</p></Card>}
          {active && !active.ready && active.jobId && <JobProgress jobId={active.jobId} onDone={() => load(active.docId)} />}
          {active?.ready && data.summary && (
            <>
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <h2 className="truncate text-lg font-medium">{active.name}</h2>
                <ExportButtons docId={active.docId} />
              </div>
              <nav className="mb-4 flex gap-1 border-b border-slate-200" role="tablist">
                {TABS.map((t) => (
                  <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
                    className={`px-4 py-2 text-sm ${tab === t ? "border-b-2 border-indigo-600 font-medium text-indigo-700" : "text-slate-600"}`}>{t}</button>
                ))}
              </nav>
              {tab === "Summary" && <SummaryView docId={active.docId} data={data.summary} />}
              {tab === "Clauses" && <ClauseTable docId={active.docId} clauses={data.clauses} />}
              {tab === "Risks" && <RiskReview docId={active.docId} data={data.risks} reviewStatus={review} onReviewed={setReview} />}
              {tab === "Q&A" && <ContractQA docId={active.docId} />}
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function ExportButtons({ docId }) {
  const [err, setErr] = useState("");
  async function go(fmt) {
    setErr("");
    try {
      await api.downloadByDoc(docId, fmt);
    } catch (e) { setErr(e.message); }
  }
  return (
    <div className="flex items-center gap-2">
      {["pdf", "docx", "md"].map((f) => (
        <button key={f} onClick={() => go(f)} className="rounded-lg border border-slate-300 px-3 py-1 text-sm uppercase hover:bg-slate-100">{f}</button>
      ))}
      {err && <span role="alert" className="text-xs text-red-700">{err}</span>}
    </div>
  );
}
