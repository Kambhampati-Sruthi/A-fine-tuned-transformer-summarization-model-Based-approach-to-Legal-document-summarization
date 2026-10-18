import React, { useState } from "react";
import { api } from "../api";
import { Card, Support } from "./ui";
import EvidenceChip from "./EvidenceChip";

const SUGGEST = ["What is the effective date and duration?", "What conditions allow either party to terminate?",
  "What notice period is specified?", "Does the agreement automatically renew?",
  "What payment deadlines and late fees are stated?", "Which confidentiality obligations continue after termination?"];

export default function ContractQA({ docId }) {
  const [q, setQ] = useState("");
  const [log, setLog] = useState([]);
  const [busy, setBusy] = useState(false);

  async function ask(text) {
    if (text.trim().length < 3) return;
    setBusy(true);
    try {
      const a = await api.ask(docId, text);
      setLog((l) => [a, ...l]);
      setQ("");
    } catch (e) {
      setLog((l) => [{ question: text, answer: `Error: ${e.message}`, answer_type: "absent", evidence: [], support: "not_verifiable" }, ...l]);
    } finally { setBusy(false); }
  }
  return (
    <div className="space-y-4">
      <Card title="Ask about this contract">
        <div className="flex gap-2">
          <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask(q)} maxLength={500}
            aria-label="Question" placeholder="e.g. What is the notice period?" className="flex-1 rounded-lg border border-slate-300 p-2 text-sm" />
          <button disabled={busy} onClick={() => ask(q)} className="rounded-lg bg-indigo-600 px-4 text-sm font-medium text-white disabled:opacity-50">
            {busy ? "..." : "Ask"}
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {SUGGEST.map((s) => <button key={s} onClick={() => ask(s)} className="rounded-full border border-slate-200 px-2 py-1 text-xs hover:bg-slate-100">{s}</button>)}
        </div>
      </Card>
      {log.map((a, i) => (
        <Card key={i}>
          <p className="text-sm font-medium">{a.question}</p>
          <p className="mt-1 text-sm">{a.answer}</p>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded bg-slate-100 px-1.5 py-0.5">{a.answer_type}</span>
            <Support value={a.support} />
            {a.evidence.map((e) => <EvidenceChip key={e.chunk_id} docId={docId} chunkId={e.chunk_id} label={e.ref} />)}
          </div>
          {a.note && <p className="mt-2 text-xs text-amber-800">{a.note}</p>}
        </Card>
      ))}
    </div>
  );
}
