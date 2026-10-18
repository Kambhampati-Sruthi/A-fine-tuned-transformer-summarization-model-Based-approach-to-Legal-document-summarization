import React, { useState } from "react";
import { Badge, Card, Disclaimer, Support } from "./ui";
import EvidenceChip from "./EvidenceChip";
import { api } from "../api";

export default function RiskReview({ docId, data, reviewStatus, onReviewed }) {
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const order = { high: 0, medium: 1, low: 2, info: 3 };
  const risks = [...data.risks].sort((a, b) => order[a.severity] - order[b.severity]);

  async function decide(status) {
    setBusy(true);
    try { await api.review(docId, status, notes); onReviewed(status); } finally { setBusy(false); }
  }
  return (
    <div className="space-y-4">
      <Disclaimer text={data.disclaimer} />
      {risks.map((r) => (
        <Card key={r.rule_id}>
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={r.severity}>{r.severity.toUpperCase()}</Badge>
            <h3 className="font-medium">{r.title}</h3>
            {r.support && <Support value={r.support} />}
            <span className="text-xs text-slate-500">{r.certainty.replaceAll("_", " ")}</span>
          </div>
          <p className="mt-2 text-sm text-slate-700">{r.rationale}</p>
          <div className="mt-2 space-x-1">
            {r.evidence_ids.map((e) => <EvidenceChip key={e} docId={docId} chunkId={e} />)}
            {!r.evidence_ids.length && <span className="text-xs text-slate-500">No source passage - based on absence of a clause.</span>}
          </div>
        </Card>
      ))}
      {!risks.length && <Card><p className="text-sm text-slate-600">No potential issues were flagged by the rule set. This does not mean the contract is risk-free.</p></Card>}
      {data.unresolved.length > 0 && (
        <Card title="Unresolved / unverified">
          <ul className="list-disc pl-5 text-sm text-slate-700">{data.unresolved.map((u, i) => <li key={i}>{u}</li>)}</ul>
        </Card>
      )}
      <Card title={`Human review (status: ${reviewStatus})`}>
        <textarea value={notes} onChange={(e) => setNotes(e.target.value)} maxLength={2000} rows={3} aria-label="Reviewer notes"
          placeholder="Reviewer notes" className="w-full rounded-lg border border-slate-300 p-2 text-sm" />
        <div className="mt-2 flex gap-2">
          {["approved", "edited", "rejected"].map((s) => (
            <button key={s} disabled={busy} onClick={() => decide(s)}
              className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm capitalize hover:bg-slate-100 disabled:opacity-50">{s}</button>
          ))}
        </div>
      </Card>
    </div>
  );
}
