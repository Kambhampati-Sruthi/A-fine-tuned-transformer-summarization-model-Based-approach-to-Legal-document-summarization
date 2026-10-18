import React from "react";
import { Card, Disclaimer } from "./ui";
import EvidenceChip from "./EvidenceChip";

export default function SummaryView({ docId, data }) {
  const { summary, key_facts, warnings, disclaimer } = data;
  return (
    <div className="space-y-4">
      <Disclaimer text={disclaimer} />
      {warnings?.length > 0 && (
        <Card title="Warnings"><ul className="list-disc pl-5 text-sm text-amber-800">{warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></Card>
      )}
      <Card title="Executive summary">
        <p className="leading-relaxed">{summary.executive_summary}</p>
        <p className="mt-2 text-xs text-slate-500">
          {summary.mode} / {summary.model} - clause coverage {(summary.coverage * 100).toFixed(0)}%
        </p>
      </Card>
      <Card title="Key facts">
        <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
          {key_facts.map((f, i) => (
            <div key={i} className="border-b border-slate-100 pb-1">
              <dt className="text-xs text-slate-500">{f.field}</dt>
              <dd className="text-sm">{f.value} {f.evidence_ids.map((e) => <EvidenceChip key={e} docId={docId} chunkId={e} />)}</dd>
            </div>
          ))}
          {!key_facts.length && <p className="text-sm text-slate-500">No key facts detected.</p>}
        </dl>
      </Card>
      <Card title="Section summaries">
        <ul className="space-y-3">
          {summary.sections.map((s, i) => (
            <li key={i} className="text-sm">
              <b>{s.heading}</b> - {s.summary}{" "}
              {s.evidence_ids.slice(0, 3).map((e) => <EvidenceChip key={e} docId={docId} chunkId={e} />)}
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
