import React, { useMemo, useState } from "react";
import { Card } from "./ui";
import EvidenceChip from "./EvidenceChip";

export default function ClauseTable({ docId, clauses }) {
  const [filter, setFilter] = useState("All");
  const types = useMemo(() => ["All", ...new Set(clauses.map((c) => c.clause_type))].sort(), [clauses]);
  const rows = clauses.filter((c) => filter === "All" || c.clause_type === filter);
  return (
    <Card title="Detected clauses">
      <label className="mb-3 block text-sm">Type{" "}
        <select value={filter} onChange={(e) => setFilter(e.target.value)} className="ml-2 rounded border border-slate-300 p-1">
          {types.map((t) => <option key={t}>{t}</option>)}
        </select>
      </label>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase text-slate-500"><tr><th className="py-1 pr-3">Type</th><th className="pr-3">Source</th><th className="pr-3">Conf.</th><th>Snippet</th></tr></thead>
          <tbody>
            {rows.map((c, i) => (
              <tr key={i} className="border-t border-slate-100 align-top">
                <td className="py-2 pr-3 font-medium">{c.clause_type}</td>
                <td className="pr-3"><EvidenceChip docId={docId} chunkId={c.chunk_id} label={c.ref} /></td>
                <td className="pr-3">{c.confidence.toFixed(2)}</td>
                <td className="max-w-md text-slate-600">{c.snippet.slice(0, 160)}...</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
