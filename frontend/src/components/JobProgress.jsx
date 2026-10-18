import React, { useEffect, useState } from "react";
import { api } from "../api";
import { Card } from "./ui";

const STAGES = ["ingest", "chunk", "summarize", "index", "analyze", "risk", "verify", "revise", "draft"];
const LABEL = { ingest: "Extracting text", chunk: "Splitting into clauses", summarize: "Summarizing", index: "Building search index",
  analyze: "Finding key terms", risk: "Checking potential issues", verify: "Verifying claims", revise: "Re-checking evidence", draft: "Drafting report" };

export default function JobProgress({ jobId, onDone }) {
  const [job, setJob] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    let alive = true, timer;
    const tick = async () => {
      try {
        const j = await api.job(jobId);
        if (!alive) return;
        setJob(j);
        if (j.status === "completed") return onDone();
        if (j.status !== "failed") timer = setTimeout(tick, 1000);
      } catch (e) { alive && setErr(e.message); }
    };
    tick();
    return () => { alive = false; clearTimeout(timer); };
  }, [jobId]);

  if (err) return <Card><p role="alert" className="text-red-700">{err}</p></Card>;
  const idx = job ? Math.max(0, STAGES.indexOf(job.stage)) : 0;
  return (
    <Card title="Processing">
      {job?.status === "failed" ? (
        <p role="alert" className="text-sm text-red-700">Analysis failed: {job.error}</p>
      ) : (
        <>
          <div className="h-2 overflow-hidden rounded bg-slate-200">
            <div className="h-full bg-indigo-600 transition-all" style={{ width: `${((idx + 1) / STAGES.length) * 100}%` }} />
          </div>
          <p className="mt-2 text-sm text-slate-600">{LABEL[job?.stage] || "Queued"}...</p>
        </>
      )}
    </Card>
  );
}
