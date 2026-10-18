import React, { useState } from "react";
import { api } from "../api";

/** Clickable source reference that opens the exact passage the claim is based on. */
export default function EvidenceChip({ docId, chunkId, label }) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");

  const toggle = async () => {
    setOpen(!open);
    if (!data && !open) {
      try { setData(await api.chunk(docId, chunkId)); } catch (e) { setErr(e.message); }
    }
  };
  return (
    <span className="inline-block align-baseline">
      <button onClick={toggle} className="rounded bg-indigo-50 px-1.5 py-0.5 text-xs font-medium text-indigo-700 hover:bg-indigo-100"
        aria-expanded={open}>
        {label || data?.ref || chunkId}
      </button>
      {open && (
        <span className="mt-1 block max-w-prose rounded-md border border-indigo-100 bg-indigo-50/60 p-2 text-xs text-slate-700">
          {err ? `Could not load source (${err})` : data ? (<><b>{data.ref}</b> - {data.text}</>) : "Loading..."}
        </span>
      )}
    </span>
  );
}
