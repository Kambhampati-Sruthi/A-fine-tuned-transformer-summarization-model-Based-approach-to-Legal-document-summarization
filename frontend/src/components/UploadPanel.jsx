import React, { useRef, useState } from "react";
import { api } from "../api";
import { Card } from "./ui";

const MAX_MB = 20;

export default function UploadPanel({ onStarted }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  const input = useRef();

  async function handle(file) {
    setError("");
    if (!file) return;
    if (!/\.(pdf|docx)$/i.test(file.name)) return setError("Only PDF and DOCX files are supported.");
    if (file.size > MAX_MB * 1024 * 1024) return setError(`File is larger than ${MAX_MB} MB.`);
    setBusy(true);
    try { onStarted(await api.upload(file), file.name); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  return (
    <Card title="New analysis">
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]); }}
        className={`flex flex-col items-center gap-2 rounded-lg border-2 border-dashed p-8 text-center transition ${drag ? "border-indigo-400 bg-indigo-50" : "border-slate-300"}`}
      >
        <p className="text-sm text-slate-600">Drop a contract here (text-based PDF or DOCX, up to {MAX_MB} MB)</p>
        <button disabled={busy} onClick={() => input.current.click()}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50">
          {busy ? "Uploading..." : "Choose file"}
        </button>
        <input ref={input} type="file" hidden accept=".pdf,.docx" onChange={(e) => handle(e.target.files[0])} />
        <p className="text-xs text-slate-500">Use public or synthetic contracts unless you have permission to process the document.</p>
      </div>
      {error && <p role="alert" className="mt-3 text-sm text-red-700">{error}</p>}
    </Card>
  );
}
