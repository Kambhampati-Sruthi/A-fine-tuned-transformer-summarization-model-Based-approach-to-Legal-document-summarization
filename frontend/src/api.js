const BASE = import.meta.env.VITE_API_URL || "";
const KEY = import.meta.env.VITE_API_KEY || "";
const headers = (extra = {}) => (KEY ? { "X-API-Key": KEY, ...extra } : extra);

async function req(path, opts = {}) {
  const res = await fetch(BASE + path, { ...opts, headers: headers(opts.headers) });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {}
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  upload(file) {
    const fd = new FormData();
    fd.append("file", file);
    return req("/api/documents", { method: "POST", body: fd });
  },
  list: () => req("/api/documents"),
  job: (id) => req(`/api/jobs/${id}`),
  summary: (d) => req(`/api/documents/${d}/summary`),
  clauses: (d) => req(`/api/documents/${d}/clauses`),
  risks: (d) => req(`/api/documents/${d}/risks`),
  chunk: (d, c) => req(`/api/documents/${d}/chunks/${c}`),
  ask: (d, question) =>
    req(`/api/documents/${d}/questions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
  review: (d, status, notes) =>
    req(`/api/documents/${d}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, notes }),
    }),
  remove: (d) => req(`/api/documents/${d}`, { method: "DELETE" }),
  async downloadByDoc(docId, format) {
    const res = await fetch(`${BASE}/api/documents/${docId}/export?format=${format}`, { headers: headers() });
    if (!res.ok) throw new Error("Download failed");
    const url = URL.createObjectURL(await res.blob());
    Object.assign(document.createElement("a"), { href: url, download: `report.${format}` }).click();
    URL.revokeObjectURL(url);
  },
};
