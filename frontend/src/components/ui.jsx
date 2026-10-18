import React from "react";

export const Card = ({ title, children, className = "" }) => (
  <section className={`rounded-xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
    {title && <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">{title}</h2>}
    {children}
  </section>
);

const SEV = {
  high: "bg-red-100 text-red-800",
  medium: "bg-amber-100 text-amber-800",
  low: "bg-sky-100 text-sky-800",
  info: "bg-slate-100 text-slate-700",
};
export const Badge = ({ tone = "info", children }) => (
  <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${SEV[tone] || SEV.info}`}>{children}</span>
);

const SUP = {
  supported: ["bg-emerald-100 text-emerald-800", "Verified"],
  partially_supported: ["bg-amber-100 text-amber-800", "Partly verified"],
  contradicted: ["bg-red-100 text-red-800", "Contradicted"],
  not_verifiable: ["bg-slate-100 text-slate-700", "Not verifiable"],
};
export const Support = ({ value }) => {
  const [cls, label] = SUP[value] || SUP.not_verifiable;
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}>{label}</span>;
};

export const Disclaimer = ({ text }) => (
  <p role="note" className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900">
    {text || "Decision-support aid only. Not legal advice - have a qualified professional review."}
  </p>
);
