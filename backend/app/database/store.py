"""Tiny SQLite persistence (stdlib). Swap for PostgreSQL/SQLAlchemy before multi-user deployment.
Stores metadata + report JSON only; raw contract text is never written to logs."""
from __future__ import annotations
import json, sqlite3, threading, time
from pathlib import Path
from typing import Optional

_lock = threading.Lock()


class Store:
    def __init__(self, database_url: str):
        self.path = database_url.replace("sqlite:///", "")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, owner TEXT, filename TEXT, stored_path TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, doc_id TEXT, status TEXT, stage TEXT, error TEXT, created REAL, updated REAL);
            CREATE TABLE IF NOT EXISTS reports(id TEXT PRIMARY KEY, doc_id TEXT, json TEXT, created REAL);
            """)

    def _c(self):
        c = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def _run(self, sql, args=(), one=False, write=False):
        with _lock:
            c = self._c()
            try:
                cur = c.execute(sql, args)
                if write:
                    c.commit()
                    return None
                rows = cur.fetchall()
                return (dict(rows[0]) if rows else None) if one else [dict(r) for r in rows]
            finally:
                c.close()

    def add_document(self, doc_id, owner, filename, path):
        self._run("INSERT INTO documents VALUES(?,?,?,?,?)", (doc_id, owner, filename, str(path), time.time()), write=True)

    def get_document(self, doc_id, owner="anon") -> Optional[dict]:
        return self._run("SELECT * FROM documents WHERE id=? AND owner=?", (doc_id, owner), one=True)

    def list_documents(self, owner="anon"):
        return self._run("""SELECT d.id, d.filename, d.created, j.status, j.id AS job_id FROM documents d
                            LEFT JOIN jobs j ON j.doc_id=d.id WHERE d.owner=? ORDER BY d.created DESC LIMIT 50""", (owner,))

    def add_job(self, job_id, doc_id):
        now = time.time()
        self._run("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", (job_id, doc_id, "queued", "queued", None, now, now), write=True)

    def update_job(self, job_id, status=None, stage=None, error=None):
        self._run("UPDATE jobs SET status=COALESCE(?,status), stage=COALESCE(?,stage), error=COALESCE(?,error), updated=? WHERE id=?",
                  (status, stage, error, time.time(), job_id), write=True)

    def get_job(self, job_id):
        return self._run("SELECT * FROM jobs WHERE id=?", (job_id,), one=True)

    def save_report(self, report_id, doc_id, report_json):
        self._run("INSERT OR REPLACE INTO reports VALUES(?,?,?,?)", (report_id, doc_id, report_json, time.time()), write=True)

    def get_report(self, report_id=None, doc_id=None):
        if report_id:
            return self._run("SELECT * FROM reports WHERE id=?", (report_id,), one=True)
        return self._run("SELECT * FROM reports WHERE doc_id=? ORDER BY created DESC LIMIT 1", (doc_id,), one=True)

    def delete_document(self, doc_id):
        self._run("DELETE FROM reports WHERE doc_id=?", (doc_id,), write=True)
        self._run("DELETE FROM jobs WHERE doc_id=?", (doc_id,), write=True)
        self._run("DELETE FROM documents WHERE id=?", (doc_id,), write=True)

    def expired_documents(self, older_than_s: float):
        return self._run("SELECT id, stored_path FROM documents WHERE created < ?", (time.time() - older_than_s,))
