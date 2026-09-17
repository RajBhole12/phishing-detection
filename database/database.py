"""
database.py
------------
SQLite persistence layer for scan history. Uses parameterized queries
throughout to avoid SQL injection, and a context-managed connection
per call to keep things simple and thread-safe for Flask's dev server.

EXTENDED (additive only) to support the new Email and Message phishing
detectors alongside the existing, untouched Website/URL scan history:
  - a new `scan_type` column ("Website" / "Email" / "Message"), added via
    a safe schema migration so existing rows/behavior are unaffected
    (existing rows automatically default to "Website")
  - a new `risk_factors` column (JSON-encoded) so email/message risk
    factors can be stored without bloating the schema with new tables
All existing function signatures remain backward compatible: every new
parameter has a default that reproduces the exact old behavior.
"""

import os
import json
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "scans.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn, table: str, column: str, coltype: str):
    """Idempotent ALTER TABLE ADD COLUMN, safe to call on every startup."""
    existing_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing_cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            prediction TEXT NOT NULL,
            risk_score REAL NOT NULL,
            confidence REAL NOT NULL,
            model_used TEXT NOT NULL,
            scanned_at TEXT NOT NULL
        )
    """)
    # Schema migration for the new Email/Message features -- existing rows
    # (and any pre-existing scans.db file) automatically get
    # scan_type='Website', preserving all current Website/URL behavior.
    _add_column_if_missing(conn, "scans", "scan_type", "TEXT NOT NULL DEFAULT 'Website'")
    _add_column_if_missing(conn, "scans", "risk_factors", "TEXT")
    conn.commit()
    conn.close()


def insert_scan(url: str, prediction: str, risk_score: float, confidence: float,
                 model_used: str, scan_type: str = "Website", risk_factors: list = None):
    """
    Unchanged for existing callers: url, prediction, risk_score, confidence,
    model_used are the same positional signature as before. scan_type and
    risk_factors are new, optional, keyword-only-in-practice parameters
    that default to the exact previous behavior ("Website", no factors).
    """
    conn = get_connection()
    conn.execute(
        "INSERT INTO scans (url, prediction, risk_score, confidence, model_used, "
        "scanned_at, scan_type, risk_factors) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (url, prediction, risk_score, confidence, model_used,
         datetime.now(timezone.utc).isoformat(), scan_type,
         json.dumps(risk_factors) if risk_factors is not None else None)
    )
    conn.commit()
    conn.close()


def get_history(search: str = None, prediction_filter: str = None,
                 sort_by: str = "scanned_at", sort_dir: str = "desc", limit: int = 500,
                 scan_type_filter: str = None):
    allowed_sort_cols = {"scanned_at", "risk_score", "confidence", "url", "prediction"}
    if sort_by not in allowed_sort_cols:
        sort_by = "scanned_at"
    sort_dir = "DESC" if sort_dir.lower() != "asc" else "ASC"

    query = "SELECT * FROM scans WHERE 1=1"
    params = []
    if search:
        query += " AND url LIKE ?"
        params.append(f"%{search}%")
    if prediction_filter and prediction_filter.lower() != "all":
        query += " AND prediction = ?"
        params.append(prediction_filter)
    if scan_type_filter and scan_type_filter.lower() != "all":
        query += " AND scan_type = ?"
        params.append(scan_type_filter)

    query += f" ORDER BY {sort_by} {sort_dir} LIMIT ?"
    params.append(limit)

    conn = get_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()

    results = [dict(r) for r in rows]
    for r in results:
        if r.get("risk_factors"):
            try:
                r["risk_factors"] = json.loads(r["risk_factors"])
            except (TypeError, ValueError):
                pass
    return results


def clear_history():
    conn = get_connection()
    conn.execute("DELETE FROM scans")
    conn.commit()
    conn.close()


def get_statistics():
    conn = get_connection()

    def count(prediction=None, scan_type=None):
        query = "SELECT COUNT(*) c FROM scans WHERE 1=1"
        params = []
        if prediction:
            query += " AND prediction = ?"
            params.append(prediction)
        if scan_type:
            query += " AND scan_type = ?"
            params.append(scan_type)
        return conn.execute(query, params).fetchone()["c"]

    # --- Existing Website/URL statistics: scoped to scan_type='Website' so
    # the numbers are IDENTICAL to before for any pre-existing database
    # (all prior rows are migrated to scan_type='Website' automatically). ---
    total = count(scan_type="Website")
    phishing = count("Phishing", "Website")
    suspicious = count("Suspicious", "Website")
    legitimate = count("Legitimate", "Website")
    recent = conn.execute(
        "SELECT * FROM scans WHERE scan_type='Website' ORDER BY scanned_at DESC LIMIT 10"
    ).fetchall()
    activity = conn.execute("""
        SELECT substr(scanned_at, 1, 10) AS day, COUNT(*) AS count
        FROM scans WHERE scan_type='Website' GROUP BY day ORDER BY day DESC LIMIT 14
    """).fetchall()

    # --- New: separate Email / Message statistics (additive) ---
    emails_analyzed = count(scan_type="Email")
    phishing_emails = count("Phishing", "Email")
    suspicious_emails = count("Suspicious", "Email")
    legitimate_emails = count("Legitimate", "Email")

    messages_analyzed = count(scan_type="Message")
    phishing_messages = count("Phishing", "Message")
    suspicious_messages = count("Suspicious", "Message")
    legitimate_messages = count("Legitimate", "Message")

    conn.close()

    return {
        # Unchanged keys -- existing dashboard code keeps working as-is.
        "total_scans": total,
        "phishing_detected": phishing,
        "suspicious_urls": suspicious,
        "legitimate_urls": legitimate,
        "recent_scans": [dict(r) for r in recent],
        "activity_over_time": [dict(r) for r in activity][::-1],
        # New keys -- ignored by any code that doesn't know about them yet.
        "emails_analyzed": emails_analyzed,
        "phishing_emails": phishing_emails,
        "suspicious_emails": suspicious_emails,
        "legitimate_emails": legitimate_emails,
        "messages_analyzed": messages_analyzed,
        "phishing_messages": phishing_messages,
        "suspicious_messages": suspicious_messages,
        "legitimate_messages": legitimate_messages,
    }
