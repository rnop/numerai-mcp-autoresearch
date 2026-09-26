"""
Shared ordering for submissions/*_meta.json.

`metas[-1]` decides which model the weekly tools QA, report on, and upload, so
"latest" has to be trustworthy. File mtime is not: the report and diagnostics
helpers rewrite meta files in place (bumping their mtime), and a fresh clone or
a cloud-drive resync rewrites every mtime at once. Era numbers are not a
substitute either — the 2026-08-09 ender_60 switch retreated the labeled
boundary from 1225 to 1218, so a later build can carry a *lower* era window.

Order by the build timestamp recorded inside each JSON, and fall back to mtime
only when that is missing or unparseable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def load_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_timestamp(meta: dict) -> float | None:
    """Epoch seconds from `built_at` (full ISO) or `built_date` (date-only)."""
    raw = meta.get("built_at") or meta.get("built_date")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        stamp = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def meta_sort_key(path: Path) -> tuple[float, str]:
    try:
        stamp = _build_timestamp(load_meta(path))
    except (OSError, ValueError):
        stamp = None
    if stamp is None:
        try:
            stamp = path.stat().st_mtime
        except OSError:
            stamp = 0.0
    # Builds from the same day tie on the timestamp; the filename carries the
    # era window, which is the right tie-break within a single day.
    return (stamp, path.name)


def sorted_metas(submissions_dir: Path) -> list[Path]:
    """All *_meta.json under `submissions_dir`, oldest -> newest by build time."""
    return sorted(submissions_dir.glob("*_meta.json"), key=meta_sort_key)
