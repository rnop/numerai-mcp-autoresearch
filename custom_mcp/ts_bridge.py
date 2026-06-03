"""Bridge helpers for the TypeScript MCP server."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "autoresearch-src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from custom_mcp.server import (  # noqa: E402
    _compute_live_prediction_diagnostics,
    _compute_live_report_metrics,
    _current_live_prediction_era,
    _current_max_labeled_era,
    _load_meta,
)
from custom_mcp.site_builder import build_dashboard, build_report_html  # noqa: E402
from prepare import refresh_data  # noqa: E402


def _emit(payload: object) -> None:
    print(json.dumps(payload))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: ts_bridge.py <command> [args...]")

    command = sys.argv[1]

    if command == "current-max-era":
        _emit({"era": _current_max_labeled_era()})
        return

    if command == "refresh-validation":
        refresh_data(include_live=False, dataset_names=["validation"])
        _emit({"status": "ok"})
        return

    if command == "compute-live-report-metrics":
        meta_path = Path(sys.argv[2])
        meta = _load_meta(meta_path)
        _emit(_compute_live_report_metrics(meta_path, meta))
        return

    if command == "compute-live-diagnostics":
        meta_path = Path(sys.argv[2])
        meta = _load_meta(meta_path)
        _emit(_compute_live_prediction_diagnostics(meta_path, meta))
        return

    if command == "current-live-era":
        meta_path = Path(sys.argv[2])
        meta = _load_meta(meta_path)
        _emit({"era": _current_live_prediction_era(meta)})
        return

    if command == "render-report-html":
        title = sys.argv[2]
        markdown_path = Path(sys.argv[3])
        html_path = Path(sys.argv[4])
        markdown_text = markdown_path.read_text(encoding="utf-8")
        html_text = build_report_html(title, markdown_text)
        html_path.write_text(html_text, encoding="utf-8")
        _emit({"html_path": str(html_path)})
        return

    if command == "build-dashboard":
        dashboard_path = build_dashboard()
        _emit({"dashboard": str(dashboard_path)})
        return

    raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
