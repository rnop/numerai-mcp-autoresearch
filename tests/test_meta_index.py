"""`sorted_metas()[-1]` picks the model that gets QA'd and uploaded."""
import json
import os

from pipeline.meta_index import sorted_metas


def _write(path, meta, mtime):
    path.write_text(json.dumps(meta))
    os.utime(path, (mtime, mtime))


def test_orders_by_recorded_build_time_not_mtime(tmp_path):
    # Newer build carries a lower era window and an older mtime (e.g. after a resync).
    _write(tmp_path / "a_1225_meta.json", {"built_at": "2026-08-02T10:00:00Z"}, mtime=2_000_000_000)
    _write(tmp_path / "b_1218_meta.json", {"built_at": "2026-08-09T10:00:00Z"}, mtime=1_000_000_000)
    assert sorted_metas(tmp_path)[-1].name == "b_1218_meta.json"


def test_falls_back_to_mtime_without_timestamp(tmp_path):
    _write(tmp_path / "old_meta.json", {}, mtime=1_000_000_000)
    _write(tmp_path / "new_meta.json", {"built_at": "not a date"}, mtime=1_500_000_000)
    assert sorted_metas(tmp_path)[-1].name == "new_meta.json"


def test_ignores_non_meta_files(tmp_path):
    _write(tmp_path / "x_meta.json", {"built_date": "2026-09-01"}, mtime=1)
    (tmp_path / "x.pkl").write_bytes(b"")
    assert [p.name for p in sorted_metas(tmp_path)] == ["x_meta.json"]
