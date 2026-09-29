"""The gates that stop a bad model from reaching the tournament."""
import json
from types import SimpleNamespace

import pytest

from pipeline import weekly


# --- CLI exit codes: 0 ok, 1 error, 2 blocked by a gate -------------------

@pytest.mark.parametrize(
    ("command", "payload", "expected"),
    [
        ("qa", {"status": "pass", "ready_for_submission": True}, 0),
        ("qa", {"status": "warn", "ready_for_submission": True}, 0),
        ("qa", {"status": "fail", "ready_for_submission": False}, 2),
        ("qa", {"status": "error"}, 2),
        ("status", {"status": "completed"}, 0),
        ("status", {"status": "failed"}, 1),
        ("status", {"status": "skipped"}, 2),
        ("summary", {"error": "No metadata JSON found"}, 1),
    ],
)
def test_exit_code(command, payload, expected):
    assert weekly._exit_code(command, payload) == expected


def test_next_step_after_qa():
    assert weekly._next_for_qa({"status": "pass"}).startswith("QA passed")
    assert weekly._next_for_qa({"status": "fail"}).startswith("STOP")
    assert weekly._next_for_qa({"status": "error"}).startswith("STOP")


def test_next_step_after_retrain():
    assert "qa" in weekly._next_for_retrain({"status": "completed"})
    assert weekly._next_for_retrain({"status": "failed"}).startswith("STOP")
    regression = weekly._next_for_retrain({"status": "skipped", "era_regression": True})
    assert regression.startswith("STOP") and "Do not upload" in regression


# --- era-window guard in the retrain worker -------------------------------

@pytest.fixture
def worker(monkeypatch, tmp_path):
    """Run the retrain worker with data refresh, metadata and training stubbed out."""
    statuses: list[dict] = []
    trained: list[list[str]] = []
    meta_path = tmp_path / "live_meta.json"
    meta_path.write_text(json.dumps({"era_window_start": "0700", "era_window_end": "1218"}))

    monkeypatch.setattr(weekly, "refresh_data", lambda **kwargs: None)
    monkeypatch.setattr(weekly, "_write_retrain_status", statuses.append)
    monkeypatch.setattr(weekly, "_sorted_metas", lambda: [meta_path])

    def fake_run(cmd, **kwargs):
        trained.append(cmd)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(weekly.subprocess, "run", fake_run)

    def run(current_max_era, force=False):
        monkeypatch.setattr(weekly, "_current_max_labeled_era", lambda: current_max_era)
        code = weekly._retrain_worker(force=force)
        return code, statuses[-1], trained

    return run


def test_guard_skips_when_no_new_era(worker):
    code, status, trained = worker("1218")
    assert code == 0
    assert status["status"] == "skipped"
    assert status["era_regression"] is False
    assert trained == []


def test_guard_refuses_backwards_era_window(worker):
    _, status, trained = worker("1211")
    assert status["status"] == "skipped"
    assert status["era_regression"] is True
    assert "BACKWARDS" in status["reason"]
    assert trained == []


def test_guard_allows_new_era(worker):
    _, status, trained = worker("1219")
    assert len(trained) == 1
    assert status["status"] != "skipped"


def test_force_bypasses_guard(worker):
    _, _, trained = worker("1211", force=True)
    assert len(trained) == 1
