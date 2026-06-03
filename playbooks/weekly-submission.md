# Weekly Numerai Submission — Playbook

> **Canonical, agent-neutral procedure.** This file is the single source of truth for the
> weekly live submission loop. Claude reads it via the `weekly-submission` skill; Codex and
> other agents read it via `AGENTS.md`. Edit the procedure here, not in the agent-specific
> wrappers.

The operational loop that keeps the live models ([ANGOSTURA](https://numer.ai/angostura),
[PIXELATED](https://numer.ai/pixelated), [TAILSPIN](https://numer.ai/tailspin)) current.

**What this does and does not do.** This loop takes the *already-chosen champion strategy* —
the config baked into `custom_mcp/make_submission.py` — and re-fits it on the latest era of
Numerai data, then submits. It does **not** search for a better model, change features by
hand, or tune hyperparameters. Feature *selection* still happens each week (the strategy
picks top-K features dynamically from the trailing window), but the *strategy* is fixed. If
the goal is to change the strategy itself — new target, new hyperparameters, new feature
pool — that's a research campaign; stop and switch to the autoresearch playbook
(`playbooks/autoresearch.md`), which ends by promoting a new champion into
`make_submission.py`.

## Servers and tools

Two MCP servers do the work. Configure them from `.mcp.example.json`.

- **`numerai-weekly`** (custom, this repo) — the pipeline. Python (`custom_mcp/server.py`)
  and TypeScript (`custom_mcp/server.js`) implementations are equivalent; use whichever is
  connected.
- **`numerai`** (official, HTTP) — tournament operations, used only for the final upload.

The custom server exposes these tools, in the order you'll generally call them:

| Tool | Purpose |
|------|---------|
| `run_weekly_retrain(force=False)` | Download latest data and retrain the champion in the **background**. Returns immediately. |
| `check_retrain_status()` | Poll the background job. Returns running / completed / failed plus a log tail. |
| `get_training_summary()` | Structured snapshot of the latest build's config (era window, top-k, target, etc.). |
| `check_live_predictions()` | Score the live split and emit a pass / warn / fail QA verdict + distribution artifacts. |
| `compare_weekly_features()` | Diff this week's selected features vs last week, grouped by feature family. |
| `generate_weekly_report()` | Write `docs/YYYY-WW_weekly_report.md` and `.html`. |

## The weekly loop

Run these in order. Each step gates the next — don't upload a build that failed QA.

### 1. Retrain
Call `run_weekly_retrain()`. It refreshes the validation data first, then **guards on the
era window**: if no new era has landed since the last submission, it returns
`status="skipped"` and does nothing. That's the correct, expected outcome when you run
before Numerai has published new data — report it and stop; there's nothing to submit.
Only pass `force=True` if explicitly asked to rebuild on unchanged data (rare — e.g.
recovering from a corrupted pickle).

The job runs in the background and returns a `pid` and `log_path` immediately. It does
**not** block.

### 2. Monitor
Poll `check_retrain_status()` until `status` is `completed` or `failed`. The retrain trains
XGBoost on GPU over ~140 eras and typically takes a few minutes. Space your polls out rather
than hammering — the tool returns a log tail each time so you can see progress. If it comes
back `failed`, read the `log_tail` for the stack trace before deciding whether to retry or
surface the error to the user.

### 3. QA the predictions  ← gate
Call `check_live_predictions()`. This scores the freshly packaged model on the live split
and returns a **pass / warn / fail** verdict plus distribution stats and artifact paths.

- **pass** — proceed.
- **warn** — proceed, but call out what's off (e.g. a skewed prediction distribution) so the
  user can decide.
- **fail** — **do not upload.** Something is wrong with the build. Read the diagnostics,
  check the retrain log, and surface the problem instead of submitting bad predictions. A bad
  live submission costs a tournament week, so this gate is not optional.

### 4. Review config and feature drift
Call `get_training_summary()` for the build's configuration, then `compare_weekly_features()`
to see which features rotated in and out versus last week. Large week-over-week feature churn
can be a sign of data drift worth flagging to the user. (On the very first run there's
nothing to diff against — the tool says so; that's fine.)

### 5. Report
Call `generate_weekly_report()`. It writes the markdown and HTML report into `docs/` for the
current ISO week and returns the content. The dashboard at `docs/index.html` links to these
automatically.

### 6. Upload
Only after a passing (or consciously accepted `warn`) QA gate: use the **official `numerai`**
MCP server to upload the packaged model artifact from `submissions/` (the `.pkl` named in
`get_training_summary()`'s `pkl_path`).

**Always upload to the TAILSPIN model slot.** This weekly artifact is the TAILSPIN live
model — upload it there, not ANGOSTURA or PIXELATED. Uploading to the wrong slot is a
live-stakes mistake, so treat TAILSPIN as fixed for this loop and do not guess from the
report header or infer another slot. Confirm the official server's upload tool
name/signature at call time, since that server is configured separately from this repo.

**Pickle/runtime caveat (don't skip).** The submission pickle contains **Python 3.11**
bytecode. It must be uploaded with the Python 3.11 docker image
(`4d39918c-a82b-42ea-8dc7-ed5a30e676c5`). Numerai's default is 3.12, which fails at load
time with `unknown opcode 0`. Confirm the 3.11 image is selected before uploading.

## Reporting back to the user

Close the loop with a short summary: the era window that was trained, the QA verdict,
notable feature changes, the report path, and the upload result. If the run was skipped (no
new data) or failed QA, lead with that — it's the most important thing for the user to know.
