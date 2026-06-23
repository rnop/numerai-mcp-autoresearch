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

## Environment and how to actually run it

**Interpreter.** Everything here must run under the **`numerai_rag_env`** conda interpreter
(`C:\Users\nopro\anaconda3\envs\numerai_rag_env\python.exe`, Python 3.11 — it has the GPU
XGBoost, `numerapi`, `fastmcp`, and `requests`). Set `NUMERAI_PYTHON` to it before running.
`make_submission.py` has a hard env guard that aborts on any other interpreter, and the
submission pickle is Python 3.11 bytecode.

**When the MCP servers aren't connected (the normal case here).** In this repo the
`numerai-weekly` and official `numerai` MCP servers are usually **not** wired up as live
session tools, so you can't call `mcp__numerai-weekly__*` / `mcp__numerai__*` directly.
Drive the same code locally instead — the results are identical:

- **Retrain (steps 1–2):** run the background worker entrypoint
  `python custom_mcp/server.py --weekly-worker` (add `--force` only when explicitly asked).
  It writes status to `docs/retrain_latest_status.json` and a log to `docs/retrain_latest.log`;
  poll those instead of `check_retrain_status()`.
- **QA / summary / features / report (steps 3–5):** import `custom_mcp.server` and call the
  underlying functions in-process (`check_live_predictions`, `get_training_summary`,
  `compare_weekly_features`, `generate_weekly_report` — unwrap `.fn` on the `@mcp.tool()`
  objects if needed).
- **Upload (step 6):** run `python custom_mcp/upload_to_tailspin.py`, which bridges to the
  official Numerai MCP over HTTP via `fastmcp.Client` and performs the full handoff. See step 6.

If the MCP servers *are* connected, prefer the named tools above; the local entrypoints are
the fallback, not a different procedure.

## Trigger phrasing — does "and submission" mean upload?

- **"run weekly retrain"** → run steps 1–5 and **stop before upload**; report that the build
  is QA-passed and ready, and offer to submit.
- **"run weekly retrain and submission"** (or "submit", "run the weekly pipeline end to end")
  → run steps 1–6 including the live upload to TAILSPIN. No extra confirmation needed beyond
  this phrasing; the QA gate (step 3) is still a hard stop.

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

### 6. Upload  (only when the request includes submission)
Skip this step for a retrain-only request (see *Trigger phrasing* above). Run it only after a
passing (or consciously accepted `warn`) QA gate.

**Run the helper:** `python custom_mcp/upload_to_tailspin.py` (with `NUMERAI_PYTHON` /
`numerai_rag_env`). With no argument it uploads the newest `submissions/*_meta.json` build;
pass an explicit `.pkl` path to override. It performs the full official-Numerai handoff so you
don't have to orchestrate it by hand:

1. resolves the **TAILSPIN** model id by name (tournament 8),
2. `get_upload_auth` → presigned S3 URL,
3. PUTs the pickle bytes to that URL (no `Content-Type` header — the signed type is empty),
4. `create` with the Python 3.11 docker image,
5. polls `list` until `validationStatus: validated`,
6. `assign`s the validated pickle as the active TAILSPIN model.

It prints each step and exits non-zero on failure; relay the final `SUCCESS`/pickle id to the
user. Credentials are read from `.env` (`NUMERAI_MCP_AUTH` for the connection header,
`API_TOKEN` = `PUBLIC_ID$SECRET_KEY` for the `apiToken` param) — never put them on the command
line or in the repo.

**Always TAILSPIN — never ANGOSTURA or PIXELATED.** The helper hard-codes the slot by name so
it can't drift; don't repoint it from a report header or any inferred slot. Uploading to the
wrong slot is a live-stakes mistake.

**Pickle/runtime caveat (baked into the helper, don't override).** The submission pickle is
**Python 3.11** bytecode and must use the Python 3.11 docker image
(`4d39918c-a82b-42ea-8dc7-ed5a30e676c5`). Numerai's default 3.12 fails at load with
`unknown opcode 0`.

**If the official `numerai` MCP *is* connected as a session tool,** you may instead call its
`upload_model` operations directly (`get_upload_auth` → PUT → `create` → `list` → `assign`)
with the same TAILSPIN id and 3.11 image — the helper just automates exactly that.

## Reporting back to the user

Close the loop with a short summary: the era window that was trained, the QA verdict,
notable feature changes, the report path, and the upload result. If the run was skipped (no
new data) or failed QA, lead with that — it's the most important thing for the user to know.
