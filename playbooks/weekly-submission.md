# Weekly Numerai Submission — Playbook

> **Canonical, agent-neutral procedure.** This file is the single source of truth for the
> weekly live submission loop. Claude reads it via the `weekly-submission` skill; Codex and
> other agents read it via `AGENTS.md`. Edit the procedure here, not in the agent-specific
> wrappers.

The operational loop that keeps the live models ([TAILSPIN](https://numer.ai/tailspin)) current.

**What this does and does not do.** This loop takes the *already-chosen champion strategy* —
the config baked into `pipeline/make_submission.py` — and re-fits it on the latest era of
Numerai data, then submits. It does **not** search for a better model, change features by
hand, or tune hyperparameters. Feature *selection* still happens each week (the strategy
picks top-K features dynamically from the trailing window), but the *strategy* is fixed. If
the goal is to change the strategy itself — new target, new hyperparameters, new feature
pool — that's a research campaign; stop and switch to the autoresearch playbook
(`playbooks/autoresearch.md`), which ends by promoting a new champion into
`make_submission.py`.

## Tools

The loop's tools split at the network boundary:

- **Local tools** (`pipeline/weekly.py`) handle retraining, QA, drift, and reporting. They
  are plain Python functions behind a CLI. Every command prints **one JSON object** with a
  `next` field that says what to do next, and the exit code tells you whether to continue.
- **The official Numerai MCP server** (`https://api-tournament.numer.ai/mcp`) handles
  tournament operations and is used only for the upload. `pipeline/upload_to_tailspin.py` is
  the MCP client that drives it.

| Command | Function | Purpose |
|---------|----------|---------|
| `python -m pipeline.weekly retrain [--force]` | `retrain()` | Start the background retrain. Returns a `pid` immediately. |
| `python -m pipeline.weekly status` | `retrain_status()` | Poll the retrain: a running phase, `completed`, `skipped`, or `failed`, plus a log tail. |
| `python -m pipeline.weekly qa [--no-refresh]` | `qa_live_predictions()` | **The upload gate.** Score the live split and return `pass` / `warn` / `fail` with the individual checks. |
| `python -m pipeline.weekly summary` | `training_summary()` | Config snapshot of the latest build (era window, top-k, target, etc.). |
| `python -m pipeline.weekly diff` | `diff_features()` | This week's feature changes vs last week, by feature family, with `churn_fraction`. |
| `python -m pipeline.weekly report` | `weekly_report()` | Write `docs/YYYY-WW_weekly_report.{md,html}` and rebuild the dashboard. |
| `python pipeline/upload_to_tailspin.py` | | Upload the newest build to TAILSPIN through the official Numerai MCP server. |

**Exit codes:** `0` means ok, `1` means an error, and `2` means a gate said stop (QA failed,
or the era guard skipped the retrain). On a non-zero exit, read the JSON's `next`, `reason`,
and `checks` before doing anything else.

## Environment

Every command must run under the **`numerai_rag_env`** conda interpreter
(`C:\Users\nopro\anaconda3\envs\numerai_rag_env\python.exe`, Python 3.11 — it has the GPU
XGBoost, `numerapi`, `fastmcp`, and `requests`). Set `NUMERAI_PYTHON` to it, and run from the
repo root so `-m pipeline.weekly` resolves. `make_submission.py` has a hard env guard that
aborts on any other interpreter, and the submission pickle is Python 3.11 bytecode.

## Decision rules

The loop branches at these points. Everything else runs in sequence.

| You see | Do |
|---------|----|
| `status` → `skipped`, `era_regression: false` | Stop. There's no new data this week. Report it; it's not an error. |
| `status` → `skipped`, `era_regression: true` | Stop and flag it: labeled data moved backwards. Never use `--force` without the user's go-ahead. |
| `status` → `failed` | Stop. Relay the error from `log_tail` and don't retry blindly. |
| `status` → a running phase | Wait a minute or two and poll again. Don't hammer it. |
| `qa` → `fail` or `error` (exit 2) | **Do not upload.** Report the failing checks. |
| `qa` → `warn` | Continue, but name the warning checks in your summary so the user can decide. |
| `diff` → unusually high `churn_fraction` | Continue, but call out the drift. A typical week rotates about 10% of features. |
| Upload fails | Stop. Never retry into a different model slot. |

## Trigger phrasing — does "and submission" mean upload?

- **"run weekly retrain"** → run steps 1–5 and **stop before upload**; report that the build
  is QA-passed and ready, and offer to submit.
- **"run weekly retrain and submission"** (or "submit", "run the weekly pipeline end to end")
  → run steps 1–7 including the live upload to TAILSPIN. No extra confirmation needed beyond
  this phrasing; the QA gate (step 3) is still a hard stop.
- **"run weekly retrain, submission, and commits"** → same as above, and step 7 is
  **non-negotiable**: commit *and* `git push origin master`. Don't stop at the commit and
  don't ask whether to push — this phrasing is the standing authorization.

## The weekly loop

Run these in order. Each step gates the next — don't upload a build that failed QA.

### 1. Retrain
Run `python -m pipeline.weekly retrain`. It refreshes the validation data first, then **guards on the
era window**: if no new era has landed since the last submission, it ends
as `skipped` and does nothing. That's the correct, expected outcome when you run
before Numerai has published new data — report it and stop; there's nothing to submit.
Only pass `--force` if explicitly asked to rebuild on unchanged data (rare — e.g.
recovering from a corrupted pickle).

The job runs in the background and returns a `pid` and `log_path` immediately. It does
**not** block.

### 2. Monitor
Poll `python -m pipeline.weekly status` until it reports `completed`, `skipped`, or `failed`. The retrain trains
XGBoost on GPU over ~140 eras and typically takes a few minutes. Space your polls out rather
than hammering — the tool returns a log tail each time so you can see progress. If it comes
back `failed`, read the `log_tail` for the stack trace before deciding whether to retry or
surface the error to the user.

### 3. QA the predictions  ← gate
Run `python -m pipeline.weekly qa`. It re-downloads the live batch, scores the freshly packaged model on the live split
and returns a **pass / warn / fail** verdict plus the individual checks, distribution stats, and artifact paths.

- **pass** — proceed.
- **warn** — proceed, but call out what's off (e.g. a skewed prediction distribution) so the
  user can decide.
- **fail** — **do not upload.** Something is wrong with the build. Read the checks,
  check the retrain log, and surface the problem instead of submitting bad predictions. A bad
  live submission costs a tournament week, so this gate is not optional.

### 4. Review config and feature drift
Run `python -m pipeline.weekly summary` for the build's configuration, then `python -m pipeline.weekly diff`
to see which features rotated in and out versus last week. Large week-over-week feature churn
can be a sign of data drift worth flagging to the user. (On the very first run there's
nothing to diff against — the tool says so; that's fine.)

### 5. Report
Run `python -m pipeline.weekly report`. It writes the markdown and HTML report into `docs/` for the
current ISO week, rebuilds the `docs/index.html` dashboard that links to them, and returns
the paths.

### 6. Upload  (only when the request includes submission)
Skip this step for a retrain-only request (see *Trigger phrasing* above). Run it only after a
passing (or consciously accepted `warn`) QA gate.

**Run the helper:** `python pipeline/upload_to_tailspin.py` (with `NUMERAI_PYTHON` /
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

### 7. Commit and push

The run isn't finished until it's on GitHub. Unpushed commits earn no contribution squares,
which is how two weekly runs once went missing from the graph.

Stage only what the loop produced — `docs/<ISO-week>_weekly_report.md`, the matching
`.html`, `docs/retrain_latest_status.json`, and `docs/index.html` if it changed. Pickles and
`artifacts/` are gitignored by design; leave unrelated untracked files alone rather than
sweeping them in with `git add -A`.

```bash
git add docs/<ISO-week>_weekly_report.md docs/<ISO-week>_weekly_report.html docs/retrain_latest_status.json
git commit -F <message-file>
git push origin master
```

Commit subject: `Weekly W<NN>: retrain and submission on the v5.3 quantum champion`. When a
second build lands inside the same ISO week (a new era arrived mid-week, so the guard passed
and the report was regenerated in place), disambiguate with the live era —
`Weekly W<NN> (live era <E>): ...` — so the two commits don't read identically.

Run the loop on **master in the main repo**, not a worktree, so the push goes straight to the
default branch. Verify afterwards that `git status -sb` shows no `[ahead N]`; that residue is
the exact failure this step exists to prevent.

Two gotchas worth remembering. Write multi-line commit messages with `git commit -F <file>`
— PowerShell here-string syntax (`@'...'@`) piped through the Bash tool leaks a literal `@`
into the subject line. And GitHub's contribution calendar is keyed to the **UTC** author
date, so an evening US commit can land on the following day's square; that's cosmetic, not a
failed push.

## Reporting back to the user

Close the loop with a short summary: the era window that was trained, the QA verdict,
notable feature changes, the report path, the upload result, and the pushed commit. If the
run was skipped (no new data) or failed QA, lead with that — it's the most important thing
for the user to know.
