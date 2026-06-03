---
name: weekly-submission
description: >-
  Run the weekly Numerai live submission loop for this repo: retrain the deployed
  champion model on fresh data, QA the predictions, diff features against last week,
  generate the HTML/markdown report, and upload to the tournament. Use this whenever
  the user wants to do the weekly Numerai run, "submit this week", retrain the live
  model, refresh the submission, run the weekly pipeline, or generate the weekly
  report — even if they don't name the individual MCP tools. This is the operational
  loop, NOT research: it re-fits the already-chosen strategy on new data. If the user
  wants to change the strategy itself, that's the autoresearch skill instead.
---

# Weekly Numerai Submission

**Follow the canonical playbook: [`playbooks/weekly-submission.md`](../../../playbooks/weekly-submission.md).**
Read it now and execute the loop it describes. The full procedure lives there (and not in
this file) so the same instructions are shared across agents — Codex and others read the
same playbook via `AGENTS.md`. Keep procedural edits in the playbook, not here.

Non-negotiables to keep front of mind while you read it:

- **This is operational, not research.** It re-fits the existing champion on fresh data. If
  the user wants to change the strategy itself, switch to the `autoresearch` skill.
- **The QA gate is a hard stop.** A `fail` from `check_live_predictions()` means **do not
  upload** — a bad live submission costs a tournament week.
- **Always upload to the TAILSPIN model slot** — never ANGOSTURA or PIXELATED.
- **Honor the era-window skip** (`status="skipped"` = no new data = nothing to submit) and
  the **Python 3.11 docker image** requirement for upload.
