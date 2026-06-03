# Agent Instructions

Base project instructions live in `CLAUDE.md` — read it for interpreter, GPU, and
public-repo scope rules. (Claude Code loads it automatically via the import below; other
agents should read it directly.)

@CLAUDE.md

## Playbooks (agent-neutral procedures)

The two operational workflows are documented as canonical, agent-neutral playbooks. These
are the single source of truth — follow them regardless of which agent you are. (Claude Code
also exposes them as auto-triggering skills under `.claude/skills/`, but the content is the
same.)

- **Weekly live submission** — `playbooks/weekly-submission.md`. Run this for the routine
  weekly retrain-QA-report-and-submit of the deployed champion. It re-fits the existing
  strategy on fresh data; it does not change the strategy.
- **AutoResearch finetuning** — `playbooks/autoresearch.md`. Run this to improve/finetune the
  strategy (new features, targets, hyperparameters, models) and to promote a new champion
  into the live submission path. `program.md` is the deeper underlying reference.

Pick the playbook that matches the request, read it, and execute it. The MCP tools both
playbooks call (`custom_mcp/server.py` / `server.js`) are standard MCP and work from any
MCP-capable agent; configure them from `.mcp.example.json`.
