# Numerai AutoResearch (finetuning) — Playbook

> **Canonical, agent-neutral procedure.** This file is the single source of truth for the
> autoresearch finetuning loop. Claude reads it via the `autoresearch` skill; Codex and
> other agents read it via `AGENTS.md`. Edit the procedure here, not in the agent-specific
> wrappers. For the deep rationale and original loop design, `program.md` at the repo root
> is the underlying reference this playbook distills.

An autonomous research loop for improving the **already-deployed** main strategy. The
strategy exists and is live; the job here is to *beat it* on validation, then optionally
promote the winner. This is not a greenfield search — start from the current champion as the
baseline and try to move past it.

## Guardrails (from program.md — non-negotiable)

These keep results comparable and reproducible across the campaign:

- **Edit only `autoresearch-src/train.py`.** Everything in it is fair game: `CANDIDATE_GROUPS`,
  `EXTRA_FEATURES`, `TOP_K_FEATURES`, `TRAILING_ERAS`, `MODEL`, `XGB_PARAMS`,
  `NUM_BOOST_ROUNDS`, `MAIN_TARGET`, walk-forward and early-stopping settings.
- **Never edit `autoresearch-src/prepare.py`.** It holds the fixed evaluation functions and
  data loaders — changing it invalidates every comparison.
- **`CORR_TARGET` stays `target_ender_20`** and **`MMC_BENCHMARK_COLUMN` stays
  `v52_lgbm_ender20`.** These are fixed by tournament rules; changing them means you're no
  longer measuring the same thing.
- **GPU required.** XGBoost runs with `device: cuda`; the script guards CUDA at startup.
- **Simplicity is a tiebreaker.** A tiny gain that adds ugly complexity isn't worth it;
  equal-or-better results from *deleting* code is a win. Weigh complexity against the
  improvement magnitude.

## The objective

**Maximize `val_mmc_mean`.** Secondary: keep `val_corr_mean` comfortably positive; prefer
higher `research_score` when MMC gains are close. Walk-forward mode (~100 sequential passes,
30–90 min) is the real signal; fast mode (last 100 eras) is a quick screen.

## Campaign setup

1. **Pick a run tag** (e.g. a date like `jun02`) and create a fresh branch
   `autoresearch/<tag>` off master. It must not already exist — each campaign is its own
   branch.
2. **Read the in-scope files** for context: `README.md`, `autoresearch-src/prepare.py`
   (read-only helpers), and `autoresearch-src/train.py` (the file you edit).
3. **Verify data** exists under `data/numerai/v5.2/`. If not, run
   `python autoresearch-src/prepare.py`.
4. **Establish the baseline — and don't trust `train.py`'s current state.** The constants
   sitting in `train.py` are just whatever the *last experiment* left behind; they are **not**
   the champion. Before measuring anything, reconstruct the champion's strategy:
   - The deployed config lives in `custom_mcp/make_submission.py`, and the validated champion
     row is in `experiments/results.tsv` (the committed leaderboard). As of this writing the
     champion is the **walkforward + dynamic-features** XGBoost on `target_ender_60`, top_k=60,
     trailing=20, lookback=142, 10% neutralization — read the actual files rather than trusting
     this sentence, which can go stale.
   - Set `train.py` to that strategy and run it in the matching regime
     (`--walkforward --dynamic-features`) so the baseline `val_mmc_mean` is comparable to how
     the champion was actually validated. (Running fast-mode or a different regime gives a
     baseline that doesn't correspond to the live model — see the promotion notes on why the
     regimes differ.)
   - Create the working `results.tsv` (repo root) with just the header row and log this
     reconstructed baseline. Every later experiment is judged against this number.

## The experiment loop

Once the baseline is recorded, loop autonomously:

1. Note the current branch/commit.
2. Form one experimental idea and implement it by editing `train.py`.
3. `git commit` the change.
4. Run it, redirecting output so it doesn't flood context:
   `python autoresearch-src/train.py > run.log 2>&1` (add `--walkforward` for the strong
   signal once an idea looks promising on a fast screen).
5. Read the result: `grep "val_mmc_mean\|val_corr_mean\|research_score" run.log | tail -5`.
   Empty output means it crashed — `tail -n 50 run.log` for the traceback.
6. Log the row to `results.tsv` (tab-separated; columns: `commit`, `val_mmc_mean`,
   `val_corr_mean`, `research_score`, `status`, `description`; status is `keep` / `discard`
   / `crash`). **Leave `results.tsv` untracked — do not commit it.**
7. **Keep or revert:** if `val_mmc_mean` improved, keep the commit (the branch advances). If
   it's equal or worse, `git reset` back to where you started.

On crashes: fix-and-rerun if it's a typo or missing import; if the idea is fundamentally
broken, log `crash` and move on. Kill any run that exceeds ~2× its expected time and treat
it as a failure.

## Recording results — two files, don't confuse them

A result only reaches the published leaderboard if you take a deliberate second step. This
trips people up: an experiment can be logged yet never appear on the dashboard.

- **`results.tsv` (repo root, untracked)** — your fast working log *during* a campaign, the
  6-column format above, one row per experiment. Ephemeral and never committed (per
  `program.md`). It does **not** feed any report.
- **`experiments/results.tsv` (committed)** — the curated leaderboard that
  `custom_mcp/site_builder.py` renders into `docs/index.html`. It uses a richer schema (`run`,
  `date`, `model`, `target`, `feature_pool`, `top_k`, `trailing`, `bmark_neutralization`,
  `hyperparams`, the `val_*` metrics, `corr_era_count`, `wall_clock_s`, `notes`).

So when an experiment is a genuine keeper worth publishing, add a row to
`experiments/results.tsv` in that richer schema and regenerate the site. If a notable result
"should be on the leaderboard" but isn't, it's almost always because this curation step was
skipped — the working `results.tsv` alone never touches the dashboard.

**Autonomy.** Once the loop has begun, don't stop to ask "should I keep going?" — the user
may have left it running. Keep generating ideas: new feature groups from the metadata,
combining previous near-misses, different targets or `MAIN_TARGET` values, alternate models
(LightGBM / MLP), synthetic targets. The loop runs until the user interrupts. The one thing
that *does* warrant pausing is the promotion decision below — promoting changes the live
model, so that's the user's call, not the loop's.

## Promoting a champion into the live path

This is the handoff to the weekly-submission playbook (`playbooks/weekly-submission.md`).
When a finetuned config beats the champion **and the user decides to deploy it**, the new
config has to move from `autoresearch-src/train.py` into `custom_mcp/make_submission.py`,
which builds the live pickle. Do this deliberately — it changes what gets submitted to the
tournament.

**The two files intentionally differ.** `train.py` runs the *validation* regime;
`make_submission.py` runs the *live* regime (it trains right up to the live era with no purge
and selects features per-era from a short trailing window). So promotion is a
**reconciliation, not a copy-paste**:

- **Strategy-defining knobs — copy these across** when the winning idea changed them:
  `CANDIDATE_GROUPS`, `EXTRA_FEATURES`, `MAIN_TARGET`, `BENCHMARK_NEUTRALIZATION`,
  `XGB_PARAMS`, `NUM_BOOST_ROUNDS`, `LOOKBACK_ERAS`, the model type. These define *what the
  strategy is*. Copy only the *values the experiment actually changed* — change the specific
  winning hyperparameters in `make_submission.py`'s `XGB_PARAMS` rather than pasting
  `train.py`'s whole dict over it, so you don't disturb live-only fields. (Both files now keep
  `"seed": SEED` inside `XGB_PARAMS`, so a dict copy carries the seed — but targeted edits are
  still the safer habit.)
- **Regime-specific knobs — do NOT blindly mirror:** `TOP_K_FEATURES`, `TRAILING_ERAS`, and
  purge differ on purpose between the two files (validation ranks a wide pool over the full
  ~142-era window; live deliberately picks fewer, fresher features from a short ~20-era
  trailing window per build). A value like `TOP_K_FEATURES = 320` selected over 142 eras is
  **not the same object** as 320 selected over 20 eras. If a winning experiment hinged on one
  of these, the default is to **keep the live value as-is** and treat any change as an
  explicit, user-confirmed decision — because the validation number doesn't transfer. If the
  goal is to widen the live filter, reason in terms of *fraction of the feature pool* against
  the 20-era window rather than transplanting the raw count, and say what you chose and why.
  Don't silently change these.
- **Leave the live-only `LIVE_FALLBACK_TARGET` alone** (it's `target_ender_20`). The most
  recent live eras have no labeled 60-day target yet — and when one 60-day target is
  unlabeled, the others are too, since all 60-day targets share the same maturity lag. That's
  exactly *why* the fallback is a **20-day** target, not another 60-day one: the 20-day labels
  mature sooner and are available when the 60-day isn't. So when you promote a different
  *60-day* `MAIN_TARGET`, the fallback to `target_ender_20` stays correct — don't touch it.
  The only case that needs rethinking is promoting a `MAIN_TARGET` of a *different horizon*,
  which would change this maturity assumption. (`train.py` has no fallback because it only
  trains on labeled eras; this is live-only machinery.)
- **`CORR_TARGET` and `MMC_BENCHMARK_COLUMN` never change** in either file. Note
  `MMC_BENCHMARK_COLUMN` is load-bearing in the live path beyond metrics — it's also the
  neutralization target inside `make_submission.py`'s `predict()`.

After editing `make_submission.py`, do a dry retrain to confirm the live build still packages
cleanly (the weekly loop's `run_weekly_retrain` → `check_retrain_status` →
`check_live_predictions` chain is the right smoke test), then hand back to the normal weekly
loop. Commit the promotion on master (or via PR) with a message that references the winning
experiment commit, so the live strategy stays traceable to the research that produced it.
