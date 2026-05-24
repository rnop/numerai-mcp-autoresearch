# Numerai Autoresearch

This is an experiment to have the LLM do its own quant research on the Numerai main tournament.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may24`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `autoresearch-src/prepare.py` — fixed constants, data loading, metric helpers, evaluation functions. Do not modify.
   - `autoresearch-src/train.py` — the file you modify. Feature pool, model config, hyperparameters, training loop.
4. **Verify data exists**: Check that `data/numerai/v5.2/` contains the parquet files. If not, tell the human to run `python autoresearch-src/prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. You launch the training script as: `python autoresearch-src/train.py > run.log 2>&1` (from within the `numerai_rag_env` conda environment).

**What you CAN do:**

- Modify `train.py` — this is the only file you edit. Everything is fair game: feature pool (`CANDIDATE_GROUPS`, `EXTRA_FEATURES`), feature selection (`TOP_K_FEATURES`, `TRAILING_ERAS`), model type (`MODEL`), XGBoost hyperparameters (`XGB_PARAMS`, `NUM_BOOST_ROUNDS`), training target (`MAIN_TARGET`), walkforward settings, early stopping, etc.

**What you CANNOT do:**

- Modify `prepare.py`. It is read-only. It contains the fixed evaluation functions (`per_era_corr`, `per_era_bmc`, `numerai_corr`) and data loading helpers.
- Install new packages or add dependencies beyond what is already importable in the environment.
- Change `CORR_TARGET`. It must stay `"target_ender_20"` — this is the fixed evaluation target per tournament rules.
- Change `MMC_BENCHMARK_COLUMN`. It must stay `"v52_lgbm_ender20"` — this is the fixed MMC benchmark.

**The goal is simple: maximize `val_mmc_mean`.** Secondary objectives: keep `val_corr_mean` comfortably positive; prefer higher `research_score` when MMC gains are similar. Fast-mode runs evaluate on the last 100 validation eras. Walkforward mode runs ~100 sequential training passes and is the stronger signal but takes 30–90 minutes.

**GPU**: XGBoost runs on CUDA (`"device": "cuda"`). Do not remove this setting for XGBoost runs — the script will fail at startup if CUDA is not available. LightGBM (`--model lgbm`) runs on CPU. MLP (`--model mlp`) runs on GPU via PyTorch.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.0001 val_mmc_mean improvement that adds 20 lines of hacky code? Probably not worth it. A 0.0001 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary and a `RESULT_JSON` line:

```
Validation metrics:
  research_score: 0.018765
  val_corr_max_drawdown: -0.156300
  val_corr_mean: 0.023456
  val_corr_sharpe: 1.234500
  val_mmc_mean: 0.012345
  val_mmc_sharpe: 0.876500
  wall_clock_seconds: 312.500
  ...
RESULT_JSON: {"benchmark_neutralization": 0.1, "corr_target": "target_ender_20", ...}
```

Extract the key metrics from the log file:

```
grep "val_mmc_mean\|val_corr_mean\|research_score\|RESULT_JSON" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	val_mmc_mean	val_corr_mean	research_score	status	description
```

1. git commit hash (short, 7 chars)
2. val_mmc_mean achieved (e.g. 0.012345) — use 0.000000 for crashes
3. val_corr_mean achieved (e.g. 0.023456) — use 0.000000 for crashes
4. research_score achieved (e.g. 0.018765) — use 0.000000 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_mmc_mean	val_corr_mean	research_score	status	description
a1b2c3d	0.012345	0.023456	0.018765	keep	baseline
b2c3d4e	0.013200	0.022100	0.019385	keep	increase TOP_K_FEATURES to 300
c3d4e5f	0.011800	0.023900	0.016010	discard	add agility group to candidate pool
d4e5f6g	0.000000	0.000000	0.000000	crash	walkforward with dynamic features (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/may24`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python autoresearch-src/train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "val_mmc_mean\|val_corr_mean\|research_score" run.log | tail -5`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_mmc_mean improved (higher), you "advance" the branch, keeping the git commit
9. If val_mmc_mean is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Fast-mode runs should complete in under 10 minutes. Walkforward runs can take 30–90 minutes. If a run exceeds 2× its expected time, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, wrong environment, a bug), use your judgment: If it's something easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working _indefinitely_ until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the feature metadata for new feature groups, re-read the in-scope files for new angles, try combining previous near-misses, try more radical changes (LightGBM vs XGBoost vs MLP, new training targets, different `MAIN_TARGET` values, synthetic targets). The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each fast experiment takes ~30 minutes you can run approximately 2/hour, for a total of about 16 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
