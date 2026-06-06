"use strict";

const fs = require("node:fs");
const path = require("node:path");
const process = require("node:process");
const { spawnSync, spawn } = require("node:child_process");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const CUSTOM_MCP_DIR = __dirname;
const SUBMISSIONS_DIR = path.join(PROJECT_ROOT, "submissions");
const REPORTS_DIR = path.join(PROJECT_ROOT, "docs");
const MAKE_SUBMISSION = path.join(CUSTOM_MCP_DIR, "make_submission.py");
const TS_BRIDGE = path.join(CUSTOM_MCP_DIR, "ts_bridge.py");
const FEATURES_JSON = path.join(PROJECT_ROOT, "data", "numerai", "v5.2", "features.json");
const PID_PATH = path.join(REPORTS_DIR, "retrain_latest.pid");
const LOG_PATH = path.join(REPORTS_DIR, "retrain_latest.log");
const PYTHON_EXE = process.env.NUMERAI_PYTHON || "python";
const CANDIDATE_GROUPS = ["faith", "wisdom", "strength", "intelligence"];
const EXTRA_FEATURES = new Set([
  "feature_tonal_illuminating_porgy",
  "feature_stalworth_rotund_inflammability",
  "feature_imminent_unobserved_lengthening",
  "feature_northumbrian_outflowing_connie",
  "feature_gravitational_xeromorphic_myxoma",
  "feature_depressing_punitive_recuperation",
  "feature_crimpy_amnesiac_desalinization",
  "feature_unchecked_parented_ngultrum",
  "feature_gilbertian_heliconian_perpendicular",
  "feature_different_wilier_burweed",
  "feature_tempered_devouring_izzard",
  "feature_readier_reversed_accusal",
  "feature_bridal_fingered_pensioner",
  "feature_twaddly_eleven_fustet",
  "feature_unacted_fore_folia",
  "feature_estranging_stylish_liker",
  "feature_millennial_uncanonical_sunna",
]);

let featureGroupCache = null;
let inputBuffer = Buffer.alloc(0);

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function loadJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf-8"));
}

function sortedMetas() {
  if (!fs.existsSync(SUBMISSIONS_DIR)) {
    return [];
  }
  return fs
    .readdirSync(SUBMISSIONS_DIR)
    .filter((name) => name.endsWith("_meta.json"))
    .map((name) => path.join(SUBMISSIONS_DIR, name))
    .sort((a, b) => fs.statSync(a).mtimeMs - fs.statSync(b).mtimeMs);
}

function formatMetric(value, digits = 5) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "n/a";
}

function safeTableCell(value) {
  return String(value || "").replaceAll("|", "").trim().replace(/\s+/g, " ");
}

function formatMetricBlockRows(title, rows) {
  if (rows.length === 0) {
    return "";
  }
  const lines = [`**${title}**`, "", "| Metric | Value |", "| --- | --- |"];
  for (const [label, value] of rows) {
    lines.push(`| ${label} | ${value} |`);
  }
  return lines.join("\n");
}

function tailNonEmptyLines(filePath, count) {
  if (!fs.existsSync(filePath)) {
    return "";
  }
  const lines = fs
    .readFileSync(filePath, "utf-8")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);
  return lines.slice(-count).join("\n");
}

function isProcessAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function terminateProcess(pid) {
  try {
    process.kill(pid);
  } catch {
  }
}

function pythonJson(args) {
  const result = spawnSync(PYTHON_EXE, [TS_BRIDGE, ...args], {
    cwd: PROJECT_ROOT,
    encoding: "utf-8",
  });
  if (result.status !== 0) {
    throw new Error(result.stderr.trim() || result.stdout.trim() || "Python bridge failed.");
  }
  return JSON.parse(result.stdout.trim());
}

function loadFeatureGroupLookup() {
  if (featureGroupCache) {
    return featureGroupCache;
  }
  const lookup = {};
  if (fs.existsSync(FEATURES_JSON)) {
    const data = loadJson(FEATURES_JSON);
    const featureSets = data.feature_sets || {};
    for (const group of CANDIDATE_GROUPS) {
      for (const feature of featureSets[group] || []) {
        lookup[feature] = group;
      }
    }
  }
  for (const feature of EXTRA_FEATURES) {
    lookup[feature] = "extra";
  }
  featureGroupCache = lookup;
  return lookup;
}

function classifyFeature(name) {
  return loadFeatureGroupLookup()[name] || "other";
}

function groupFeatures(features) {
  const groups = {};
  for (const feature of features) {
    const group = classifyFeature(feature);
    if (!groups[group]) {
      groups[group] = [];
    }
    groups[group].push(feature);
  }
  return groups;
}

function currentMaxLabeledEra() {
  const payload = pythonJson(["current-max-era"]);
  return typeof payload.era === "string" ? payload.era : null;
}

function refreshValidationData() {
  pythonJson(["refresh-validation"]);
}

function currentLivePredictionEra(metaPath) {
  const payload = pythonJson(["current-live-era", metaPath]);
  return typeof payload.era === "string" ? payload.era : null;
}

function computeLiveReportMetrics(metaPath) {
  return pythonJson(["compute-live-report-metrics", metaPath]);
}

function computeLivePredictionDiagnostics(metaPath) {
  return pythonJson(["compute-live-diagnostics", metaPath]);
}

function renderReportHtml(title, markdownPath, htmlPath) {
  pythonJson(["render-report-html", title, markdownPath, htmlPath]);
}
function buildDashboard() {
  pythonJson(["build-dashboard"]);
}

function asBoolean(value, fallback = false) {
  return typeof value === "boolean" ? value : fallback;
}

function getSelectedFeatures(meta) {
  const selected = meta.selected_features;
  return Array.isArray(selected) ? selected.filter((value) => typeof value === "string") : [];
}

function runWeeklyRetrain(args = {}) {
  const force = asBoolean(args.force);
  return pythonJson(["run-weekly-retrain", ...(force ? ["--force"] : [])]);
}

function checkRetrainStatus() {
  return pythonJson(["check-retrain-status"]);
}

function getTrainingSummary() {
  const metas = sortedMetas();
  if (metas.length === 0) {
    return { error: "No metadata JSON found in submissions/. Run run_weekly_retrain first." };
  }
  const metaPath = metas[metas.length - 1];
  const meta = loadJson(metaPath);
  const pklFile = path.basename(metaPath).replace("_meta.json", ".pkl");
  const liveDiagnostics = computeLivePredictionDiagnostics(metaPath);
  return {
    built_date: meta.built_date || null,
    target: meta.target || null,
    era_window: `${meta.era_window_start} - ${meta.era_window_end}`,
    era_count: meta.era_count || null,
    lookback_eras: meta.lookback_eras || null,
    trailing_eras: meta.trailing_eras || null,
    top_k_features: meta.top_k_features || null,
    feature_pool_size: meta.feature_pool_size || null,
    fit_eras: meta.fit_eras || null,
    early_stopping_eras: meta.es_eras || null,
    best_iteration: meta.best_iteration || null,
    benchmark_neutralization: meta.benchmark_neutralization || null,
    benchmark_col: meta.benchmark_col || null,
    pickle_size_mb: meta.pickle_size_mb || null,
    wall_clock_seconds: meta.wall_clock_seconds || null,
    pkl_file: pklFile,
    pkl_path: path.join(SUBMISSIONS_DIR, pklFile),
    live_diagnostics: liveDiagnostics,
  };
}

function checkLivePredictions() {
  const metas = sortedMetas();
  if (metas.length === 0) {
    return { error: "No metadata JSON found in submissions/. Run run_weekly_retrain first." };
  }
  const metaPath = metas[metas.length - 1];
  const diagnostics = computeLivePredictionDiagnostics(metaPath);
  return {
    ...diagnostics,
    meta_path: metaPath,
    pkl_path: path.join(SUBMISSIONS_DIR, path.basename(metaPath).replace("_meta.json", ".pkl")),
  };
}

function compareWeeklyFeatures() {
  const metas = sortedMetas();
  if (metas.length === 0) {
    return { error: "No metadata JSON found in submissions/." };
  }

  if (metas.length === 1) {
    const meta = loadJson(metas[0]);
    const currentFeatures = getSelectedFeatures(meta);
    const currentByGroup = groupFeatures(currentFeatures);
    const counts = {};
    for (const [group, features] of Object.entries(currentByGroup)) {
      counts[group] = features.length;
    }
    return {
      note: "Only one training run found - no previous week to compare against.",
      current_features_total: currentFeatures.length,
      current_by_group: counts,
    };
  }

  const prevMeta = loadJson(metas[metas.length - 2]);
  const currMeta = loadJson(metas[metas.length - 1]);
  const prevSet = new Set(getSelectedFeatures(prevMeta));
  const currSet = new Set(getSelectedFeatures(currMeta));

  const added = Array.from(currSet).filter((feature) => !prevSet.has(feature)).sort();
  const removed = Array.from(prevSet).filter((feature) => !currSet.has(feature)).sort();
  const retained = Array.from(currSet).filter((feature) => prevSet.has(feature)).sort();

  const countByGroup = (features) => {
    const grouped = groupFeatures(features);
    const result = {};
    for (const [group, items] of Object.entries(grouped)) {
      result[group] = items.length;
    }
    return result;
  };

  return {
    previous_build: prevMeta.built_date || null,
    current_build: currMeta.built_date || null,
    previous_era_window: `${prevMeta.era_window_start} - ${prevMeta.era_window_end}`,
    current_era_window: `${currMeta.era_window_start} - ${currMeta.era_window_end}`,
    summary: {
      total_features: currSet.size,
      added: added.length,
      removed: removed.length,
      retained: retained.length,
    },
    added_by_group: countByGroup(added),
    removed_by_group: countByGroup(removed),
    retained_by_group: countByGroup(retained),
    added_features: added,
    removed_features: removed,
  };
}

function isoWeek(date) {
  const value = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const day = value.getUTCDay() || 7;
  value.setUTCDate(value.getUTCDate() + 4 - day);
  const yearStart = new Date(Date.UTC(value.getUTCFullYear(), 0, 1));
  const week = Math.ceil((((value.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
  return { year: value.getUTCFullYear(), week };
}

function formatTimestamp(date) {
  const parts = new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(date);
  const byType = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${byType.year}-${byType.month}-${byType.day} ${byType.hour}:${byType.minute}`;
}

function generateWeeklyReport() {
  ensureDir(REPORTS_DIR);
  const metas = sortedMetas();
  if (metas.length === 0) {
    return { error: "No metadata found. Run run_weekly_retrain first." };
  }

  const currMetaPath = metas[metas.length - 1];
  const currMeta = loadJson(currMetaPath);
  const reportMetrics = computeLiveReportMetrics(currMetaPath);
  const liveDiagnostics = computeLivePredictionDiagnostics(currMetaPath);
  const livePredictionEra = currentLivePredictionEra(currMetaPath);
  const now = new Date();
  const iso = isoWeek(now);
  const weekLabel = `${iso.year}-W${String(iso.week).padStart(2, "0")}`;
  const reportPath = path.join(REPORTS_DIR, `${weekLabel}_weekly_report.md`);
  const htmlReportPath = path.join(REPORTS_DIR, `${weekLabel}_weekly_report.html`);

  let featSection = "";
  if (metas.length >= 2) {
    const prevMeta = loadJson(metas[metas.length - 2]);
    const prevSet = new Set(getSelectedFeatures(prevMeta));
    const currSet = new Set(getSelectedFeatures(currMeta));
    const added = Array.from(currSet).filter((feature) => !prevSet.has(feature)).sort();
    const removed = Array.from(prevSet).filter((feature) => !currSet.has(feature)).sort();
    const retained = Array.from(currSet).filter((feature) => prevSet.has(feature)).sort();

    const groupTable = (title, groups) => {
      const values = Object.values(groups);
      if (values.every((items) => items.length === 0)) {
        return `**${title}:** none\n\n`;
      }
      const total = values.reduce((sum, items) => sum + items.length, 0);
      const rows = ["| Group | Count | Sample Features |", "| --- | --- | --- |"];
      for (const [group, features] of Object.entries(groups).sort(([a], [b]) => a.localeCompare(b))) {
        const sample = features.slice(0, 4).map((feature) => `\`${feature}\``).join(", ");
        const suffix = features.length > 4 ? `, +${features.length - 4} more` : "";
        rows.push(`| ${group} | ${features.length} | ${sample}${suffix} |`);
      }
      return `**${title}** (${total} features)\n\n${rows.join("\n")}\n\n`;
    };

    featSection =
      "## Feature Changes vs Previous Week\n\n" +
      `> Previous build: **${prevMeta.built_date}** - era window ${prevMeta.era_window_start} - ${prevMeta.era_window_end}\n\n` +
      "| | Count |\n| --- | --- |\n" +
      `| Total features (current) | ${currSet.size} |\n` +
      `| Added this week | ${added.length} |\n` +
      `| Removed this week | ${removed.length} |\n` +
      `| Retained | ${retained.length} |\n\n` +
      groupTable("Added", groupFeatures(added)) +
      groupTable("Removed", groupFeatures(removed)) +
      groupTable("Retained", groupFeatures(retained));
  } else {
    const grouped = groupFeatures(getSelectedFeatures(currMeta));
    const lines = Object.entries(grouped)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([group, features]) => `- **${group}**: ${features.length}`);
    featSection =
      "## Feature Changes\n\n" +
      "_No previous week to compare - this is the first recorded training run._\n\n" +
      `**Selected features by group:**\n\n${lines.join("\n")}\n\n`;
  }

  const topSnapshot = formatMetricBlockRows("Model Snapshot", [
    ["Live training target", `\`${String(reportMetrics.training_target || currMeta.target || "")}\``],
    ["Validation target", `\`${String(reportMetrics.validation_target || currMeta.fallback_target || "target_ender_20")}\``],
    ["MMC benchmark", `\`${String(reportMetrics.benchmark_col || currMeta.benchmark_col || "")}\``],
    ["Training: CORR mean", formatMetric(reportMetrics.fit_corr_mean)],
    ["Training: MMC mean", formatMetric(reportMetrics.fit_mmc_mean)],
    ["Training Sharpe", formatMetric(reportMetrics.fit_corr_sharpe, 3)],
    ["Validation: CORR mean", formatMetric(reportMetrics.val_corr_mean)],
    ["Validation: MMC mean", formatMetric(reportMetrics.val_mmc_mean)],
    ["Validation Sharpe", formatMetric(reportMetrics.val_corr_sharpe, 3)],
  ]);

  const titleSuffix = livePredictionEra ? ` | Live Era ${livePredictionEra}` : "";
  const liveEraLine = livePredictionEra ? ` | **Live submission era:** ${livePredictionEra}` : "";
  const liveDiagnosticRows = [
    ["Verdict", String(liveDiagnostics.status || "n/a").toUpperCase()],
    ["Ready for submission", liveDiagnostics.ready_for_submission ? "yes" : "no"],
    ["Rows scored", String(liveDiagnostics.row_count || "n/a")],
    ["Prediction std", formatMetric(liveDiagnostics.prediction_std)],
    ["Prediction p99-p01 spread", formatMetric(liveDiagnostics.prediction_spread_p99_p01)],
    ["Duplicate fraction", formatMetric(liveDiagnostics.duplicate_fraction)],
    ["Benchmark corr", formatMetric(liveDiagnostics.benchmark_corr)],
  ];
  const liveDiagnosticChecks = Array.isArray(liveDiagnostics.checks) && liveDiagnostics.checks.length > 0
    ? liveDiagnostics.checks
        .map((row) => {
          const item = row;
          return `| ${safeTableCell(item.name || "check")} | ${safeTableCell(String(item.status || "n/a").toUpperCase())} | ${safeTableCell(item.detail || "")} |`;
        })
        .join("\n")
    : "| n/a | n/a | No live diagnostic checks were generated. |";
  const liveDiagnosticArtifacts =
    liveDiagnostics.artifacts && typeof liveDiagnostics.artifacts === "object"
      ? liveDiagnostics.artifacts
      : {};
  const plotPath = typeof liveDiagnosticArtifacts.plot_path === "string" ? liveDiagnosticArtifacts.plot_path : null;
  const plotRel = plotPath ? path.relative(REPORTS_DIR, plotPath).replace(/\\/g, "/") : null;
  const csvPath = typeof liveDiagnosticArtifacts.csv_path === "string" ? liveDiagnosticArtifacts.csv_path : null;
  const csvRel = csvPath ? path.relative(REPORTS_DIR, csvPath).replace(/\\/g, "/") : null;
  const summaryPath = typeof liveDiagnosticArtifacts.summary_path === "string" ? liveDiagnosticArtifacts.summary_path : null;
  const summaryRel = summaryPath ? path.relative(REPORTS_DIR, summaryPath).replace(/\\/g, "/") : null;
  const liveVisualizationBlock = plotRel
    ? `
### Visualization

![Live prediction QA plot](${plotRel})

The chart combines the raw histogram, sorted prediction curve, benchmark exposure scatter, and percentile-ranked distribution for the current live batch.
`
    : "";

  const report = `# Numerai Weekly Report - ${weekLabel}${titleSuffix}

**Model:** tailspin | **Built:** ${currMeta.built_date} | **Era window:** ${currMeta.era_window_start} - ${currMeta.era_window_end}${liveEraLine}

---

${featSection}
---

## Target Analysis

**Current target:** \`${currMeta.target}\`

This model is trained on \`target_ender_60\` as established by the v5.2 feature analysis. This target
was selected because it provides the best generalization for MMC in walk-forward testing.

> A dynamic target recommendation system is planned for a future update. Until then,
> \`target_ender_60\` remains the fixed default.

---

## Top Statistics

${topSnapshot}

---

## Live Prediction QA

${liveVisualizationBlock}

${formatMetricBlockRows("Distribution Check", liveDiagnosticRows)}

| Check | Status | Details |
| --- | --- | --- |
${liveDiagnosticChecks}

| Artifact | Path |
| --- | --- |
| Distribution plot | \`${String(plotRel || liveDiagnosticArtifacts.plot_path || "n/a")}\` |
| Scored CSV | \`${String(csvRel || liveDiagnosticArtifacts.csv_path || "n/a")}\` |
| Summary JSON | \`${String(summaryRel || liveDiagnosticArtifacts.summary_path || "n/a")}\` |

---

## Artifact Details

| Metric | Value |
| --- | --- |
| Built date | ${currMeta.built_date} |
| Model type | XGBoost (GPU) |
| Best iteration | ${currMeta.best_iteration} |
| Wall clock time | ${currMeta.wall_clock_seconds}s |
| Pickle size | ${currMeta.pickle_size_mb} MB |

---

## Training Configuration

| Parameter | Value |
| --- | --- |
| Target | \`${currMeta.target}\` |
| Era window | ${currMeta.era_window_start} - ${currMeta.era_window_end} |
| Era count | ${currMeta.era_count} |
| Lookback eras | ${currMeta.lookback_eras} |
| Trailing eras (feature ranking) | ${currMeta.trailing_eras} |
| Top-K features selected | ${currMeta.top_k_features} |
| Feature pool size | ${currMeta.feature_pool_size} |
| Fit eras | ${currMeta.fit_eras} |
| Early stopping eras | ${currMeta.es_eras} |
| Best iteration | ${currMeta.best_iteration} |
| Benchmark neutralization | ${currMeta.benchmark_neutralization} vs \`${currMeta.benchmark_col}\` |

---

_Generated by numerai-weekly TypeScript MCP on ${formatTimestamp(now)}._
`;

  fs.writeFileSync(reportPath, report, "utf-8");
  renderReportHtml(`Numerai Weekly Report - ${weekLabel}`, reportPath, htmlReportPath);
  buildDashboard();

  return {
    report_path: reportPath,
    html_report_path: htmlReportPath,
    dashboard_path: path.join(REPORTS_DIR, "index.html"),
    week: weekLabel,
    live_diagnostics: liveDiagnostics,
    content: report,
    html_content: fs.readFileSync(htmlReportPath, "utf-8"),
  };
}

const TOOLS = [
  {
    name: "run_weekly_retrain",
    description: "Run custom_mcp/make_submission.py in the background to retrain the weekly Numerai model.",
    inputSchema: {
      type: "object",
      properties: {
        force: {
          type: "boolean",
          description: "Retrain even if the latest labeled era matches the most recent submission window.",
        },
      },
      additionalProperties: false,
    },
  },
  {
    name: "check_retrain_status",
    description: "Poll the background retraining job and return a status snapshot plus the recent log tail.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
  {
    name: "get_training_summary",
    description: "Read the latest submission metadata and return the current training configuration.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
  {
    name: "check_live_predictions",
    description: "Score the current live split with the latest packaged submission model and write QA artifacts into artifacts/.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
  {
    name: "compare_weekly_features",
    description: "Compare the latest selected feature set against the previous run and group additions/removals.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
  {
    name: "generate_weekly_report",
    description: "Generate the current weekly Markdown and HTML report into docs/.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
];

function invokeTool(name, args) {
  switch (name) {
    case "run_weekly_retrain":
      return runWeeklyRetrain(args);
    case "check_retrain_status":
      return checkRetrainStatus();
    case "get_training_summary":
      return getTrainingSummary();
    case "check_live_predictions":
      return checkLivePredictions();
    case "compare_weekly_features":
      return compareWeeklyFeatures();
    case "generate_weekly_report":
      return generateWeeklyReport();
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

function writeMessage(payload) {
  const body = Buffer.from(JSON.stringify(payload), "utf-8");
  const header = Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, "utf-8");
  process.stdout.write(Buffer.concat([header, body]));
}

function writeResponse(id, result) {
  writeMessage({ jsonrpc: "2.0", id, result });
}

function writeError(id, code, message) {
  writeMessage({
    jsonrpc: "2.0",
    id,
    error: { code, message },
  });
}

function toolResult(payload) {
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(payload, null, 2),
      },
    ],
    structuredContent: payload,
  };
}

function handleRequest(request) {
  const id = request.id ?? null;
  const method = request.method;
  const params = request.params || {};

  try {
    switch (method) {
      case "initialize":
        writeResponse(id, {
          protocolVersion: "2024-11-05",
          capabilities: {
            tools: {},
          },
          serverInfo: {
            name: "numerai-weekly-ts",
            version: "0.1.0",
          },
        });
        return;
      case "notifications/initialized":
        return;
      case "ping":
        writeResponse(id, {});
        return;
      case "tools/list":
        writeResponse(id, { tools: TOOLS });
        return;
      case "tools/call": {
        const name = typeof params.name === "string" ? params.name : "";
        const args = params.arguments || {};
        const payload = invokeTool(name, args);
        writeResponse(id, toolResult(payload));
        return;
      }
      default:
        writeError(id, -32601, `Method not found: ${method}`);
        return;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (method === "tools/call") {
      writeResponse(id, {
        content: [{ type: "text", text: message }],
        isError: true,
      });
      return;
    }
    writeError(id, -32000, message);
  }
}

function processBuffer() {
  while (true) {
    const separator = inputBuffer.indexOf("\r\n\r\n");
    if (separator === -1) {
      return;
    }

    const headerText = inputBuffer.subarray(0, separator).toString("utf-8");
    const headers = headerText.split("\r\n");
    let contentLength = 0;
    for (const header of headers) {
      const [name, value] = header.split(":");
      if (name.toLowerCase() === "content-length") {
        contentLength = Number(value.trim());
      }
    }
    const totalLength = separator + 4 + contentLength;
    if (inputBuffer.length < totalLength) {
      return;
    }

    const body = inputBuffer.subarray(separator + 4, totalLength).toString("utf-8");
    inputBuffer = inputBuffer.subarray(totalLength);
    handleRequest(JSON.parse(body));
  }
}

process.stdin.on("data", (chunk) => {
  inputBuffer = Buffer.concat([inputBuffer, chunk]);
  processBuffer();
});

process.stdin.on("end", () => {
  process.exit(0);
});
