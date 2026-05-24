from __future__ import annotations

import csv
import html
import json
import re
import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "autoresearch-src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

REPORTS_DIR = ROOT / "reports"
EXPERIMENTS_TSV = ROOT / "experiments" / "results.tsv"
FEATURE_ANALYSIS_DIR = ROOT / "artifacts" / "feature_analysis"
GROUP_SUMMARY_CSV = FEATURE_ANALYSIS_DIR / "group_summary_validation.csv"
UNIQUENESS_CSV = FEATURE_ANALYSIS_DIR / "uniqueness_report.csv"
FEATURE_ANALYSIS_README = FEATURE_ANALYSIS_DIR / "README.md"
RECOMMENDED_FEATURES_CSV = FEATURE_ANALYSIS_DIR / "recommended_features_target_ender_60.csv"
RECOMMENDED_UNIQUE_CSV = FEATURE_ANALYSIS_DIR / "recommended_unique_target_ender_60.csv"
TARGET_INTERCORR_CSV = FEATURE_ANALYSIS_DIR / "target_intercorr.csv"
REPORT_INDEX_NAME = "research_overview.html"
DEFAULT_TARGET = "target_ender_60"
ROOT_ARTIFACTS_DIR = ROOT / "artifacts"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _format_float(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}"


def _format_seconds(value: str | None) -> str:
    seconds = _to_float(value, 0.0)
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.0f}s"


def _format_percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _resolve_feature_analysis_data_dir() -> Path | None:
    candidates = [
        ROOT / "data" / "numerai" / "v5.2",
    ]
    for path in candidates:
        if (path / "validation.parquet").exists() and (path / "features.json").exists():
            return path
    return None


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_public_train_feature_config() -> tuple[list[str], list[str]]:
    tree = ast.parse((ROOT / "autoresearch-src" / "train.py").read_text(encoding="utf-8"))
    candidate_groups: list[str] = []
    extra_features: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "CANDIDATE_GROUPS":
                candidate_groups = ast.literal_eval(node.value)
            if target.id == "EXTRA_FEATURES":
                extra_features = ast.literal_eval(node.value)
    return candidate_groups, extra_features


def _load_latest_selected_features() -> list[str]:
    submissions_dir = ROOT / "submissions"
    meta_paths = sorted(submissions_dir.glob("*_meta.json"), key=lambda p: p.stat().st_mtime)
    if not meta_paths:
        return []
    meta = _load_json(meta_paths[-1])
    return list(meta.get("selected_features", []))


def _load_latest_submission_meta() -> dict | None:
    submissions_dir = ROOT / "submissions"
    meta_paths = sorted(submissions_dir.glob("*_meta.json"), key=lambda p: p.stat().st_mtime)
    if not meta_paths:
        return None
    return _load_json(meta_paths[-1])


def _load_latest_submission_pickle() -> Path | None:
    submissions_dir = ROOT / "submissions"
    pickle_paths = sorted(submissions_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    return pickle_paths[-1] if pickle_paths else None


def _inline_markdown(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    return escaped


def markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    chunks: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    blockquote: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            chunks.append(f"<p>{_inline_markdown(' '.join(paragraph).strip())}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            chunks.append("<ul>" + "".join(f"<li>{item}</li>" for item in list_items) + "</ul>")
            list_items = []

    def flush_blockquote() -> None:
        nonlocal blockquote
        if blockquote:
            body = " ".join(blockquote).strip()
            chunks.append(f"<blockquote><p>{_inline_markdown(body)}</p></blockquote>")
            blockquote = []

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            flush_list()
            flush_blockquote()
            i += 1
            continue

        if stripped == "---":
            flush_paragraph()
            flush_list()
            flush_blockquote()
            chunks.append("<hr />")
            i += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            flush_list()
            flush_blockquote()
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            if len(table_lines) >= 2:
                headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
                rows = table_lines[2:] if re.fullmatch(r"[\|\-\s:]+", table_lines[1]) else table_lines[1:]
                thead = "".join(f"<th>{_inline_markdown(cell)}</th>" for cell in headers)
                tbody_rows = []
                for row in rows:
                    cells = [cell.strip() for cell in row.strip("|").split("|")]
                    cells_html = "".join(f"<td>{_inline_markdown(cell)}</td>" for cell in cells)
                    tbody_rows.append(f"<tr>{cells_html}</tr>")
                chunks.append(
                    "<div class=\"table-wrap\"><table>"
                    f"<thead><tr>{thead}</tr></thead>"
                    f"<tbody>{''.join(tbody_rows)}</tbody>"
                    "</table></div>"
                )
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            flush_list()
            blockquote.append(stripped[1:].strip())
            i += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            flush_blockquote()
            list_items.append(_inline_markdown(stripped[2:].strip()))
            i += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            flush_list()
            flush_blockquote()
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            anchor = _slugify(re.sub(r"<[^>]+>", "", _inline_markdown(text)))
            chunks.append(f"<h{level} id=\"{anchor}\">{_inline_markdown(text)}</h{level}>")
            i += 1
            continue

        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    flush_list()
    flush_blockquote()
    return "\n".join(chunks)


def _layout_weekly_report_html(body: str) -> str:
    """Promote each H2 section into overview-style panels for weekly reports."""
    trailing_rule_pattern = r"\s*<hr\b[^>]*>\s*$"
    parts = re.split(r"(<h2 id=\"[^\"]+\">.*?</h2>)", body, flags=re.DOTALL)
    if len(parts) <= 1:
        return f'<div class="report-stack">{body}</div>'

    intro_html = re.sub(trailing_rule_pattern, "", parts[0].strip())
    section_chunks: list[str] = []
    for idx in range(1, len(parts), 2):
        heading = parts[idx]
        content = parts[idx + 1] if idx + 1 < len(parts) else ""
        content = re.sub(trailing_rule_pattern, "", content.strip())
        if 'id="training-configuration"' in heading:
            content = _split_training_configuration_table(content)
        section_class = (
            "panel report-panel report-panel-wide"
            if content.count("<table>") > 1
            else "panel report-panel"
        )
        section_chunks.append(
            f'<section class="{section_class}">{heading}{content}</section>'
        )

    intro_block = (
        f'<header class="report-hero">{intro_html}</header>'
        if intro_html
        else ""
    )
    sections_html = "".join(section_chunks)
    if not sections_html:
        return intro_block
    return f"{intro_block}<div class=\"layout report-layout\">{sections_html}</div>"


def _split_training_configuration_table(content: str) -> str:
    table_match = re.search(
        r'(<div class="table-wrap"><table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>)(.*?)(</tbody></table></div>)',
        content,
        flags=re.DOTALL,
    )
    if not table_match:
        return content

    rows = re.findall(r"<tr>.*?</tr>", table_match.group(2), flags=re.DOTALL)
    if len(rows) < 4:
        return content

    midpoint = (len(rows) + 1) // 2
    left_rows = "".join(rows[:midpoint])
    right_rows = "".join(rows[midpoint:])
    table_open = '<div class="table-wrap training-table"><table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>'
    table_suffix = table_match.group(3)
    split_tables = (
        '<div class="training-grid">'
        f"{table_open}{left_rows}{table_suffix}"
        f"{table_open}{right_rows}{table_suffix}"
        "</div>"
    )
    return content[: table_match.start()] + split_tables + content[table_match.end() :]


def _nav_links_html(active: str = "") -> str:
    weekly_paths = sorted(
        (p for p in REPORTS_DIR.glob("*_weekly_report.html") if "example" not in p.stem),
        key=lambda p: p.name,
    )
    latest_weekly = f"./{weekly_paths[-1].name}" if weekly_paths else "#"
    items = [
        ("Research Experiments Overview", f"./{REPORT_INDEX_NAME}", active == "overview"),
        ("Latest weekly report", latest_weekly, active == "weekly"),
        ("Feature Analysis", "./feature_analysis_report.html", active == "feature"),
        ("Read the repo guide", "../README.md", False),
    ]
    parts = []
    for label, href, is_active in items:
        cls = ' class="primary"' if is_active else ""
        parts.append(f'<a{cls} href="{html.escape(href)}">{html.escape(label)}</a>')
    return "\n        ".join(parts)


def build_report_html(title: str, markdown_text: str) -> str:
    body = _layout_weekly_report_html(markdown_to_html(markdown_text))

    # Split the intro <header> from the panels grid so the nav can sit
    # right after the title (matching the other report pages' structure).
    split_idx = body.find("</header>")
    if split_idx != -1:
        intro_html = body[: split_idx + len("</header>")]
        panels_html = body[split_idx + len("</header>") :].strip()
    else:
        intro_html = ""
        panels_html = body
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f3ec;
      --paper: rgba(255, 252, 246, 0.84);
      --paper-strong: #fffdf9;
      --ink: #161514;
      --muted: #615a52;
      --line: rgba(123, 107, 83, 0.18);
      --accent: #0f766e;
      --accent-2: #b45309;
      --accent-soft: rgba(15, 118, 110, 0.14);
      --shadow: 0 20px 50px rgba(61, 47, 25, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(180, 83, 9, 0.16), transparent 26%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.14), transparent 30%),
        linear-gradient(180deg, #efe6d6 0%, var(--bg) 38%, #f4efe7 100%);
    }}
    a {{
      color: inherit;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 26px 18px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.72), rgba(255,248,237,0.94));
      border: 1px solid rgba(255,255,255,0.55);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      border-radius: 28px;
      padding: 30px;
      overflow: hidden;
      position: relative;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      right: -80px;
      top: -80px;
      width: 220px;
      height: 220px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(15, 118, 110, 0.24), transparent 70%);
    }}
    .eyebrow {{
      display: inline-block;
      padding: 0.4rem 0.7rem;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.1);
      color: var(--accent);
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      font-size: 0.76rem;
    }}
    .hero-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
    }}
    .hero-top a {{
      text-decoration: none;
      border-radius: 999px;
      padding: 0.7rem 1rem;
      font-weight: 700;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.74);
      white-space: nowrap;
    }}
    .hero-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }}
    .hero-links a {{
      text-decoration: none;
      border-radius: 999px;
      padding: 0.85rem 1.1rem;
      font-weight: 700;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.7);
    }}
    .hero-links a.primary {{
      background: var(--accent);
      color: #fff;
      border-color: transparent;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-top: 20px;
    }}
    .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 22px;
    }}
    .report-panel-wide {{
      grid-column: 1 / -1;
    }}
    .report-panel > :last-child,
    .hero > :last-child {{
      margin-bottom: 0;
    }}
    h1, h2, h3 {{
      letter-spacing: -0.03em;
      line-height: 1.05;
    }}
    h1 {{
      margin: 16px 0 12px;
      font-size: clamp(2rem, 3.2vw, 3.8rem);
      line-height: 0.98;
      letter-spacing: -0.05em;
      max-width: none;
      white-space: nowrap;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 1.35rem;
    }}
    h3 {{
      margin: 24px 0 10px;
      font-size: 1.15rem;
    }}
    p, li, td, th, blockquote {{
      font-size: 1rem;
      line-height: 1.65;
    }}
    p {{
      margin: 0 0 16px;
      color: var(--muted);
    }}
    ul {{
      margin: 0 0 18px 1.2rem;
      padding: 0;
    }}
    li + li {{ margin-top: 6px; }}
    hr {{
      border: 0;
      border-top: 1px solid var(--line);
      margin: 22px 0;
    }}
    code {{
      font-family: "Consolas", "SFMono-Regular", monospace;
      background: rgba(244, 240, 232, 0.85);
      border-radius: 6px;
      padding: 0.15rem 0.35rem;
      font-size: 0.95em;
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    blockquote {{
      margin: 18px 0;
      padding: 0 0 0 16px;
      border-left: 4px solid var(--accent-soft);
      color: var(--muted);
      background: rgba(255,255,255,0.28);
      border-radius: 0 12px 12px 0;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin: 18px 0 22px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--paper-strong);
      max-width: 100%;
    }}
    .training-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      align-items: start;
    }}
    .training-table {{
      margin: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 0;
      table-layout: fixed;
    }}
    th, td {{
      padding: 14px 16px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    th {{
      background: rgba(244, 240, 232, 0.85);
      font-size: 0.82rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    strong {{
      color: var(--ink);
    }}
    .report-hero p:first-of-type {{
      font-size: 1.02rem;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    @media (max-width: 900px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .training-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 680px) {{
      .page {{
        padding: 14px 10px 28px;
      }}
      .hero, .panel {{
        padding: 18px;
      }}
      .hero-top {{
        flex-direction: column;
        align-items: flex-start;
      }}
      h1 {{
        white-space: normal;
      }}
      .hero-links a {{
        width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero report-hero">
      <div class="hero-top">
        <span class="eyebrow">Weekly Report</span>
      </div>
      {intro_html}
      <div class="hero-links">
        {_nav_links_html("weekly")}
      </div>
    </section>
    {panels_html}
  </div>
</body>
</html>
"""


def _chart_svg(
    series_map: dict[str, list[float]],
    labels: list[str],
    title: str,
    subtitle: str,
    palette: list[str] | None = None,
) -> str:
    width = 920
    height = 280
    pad_left = 62
    pad_right = 22
    pad_top = 36
    pad_bottom = 48
    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom
    palette = palette or ["#0f766e", "#b45309", "#1d4ed8", "#be123c"]

    all_values = [value for values in series_map.values() for value in values]
    y_min = min(all_values + [-0.01])
    y_max = max(all_values + [0.01])
    if y_max == y_min:
        y_max += 0.01
        y_min -= 0.01
    span = y_max - y_min
    y_min -= span * 0.08
    y_max += span * 0.08

    def x_at(index: int, total: int) -> float:
        if total <= 1:
            return pad_left + inner_w / 2
        return pad_left + (index / (total - 1)) * inner_w

    def y_at(value: float) -> float:
        return pad_top + (y_max - value) / (y_max - y_min) * inner_h

    y_ticks = []
    for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = y_min + (y_max - y_min) * pct
        y = y_at(value)
        y_ticks.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" stroke="#e6ddd0" stroke-width="1" />'
            f'<text x="{pad_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#6a645d">{value:.3f}</text>'
        )

    x_ticks = []
    tick_indexes = sorted({0, len(labels) // 3, (2 * len(labels)) // 3, len(labels) - 1})
    for idx in tick_indexes:
        x = x_at(idx, len(labels))
        x_ticks.append(
            f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{height - pad_bottom}" stroke="#f0e7da" stroke-width="1" />'
            f'<text x="{x:.1f}" y="{height - pad_bottom + 20}" text-anchor="middle" font-size="11" fill="#6a645d">{html.escape(str(labels[idx]))}</text>'
        )

    zero_line = ""
    if y_min < 0 < y_max:
        zero_y = y_at(0.0)
        zero_line = f'<line x1="{pad_left}" y1="{zero_y:.1f}" x2="{width - pad_right}" y2="{zero_y:.1f}" stroke="#c2410c" stroke-dasharray="4 4" stroke-width="1.2" />'

    paths = []
    legends = []
    for idx, (name, values) in enumerate(series_map.items()):
        color = palette[idx % len(palette)]
        coords = [f"{x_at(i, len(values)):.1f},{y_at(v):.1f}" for i, v in enumerate(values)]
        points = " ".join(coords)
        paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}" />')
        final_x = x_at(len(values) - 1, len(values))
        final_y = y_at(values[-1])
        paths.append(f'<circle cx="{final_x:.1f}" cy="{final_y:.1f}" r="4.5" fill="{color}" />')
        legends.append(
            f'<div class="legend-item"><span class="legend-swatch" style="background:{color};"></span>{html.escape(name)}</div>'
        )

    return f"""
    <section class="chart-card">
      <div class="chart-copy">
        <h2>{html.escape(title)}</h2>
        <p>{html.escape(subtitle)}</p>
      </div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        <rect x="0" y="0" width="{width}" height="{height}" rx="22" fill="#fffdf8"></rect>
        {''.join(y_ticks)}
        {''.join(x_ticks)}
        {zero_line}
        {''.join(paths)}
      </svg>
      <div class="legend">{''.join(legends)}</div>
    </section>
    """


def _build_feature_set(ordered_rows: list[dict[str, str]], limit: int) -> list[str]:
    seen: set[str] = set()
    chosen: list[str] = []
    for row in ordered_rows:
        feature = row["feature"]
        if feature in seen:
            continue
        seen.add(feature)
        chosen.append(feature)
        if len(chosen) >= limit:
            break
    return chosen


def _compute_feature_chart_payload() -> dict[str, object] | None:
    data_dir = _resolve_feature_analysis_data_dir()
    if data_dir is None:
        return None

    try:
        import pandas as pd
        from feature_analysis import per_era_feature_corr
    except Exception:
        return None

    recommended_rows = _read_csv(RECOMMENDED_FEATURES_CSV)
    unique_rows = _read_csv(RECOMMENDED_UNIQUE_CSV)
    features_meta = _load_json(data_dir / "features.json")
    feature_sets = features_meta["feature_sets"]
    candidate_groups, extra_features = _load_public_train_feature_config()
    selected_features = _load_latest_selected_features()

    target_rows = [row for row in recommended_rows if row.get("target") == DEFAULT_TARGET]
    unique_target_rows = [row for row in unique_rows if row.get("target") == DEFAULT_TARGET]
    if not target_rows:
        return None

    best_feature_row = max(target_rows, key=lambda row: _to_float(row.get("val_sharpe")))
    best_feature = best_feature_row["feature"]
    top_recommended_set = _build_feature_set(
        sorted(target_rows, key=lambda row: _to_float(row.get("val_sharpe")), reverse=True),
        limit=12,
    )
    top_unique_set = _build_feature_set(
        sorted(unique_target_rows, key=lambda row: _to_float(row.get("combined_score")), reverse=True),
        limit=12,
    )
    training_pool = []
    for group in candidate_groups:
        training_pool.extend(feature_sets.get(group, []))
    training_pool.extend(extra_features)
    training_pool = list(dict.fromkeys(training_pool))

    set_map: dict[str, dict[str, object]] = {
        "small": {"label": f"Numerai small ({len(feature_sets['small'])})", "features": feature_sets["small"]},
        "medium": {"label": f"Numerai medium ({len(feature_sets['medium'])})", "features": feature_sets["medium"]},
        "faith": {"label": f"Faith ({len(feature_sets['faith'])})", "features": feature_sets["faith"]},
        "wisdom": {"label": f"Wisdom ({len(feature_sets['wisdom'])})", "features": feature_sets["wisdom"]},
        "strength": {"label": f"Strength ({len(feature_sets['strength'])})", "features": feature_sets["strength"]},
        "intelligence": {"label": f"Intelligence ({len(feature_sets['intelligence'])})", "features": feature_sets["intelligence"]},
        "rain": {"label": f"Rain ({len(feature_sets['rain'])})", "features": feature_sets["rain"]},
        "sunshine": {"label": f"Sunshine ({len(feature_sets['sunshine'])})", "features": feature_sets["sunshine"]},
        "training_pool": {"label": f"Current training pool ({len(training_pool)})", "features": training_pool},
        "recommended_top12": {"label": f"Recommended top 12 ({len(top_recommended_set)})", "features": top_recommended_set},
        "unique_top12": {"label": f"Unique top 12 ({len(top_unique_set)})", "features": top_unique_set},
    }
    if selected_features:
        set_map["selected_60"] = {
            "label": f"Latest selected training features ({len(selected_features)})",
            "features": selected_features,
        }

    feature_option_rows = sorted(target_rows, key=lambda row: _to_float(row.get("val_sharpe")), reverse=True)[:18]
    feature_option_rows.extend(
        sorted(unique_target_rows, key=lambda row: _to_float(row.get("combined_score")), reverse=True)[:18]
    )
    if selected_features:
        for feature in selected_features[:24]:
            feature_option_rows.append({"feature": feature})

    feature_options: list[str] = []
    seen_features: set[str] = set()
    for row in feature_option_rows:
        feature = row["feature"]
        if feature in seen_features:
            continue
        seen_features.add(feature)
        feature_options.append(feature)

    all_features_for_read = set(feature_options)
    for item in set_map.values():
        all_features_for_read.update(item["features"])
    feature_cols = sorted(all_features_for_read)

    validation_df = pd.read_parquet(
        data_dir / "validation.parquet",
        columns=["era", DEFAULT_TARGET, *feature_cols],
    ).reset_index(drop=True)

    all_val_eras = sorted(validation_df["era"].astype(str).unique())
    tail_eras = set(all_val_eras[-100:])
    validation_df = validation_df[validation_df["era"].astype(str).isin(tail_eras)].reset_index(drop=True)

    corr_df = per_era_feature_corr(validation_df, feature_cols, DEFAULT_TARGET)
    corr_df.index = corr_df.index.astype(str)
    corr_df = corr_df.sort_index()

    if best_feature not in corr_df.columns:
        return None

    feature_series = {
        feature: corr_df[feature].tolist()
        for feature in feature_options
        if feature in corr_df.columns
    }

    set_series = {}
    set_labels = {}
    for key, item in set_map.items():
        features = [feature for feature in item["features"] if feature in corr_df.columns]
        if not features:
            continue
        set_series[key] = corr_df[features].mean(axis=1).tolist()
        set_labels[key] = item["label"]

    return {
        "target": DEFAULT_TARGET,
        "eras": corr_df.index.tolist(),
        "feature_options": feature_options,
        "feature_labels": {feature: feature.replace("feature_", "") for feature in feature_options},
        "feature_series": feature_series,
        "default_feature": best_feature,
        "set_series": set_series,
        "set_labels": set_labels,
        "default_primary_set": "selected_60" if "selected_60" in set_series else "training_pool",
        "default_secondary_set": "medium" if "medium" in set_series else next(iter(set_series)),
    }



def _make_markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, divider, *body])


def _build_feature_analysis_report_markdown() -> str:
    intro = FEATURE_ANALYSIS_README.read_text(encoding="utf-8").split("## Reproducibility")[0].strip()
    group_rows = _read_csv(GROUP_SUMMARY_CSV)
    unique_rows = _read_csv(UNIQUENESS_CSV)
    recommended_rows = _read_csv(RECOMMENDED_FEATURES_CSV)
    unique_feature_rows = _read_csv(RECOMMENDED_UNIQUE_CSV)
    intercorr_rows = _read_csv(TARGET_INTERCORR_CSV)

    top_recommended = sorted(
        recommended_rows,
        key=lambda row: _to_float(row.get("val_sharpe")),
        reverse=True,
    )[:10]
    top_unique = sorted(
        unique_feature_rows,
        key=lambda row: _to_float(row.get("combined_score")),
        reverse=True,
    )[:10]

    best_group = max(group_rows, key=lambda row: _to_float(row.get("sharpe")))
    unique_outside_medium = sum(1 for row in unique_rows if row.get("in_medium") == "False")
    rain_candidates = sum(1 for row in unique_feature_rows if row.get("group") == "rain")

    group_summary_rows = []
    for row in group_rows:
        group_summary_rows.append(
            {
                "target": row["target"],
                "mean": _format_float(_to_float(row["mean"])),
                "sharpe": _format_float(_to_float(row["sharpe"]), 3),
                "positive eras": _format_percent(_to_float(row["pct_pos"])),
            }
        )

    top_recommended_rows = []
    for row in top_recommended:
        top_recommended_rows.append(
            {
                "feature": f"`{row['feature']}`",
                "group": row["group"],
                "val mean": _format_float(_to_float(row["val_mean"])),
                "val sharpe": _format_float(_to_float(row["val_sharpe"]), 3),
                "positive eras": _format_percent(_to_float(row["val_pct_pos"])),
            }
        )

    top_unique_rows = []
    for row in top_unique:
        top_unique_rows.append(
            {
                "feature": f"`{row['feature']}`",
                "group": row["group"],
                "combined score": _format_float(_to_float(row["combined_score"]), 3),
                "val sharpe": _format_float(_to_float(row["sharpe_val"]), 3),
                "in medium": row["in_medium"],
            }
        )

    target_columns = list(intercorr_rows[0].keys())
    target_corr_table = _make_markdown_table(intercorr_rows, target_columns)

    return f"""# Feature Analysis Report

This report turns the public feature-analysis artifacts into a browser-readable summary. The raw CSVs remain in `artifacts/feature_analysis/`; this report is the narrative layer that explains what they imply for the live candidate pool.

## Why this exists

{intro.split("## Why this matters")[0].replace("# Feature Analysis", "").strip()}

## Headline Takeaways

- The strongest validation summary in the published slice is **{best_group['target']}** with Sharpe **{_format_float(_to_float(best_group['sharpe']), 3)}** and **{_format_percent(_to_float(best_group['pct_pos']))}** positive eras.
- The public uniqueness artifact surfaces **{unique_outside_medium}** features outside the standard medium set, while the curated unique shortlist still highlights **{rain_candidates}** `rain` candidates.
- The feature shortlist for `target_ender_60` shows why the live workflow leans into a broader pool than the default Numerai feature sets.

## Validation Group Summary

{_make_markdown_table(group_summary_rows, ["target", "mean", "sharpe", "positive eras"])}

## Top Recommended Features For `target_ender_60`

{_make_markdown_table(top_recommended_rows, ["feature", "group", "val mean", "val sharpe", "positive eras"])}

## Top Unique Features Worth Inspecting

{_make_markdown_table(top_unique_rows, ["feature", "group", "combined score", "val sharpe", "in medium"])}

## Target Inter-Correlation

These correlations help explain why several targets cluster together and why `target_ender_60` can act as a stable live-training anchor while `target_ender_20` remains the fixed CORR evaluation target.

{target_corr_table}

## Artifact Links

- Source folder: `artifacts/feature_analysis/`
- Shortlist CSV: `artifacts/feature_analysis/recommended_features_target_ender_60.csv`
- Unique shortlist CSV: `artifacts/feature_analysis/recommended_unique_target_ender_60.csv`
- Uniqueness scan: `artifacts/feature_analysis/uniqueness_report.csv`
- Correlation matrix: `artifacts/feature_analysis/target_intercorr.csv`
"""




def _build_feature_analysis_report_html(markdown_text: str) -> str:
    chart_payload = _compute_feature_chart_payload()
    charts_html = ""
    chart_script = ""
    availability_note = ""

    if chart_payload:
        feature_options_html = "".join(
            f'<option value="{html.escape(feature)}"{(" selected" if feature == chart_payload["default_feature"] else "")}>{html.escape(chart_payload["feature_labels"][feature])}</option>'
            for feature in chart_payload["feature_options"]
        )
        set_options_html = "".join(
            f'<option value="{html.escape(key)}"{(" selected" if key == chart_payload["default_primary_set"] else "")}>{html.escape(label)}</option>'
            for key, label in chart_payload["set_labels"].items()
        )
        compare_options_html = "".join(
            f'<option value="{html.escape(key)}"{(" selected" if key == chart_payload["default_secondary_set"] else "")}>{html.escape(label)}</option>'
            for key, label in chart_payload["set_labels"].items()
        )
        charts_html = f"""
        <section class="chart-card">
          <div class="chart-copy">
            <div class="chart-header">
              <div>
                <h2>Single Feature Performance Across Eras</h2>
                <p>Pick an individual feature and inspect its per-era Spearman correlation against <code>{html.escape(chart_payload["target"])}</code> over the last 100 validation eras.</p>
              </div>
              <label class="control">
                <span>Feature</span>
                <select id="feature-select">{feature_options_html}</select>
              </label>
            </div>
          </div>
          <div class="interactive-chart">
            <svg id="feature-chart" viewBox="0 0 920 300" role="img" aria-label="Interactive single-feature performance chart"></svg>
            <div id="feature-tooltip" class="chart-tooltip" hidden></div>
          </div>
        </section>
        <section class="chart-card">
          <div class="chart-copy">
            <div class="chart-header">
              <div>
                <h2>Feature Set Performance Across Eras</h2>
                <p>Compare sets like <code>small</code>, <code>medium</code>, <code>faith</code>, the current training pool, and the latest selected training features.</p>
              </div>
              <div class="control-row">
                <label class="control">
                  <span>Primary set</span>
                  <select id="set-primary-select">{set_options_html}</select>
                </label>
                <label class="control">
                  <span>Compare with</span>
                  <select id="set-secondary-select">{compare_options_html}</select>
                </label>
              </div>
            </div>
          </div>
          <div class="interactive-chart">
            <svg id="set-chart" viewBox="0 0 920 300" role="img" aria-label="Interactive feature-set performance chart"></svg>
            <div id="set-tooltip" class="chart-tooltip" hidden></div>
          </div>
        </section>
        """
        chart_json = json.dumps(chart_payload)
        chart_script = f"""
  <script id="feature-chart-data" type="application/json">{chart_json}</script>
  <script>
    (function() {{
      const payload = JSON.parse(document.getElementById("feature-chart-data").textContent);
      const eras = payload.eras;
      const palette = ["#0f766e", "#b45309", "#1d4ed8", "#be123c"];

      function renderChart(svgId, tooltipId, seriesEntries) {{
        const svg = document.getElementById(svgId);
        const tooltip = document.getElementById(tooltipId);
        const width = 920;
        const height = 300;
        const pad = {{ left: 58, right: 20, top: 24, bottom: 42 }};
        const innerW = width - pad.left - pad.right;
        const innerH = height - pad.top - pad.bottom;
        const allValues = seriesEntries.flatMap((entry) => entry.values);
        const rawMin = Math.min(...allValues, -0.01);
        const rawMax = Math.max(...allValues, 0.01);
        const span = Math.max(rawMax - rawMin, 0.001);
        const yMin = rawMin - span * 0.08;
        const yMax = rawMax + span * 0.08;

        const xAt = (index) => pad.left + (eras.length <= 1 ? innerW / 2 : (index / (eras.length - 1)) * innerW);
        const yAt = (value) => pad.top + ((yMax - value) / (yMax - yMin)) * innerH;
        const yLabel = (value) => value.toFixed(3);

        const grid = [];
        [0, 0.25, 0.5, 0.75, 1].forEach((pct) => {{
          const value = yMin + (yMax - yMin) * pct;
          const y = yAt(value);
          grid.push(`<line x1="${{pad.left}}" y1="${{y.toFixed(1)}}" x2="${{width - pad.right}}" y2="${{y.toFixed(1)}}" stroke="#e6ddd0" stroke-width="1" />`);
          grid.push(`<text x="${{pad.left - 10}}" y="${{(y + 4).toFixed(1)}}" text-anchor="end" font-size="11" fill="#6a645d">${{yLabel(value)}}</text>`);
        }});

        [0, Math.floor(eras.length / 3), Math.floor((2 * eras.length) / 3), eras.length - 1]
          .filter((value, index, self) => self.indexOf(value) === index)
          .forEach((idx) => {{
            const x = xAt(idx);
            grid.push(`<line x1="${{x.toFixed(1)}}" y1="${{pad.top}}" x2="${{x.toFixed(1)}}" y2="${{height - pad.bottom}}" stroke="#f1e8db" stroke-width="1" />`);
            grid.push(`<text x="${{x.toFixed(1)}}" y="${{height - pad.bottom + 20}}" text-anchor="middle" font-size="11" fill="#6a645d">${{eras[idx]}}</text>`);
          }});

        let zeroLine = "";
        if (yMin < 0 && yMax > 0) {{
          const y = yAt(0);
          zeroLine = `<line x1="${{pad.left}}" y1="${{y.toFixed(1)}}" x2="${{width - pad.right}}" y2="${{y.toFixed(1)}}" stroke="#c2410c" stroke-dasharray="4 4" stroke-width="1.2" />`;
        }}

        const paths = seriesEntries.map((entry) => {{
          const points = entry.values.map((value, index) => `${{xAt(index).toFixed(1)}},${{yAt(value).toFixed(1)}}`).join(" ");
          return `<polyline fill="none" stroke="${{entry.color}}" stroke-width="3" points="${{points}}" />`;
        }}).join("");

        svg.innerHTML = `
          <rect x="0" y="0" width="${{width}}" height="${{height}}" rx="22" fill="#fffdf8"></rect>
          ${{grid.join("")}}
          ${{zeroLine}}
          ${{paths}}
          <line id="${{svgId}}-hover-line" x1="${{pad.left}}" y1="${{pad.top}}" x2="${{pad.left}}" y2="${{height - pad.bottom}}" stroke="#7c6f60" stroke-dasharray="3 4" stroke-width="1.2" opacity="0"></line>
          <rect id="${{svgId}}-overlay" x="${{pad.left}}" y="${{pad.top}}" width="${{innerW}}" height="${{innerH}}" fill="transparent"></rect>
        `;

        const wrapper = svg.parentElement;
        let legendEl = wrapper.querySelector(".legend");
        if (!legendEl) {{
          legendEl = document.createElement("div");
          legendEl.className = "legend";
          wrapper.appendChild(legendEl);
        }}
        legendEl.innerHTML = seriesEntries.map((entry) =>
          `<div class="legend-item"><span class="legend-swatch" style="background:${{entry.color}};"></span>${{entry.label}}</div>`
        ).join("");

        const overlay = document.getElementById(`${{svgId}}-overlay`);
        const hoverLine = document.getElementById(`${{svgId}}-hover-line`);
        overlay.addEventListener("mousemove", (event) => {{
          const bounds = svg.getBoundingClientRect();
          const relativeX = Math.max(pad.left, Math.min(width - pad.right, ((event.clientX - bounds.left) / bounds.width) * width));
          const ratio = (relativeX - pad.left) / innerW;
          const index = Math.max(0, Math.min(eras.length - 1, Math.round(ratio * (eras.length - 1))));
          const x = xAt(index);
          hoverLine.setAttribute("x1", x.toFixed(1));
          hoverLine.setAttribute("x2", x.toFixed(1));
          hoverLine.setAttribute("opacity", "1");
          const lines = [`<strong>Era ${{eras[index]}}</strong>`];
          seriesEntries.forEach((entry) => {{
            lines.push(`<span style="color:${{entry.color}};">${{entry.label}}: ${{entry.values[index].toFixed(4)}}</span>`);
          }});
          tooltip.innerHTML = lines.join("<br />");
          tooltip.hidden = false;
          tooltip.style.left = `${{event.offsetX + 18}}px`;
          tooltip.style.top = `${{event.offsetY + 12}}px`;
        }});
        overlay.addEventListener("mouseleave", () => {{
          hoverLine.setAttribute("opacity", "0");
          tooltip.hidden = true;
        }});
      }}

      function renderFeatureChart() {{
        const feature = document.getElementById("feature-select").value;
        renderChart("feature-chart", "feature-tooltip", [{{
          label: payload.feature_labels[feature],
          values: payload.feature_series[feature],
          color: palette[0],
        }}]);
      }}

      function renderSetChart() {{
        const primary = document.getElementById("set-primary-select").value;
        const secondary = document.getElementById("set-secondary-select").value;
        const entries = [{{
          label: payload.set_labels[primary],
          values: payload.set_series[primary],
          color: palette[1],
        }}];
        if (secondary && secondary !== primary) {{
          entries.push({{
            label: payload.set_labels[secondary],
            values: payload.set_series[secondary],
            color: palette[2],
          }});
        }}
        renderChart("set-chart", "set-tooltip", entries);
      }}

      document.getElementById("feature-select").addEventListener("change", renderFeatureChart);
      document.getElementById("set-primary-select").addEventListener("change", renderSetChart);
      document.getElementById("set-secondary-select").addEventListener("change", renderSetChart);
      renderFeatureChart();
      renderSetChart();
    }})();
  </script>
        """
    else:
        availability_note = (
            "<section class=\"chart-card\">"
            "<div class=\"chart-copy\">"
            "<h2>Per-era charts unavailable</h2>"
            "<p>The public export does not include parquet data, so these charts only render when local Numerai data is available in the workspace.</p>"
            "</div>"
            "</section>"
        )

    body = markdown_to_html(markdown_text)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Feature Analysis Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f3ec;
      --paper: rgba(255, 252, 246, 0.84);
      --paper-strong: #fffdf9;
      --ink: #161514;
      --muted: #615a52;
      --line: rgba(123, 107, 83, 0.18);
      --accent: #0f766e;
      --accent-2: #b45309;
      --accent-soft: rgba(15, 118, 110, 0.14);
      --shadow: 0 20px 50px rgba(61, 47, 25, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(180, 83, 9, 0.16), transparent 26%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.14), transparent 30%),
        linear-gradient(180deg, #efe6d6 0%, var(--bg) 38%, #f4efe7 100%);
    }}
    a {{ color: inherit; }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 26px 18px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.72), rgba(255,248,237,0.94));
      border: 1px solid rgba(255,255,255,0.55);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      border-radius: 28px;
      padding: 30px;
      overflow: hidden;
      position: relative;
      margin-bottom: 18px;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      right: -80px;
      top: -80px;
      width: 220px;
      height: 220px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(15, 118, 110, 0.24), transparent 70%);
    }}
    .eyebrow {{
      display: inline-block;
      padding: 0.4rem 0.7rem;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.1);
      color: var(--accent);
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      font-size: 0.76rem;
    }}
    .hero-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
    }}
    .hero-top a {{
      text-decoration: none;
      border-radius: 999px;
      padding: 0.7rem 1rem;
      font-weight: 700;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.74);
      white-space: nowrap;
    }}
    .hero h1 {{
      margin: 16px 0 12px;
      font-size: clamp(2rem, 3.2vw, 3.8rem);
      line-height: 0.98;
      letter-spacing: -0.05em;
    }}
    .hero p {{
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.65;
      font-size: 1.02rem;
    }}
    .hero-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }}
    .hero-links a {{
      text-decoration: none;
      border-radius: 999px;
      padding: 0.85rem 1.1rem;
      font-weight: 700;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.7);
    }}
    .hero-links a.primary {{
      background: var(--accent);
      color: #fff;
      border-color: transparent;
    }}
    .chart-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      flex-wrap: wrap;
    }}
    .control-row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .control {{
      display: grid;
      gap: 6px;
      min-width: 230px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .control span {{ font-weight: 600; }}
    .control select {{
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      padding: 0.7rem 0.85rem;
      color: var(--ink);
      font-size: 0.95rem;
    }}
    .chart-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 24px;
      margin-bottom: 18px;
    }}
    .chart-copy h2 {{
      margin: 0 0 8px;
      font-size: 1.35rem;
      letter-spacing: -0.03em;
    }}
    .chart-copy p {{
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.65;
    }}
    svg {{ width: 100%; height: auto; display: block; }}
    .interactive-chart {{ position: relative; }}
    .chart-tooltip {{
      position: absolute;
      pointer-events: none;
      background: rgba(28, 27, 24, 0.92);
      color: #fff;
      padding: 10px 12px;
      border-radius: 12px;
      font-size: 0.85rem;
      line-height: 1.5;
      box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
      max-width: 300px;
      z-index: 2;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.94rem;
    }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 8px; }}
    .legend-swatch {{ width: 12px; height: 12px; border-radius: 999px; }}
    article {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 24px;
    }}
    article h1 {{
      margin: 0 0 12px;
      font-size: clamp(2rem, 3.5vw, 3rem);
      letter-spacing: -0.04em;
      line-height: 0.98;
    }}
    article h2 {{ margin: 28px 0 10px; font-size: 1.35rem; letter-spacing: -0.03em; }}
    article p, article li, article td, article th {{ font-size: 1rem; line-height: 1.65; }}
    article p {{ margin: 0 0 16px; color: var(--muted); }}
    article ul {{ margin: 0 0 18px 1.2rem; padding: 0; }}
    article li + li {{ margin-top: 6px; }}
    article code {{
      font-family: "Consolas", "SFMono-Regular", monospace;
      background: rgba(244, 240, 232, 0.85);
      border-radius: 6px;
      padding: 0.15rem 0.35rem;
      font-size: 0.95em;
    }}
    article strong {{ color: var(--ink); }}
    article .table-wrap {{
      overflow-x: auto;
      margin: 18px 0 22px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--paper-strong);
      max-width: 100%;
    }}
    article table {{ width: 100%; border-collapse: collapse; min-width: 0; table-layout: fixed; }}
    article th, article td {{
      padding: 14px 16px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    article th {{
      background: rgba(244, 240, 232, 0.85);
      font-size: 0.82rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    article tr:last-child td {{ border-bottom: 0; }}
    @media (max-width: 680px) {{
      .page {{ padding: 14px 10px 28px; }}
      .hero, .chart-card, article {{ padding: 18px; }}
      .hero-top {{ flex-direction: column; align-items: flex-start; }}
      .hero h1 {{ white-space: normal; }}
      .hero-links a {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-top">
        <span class="eyebrow">Feature Analysis</span>
      </div>
      <h1>Feature Analysis Report</h1>
      <p>
        The original public export already had strong CSVs, but this version adds per-era
        visual evidence. The charts below use validation-era correlations so we can see how
        individual features and curated feature sets behave through time instead of just reading averages.
      </p>
      <div class="hero-links">
        {_nav_links_html("feature")}
      </div>
    </section>
    {charts_html or availability_note}
    <article>
      {body}
    </article>
  </div>
  {chart_script}
</body>
</html>
"""


def _metric_card(label: str, value: str, detail: str) -> str:
    return (
        "<div class=\"metric-card\">"
        f"<span class=\"metric-label\">{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        f"<p>{html.escape(detail)}</p>"
        "</div>"
    )


def _build_dashboard_html() -> str:
    experiments = _read_tsv(EXPERIMENTS_TSV)
    report_paths = sorted(REPORTS_DIR.glob("*.html"))

    best_research = max(experiments, key=lambda row: _to_float(row.get("research_score")))
    best_mmc = max(experiments, key=lambda row: _to_float(row.get("val_mmc_mean")))
    best_corr = max(experiments, key=lambda row: _to_float(row.get("val_corr_mean")))
    best_sharpe = max(experiments, key=lambda row: _to_float(row.get("val_corr_sharpe")))

    leaderboard_rows = sorted(
        experiments,
        key=lambda row: _to_float(row.get("research_score")),
        reverse=True,
    )[:6]

    weekly_paths = [
        p for p in report_paths
        if "_weekly_report" in p.stem and "example" not in p.stem
    ]
    report_cards = []
    for path in weekly_paths:
        report_cards.append(
            "<a class=\"report-card\" href=\"./{name}\">"
            f"<span>{html.escape(path.stem.replace('_', ' '))}</span>"
            "<strong>Open HTML report</strong>"
            "</a>".format(name=path.name)
        )

    if not report_cards:
        report_cards.append("<div class=\"empty-card\">No weekly reports generated yet.</div>")

    timeline_rows = []
    for row in experiments[-8:]:
        timeline_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(row['run'])}</strong><br /><span>{html.escape(row['notes'])}</span></td>"
            f"<td>{html.escape(row['target'])}</td>"
            f"<td>{_format_float(_to_float(row['val_corr_mean']), 5)}</td>"
            f"<td>{_format_float(_to_float(row['val_mmc_mean']), 5)}</td>"
            f"<td>{_format_seconds(row.get('wall_clock_s'))}</td>"
            "</tr>"
        )

    leaderboard_cards = []
    for row in leaderboard_rows:
        leaderboard_cards.append(
            "<div class=\"leader-card\">"
            f"<div><strong>{html.escape(row['run'])}</strong><p>{html.escape(row['notes'])}</p></div>"
            f"<span>{_format_float(_to_float(row['research_score']), 5)}</span>"
            "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Numerai Research Experiments Overview</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f3ec;
      --paper: rgba(255, 252, 246, 0.84);
      --paper-strong: #fffdf9;
      --ink: #161514;
      --muted: #615a52;
      --line: rgba(123, 107, 83, 0.18);
      --accent: #0f766e;
      --accent-2: #b45309;
      --accent-soft: rgba(15, 118, 110, 0.14);
      --shadow: 0 20px 50px rgba(61, 47, 25, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(180, 83, 9, 0.16), transparent 26%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.14), transparent 30%),
        linear-gradient(180deg, #efe6d6 0%, var(--bg) 38%, #f4efe7 100%);
    }}
    a {{
      color: inherit;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 26px 18px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.72), rgba(255,248,237,0.94));
      border: 1px solid rgba(255,255,255,0.55);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      border-radius: 28px;
      padding: 30px;
      overflow: hidden;
      position: relative;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      right: -80px;
      top: -80px;
      width: 220px;
      height: 220px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(15, 118, 110, 0.24), transparent 70%);
    }}
    .eyebrow {{
      display: inline-block;
      padding: 0.4rem 0.7rem;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.1);
      color: var(--accent);
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      font-size: 0.76rem;
    }}
    h1 {{
      margin: 16px 0 12px;
      font-size: clamp(2rem, 3.2vw, 3.8rem);
      line-height: 0.98;
      letter-spacing: -0.05em;
      white-space: nowrap;
    }}
    .hero p {{
      max-width: 720px;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.7;
    }}
    .hero-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }}
    .hero-links a {{
      text-decoration: none;
      border-radius: 999px;
      padding: 0.85rem 1.1rem;
      font-weight: 700;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.7);
    }}
    .hero-links a.primary {{
      background: var(--accent);
      color: #fff;
      border-color: transparent;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
      margin: 20px 0 28px;
    }}
    .metric-card, .panel, .signal-card, .leader-card, .report-card, .empty-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .metric-card {{
      padding: 18px;
    }}
    .metric-label {{
      display: block;
      color: var(--muted);
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }}
    .metric-card strong {{
      display: block;
      font-size: 1.8rem;
      margin-bottom: 6px;
    }}
    .metric-card p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 0.95rem;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.3fr 0.9fr;
      gap: 18px;
    }}
    .panel {{
      padding: 22px;
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 1.35rem;
      letter-spacing: -0.03em;
    }}
    .panel p {{
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.7;
    }}
    .leader-grid, .signal-grid, .report-grid {{
      display: grid;
      gap: 12px;
    }}
    .leader-card {{
      padding: 14px 16px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
    }}
    .leader-card p {{
      margin: 6px 0 0;
      font-size: 0.92rem;
      color: var(--muted);
    }}
    .leader-card span {{
      font-size: 1.1rem;
      font-weight: 800;
      color: var(--accent);
      white-space: nowrap;
    }}
    .signal-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .signal-card {{
      padding: 16px;
    }}
    .signal-card h3 {{
      margin: 0 0 8px;
      font-size: 1rem;
    }}
    .signal-card p, .signal-card small {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}
    .signal-bar {{
      height: 10px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.08);
      margin: 12px 0 8px;
      overflow: hidden;
    }}
    .signal-bar span {{
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), #14b8a6);
    }}
    .report-card {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 16px 18px;
      text-decoration: none;
    }}
    .report-card span {{
      text-transform: capitalize;
      font-weight: 700;
    }}
    .report-card strong {{
      color: var(--accent);
      font-size: 0.95rem;
    }}
    .empty-card {{
      padding: 18px;
      color: var(--muted);
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--paper-strong);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 640px;
    }}
    th, td {{
      text-align: left;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: rgba(244, 240, 232, 0.85);
    }}
    td span {{
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.5;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .footer-note {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.95rem;
      line-height: 1.7;
    }}
    @media (max-width: 980px) {{
      .metrics, .layout {{
        grid-template-columns: 1fr 1fr;
      }}
      .layout {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 680px) {{
      .page {{
        padding: 14px 10px 28px;
      }}
      .hero, .panel, .metric-card {{
        padding: 18px;
      }}
      .metrics {{
        grid-template-columns: 1fr;
      }}
      h1 {{
        white-space: normal;
      }}
      .leader-card, .report-card {{
        flex-direction: column;
        align-items: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <span class="eyebrow">Autoresearch</span>
      <h1>Numerai Autoresearch</h1>
      <p>
        This static dashboard turns the public repo's TSV, CSV, and weekly markdown
        artifacts into a browsable visual surface. The source of truth stays in the
        repo, while the HTML layer makes the research loop easier to scan, demo, and share.
      </p>
      <div class="hero-links">
        {_nav_links_html("overview")}
      </div>
    </section>

    <section class="metrics">
      {_metric_card("Best research score", _format_float(_to_float(best_research["research_score"])), best_research["run"])}
      {_metric_card("Best MMC", _format_float(_to_float(best_mmc["val_mmc_mean"])), best_mmc["run"])}
      {_metric_card("Best CORR", _format_float(_to_float(best_corr["val_corr_mean"])), best_corr["run"])}
      {_metric_card("Best Sharpe", _format_float(_to_float(best_sharpe["val_corr_sharpe"])), best_sharpe["run"])}
    </section>

    <section class="layout">
      <div class="panel">
        <h2>Current strategy snapshot</h2>
        <p>
          The leading public configuration is <strong>{html.escape(best_research["run"])}</strong>:
          a {html.escape(best_research["model"])} model on <code>{html.escape(best_research["target"])}</code>
          with top-{html.escape(best_research["top_k"])} dynamic features over a trailing
          {html.escape(best_research["trailing"])}-era ranking window.
        </p>
        <div class="leader-grid">
          {''.join(leaderboard_cards)}
        </div>
      </div>

      <div class="panel">
        <h2>Weekly reports</h2>
        <p>
          The public repo keeps its browser-native outputs together in this reports
          folder. These links open the rendered artifacts directly.
        </p>
        <div class="report-grid">
          {''.join(report_cards)}
        </div>
      </div>
    </section>

    <section class="panel" style="margin-top: 18px;">
      <h2>Experiment timeline</h2>
      <p>
        A quick-reading slice of the retained public ledger. The raw source remains
        <code>experiments/results.tsv</code>; this view is just a visual index.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Target</th>
              <th>CORR</th>
              <th>MMC</th>
              <th>Runtime</th>
            </tr>
          </thead>
          <tbody>
            {''.join(timeline_rows)}
          </tbody>
        </table>
      </div>
      <p class="footer-note">
        The site is generated from checked-in artifacts, which keeps it friendly to GitHub,
        code review, and reproducible local browsing without adding a frontend build step.
      </p>
    </section>
  </div>
</body>
</html>
"""


def render_report_file(markdown_path: Path) -> Path:
    markdown_text = markdown_path.read_text(encoding="utf-8")
    html_path = markdown_path.with_suffix(".html")
    if markdown_path.name == "feature_analysis_report.md":
        html_output = _build_feature_analysis_report_html(markdown_text)
    else:
        title = markdown_path.stem.replace("_", " ").title()
        html_output = build_report_html(title, markdown_text)
    html_path.write_text(html_output, encoding="utf-8")
    return html_path


def build_site() -> dict[str, list[str] | str]:
    REPORTS_DIR.mkdir(exist_ok=True)

    feature_report_md = REPORTS_DIR / "feature_analysis_report.md"
    feature_report_md.write_text(_build_feature_analysis_report_markdown(), encoding="utf-8")

    generated_reports = []
    for markdown_path in sorted(REPORTS_DIR.glob("*.md")):
        generated_reports.append(str(render_report_file(markdown_path)))

    dashboard_path = REPORTS_DIR / REPORT_INDEX_NAME
    dashboard_path.write_text(_build_dashboard_html(), encoding="utf-8")

    return {
        "dashboard": str(dashboard_path),
        "reports": generated_reports,
    }


if __name__ == "__main__":
    result = build_site()
    print(json.dumps(result, indent=2))
