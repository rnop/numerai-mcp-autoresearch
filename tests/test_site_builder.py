"""The markdown renderer behind the published weekly reports."""
from pipeline.site_builder import markdown_to_html


def test_headings_lists_and_inline_markup():
    out = markdown_to_html("# Weekly Report\n\n- **mmc** is `0.003`\n- second")
    assert '<h1 id="weekly-report">Weekly Report</h1>' in out
    assert "<ul><li><strong>mmc</strong> is <code>0.003</code></li><li>second</li></ul>" in out


def test_table_skips_separator_row():
    out = markdown_to_html("| metric | value |\n|---|---|\n| corr | 0.02 |")
    assert "<th>metric</th><th>value</th>" in out
    assert "<td>corr</td><td>0.02</td>" in out
    assert "---" not in out


def test_html_is_escaped():
    out = markdown_to_html("a <script>alert(1)</script> b")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_image_becomes_figure():
    out = markdown_to_html("![Live QA](assets/qa.png)")
    assert '<img src="assets/qa.png" alt="Live QA"' in out
    assert "<figcaption>Live QA</figcaption>" in out
