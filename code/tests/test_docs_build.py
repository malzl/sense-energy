"""HTML documentation builder: shortcodes, figure lookup, tables, charts, page assembly."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sense_energy.docs import build as db


def _setup(tmp_path, monkeypatch):
    figs = tmp_path / "figures" / "weather"
    figs.mkdir(parents=True)
    (figs / "demo_fig.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (figs / "demo_fig.pdf").write_bytes(b"%PDF-1.4")
    data = tmp_path / "reports" / "demo.csv"
    data.parent.mkdir(parents=True)
    pd.DataFrame({"model": ["a", "b"], "nmae": [0.1234, 0.5], "day_class": ["hot", "hot"]}).to_csv(
        data, index=False
    )
    docs = tmp_path / "doc"
    docs.mkdir()
    (docs / "page.md").write_text(
        "# A page\n\nIntro paragraph.\n\n## 1. Section (`demo_fig`)\n\nText.\n\n"
        "{{table: reports/demo.csv | cols=model,nmae | round=2}}\n\n"
        "{{chart: reports/demo.csv | kind=bar | x=nmae | y=model | color=day_class}}\n\n"
        "## Figures\n\nAll under figures: `demo_fig`, `missing_fig`.\n"
    )
    monkeypatch.setattr(db, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(db, "FIGURE_ROOT", tmp_path / "figures")
    monkeypatch.setattr(db, "DOC_DIR", docs)
    return docs


def test_args_parsing():
    target, opts = db._args(" reports/x.csv | cols=a,b | round=2 ")
    assert target == "reports/x.csv" and opts == {"cols": "a,b", "round": "2"}


def test_build_embeds_figures_tables_and_charts(tmp_path, monkeypatch):
    docs = _setup(tmp_path, monkeypatch)
    out = tmp_path / "html"
    written = db.build(out_dir=out, sources=[docs / "page.md"])
    html = Path(written[0]).read_text()
    assert html.count('src="figures/demo_fig.png"') == 2  # under the heading and in the figure list
    table = html[html.index("<table") : html.index("</table>")]
    assert 'class="dataframe data"' in table and "0.12" in table and "day_class" not in table
    assert "vegaEmbed" in html and '"mark"' in html
    assert (out / "figures" / "demo_fig.png").exists() and (
        out / "figures" / "demo_fig.pdf"
    ).exists()
    assert (out / "data" / "demo.csv").exists()
    assert (out / "index.html").exists() and (out / "style.css").exists()
    assert 'id="1-section-demo-fig"' in html  # anchored headings for the table of contents


def test_missing_targets_degrade_gracefully(tmp_path, monkeypatch):
    docs = _setup(tmp_path, monkeypatch)
    page = db.Page(docs / "page.md")
    assert "not found" in page.figure("nope", {})
    assert "not found" in page.table("reports/nope.csv", {})
    assert "needs x= and y=" in page.chart("reports/demo.csv", {})
