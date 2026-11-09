"""Build the project's HTML documentation from the write-ups and the analysis outputs.

The prose lives in ``doc/*.md``; the pages are built to ``doc/html/`` with the result tables,
figures and charts embedded, so the write-up and its evidence travel together:

* ``{{figure: name | caption=...}}`` embeds ``code/reports/figures/**/<name>.png`` (found by
  name; the PDF stays next to it for print). A heading that names figures in backticks, as the
  write-ups do, gets those figures embedded under it without a shortcode.
* ``{{table: path.csv | cols=a,b | round=2 | head=30 | sort=-col | index=col}}`` embeds a CSV
  from the repository as an HTML table with a download link.
* ``{{chart: path.csv | kind=bar|line|scatter|point | x=col | y=col | color=col | title=... |
  head=5000}}`` embeds an interactive Vega-Lite chart (vega-embed from a CDN, data inline).

Markdown is rendered with mistune (tables, footnotes, strikethrough); the template is jinja2.
``sense-energy build-docs`` writes every page plus ``index.html`` and copies the figures and
data files it references. Pages work offline except for the interactive charts.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT
from ..logging_utils import get_logger

logger = get_logger(__name__)

DOC_DIR = PROJECT_ROOT / "doc"
OUT_DIR = DOC_DIR / "html"
FIGURE_ROOT = PROJECT_ROOT / "code" / "reports" / "figures"
VEGA_CDN = (
    '<script src="https://cdn.jsdelivr.net/npm/vega@5.30.0"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-lite@5.21.0"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.26.0"></script>'
)
SHORTCODE = re.compile(r"\{\{\s*(figure|table|chart)\s*:\s*([^}]*)\}\}")
SKIP = {"README.md"}
LONG_TABLE = 25  # longer tables fold into an expandable block
#: chart colours in the order of the figure specification (Okabe-Ito)
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442", "#7A7A7A"]

CSS = """
:root { --ink: #1f1f1f; --muted: #5a5a5a; --rule: #d9d9d9; --accent: #0072b2; --bg: #ffffff; }
* { box-sizing: border-box; }
body { margin: 0; font: 16px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--ink); background: var(--bg); }
.wrap { display: grid; grid-template-columns: 240px minmax(0, 860px); gap: 40px; max-width: 1180px; margin: 0 auto; padding: 24px 20px 80px; }
nav.site { font-size: 14px; position: sticky; top: 16px; align-self: start; max-height: calc(100vh - 32px); overflow: auto; }
nav.site a { display: block; color: var(--muted); text-decoration: none; padding: 3px 0; }
nav.site a.current { color: var(--accent); font-weight: 600; }
nav.site .toc { margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--rule); }
nav.site .toc a.h3 { padding-left: 14px; font-size: 13px; }
main h1 { font-size: 30px; line-height: 1.2; margin: 0 0 6px; }
main h2 { font-size: 22px; margin: 40px 0 10px; padding-top: 10px; border-top: 1px solid var(--rule); }
main h3 { font-size: 18px; margin: 28px 0 8px; }
main p, main li { max-width: 72ch; }
main code { font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }
main pre { background: #f6f6f6; padding: 12px 14px; overflow: auto; border-radius: 4px; }
main pre code { background: none; padding: 0; }
table.data { border-collapse: collapse; font-size: 13px; margin: 12px 0 4px; display: block; overflow-x: auto; max-width: 100%; }
table.data th, table.data td { border-bottom: 1px solid var(--rule); padding: 5px 10px; text-align: right; white-space: nowrap; }
table.data th { text-align: right; font-weight: 600; background: #fafafa; position: sticky; top: 0; }
table.data td:first-child, table.data th:first-child { text-align: left; }
table.data tr:nth-child(even) td { background: #fbfbfb; }
main table:not(.data) { border-collapse: collapse; font-size: 14px; margin: 12px 0; }
main table:not(.data) th, main table:not(.data) td { border-bottom: 1px solid var(--rule); padding: 5px 10px; text-align: left; vertical-align: top; }
.source { font-size: 12px; color: var(--muted); margin: 0 0 18px; }
.source a { color: var(--accent); }
figure { margin: 18px 0 26px; }
figure img { max-width: 100%; height: auto; border: 1px solid #eee; background: #fff; }
figure figcaption { font-size: 13px; color: var(--muted); margin-top: 6px; max-width: 72ch; }
.figrow { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.figrow figure { flex: 1 1 320px; max-width: calc(50% - 8px); margin: 10px 0 18px; }
figure.single { max-width: 640px; }
details.table { margin: 10px 0 18px; }
details.table summary { cursor: pointer; color: var(--accent); font-size: 14px; }
.chart { margin: 18px 0 26px; width: 100%; min-width: 280px; }
.meta { color: var(--muted); font-size: 13px; margin-bottom: 26px; }
.index li { margin: 10px 0; }
.index .desc { color: var(--muted); font-size: 14px; display: block; }
@media (max-width: 900px) { .wrap { grid-template-columns: 1fr; } nav.site { position: static; max-height: none; } }
@media print { nav.site { display: none; } .wrap { display: block; } }
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} · SENSE energy</title>
<link rel="stylesheet" href="style.css">
{% if charts %}{{ vega }}{% endif %}
</head>
<body>
<div class="wrap">
<nav class="site">
  <a href="index.html"{% if current == 'index' %} class="current"{% endif %}>SENSE energy: documentation</a>
  {% for p in pages %}<a href="{{ p.file }}"{% if p.stem == current %} class="current"{% endif %}>{{ p.title }}</a>
  {% endfor %}
  {% if toc %}<div class="toc">{% for t in toc %}<a class="{{ t.level }}" href="#{{ t.id }}">{{ t.text }}</a>
  {% endfor %}</div>{% endif %}
</nav>
<main>
{{ body }}
</main>
</div>
{% if charts %}<script>
window.addEventListener("load", function () {
{% for c in charts %}  (function () {
    var el = document.getElementById("{{ c.id }}"), spec = {{ c.spec }};
    spec.width = Math.max(280, el.clientWidth - 90);
    vegaEmbed(el, spec, {actions: {export: true, source: false, compiled: false, editor: false}});
  })();
{% endfor %}});
</script>{% endif %}
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Shortcodes
# --------------------------------------------------------------------------- #


def _args(text: str) -> tuple[str, dict[str, str]]:
    parts = [p.strip() for p in text.split("|")]
    opts = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            opts[k.strip()] = v.strip()
    return parts[0], opts


def find_figure(name: str) -> Path | None:
    hits = sorted(FIGURE_ROOT.glob(f"**/{name}.png"))
    return hits[0] if hits else None


class Page:
    def __init__(self, src: Path):
        self.src = src
        self.stem = src.stem
        self.file = f"{src.stem}.html"
        self.charts: list[dict] = []
        self.assets: set[Path] = set()
        self.data: set[Path] = set()
        self.title = src.stem.replace("_", " ")

    # -- figure ----------------------------------------------------------- #
    def figure(self, name: str, opts: dict[str, str]) -> str:
        path = find_figure(name)
        if path is None:
            return f'<p class="source">figure <code>{name}</code> not found</p>'
        self.assets.add(path)
        cap = opts.get("caption", "")
        pdf = path.with_suffix(".pdf")
        link = f' <a href="figures/{pdf.name}">PDF</a>' if pdf.exists() else ""
        return (
            f'<figure class="{opts.get("class", "single")}"><img src="figures/{path.name}" alt="{name}" loading="lazy">'
            f"<figcaption>{cap}{' ' if cap else ''}<code>{name}</code>{link}</figcaption></figure>"
        )

    # -- table ------------------------------------------------------------ #
    def table(self, rel: str, opts: dict[str, str]) -> str:
        path = PROJECT_ROOT / rel
        if not path.exists():
            return f'<p class="source">table <code>{rel}</code> not found</p>'
        df = pd.read_csv(path)
        if "cols" in opts:
            cols = [c for c in opts["cols"].split(",") if c.strip() in df.columns]
            df = df[[c.strip() for c in cols]]
        if "sort" in opts:
            col = opts["sort"].lstrip("-")
            if col in df.columns:
                df = df.sort_values(col, ascending=not opts["sort"].startswith("-"))
        if "head" in opts:
            df = df.head(int(opts["head"]))
        if "index" in opts and opts["index"] in df.columns:
            df = df.set_index(opts["index"])
        nd = int(opts.get("round", 3))
        num = df.select_dtypes("number").columns
        df[num] = df[num].round(nd)
        self.data.add(path)
        html = df.to_html(
            classes="data",
            border=0,
            index="index" in opts,
            na_rep="",
            float_format=lambda v: f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) >= 1e-9 else "0",
        )
        src = f'<p class="source">source: <a href="data/{path.name}">{rel}</a> ({len(df)} rows shown)</p>'
        if len(df) > LONG_TABLE:
            return f'<details class="table"><summary>Table: {len(df)} rows</summary>{html}</details>{src}'
        return f"{html}{src}"

    # -- chart ------------------------------------------------------------ #
    def chart(self, rel: str, opts: dict[str, str]) -> str:
        path = PROJECT_ROOT / rel
        if not path.exists():
            return f'<p class="source">chart data <code>{rel}</code> not found</p>'
        df = pd.read_csv(path)
        if "cols" in opts:
            df = df[[c.strip() for c in opts["cols"].split(",") if c.strip() in df.columns]]
        df = df.head(int(opts.get("head", 5000)))
        kind = opts.get("kind", "bar")
        mark = {
            "bar": "bar",
            "line": "line",
            "scatter": "point",
            "point": "point",
            "area": "area",
        }.get(kind, "bar")
        x, y = opts.get("x"), opts.get("y")
        if x is None or y is None:
            return '<p class="source">chart needs x= and y=</p>'
        qx = "quantitative" if pd.api.types.is_numeric_dtype(df[x]) else "nominal"
        qy = "quantitative" if pd.api.types.is_numeric_dtype(df[y]) else "nominal"
        enc = {
            "x": {"field": x, "type": qx, "title": opts.get("xlabel", x)},
            "y": {"field": y, "type": qy, "title": opts.get("ylabel", y)},
        }
        if "color" in opts and opts["color"] in df.columns:
            enc["color"] = {
                "field": opts["color"],
                "type": "nominal",
                "title": opts.get("colorlabel", opts["color"]),
            }
            if mark == "bar" and qy == "nominal":
                enc["yOffset"] = {"field": opts["color"]}
            if mark == "bar" and qx == "nominal":
                enc["xOffset"] = {"field": opts["color"]}
        tooltip = [
            {
                "field": c,
                "type": "quantitative" if pd.api.types.is_numeric_dtype(df[c]) else "nominal",
            }
            for c in df.columns[:12]
        ]
        enc["tooltip"] = tooltip
        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": opts.get("title", ""),
            "width": "container",
            "height": int(opts.get("height", 320)),
            "data": {"values": json.loads(df.to_json(orient="records", date_format="iso"))},
            "mark": {"type": mark, "tooltip": True}
            if mark != "line"
            else {"type": "line", "point": True, "tooltip": True},
            "encoding": enc,
            "config": {
                "font": "Helvetica, Arial, sans-serif",
                "range": {"category": OKABE_ITO},
                "axis": {"labelFontSize": 12, "titleFontSize": 13},
                "legend": {"orient": "bottom"},
            },
        }
        cid = f"chart{len(self.charts) + 1}"
        self.charts.append({"id": cid, "spec": json.dumps(spec)})
        self.data.add(path)
        return f'<div class="chart" id="{cid}"></div><p class="source">interactive; source: <a href="data/{path.name}">{rel}</a></p>'

    def expand(self, kind: str, body: str) -> str:
        target, opts = _args(body)
        return getattr(self, kind)(target, opts)


# --------------------------------------------------------------------------- #
# Markdown handling
# --------------------------------------------------------------------------- #

HEADING_FIGS = re.compile(r"^(#{2,3} .*?)(\n)", re.M)


def auto_figures(text: str) -> str:
    """Under a heading that names figures in backticks, embed them as a row."""

    def repl(m):
        heading = m.group(1)
        names = [n for n in re.findall(r"`([A-Za-z0-9_]+)`", heading) if find_figure(n)]
        if not names:
            return m.group(0)
        row = (
            '<div class="figrow">'
            + "".join(f"{{{{figure: {n} | class=inrow}}}}" for n in names)
            + "</div>"
        )
        return f"{heading}\n\n{row}\n"

    return HEADING_FIGS.sub(repl, text)


FIGURE_SECTION = re.compile(r"(^## Figures[^\n]*\n)(.*?)(?=^## |\Z)", re.M | re.S)


def figure_sections(text: str) -> str:
    """A ``## Figures`` section that lists figure names in backticks gets them all embedded."""

    def repl(m):
        body = m.group(2)
        names = []
        for n in re.findall(r"`([A-Za-z0-9_]+)`", body):
            if find_figure(n) and n not in names:
                names.append(n)
        if not names:
            return m.group(0)
        row = (
            '<div class="figrow">'
            + "".join(f"{{{{figure: {n} | class=inrow}}}}" for n in names)
            + "</div>"
        )
        return f"{m.group(1)}{body.rstrip()}\n\n{row}\n\n"

    return FIGURE_SECTION.sub(repl, text)


def render_markdown(text: str) -> str:
    import mistune

    md = mistune.create_markdown(
        escape=False, plugins=["table", "strikethrough", "footnotes", "task_lists", "url"]
    )
    return md(text)


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "section"


def add_anchors(html: str) -> tuple[str, list[dict]]:
    toc = []

    def repl(m):
        level, inner = m.group(1), m.group(2)
        text = re.sub(r"<[^>]+>", "", inner)
        sid = slug(text)[:60]
        toc.append({"level": f"h{level}", "id": sid, "text": text})
        return f'<h{level} id="{sid}">{inner}</h{level}>'

    html = re.sub(r"<h([23])>(.*?)</h\1>", repl, html, flags=re.S)
    return html, toc


def build_page(page: Page, pages: list[Page]) -> str:
    from jinja2 import Template

    text = page.src.read_text()
    m = re.search(r"^# (.+)$", text, re.M)
    page.title = m.group(1).strip() if m else page.title
    text = figure_sections(auto_figures(text))
    text = SHORTCODE.sub(lambda mm: page.expand(mm.group(1), mm.group(2)), text)
    body = render_markdown(text)
    body, toc = add_anchors(body)
    body += f'<p class="meta">Built from <code>doc/{page.src.name}</code> by <code>sense-energy build-docs</code>.</p>'
    return Template(TEMPLATE).render(
        title=page.title,
        body=body,
        pages=pages,
        current=page.stem,
        toc=toc,
        charts=page.charts,
        vega=VEGA_CDN,
    )


def build_index(pages: list[Page]) -> str:
    from jinja2 import Template

    items = []
    for p in pages:
        text = p.src.read_text()
        paras = [
            b.strip()
            for b in re.split(r"\n\s*\n", text)
            if b.strip() and not b.strip().startswith(("#", "|", "{{", "<", "-", "*"))
        ]
        desc = re.sub(r"[`*_\[\]]", "", paras[0])[:220] if paras else ""
        items.append(f'<li><a href="{p.file}">{p.title}</a><span class="desc">{desc}</span></li>')
    body = (
        '<h1>SENSE energy: documentation</h1><p class="meta">Research threads, data and results of the NHS site '
        "energy project, with the result tables and figures embedded. The printable summary is the PI report.</p>"
        f'<ul class="index">{"".join(items)}</ul>'
    )
    return Template(TEMPLATE).render(
        title="Documentation", body=body, pages=pages, current="index", toc=[], charts=[], vega=""
    )


def clean(html: str) -> str:
    """No trailing whitespace, one final newline (what the repository's hooks enforce)."""
    return "\n".join(line.rstrip() for line in html.splitlines()).rstrip("\n") + "\n"


def build(out_dir: Path = OUT_DIR, sources: list[Path] | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)
    srcs = sources or sorted(p for p in DOC_DIR.glob("*.md") if p.name not in SKIP)
    pages = [Page(p) for p in srcs]
    for p in pages:  # titles first, for the navigation
        m = re.search(r"^# (.+)$", p.src.read_text(), re.M)
        p.title = m.group(1).strip() if m else p.title
    written = []
    for p in pages:
        html = build_page(p, pages)
        (out_dir / p.file).write_text(clean(html))
        for a in p.assets:
            shutil.copy2(a, out_dir / "figures" / a.name)
            pdf = a.with_suffix(".pdf")
            if pdf.exists():
                shutil.copy2(pdf, out_dir / "figures" / pdf.name)
        for d in p.data:
            shutil.copy2(d, out_dir / "data" / d.name)
        written.append(out_dir / p.file)
        logger.info(
            "%s: %d figures, %d tables/charts data files, %d charts",
            p.file,
            len(p.assets),
            len(p.data),
            len(p.charts),
        )
    (out_dir / "index.html").write_text(clean(build_index(pages)))
    (out_dir / "style.css").write_text(clean(CSS))
    return written
