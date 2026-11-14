"""Shared style for the kickoff-deck figures (Oxford / SENSE slide template).

The deck has its own look: Oxford blue, Helvetica for charts, the template's chart palette in
its fixed order. Figures are drawn at the size they have on the 160 x 90 mm slide, so the font sizes
are true. Each script writes ``<name>.pdf`` (vector, for LaTeX), ``<name>.png`` and
``<name>_data.csv`` to ``doc/kickoff_deck/figures/``. No titles and no text inside the plots
beyond axis labels and a legend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from sense_energy.config import PROJECT_ROOT

DECK = PROJECT_ROOT / "doc" / "kickoff_deck"
OUT = DECK / "figures"
REPORTS = PROJECT_ROOT / "code" / "reports"

OXFORD_BLUE = "#002147"
SHELL = "#F1EEE9"
CHARCOAL = "#211D1C"
ASH = "#61615F"
STONE = "#D9D8D6"
GOLD = "#E0C040"
#: Chart palette of the template, in its order of use.
PALETTE = ("#274EB3", "#FB5607", "#00AAB4", "#D0AE2C", "#8F1D6D", "#95C11F")
BLUE, ORANGE, TEAL, YELLOW, PURPLE, GREEN = PALETTE
LIGHT_BLUE = "#95B3E9"
MID_GREY = "#A9A8A5"
MM = 1 / 25.4
#: Helvetica and its metric clones, in order of preference (the slide text is TeX Gyre Heros).
FONTS = ("Helvetica", "TeX Gyre Heros", "Nimbus Sans", "Liberation Sans")


def setup() -> None:
    have = {f.name for f in font_manager.fontManager.ttflist}
    family = next(f for f in (*FONTS, "DejaVu Sans") if f in have)
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 6.5,
            "axes.labelsize": 6.8,
            "axes.labelcolor": CHARCOAL,
            "axes.edgecolor": ASH,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": False,
            "grid.color": STONE,
            "grid.linewidth": 0.4,
            "xtick.color": ASH,
            "ytick.color": ASH,
            "xtick.labelcolor": CHARCOAL,
            "ytick.labelcolor": CHARCOAL,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.fontsize": 6.3,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 0.9,
            "legend.borderaxespad": 0.2,
            "text.color": CHARCOAL,
            "pdf.fonttype": 42,
            "savefig.transparent": True,
        }
    )


def figure(width_mm: float, height_mm: float, **kw):
    """A figure of exactly this size on the slide; the layout engine keeps labels and legends
    inside it, so the saved page is not cropped and the font sizes stay true."""
    setup()
    return plt.subplots(figsize=(width_mm * MM, height_mm * MM), layout="constrained", **kw)


def legend_below(fig, ax, ncol: int, reverse: bool = False, **kw):
    """Legend under the plot, inside the figure."""
    h, lab = ax.get_legend_handles_labels()
    if reverse:
        h, lab = h[::-1], lab[::-1]
    kw = {"handlelength": 0.8, "columnspacing": 0.8, "fontsize": 6.0, **kw}
    return fig.legend(h, lab, loc="outside lower center", ncol=ncol, **kw)


def england_map(ax, face: str = SHELL, edge: str = STONE, lw: float = 0.4) -> None:
    """England outline on ``ax``, aspect-corrected, axis off."""
    from sense_energy.eda import common as C

    eng = C.england()
    eng.plot(ax=ax, facecolor=face, edgecolor=edge, linewidth=lw, zorder=1)
    minx, miny, maxx, maxy = eng.total_bounds
    ax.set_xlim(minx - 0.15, maxx + 0.15)
    ax.set_ylim(miny - 0.1, maxy + 0.1)
    ax.set_aspect(1 / np.cos(np.deg2rad(53)))
    ax.set_axis_off()


def save(fig, name: str, data=None, facecolor: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kw = {"facecolor": facecolor, "transparent": False} if facecolor else {}
    fig.savefig(OUT / f"{name}.pdf", **kw)
    fig.savefig(OUT / f"{name}.png", dpi=400, transparent=False, facecolor=facecolor or "white")
    if data is not None:
        data.to_csv(Path(OUT / f"{name}_data.csv"), index=False)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")
