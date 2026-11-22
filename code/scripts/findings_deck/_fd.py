"""Shared style for the findings-deck figures (doc/findings_deck).

The deck follows the kickoff deck: the same slide template, palette and fonts
(``code/scripts/deck/_deck.py``), figures drawn at their exact size on the 160 x 90 mm slide.
Each slide has one figure across the full width (145.6 mm); its panels carry a short bold title
at the top left, so the slide needs no caption. Colours keep one meaning across the deck:
cold blue, heat orange, sites with a plant gold, the zero-shot forecast blue, trained models
dark teal, reference forecasts grey.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deck"))

import _deck  # noqa: E402
from _deck import (  # noqa: E402, F401
    ASH,
    BLUE,
    CHARCOAL,
    GOLD,
    GREEN,
    LIGHT_BLUE,
    MID_GREY,
    ORANGE,
    OXFORD_BLUE,
    REPORTS,
    SHELL,
    STONE,
    TEAL,
    TEXT_GOLD,
    TEXT_ORANGE,
    TEXT_TEAL,
    YELLOW,
    england_map,
    figure,
    legend_below,
)

from sense_energy.config import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "doc" / "findings_deck" / "figures"
#: full width of the content area and the height of a figure below a one-line subtitle (mm)
WIDTH, HEIGHT = 145.6, 55.5
COLD, HEAT = BLUE, ORANGE
PLANT, PLANT_EDGE = YELLOW, TEXT_GOLD
MODEL, TRAINED, NAIVE = BLUE, TEXT_TEAL, MID_GREY


def save(fig, name: str, data=None, facecolor: str | None = None) -> None:
    _deck.save(fig, name, data=data, facecolor=facecolor, out=OUT)


def title(ax, text: str, pad: float = 4.0) -> None:
    """The panel's name at its top left (bold, charcoal), in place of a caption."""
    ax.set_title(text, loc="left", fontsize=6.7, fontweight="bold", color=CHARCOAL, pad=pad)


def note(ax, x: float, y: float, text: str, color: str = ASH, **kw) -> None:
    """A direct label in axes coordinates."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=6.0, color=color, **kw)
