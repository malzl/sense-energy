"""The shared figure style must encode the master specification exactly."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from sense_energy.visualization import style  # noqa: E402


def test_configure_sets_typography_and_axes():
    style.configure_matplotlib()
    assert mpl.rcParams["font.weight"] == "normal" and mpl.rcParams["axes.labelweight"] == "normal"
    assert mpl.rcParams["pdf.fonttype"] == 42 and mpl.rcParams["text.usetex"] is False
    assert mpl.rcParams["axes.labelsize"] == 8 and mpl.rcParams["xtick.labelsize"] == 7
    assert mpl.rcParams["axes.linewidth"] == 0.7 and mpl.rcParams["xtick.major.width"] == 0.6
    assert mpl.rcParams["xtick.direction"] == "out" and mpl.rcParams["axes.spines.top"] is False


def test_allowed_sizes_only():
    fig, _ = style.figure(style.SINGLE_COLUMN)
    assert tuple(fig.get_size_inches()) == (3.5, 2.8)
    plt.close(fig)
    with pytest.raises(ValueError):
        style.figure((6.0, 4.0))


def test_role_colors_are_the_specified_okabe_ito_values():
    assert style.ROLE_COLORS == {
        "reference": "#000000",
        "simple_baseline": "#7A7A7A",
        "classical_model": "#56B4E9",
        "foundation_model": "#0072B2",
        "specialist_model": "#009E73",
        "exact_bayes": "#E69F00",
        "pfn": "#D55E00",
        "zero_effect": "#B8B8B8",
    }


def test_every_method_has_a_role_and_fixed_order_never_reorders_by_value():
    for method in style.METHOD_ORDER:
        style.method_color(method)
    assert style.ordered(["LightGBM", "Seasonal naive", "Ridge"]) == [
        "Seasonal naive",
        "Ridge",
        "LightGBM",
    ]
    with pytest.raises(KeyError):
        style.ordered(["lgbm_v1_ckpt"])


def test_method_names_cover_the_model_registry():
    from sense_energy.models.train import MODEL_REGISTRY

    assert set(style.METHOD_NAMES) == set(MODEL_REGISTRY)


def test_bottom_legend_is_outside_below_and_borderless():
    fig, ax = style.figure()
    ax.plot([0, 1], [0, 1], label="Seasonal naive")
    legend = style.add_bottom_legend(ax)
    assert legend.get_frame_on() is False
    assert legend._loc == 9  # upper center
    bbox = legend.get_bbox_to_anchor()._bbox
    assert (bbox.x0, bbox.y0) == pytest.approx((0.5, -0.22))
    assert legend.get_texts()[0].get_text() == "seasonal naive"  # lowercase amendment
    plt.close(fig)


def test_save_figure_writes_pdf_png_and_data_then_closes(tmp_path):
    fig, ax = style.figure()
    ax.plot([0, 1], [0, 1])
    out = style.save_figure(
        fig, tmp_path / "fig", png_dpi=300, data=pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    )
    assert out["pdf"].exists() and out["png"].exists() and out["data"].name == "fig_data.csv"
    assert not plt.get_fignums()


def test_fixed_horizon_order_and_labels():
    assert style.HORIZON_ORDER_HOURS == [0.5, 1, 3, 6, 12, 24, 48, 168]
    assert style.QUANTITY_LABELS["coverage"] == "Coverage (%, target 80)"
    assert np.all([label[0].isupper() for label in style.QUANTITY_LABELS.values()])
