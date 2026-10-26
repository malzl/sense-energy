"""TNP experiment: name parsing, calendar features and a forward pass of each architecture."""

from __future__ import annotations

import numpy as np
import pytest

from sense_energy.experiments import tnp_forecast as tf

torch = pytest.importorskip("torch")
pytest.importorskip("tnp")

SPEC = {
    "embed_dim": 16,
    "num_heads": 2,
    "head_dim": 8,
    "num_layers": 2,
    "num_latents": 8,
    "grid_spacing_periods": 2,
    "window_tokens": 8,
    "top_k_ctot": 2,
}


def test_parse_name():
    assert tf.parse_name("tnp") == ("tnp", "none")
    assert tf.parse_name("pttnp_ifs") == ("pttnp", "ifs")
    assert tf.parse_name("gtnp_era5") == ("gtnp", "era5")
    with pytest.raises(KeyError):
        tf.parse_name("tnp_gfs")


@pytest.mark.parametrize("arch", tf.ARCHS)
def test_each_architecture_returns_a_gaussian_per_target(arch):
    from check_shapes import ShapeCheckingState, set_enable_check_shapes

    set_enable_check_shapes(ShapeCheckingState.ENABLED)
    torch.manual_seed(0)
    L, H, dx, B = 96, 12, 7, 3
    model = tf.build_model(arch, dx, SPEC, L, H)
    tau = torch.arange(-L + 1, H + 1, dtype=torch.float32) / tf.WEEK
    x = torch.cat([tau[None, :, None].expand(B, -1, -1), torch.randn(B, L + H, dx - 1)], dim=-1)
    xc, xt = x[:, :L], x[:, L:]
    yc = torch.randn(B, L, 1)
    dist = model(xc, yc, xt)
    assert dist.loc.shape == (B, H, 1) and dist.scale.shape == (B, H, 1)
    assert torch.all(dist.scale > 0)
    nll = tf._gaussian_nll(dist.loc[..., 0], dist.scale[..., 0], torch.randn(B, H))
    nll.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_gaussian_nll_ignores_missing_targets():
    loc, scale = torch.zeros(1, 4), torch.ones(1, 4)
    y = torch.tensor([[0.0, float("nan"), 0.0, float("nan")]])
    assert float(tf._gaussian_nll(loc, scale, y)) == pytest.approx(0.5 * np.log(2 * np.pi))
