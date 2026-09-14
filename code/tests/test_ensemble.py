"""Mixture of member quantile forecasts."""

import numpy as np

from sense_energy.experiments.ensemble import mixture_quantiles

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _gauss_q(mu, sd):
    from scipy.stats import norm

    return mu + sd * norm.ppf(LEVELS)


def test_identical_members_return_the_member_quantiles():
    q = np.tile(_gauss_q(100, 5)[None, None, :], (7, 3, 1))
    out = mixture_quantiles(q, LEVELS)
    assert out.shape == (3, 9)
    assert np.allclose(out, q[0], atol=0.2)


def test_disagreeing_members_widen_the_mixture_and_keep_monotone_quantiles():
    q = np.stack([_gauss_q(mu, 5)[None, :] for mu in (90, 100, 110)])  # 3 members x 1 step
    out = mixture_quantiles(q, LEVELS)
    assert (np.diff(out[0]) > 0).all()
    assert out[0, -1] - out[0, 0] > (_gauss_q(100, 5)[-1] - _gauss_q(100, 5)[0]) + 10
    assert abs(out[0, 4] - 100) < 1.0
