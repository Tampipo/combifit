"""pois_nps_aux from the summaries equals the joint fit of all the raw data (sufficiency)."""

import numpy as np
import pytest
from helpers import RawModel

from combifit import combine

POIS = ["mu1", "mu2"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_summaries_reproduce_the_joint_fit(seed):
    layout = [(POIS, ["s1", "s2"], "A"), (["mu1"], ["s1", "s3"], "A"),
              (["mu2"], ["s2", "s3"], "A"), (["mu1"], ["s1", "s3"], "C")]
    model = RawModel(layout, POIS, seed=seed)
    res = combine(model.measurements, POIS, type="pois_nps_aux", auxiliary=model.auxiliary)
    names, theta, cov = model.joint_fit()
    idx = [names.index(x) for x in res.names]
    np.testing.assert_allclose(res.theta, theta[idx], atol=1e-9)
    np.testing.assert_allclose(res.cov, cov[np.ix_(idx, idx)], atol=1e-9)


def test_only_blind_impacts_reproduce_the_joint_fit():
    """Measurements whose data do not constrain the NPs, given by their impacts only:
    pois_only is already exact, and pois_nps_aux gives the same."""
    layout = [(["mu1"], ["s1", "s2"], "C"), (["mu1"], ["s1"], "C"), (["mu1"], ["s2"], "C")]
    model = RawModel(layout, ["mu1"], seed=5)
    names, theta, cov = model.joint_fit()
    for method in ("pois_only", "pois_nps_aux"):
        res = combine(model.measurements, ["mu1"], type=method, auxiliary=model.auxiliary)
        assert res.get_value("mu1") == pytest.approx(theta[0], abs=1e-9)
        assert res.cov[0, 0] == pytest.approx(cov[0, 0], abs=1e-9)


def test_auxiliary_for_unknown_np_raises():
    model = RawModel([(POIS, ["s1"], "A"), (POIS, ["s1"], "A")], POIS, seed=3)
    with pytest.raises(ValueError):
        combine(model.measurements, POIS, type="pois_nps_aux", auxiliary={"p0": 0.1})


def test_auxiliary_pulls_are_np_pulls():
    """The pull of the auxiliary entry is (a - alpha_hat) / sqrt(1 - var(alpha_hat))."""
    model = RawModel([(POIS, ["s1", "s2"], "A"), (POIS, ["s1", "s2"], "A")], POIS, seed=8)
    res = combine(model.measurements, POIS, type="pois_nps_aux", auxiliary=model.auxiliary)
    for x in ("s1", "s2"):
        expected = (model.a[x] - res.get_value(x)) / np.sqrt(1 - res.get_uncertainty(x) ** 2)
        assert res.pulls["auxiliary"][x] == pytest.approx(expected, rel=1e-8)
