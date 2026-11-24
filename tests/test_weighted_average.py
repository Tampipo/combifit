import numpy as np
import pytest
from helpers import stacked_reference
from scipy.stats import chi2 as chi2_distribution

from combifit import Measurement, combine


@pytest.fixture
def scalars():
    x, s = np.array([1.0, 1.6, 0.7, 1.2]), np.array([0.5, 0.8, 1.1, 0.6])
    return x, s, [Measurement(f"S{k}", [x[k]], [[s[k] ** 2]], ["mu"]) for k in range(len(x))]


def test_scalar_textbook(scalars):
    x, s, meas = scalars
    res = combine(meas, ["mu"], type="weighted_average")
    w = 1 / s**2
    mean = np.sum(w * x) / np.sum(w)
    assert res.get_value("mu") == pytest.approx(mean, rel=1e-12)
    assert res.get_uncertainty("mu") == pytest.approx(1 / np.sqrt(np.sum(w)), rel=1e-12)
    np.testing.assert_allclose(np.hstack(res.weights)[0], w / np.sum(w), rtol=1e-12)
    chi2 = np.sum((x - mean) ** 2 * w)
    assert res.chi2 == pytest.approx(chi2, rel=1e-12)
    assert res.ndf == len(x) - 1
    assert res.p_value == pytest.approx(chi2_distribution.sf(chi2, len(x) - 1), rel=1e-12)


def test_scalar_pulls(scalars):
    """Pull of measurement k: (x_k - mean) / sqrt(s_k^2 - var(mean))."""
    x, s, meas = scalars
    res = combine(meas, ["mu"], type="weighted_average")
    mean, var = res.get_value("mu"), res.cov[0, 0]
    for k, m in enumerate(meas):
        expected = (x[k] - mean) / np.sqrt(s[k] ** 2 - var)
        assert res.pulls[m.name]["mu"] == pytest.approx(expected, rel=1e-10)


def test_partial_pois_match_stacked_fit():
    rng = np.random.default_rng(4)
    pois = ["mu1", "mu2", "mu3"]
    layout = [pois, ["mu1", "mu2"], ["mu2", "mu3"], ["mu3"]]
    meas = []
    for k, sub in enumerate(layout):
        names = sub + [f"np{k}"]
        A = rng.normal(size=(len(names), len(names)))
        meas.append(Measurement(f"M{k}", rng.normal(size=len(names)), A @ A.T + np.eye(len(names)),
                                names))
    res = combine(meas, pois, type="weighted_average")
    (theta, cov, gain, chi2, ndf), _, _ = stacked_reference(meas, pois, correlated=False)
    np.testing.assert_allclose(res.theta, theta, atol=1e-12)
    np.testing.assert_allclose(res.cov, cov, atol=1e-12)
    np.testing.assert_allclose(np.hstack(res.weights), gain, atol=1e-12)
    assert res.chi2 == pytest.approx(chi2, rel=1e-10) and res.ndf == ndf
