"""pois_only and pois_nps against a stacked fit built block by block, without the package."""

import numpy as np
import pytest
from helpers import RawModel, stacked_reference

from combifit import combine

POIS = ["mu1", "mu2"]
LAYOUT = [(POIS, ["s1", "s2"], "A"), (["mu1"], ["s1", "s3"], "A"), (["mu2"], ["s2", "s3"], "A"),
          (["mu1"], ["s1", "s2", "s4"], "C"), (["mu2"], ["s3", "s4"], "C")]


@pytest.fixture
def model():
    return RawModel(LAYOUT, POIS, seed=11)


@pytest.mark.parametrize("method, nps", [("pois_only", ()), ("pois_nps", ("s1", "s2", "s3"))])
def test_against_stacked_reference(model, method, nps):
    res = combine(model.measurements, POIS, type=method)
    (theta, cov, gain, chi2, ndf), rows, names = stacked_reference(model.measurements, POIS, nps)
    assert res.names == names
    np.testing.assert_allclose(res.theta, theta, atol=1e-10)
    np.testing.assert_allclose(res.cov, cov, atol=1e-10)
    np.testing.assert_allclose(np.hstack(res.weights), gain, atol=1e-10)
    assert res.chi2 == pytest.approx(chi2, rel=1e-8) and res.ndf == ndf


@pytest.mark.parametrize("method", ["weighted_average", "pois_only", "pois_nps", "pois_nps_aux"])
def test_weights_sum_to_one(model, method):
    res = combine(model.measurements, POIS, type=method, **(
        {"auxiliary": model.auxiliary} if method == "pois_nps_aux" else {}))
    eye = np.eye(len(res.names))
    total = np.zeros_like(eye)
    for w, m in zip(res.weights, res.measurements):
        rows = [x for x in res.names if x in m.names and (x in POIS or m.type != "C")]
        if m.type == "auxiliary":
            rows = [x for x in res.names if x in m.names]
        total += w @ eye[[res.names.index(x) for x in rows]]
    np.testing.assert_allclose(total, eye, atol=1e-10)


def test_pois_nps_without_fitted_nps_is_pois_only(model):
    only_c = [m for m in model.measurements if m.type == "C"]
    a = combine(only_c, POIS, type="pois_nps")
    b = combine(only_c, POIS, type="pois_only")
    assert a.nps == []
    np.testing.assert_allclose(a.theta, b.theta, atol=1e-12)
    np.testing.assert_allclose(a.cov, b.cov, atol=1e-12)


def test_unknown_type_and_unmeasured_poi_raise(model):
    with pytest.raises(ValueError):
        combine(model.measurements, POIS, type="blue")
    with pytest.raises(ValueError):
        combine(model.measurements, POIS + ["mu9"], type="pois_only")
    with pytest.raises(ValueError):
        combine([], POIS)
