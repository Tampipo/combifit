"""Limiting cases where the methods meet (see examples/limit_*.py)."""

import numpy as np
import pytest

from combifit import Measurement, combine, impacts_only, linear_gaussian_fit

METHODS = ["weighted_average", "pois_only", "pois_nps", "pois_nps_aux"]


def test_nothing_shared_all_methods_are_the_weighted_average():
    meas = [Measurement("M1", [10.2], [[0.25]], ["mu"]), Measurement("M2", [9.6], [[0.64]], ["mu"]),
            linear_gaussian_fit("M3", [[1.0, 0.7]], 0.6, [10.5], ["mu", "p"], 1)]
    ref = combine(meas, ["mu"], type="weighted_average")
    for method in METHODS[1:]:
        res = combine(meas, ["mu"], type=method)
        assert res.get_value("mu") == pytest.approx(ref.get_value("mu"), abs=1e-12)
        assert res.get_uncertainty("mu") == pytest.approx(ref.get_uncertainty("mu"), abs=1e-12)


def common_systematic(tau=None):
    """y_K = mu + s_K alpha + noise(sigma_K), with a control region z_K = alpha + noise(tau)."""
    sigma, s, y = [0.6, 0.9], [1.0, 0.5], [10.3, 9.1]
    out = []
    for k in range(2):
        J, sig, data = [[1.0, s[k]]], [sigma[k]], [y[k]]
        if tau is not None:
            J, sig, data = J + [[0.0, 1.0]], sig + [tau], data + [0.0]
        out.append(linear_gaussian_fit(f"M{k}", J, sig, data, ["mu", "alpha"], 1))
    return out


def test_unconstrained_common_systematic_is_the_textbook_blue():
    meas = common_systematic()
    V = np.array([[0.36 + 1.0, 0.5], [0.5, 0.81 + 0.25]])
    y = np.array([m.theta[0] for m in meas])
    one = np.ones(2)
    var = 1 / (one @ np.linalg.solve(V, one))
    blue = var * (one @ np.linalg.solve(V, y))
    type_c = [impacts_only(m.name, "mu", m.theta[0], np.sqrt(m.cov[0, 0]), {"alpha": m.cov[0, 1]})
              for m in meas]
    for method in METHODS[1:]:
        for inputs in (meas, type_c):
            res = combine(inputs, ["mu"], type=method)
            assert res.get_value("mu") == pytest.approx(blue, abs=1e-10)
            assert res.cov[0, 0] == pytest.approx(var, abs=1e-10)
    wa = combine(meas, ["mu"], type="weighted_average")
    assert wa.cov[0, 0] < var       # ignores the positive correlation


def test_methods_meet_at_both_ends_and_are_ordered_in_between():
    errors = {}
    for tau in (1e4, 1.0, 1e-4):
        meas = common_systematic(tau)
        errors[tau] = {m: combine(meas, ["mu"], type=m).get_uncertainty("mu") for m in METHODS}
    loose, mid, tight = errors[1e4], errors[1.0], errors[1e-4]
    for m in ("pois_nps", "pois_nps_aux"):
        assert loose[m] == pytest.approx(loose["pois_only"], rel=1e-6)
    for m in METHODS[1:]:
        assert tight[m] == pytest.approx(tight["weighted_average"], rel=1e-6)
    assert mid["pois_nps_aux"] < mid["pois_nps"] < mid["pois_only"]
