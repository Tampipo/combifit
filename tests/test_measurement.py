import numpy as np
import pytest

from combifit import Measurement, impacts_only, linear_gaussian_fit


def test_lists_are_converted():
    m = Measurement("A", [1.0, 2.0], [[1.0, 0.1], [0.1, 2.0]], ("mu", "x"))
    assert isinstance(m.theta, np.ndarray) and isinstance(m.cov, np.ndarray)
    assert m.names == ["mu", "x"]
    assert m.get_parameter_value("x") == 2.0
    assert m.get_covariance("mu", "x") == 0.1
    assert m.get_uncertainty("x") == pytest.approx(np.sqrt(2.0))


@pytest.mark.parametrize("theta, cov, names", [
    ([1.0, 2.0], np.eye(3), ["a", "b"]),          # cov too large
    ([1.0, 2.0], np.eye(2), ["a"]),               # names too short
    ([[1.0]], np.eye(1), ["a"]),                  # theta not a vector
])
def test_inconsistent_shapes_raise(theta, cov, names):
    with pytest.raises(ValueError):
        Measurement("A", theta, cov, names)


def test_unknown_parameter_raises():
    m = Measurement("A", [1.0], [[1.0]], ["mu"])
    assert not m.has_parameter("x")
    with pytest.raises(ValueError):
        m.get_parameter_index("x")


def test_impacts_only_layout():
    m = impacts_only("M", "mu", 3.0, 2.0, {"s1": 1.0, "s2": -0.5})
    assert m.type == "C" and m.names == ["mu", "s1", "s2"]
    np.testing.assert_allclose(m.theta, [3.0, 0.0, 0.0])
    np.testing.assert_allclose(m.cov[0], [4.0, 1.0, -0.5])
    np.testing.assert_allclose(m.cov[1:, 1:], np.eye(2))


def test_impacts_larger_than_total_raise():
    with pytest.raises(ValueError):
        impacts_only("M", "mu", 3.0, 1.0, {"s1": 1.0, "s2": 0.5})


def test_linear_gaussian_fit_unconstrained_np():
    """One datum y = mu + s alpha: the NP stays at its prior, Cov(mu, alpha) = -s."""
    s, sigma, a = 0.7, 0.4, 0.3
    m = linear_gaussian_fit("A", [[1.0, s]], sigma, [2.0], ["mu", "alpha"], 1, auxiliary=[a])
    np.testing.assert_allclose(m.theta, [2.0 - s * a, a])
    np.testing.assert_allclose(m.cov, [[sigma**2 + s**2, -s], [-s, 1.0]])
