#!/usr/bin/env python3
"""Limit 3: the data constrain the common NP more and more.

Two measurements y_K = mu + s_K alpha + noise(sigma_K), each with a control region
z_K = alpha + noise(tau). The uncertainties of the methods meet at both ends:
  - tau large: the control regions say nothing, back to limit 2 (all correlated methods equal);
  - tau small: each measurement pins alpha down, the correlation through alpha disappears and
    every method tends to the weighted average.
In between, using the fitted NPs helps: pois_nps_aux <= pois_nps <= pois_only. The weighted
average quotes too small an uncertainty: its true spread is given in the last column.

    python examples/limit_profiled_np.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from combifit import combine, linear_gaussian_fit  # noqa: E402

METHODS = ["weighted_average", "pois_only", "pois_nps", "pois_nps_aux"]
sigma, s = np.array([0.6, 0.9]), np.array([1.0, 0.5])


def measurements(tau):
    return [linear_gaussian_fit(f"M{k}", [[1.0, s[k]], [0.0, 1.0]], [sigma[k], tau],
                                [10.0, 0.0], ["mu", "alpha"], n_pois=1) for k in range(2)]


def true_error_of_weighted_average(meas, res):
    """sqrt(w^T V w), V the true covariance of the two mu_hat (correlated through alpha)."""
    w = np.array([wk[0, 0] for wk in res.weights])
    lam = np.array([m.cov[0, 1] for m in meas])
    V = np.diag([m.cov[0, 0] for m in meas]) + np.outer(lam, lam) - np.diag(lam**2)
    return np.sqrt(w @ V @ w)


def main():
    print(f"{'tau':>8s} " + " ".join(f"{m:>16s}" for m in METHODS) + f" {'w. avg (true)':>14s}")
    for tau in [1e3, 10.0, 3.0, 1.0, 0.3, 0.1, 1e-3]:
        meas = measurements(tau)
        res = {m: combine(meas, ["mu"], type=m) for m in METHODS}
        true = true_error_of_weighted_average(meas, res["weighted_average"])
        print(f"{tau:8.3g} " + " ".join(f"{res[m].get_uncertainty('mu'):16.4f}" for m in METHODS)
              + f" {true:14.4f}")


if __name__ == "__main__":
    main()
