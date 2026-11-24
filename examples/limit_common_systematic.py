#!/usr/bin/env python3
"""Limit 2: a common systematic that the data do not constrain.

Two measurements y_K = mu + s_K alpha + noise(sigma_K), one datum each: the data cannot separate
mu from alpha, so the post-fit alpha is the prior (value a, variance 1). Then:
  - pois_only, pois_nps and pois_nps_aux all give the textbook BLUE with the correlated
    covariance V = [[sigma_1^2 + s_1^2, s_1 s_2], [s_1 s_2, sigma_2^2 + s_2^2]];
  - giving the measurements as type A (post-fit NPs) or as type C (impacts only) is the same
    (with type C only, no measurement fits alpha and pois_nps reduces to pois_only);
  - the weighted average ignores the correlation: wrong value and too small an uncertainty.

    python examples/limit_common_systematic.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from combifit import combine, impacts_only, linear_gaussian_fit  # noqa: E402

sigma, s, y = np.array([0.6, 0.9]), np.array([1.0, 0.5]), np.array([10.3, 9.1])

type_a = [linear_gaussian_fit(f"M{k}", [[1.0, s[k]]], sigma[k], [y[k]], ["mu", "alpha"], 1)
          for k in range(2)]
# The same measurements quoted as value, total uncertainty and impact Cov(mu_hat, a) = -s_K
type_c = [impacts_only(m.name, "mu", m.theta[0], np.sqrt(m.cov[0, 0]), {"alpha": m.cov[0, 1]})
          for m in type_a]

V = np.diag(sigma**2 + s**2) + np.outer(s, s) - np.diag(s**2)
one = np.ones(2)
blue_var = 1.0 / (one @ np.linalg.solve(V, one))
blue = blue_var * (one @ np.linalg.solve(V, y))
print(f"textbook BLUE      mu = {blue:.4f} +- {np.sqrt(blue_var):.4f}\n")

print(f"{'method':18s} {'type A':>18s} {'type C':>18s}")
for method in ["weighted_average", "pois_only", "pois_nps", "pois_nps_aux"]:
    cells = []
    for meas in (type_a, type_c):
        res = combine(meas, ["mu"], type=method)
        cells.append(f"{res.get_value('mu'):8.4f} +- {res.get_uncertainty('mu'):.4f}")
    print(f"{method:18s} {cells[0]:>18s} {cells[1]:>18s}")
