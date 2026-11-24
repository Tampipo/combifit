#!/usr/bin/env python3
"""Limit 1: nothing is shared, so every method gives the weighted average.

Three measurements of mu. M3 also fits a private NP: it is profiled out exactly (its effect is
already in M3's uncertainty on mu), so it changes nothing either.

    python examples/limit_independent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from combifit import Measurement, combine, linear_gaussian_fit  # noqa: E402

measurements = [
    Measurement("M1", [10.2], [[0.5**2]], ["mu"]),
    Measurement("M2", [9.6], [[0.8**2]], ["mu"]),
    # M3: y = mu + 0.7 p + noise(0.6), its private NP p constrained by its own auxiliary
    linear_gaussian_fit("M3", [[1.0, 0.7]], 0.6, [10.5], ["mu", "p"], n_pois=1),
]

print(f"{'method':18s} {'mu':>8s} {'error':>7s}")
for method in ["weighted_average", "pois_only", "pois_nps", "pois_nps_aux"]:
    res = combine(measurements, ["mu"], type=method)
    print(f"{method:18s} {res.get_value('mu'):8.4f} {res.get_uncertainty('mu'):7.4f}")
