#!/usr/bin/env python3
"""Toys and compatibility.

A synthetic set of measurements drawn from the linear Gaussian model (type A with post-fit NPs,
type C with impacts only, shared and private NPs). For each method: compatibility of the observed
measurements, then toys around the combined values to check the bias, the coverage of the quoted
uncertainty and the chi2 laws of the goodness of fit.

    python examples/toys_and_compatibility.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from combifit import Measurement, combine, compatibility, generate_toys, run_toys  # noqa: E402

METHODS = ["weighted_average", "pois_only", "pois_nps", "pois_nps_aux"]


def synthetic(seed=3):
    """One POI, 3 measurements of type A and 2 of type C, NPs in units of their prior.
    cov_K = (H_K + S S^T)^-1 with H_K = J^T J the information of the data of K; the NP columns
    of J are half aligned with the POI one, so that the NPs have large impacts on the POI."""
    rng = np.random.default_rng(seed)
    layout = [(["s1", "s2", "s3", "p0"], "A"), (["s1", "s2", "p1"], "A"), (["s2", "s3"], "A"),
              (["s1", "s2", "s3"], "C"), (["s1", "s3", "p4"], "C")]
    meas = []
    for k, (nps, t) in enumerate(layout):
        p = 1 + len(nps)
        J = rng.normal(size=(p + 4, p))
        J[:, 1:] = 0.5 * J[:, [0]] + 0.5 * J[:, 1:]
        cov = np.linalg.inv(J.T @ J + np.diag([0.0] + [1.0] * len(nps)))
        meas.append(Measurement(f"M{k}", rng.multivariate_normal(np.zeros(p), cov), cov,
                                ["mu"] + nps, type=t))
    return meas


def report(measurements, pois, n_toys, seed, label):
    print(f"\n=== {label}")
    truth = combine(measurements, pois, type="pois_nps_aux")
    print(f"truth for the toys = pois_nps_aux combination: {pois[0]} = {truth.theta[0]:.4f}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        toys = generate_toys(measurements, pois, truth, n_toys=n_toys, seed=seed)
    for w in caught:
        print("warning:", str(w.message)[:160])
    print(f"{'method':17s} {'value':>12s} {'quoted':>8s} | {'total':>17s} {'POIs':>15s}"
          f" {'NPs':>17s} | toys: {'bias':>7s} {'rms':>7s} {'quoted':>7s} {'pull std':>8s}"
          f" {'<chi2>/ndf':>13s} {'KS':>5s} {'<dchi2>/ndf':>11s} {'KS':>5s}")
    for method in METHODS:
        c = compatibility(measurements, pois, type=method)
        r = run_toys(measurements, pois, truth, type=method, toys=toys, with_compatibility=True)
        cov = r.coverage(pois[0])
        tot, poi = r.chi2_check(), r.chi2_check(pois=True)
        fmt = lambda chi2, ndf, p: f"{chi2:7.2f}/{ndf:<3d}" + (f" {p:.0e}" if p is not None else "  -  ")
        print(f"{method:17s} {c.combined.theta[0]:12.4f} {c.combined.get_uncertainty(pois[0]):8.4f} |"
              f" {fmt(c.chi2, c.ndf, c.p_value):>17s} {fmt(c.chi2_pois, c.ndf_pois, c.p_value_pois):>15s}"
              f" {fmt(c.chi2_nps, c.ndf_nps, c.p_value_nps):>17s} |"
              f" {cov['bias']:+7.3f} {cov['rms']:7.3f} {cov['quoted']:7.3f} {cov['pull_std']:8.3f}"
              f" {tot['mean_chi2']:7.2f}/{tot['ndf']:<5.0f} {tot['ks_p_value']:5.2f}"
              f" {poi['mean_chi2']:5.2f}/{poi['ndf']:<5.0f} {poi['ks_p_value']:5.2f}")
    print("largest pulls (pois_nps_aux):",
          ", ".join(f"{m}:{x} {v:+.2f}" for m, x, v in
                    compatibility(measurements, pois, type="pois_nps_aux").worst_pulls(4)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--toys", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    report(synthetic(), ["mu"], args.toys, args.seed, "synthetic measurements")


if __name__ == "__main__":
    main()
