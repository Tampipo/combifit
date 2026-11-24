"""Ways to build measurements.

    impacts_only("LHCb", "mW", 80354.0, 32.0, {"pdf1": 5.0, "pdf2": -3.0})   # type C
    linear_gaussian_fit("A", design, sigma, data, ["mu", "alpha"], n_pois=1)  # type A
"""

from __future__ import annotations

import numpy as np

from .measurement import Measurement


def impacts_only(name: str, poi: str, value: float, total: float, impacts: dict[str, float],
                 type: str = "C") -> Measurement:
    """A measurement known by its POI, its total uncertainty and its impacts on NPs (type C).

    The impact of an NP is the shift of the POI when the auxiliary measurement of that NP moves
    by one standard deviation, i.e. Cov(POI, a). The NP entries are at the prior (value 0,
    covariance I): they carry the impacts, not a measurement of the NPs.

    Raises:
        ValueError: If the impacts exceed the total uncertainty.
    """
    nps = list(impacts)
    lam = np.array([impacts[x] for x in nps], dtype=float)
    if lam @ lam > total**2:
        raise ValueError(f"{name}: the impacts ({np.sqrt(lam @ lam):.4g} in quadrature) exceed the "
                         f"total uncertainty ({total:.4g}).")
    cov = np.block([[np.array([[total**2]]), lam[None, :]], [lam[:, None], np.eye(len(nps))]])
    return Measurement(name, np.concatenate([[value], np.zeros(len(nps))]), cov, [poi] + nps,
                       type=type)


def linear_gaussian_fit(name: str, design, sigma, data, names: list[str], n_pois: int,
                        auxiliary=None, type: str = "A") -> Measurement:
    """Fit of data = design @ theta + noise (independent Gaussian noise of width sigma), the
    parameters after the first n_pois being NPs constrained by auxiliary ~ N(alpha, I).

    Returns the post-fit measurement: theta_hat and (J^T V^-1 J + S S^T)^-1. Useful for toy
    models, tests and examples.

    Args:
        name: Name of the measurement.
        design: Matrix J, shape (n_data, len(names)).
        sigma: Noise of each datum (scalar or vector).
        data: The observed data.
        names: Parameter names, the POIs first.
        n_pois: Number of POIs.
        auxiliary: Observed auxiliary values of the NPs (default 0).
        type: Type of the measurement.
    """
    J = np.atleast_2d(np.asarray(design, dtype=float))
    w = 1.0 / np.broadcast_to(np.asarray(sigma, dtype=float), (J.shape[0],)) ** 2
    n_nps = len(names) - n_pois
    a = np.zeros(n_nps) if auxiliary is None else np.asarray(auxiliary, dtype=float)
    S = np.vstack([np.zeros((n_pois, n_nps)), np.eye(n_nps)])
    info = J.T @ (w[:, None] * J) + S @ S.T
    cov = np.linalg.inv(info)
    theta = cov @ (J.T @ (w * np.asarray(data, dtype=float)) + S @ a)
    return Measurement(name, theta, 0.5 * (cov + cov.T), list(names), type=type)
