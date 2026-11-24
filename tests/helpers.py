"""Independent references for the tests: written without the code of the package."""

from __future__ import annotations

import numpy as np

from combifit import Measurement, linear_gaussian_fit


def gls(Y, U, V):
    """theta, cov, gain (theta = gain @ Y), chi2 and ndf of the GLS fit of Y = U theta."""
    W = np.linalg.inv(V)
    cov = np.linalg.inv(U.T @ W @ U)
    gain = cov @ U.T @ W
    theta = gain @ Y
    r = Y - U @ theta
    return theta, cov, gain, float(r @ W @ r), len(Y) - U.shape[1]


def stacked_reference(measurements, pois, nps=(), correlated=True):
    """Stack, for each measurement, its POIs and (unless of type C) its NPs among `nps`; build V
    block by block: V_KK = cov_K on these rows and, if `correlated`,
    V_KL = sum over the NPs x present in both of cov_K[rows_K, x] cov_L[rows_L, x]^T.

    Returns (theta, cov, gain, chi2, ndf), the rows of each measurement and the names."""
    names = list(pois) + list(nps)
    rows = [[x for x in (pois if m.type == "C" else names) if x in m.names] for m in measurements]
    start = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
    Y, U = np.zeros(start[-1]), np.zeros((start[-1], len(names)))
    V = np.zeros((start[-1], start[-1]))
    for k, (m, rk) in enumerate(zip(measurements, rows)):
        blk = slice(start[k], start[k + 1])
        idx = [m.names.index(x) for x in rk]
        Y[blk] = m.theta[idx]
        U[blk, [names.index(x) for x in rk]] = np.eye(len(rk))
        V[blk, blk] = m.cov[np.ix_(idx, idx)]
        if not correlated:
            continue
        for j, (o, rj) in enumerate(zip(measurements, rows)):
            if j == k:
                continue
            jdx = [o.names.index(x) for x in rj]
            for x in [x for x in m.names if x in o.names and x not in pois]:
                V[blk, start[j]:start[j + 1]] += np.outer(m.cov[idx, m.names.index(x)],
                                                          o.cov[jdx, o.names.index(x)])
    return gls(Y, U, V), rows, names


class RawModel:
    """Measurements built from raw data in the linear Gaussian model, and the joint fit of all
    the raw data with each auxiliary measurement counted once (the target of pois_nps_aux).

    Each measurement k has data y_k = J_k theta_k + noise(1) over its POIs, shared NPs and one
    private NP; every NP has an auxiliary measurement a ~ N(alpha, I).
    """

    def __init__(self, layout, pois, seed=0, n_extra=3):
        """layout: list of (pois of k, shared NPs of k, type of k). Type "C" measurements get a
        single datum: their data do not constrain their NPs, so the impacts-only summary loses
        nothing."""
        rng = np.random.default_rng(seed)
        self.pois = list(pois)
        self.shared = sorted({x for _, nps, _ in layout for x in nps})
        self.a = {x: float(rng.normal()) for x in self.shared}
        self.blocks, self.measurements = [], []
        for k, (pk, sk, t) in enumerate(layout):
            names = list(pk) + list(sk) + [f"p{k}"]
            self.a[f"p{k}"] = float(rng.normal())
            n_data = 1 if t == "C" else len(names) + n_extra
            J = rng.normal(size=(n_data, len(names)))
            y = rng.normal(size=n_data)
            self.blocks.append((names, J, y))
            m = linear_gaussian_fit(f"M{k}", J, 1.0, y, names, n_pois=len(pk),
                                    auxiliary=[self.a[x] for x in names[len(pk):]], type=t)
            if t == "C":
                m = Measurement(m.name, np.concatenate([m.theta[:len(pk)], np.zeros(len(sk) + 1)]),
                                np.block([[m.cov[:len(pk), :len(pk)], m.cov[:len(pk), len(pk):]],
                                          [m.cov[len(pk):, :len(pk)], np.eye(len(sk) + 1)]]),
                                names, type="C")
            self.measurements.append(m)

    def joint_fit(self):
        """theta and cov of the joint fit of all the raw data and all the auxiliary values,
        over the POIs, the shared NPs and the private NPs."""
        names = self.pois + self.shared + [f"p{k}" for k in range(len(self.blocks))]
        rows, data = [], []
        for nk, J, y in self.blocks:
            full = np.zeros((J.shape[0], len(names)))
            full[:, [names.index(x) for x in nk]] = J
            rows.append(full)
            data.append(y)
        nps = names[len(self.pois):]
        aux = np.zeros((len(nps), len(names)))
        aux[np.arange(len(nps)), [names.index(x) for x in nps]] = 1.0
        X, Y = np.vstack(rows + [aux]), np.concatenate(data + [[self.a[x] for x in nps]])
        cov = np.linalg.inv(X.T @ X)
        return names, cov @ X.T @ Y, cov

    @property
    def auxiliary(self):
        return {x: self.a[x] for x in self.shared}
