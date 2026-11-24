"""Pseudo-experiments (toys) of the summary statistics.

In the linear Gaussian model, the fitted vectors theta_K of all the measurements and the
auxiliary measurement a of the shared NPs are jointly Gaussian:

    E[theta_K] = true values,        Cov(theta_K) = Sigma_K,
    E[a] = true shared NPs,          Cov(a) = I,
    Cov(theta_K, theta_L) = Lambda_K Lambda_L^T,   Cov(theta_K, a) = Lambda_K   (K != L),

with Lambda_K = Sigma_K[:, shared NPs] (master formula). Toys are drawn from this law, keeping the
covariances of the measurements: they test the combination itself (bias, coverage of the quoted
uncertainty, chi2 law of the goodness of fit), not the individual fits.

For a measurement of type C (impacts only, NP block = prior), the NP entries come out as exact
copies of a, which is what "post-fit = prior" means.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import kstest

from .combifit import CombinedMeasurement
from .combine import _get_shared_parameters, combine
from .compatibility import compatibility
from .measurement import Measurement


def _truth_dict(truth) -> dict[str, float]:
    if isinstance(truth, CombinedMeasurement):
        return dict(zip(truth.names, (float(v) for v in truth.theta)))
    return dict(truth or {})


def joint_model(measurements: list[Measurement], pois: list[str], truth=None):
    """Mean and covariance of the stacked (theta_1, ..., theta_N, a).

    Args:
        measurements: The measurements.
        pois: The POIs (every other shared parameter is an NP constrained by a).
        truth: True values, as a dict or a CombinedMeasurement. Parameters not given take, for
            each measurement, its own fitted value; shared NPs not given are at 0 in a.

    Returns:
        mean, cov and the list of shared NPs (the components of a, at the end of the vector).
    """
    truth = _truth_dict(truth)
    shared = sorted(_get_shared_parameters(measurements) - set(pois))
    means, blocks, lambdas = [], [], []
    for m in measurements:
        means.append([truth.get(x, float(v)) for x, v in zip(m.names, m.theta)])
        blocks.append(m.cov)
        lam = np.zeros((len(m.names), len(shared)))
        for c, x in enumerate(shared):
            if m.has_parameter(x):
                lam[:, c] = m.cov[:, m.get_parameter_index(x)]
        lambdas.append(lam)
    means.append([truth.get(x, 0.0) for x in shared])
    blocks.append(np.eye(len(shared)))
    lambdas.append(np.eye(len(shared)))
    L = np.vstack(lambdas)
    cov = block_diag(*blocks) + L @ L.T - block_diag(*[lam @ lam.T for lam in lambdas])
    return np.concatenate([np.asarray(x, dtype=float) for x in means]), 0.5 * (cov + cov.T), shared


def generate_toys(measurements: list[Measurement], pois: list[str], truth=None,
                  n_toys: int = 1000, seed=None, psd_tol: float = 1e-8):
    """Draw pseudo-experiments of the measurements and of the auxiliary measurement.

    Args:
        measurements: The measurements (their covariances are kept).
        pois: The POIs.
        truth: True values, see joint_model. Typically a combination result.
        n_toys: Number of toys.
        seed: Seed or numpy Generator.
        psd_tol: Relative tolerance on negative eigenvalues of the joint covariance. Inputs that
            do not come from one linear Gaussian model (e.g. a post-fit NP variance above its
            prior) give negative eigenvalues; they are set to 0 with a warning beyond this.

    Returns:
        A list of (measurements, auxiliary) pairs, ready for combine(..., auxiliary=auxiliary).
    """
    rng = np.random.default_rng(seed)
    mean, cov, shared = joint_model(measurements, pois, truth)
    eigval, eigvec = np.linalg.eigh(cov)
    if eigval.min() < -psd_tol * eigval.max():
        warnings.warn(f"Joint covariance not positive semi-definite (smallest eigenvalue "
                      f"{eigval.min():.3g}, largest {eigval.max():.3g}): the inputs are not "
                      "those of one linear Gaussian model; negative eigenvalues set to 0.")
    root = eigvec * np.sqrt(np.clip(eigval, 0.0, None))
    draws = mean + rng.standard_normal((n_toys, len(mean))) @ root.T

    sizes = np.cumsum([0] + [len(m.names) for m in measurements])
    toys = []
    for draw in draws:
        toy = [Measurement(m.name, draw[sizes[k]:sizes[k + 1]], m.cov, m.names, type=m.type)
               for k, m in enumerate(measurements)]
        toys.append((toy, dict(zip(shared, draw[sizes[-1]:]))))
    return toys


@dataclass
class ToyResults:
    """Combinations of the toys.

    Attributes:
        method: Combination method.
        names: Combined parameters (columns of theta and errors).
        truth: True values of `names`.
        theta: Combined values, shape (n_toys, len(names)).
        errors: Quoted uncertainties, same shape.
        chi2, ndf: Goodness of fit of each toy.
        chi2_pois, ndf_pois: POI compatibility of each toy (None unless requested).
    """

    method: str
    names: list[str]
    truth: np.ndarray
    theta: np.ndarray
    errors: np.ndarray
    chi2: np.ndarray
    ndf: np.ndarray
    chi2_pois: np.ndarray | None = None
    ndf_pois: np.ndarray | None = None

    def save(self, path) -> None:
        """Save to a .npz file (reload with ToyResults.load)."""
        arrays = dict(method=np.array(self.method), names=np.array(self.names, dtype=str),
                      truth=self.truth, theta=self.theta, errors=self.errors,
                      chi2=self.chi2, ndf=self.ndf)
        if self.chi2_pois is not None:
            arrays.update(chi2_pois=self.chi2_pois, ndf_pois=self.ndf_pois)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path) -> "ToyResults":
        with np.load(path, allow_pickle=False) as d:
            return cls(method=str(d["method"]), names=[str(x) for x in d["names"]],
                       truth=d["truth"], theta=d["theta"], errors=d["errors"], chi2=d["chi2"],
                       ndf=d["ndf"],
                       chi2_pois=d["chi2_pois"] if "chi2_pois" in d else None,
                       ndf_pois=d["ndf_pois"] if "ndf_pois" in d else None)

    @property
    def pulls(self) -> np.ndarray:
        """(theta - truth) / quoted error: N(0, 1) if the combination is unbiased and its
        uncertainty correct."""
        return (self.theta - self.truth) / self.errors

    def coverage(self, parameter: str) -> dict[str, float]:
        """Bias, true spread and mean quoted uncertainty of one parameter."""
        j = self.names.index(parameter)
        return dict(bias=float(np.mean(self.theta[:, j]) - self.truth[j]),
                    rms=float(np.std(self.theta[:, j], ddof=1)),
                    quoted=float(np.mean(self.errors[:, j])),
                    pull_mean=float(np.mean(self.pulls[:, j])),
                    pull_std=float(np.std(self.pulls[:, j], ddof=1)))

    def chi2_check(self, pois: bool = False) -> dict[str, float]:
        """Mean chi2 against ndf, and Kolmogorov-Smirnov test that the p-values are uniform."""
        chi2, ndf = (self.chi2_pois, self.ndf_pois) if pois else (self.chi2, self.ndf)
        from scipy.stats import chi2 as chi2_distribution
        p = chi2_distribution.sf(chi2, ndf)
        return dict(mean_chi2=float(np.mean(chi2)), ndf=float(np.mean(ndf)),
                    ks_p_value=float(kstest(p, "uniform").pvalue))

    def summary(self, parameters: list[str] | None = None) -> str:
        n = len(self.chi2)
        lines = [f"Toys ({self.method}, {n} toys)"]
        for x in parameters or self.names[:1]:
            c = self.coverage(x)
            lines.append(f"  {x}: bias {c['bias']:+.3g}, rms {c['rms']:.4g} vs quoted "
                         f"{c['quoted']:.4g}, pull {c['pull_mean']:+.3f} +- {c['pull_std']:.3f}")
        c = self.chi2_check()
        lines.append(f"  total chi2: mean {c['mean_chi2']:.2f} vs ndf {c['ndf']:.0f}, "
                     f"KS p-value of the p-values {c['ks_p_value']:.3f}")
        if self.chi2_pois is not None:
            c = self.chi2_check(pois=True)
            lines.append(f"  POI chi2:   mean {c['mean_chi2']:.2f} vs ndf {c['ndf']:.0f}, "
                         f"KS p-value of the p-values {c['ks_p_value']:.3f}")
        return "\n".join(lines)


def run_toys(measurements: list[Measurement], pois: list[str], truth=None, n_toys: int = 1000,
             type: str = "pois_nps_aux", seed=None, with_compatibility: bool = False,
             toys=None) -> ToyResults:
    """Generate toys and combine each of them.

    Args:
        measurements, pois, truth, n_toys, seed: As in generate_toys.
        type: Combination method, as in combine().
        with_compatibility: Also compute the POI compatibility of each toy (twice slower).
        toys: Toys already generated by generate_toys, e.g. to run several methods on the same
            toys; then n_toys and seed are ignored.
    """
    truth = _truth_dict(truth)
    if toys is None:
        toys = generate_toys(measurements, pois, truth, n_toys, seed)
    rows = []
    for toy, auxiliary in toys:
        if with_compatibility:
            comp = compatibility(toy, pois, type=type, auxiliary=auxiliary)
            res = comp.combined
            rows.append((res, comp.chi2_pois, comp.ndf_pois))
        else:
            res = combine(toy, pois, type=type, auxiliary=auxiliary)
            rows.append((res, None, None))
    names = rows[0][0].names
    return ToyResults(
        method=type,
        names=names,
        truth=np.array([truth.get(x, 0.0) for x in names]),
        theta=np.array([r.theta for r, _, _ in rows]),
        errors=np.array([np.sqrt(np.diag(r.cov)) for r, _, _ in rows]),
        chi2=np.array([r.chi2 for r, _, _ in rows]),
        ndf=np.array([r.ndf for r, _, _ in rows]),
        chi2_pois=np.array([c for _, c, _ in rows]) if with_compatibility else None,
        ndf_pois=np.array([d for _, _, d in rows]) if with_compatibility else None,
    )
