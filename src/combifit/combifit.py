"""Combined measurements.

The result of a combination.
Contains additional information about the combination, such as the list of measurements that were combined and the weights used for the combination.

"""


from dataclasses import dataclass, field
import numpy as np
from scipy.stats import chi2 as chi2_distribution

from .measurement import Measurement

@dataclass
class CombinedMeasurement:
    """Combined measurement.

    Attributes:
        measurements: List of measurements that were combined.
        pois: List of parameters of interest.
        nps: Nuisance parameters combined together with the POIs (empty unless the method
            fits them). theta and cov are ordered as `names` = pois + nps.
        method: Combination method: "weighted_average", "pois_only" (marginalised),
            "pois_nps" (full) or "pois_nps_aux" (likelihood sum).
        theta: Combined best fit values of `names`.
        cov: Covariance matrix of the combined `names`.
        weights: One matrix per measurement, aligned with `measurements`. weights[k] has shape
            (len(names), n_k), its columns being the parameters of measurement k used in the
            combination, in the order of `names`, such that theta = sum_k weights[k] @ theta_k
            (theta_k restricted to these parameters). For each parameter the weights sum to one.
        chi2: chi2 at the minimum, the goodness of fit (chi2_{Y~,min} for "pois_nps_aux").
        ndf: Number of degrees of freedom of chi2, rank of the stacked covariance minus the
            number of combined parameters.
        p_value: Probability of a chi2 at least as large, computed from chi2 and ndf when not
            given; None if ndf <= 0.
        pulls: {measurement name: {parameter: pull}}, the residual of each input entry used in
            the combination divided by the standard deviation of that residual, so that each
            pull is N(0, 1) if the model holds. For "pois_nps_aux" the pulls of the "auxiliary"
            entry are the usual NP pulls (a - alpha_hat) / sqrt(1 - var(alpha_hat)).
    """


    measurements: list[Measurement] = field(repr=False)
    pois: list[str] = field(default_factory=list, repr=False)
    nps: list[str] = field(default_factory=list, repr=False)
    method: str | None = None
    theta: np.ndarray | None = field(default=None, repr=False)
    cov: np.ndarray | None = field(default=None, repr=False)
    weights: list[np.ndarray] | None = field(default=None, repr=False)
    chi2: float | None = None
    ndf: int | None = None
    p_value: float | None = None
    pulls: dict[str, dict[str, float]] | None = field(default=None, repr=False)



    def _set_parameters(self) -> None:
        """ Get all labels of parameters."""
        self.parameters = set()
        for measurement in self.measurements:
            self.parameters.update(measurement.names)


    def __post_init__(self) -> None:
        """Post-initialization validation."""
        self._set_parameters()
        if self.pois:
            self.set_pois(self.pois)
        if not all(nuisance in self.parameters for nuisance in self.nps):
            raise ValueError("All combined NPs must be in the set of parameters.")
        self.nps = list(self.nps)
        if self.theta is not None:
            self.theta = np.asarray(self.theta, dtype=float)
        if self.cov is not None:
            self.cov = np.asarray(self.cov, dtype=float)
        if self.p_value is None and self.chi2 is not None and self.ndf is not None and self.ndf > 0:
            self.p_value = float(chi2_distribution.sf(self.chi2, self.ndf))

    def set_pois(self, pois: list[str]) -> None:
        """Set the parameters of interest (POIs).

        Args:
            pois: List of parameter names to be set as POIs.
        """
        if not all(poi in self.parameters for poi in pois):
            raise ValueError("All POIs must be in the set of parameters.")
        self.pois = list(pois)

    @property
    def names(self) -> list[str]:
        """Names of the combined parameters, in the order of theta and cov."""
        return self.pois + self.nps

    def get_value(self, parameter: str) -> float:
        """Combined value of one parameter (POI or combined NP)."""
        return float(self.theta[self.names.index(parameter)])

    def get_uncertainty(self, parameter: str) -> float:
        """Combined uncertainty (standard deviation) of one parameter (POI or combined NP)."""
        index = self.names.index(parameter)
        return float(np.sqrt(self.cov[index, index]))

    def as_measurement(self, name: str = "Combined Measurement") -> Measurement:
        """The combined result as a Measurement, e.g. to use it as input of another combination."""
        return Measurement(name=name, theta=self.theta, cov=self.cov, names=self.names,
                           type=self.method, chi2=self.chi2, ndof=self.ndf)

    # ------------------------------------------------------------------ outputs

    def summary(self) -> str:
        """Combined POIs with their uncertainty, and the goodness of fit."""
        lines = [f"Combination ({self.method}) of {len(self.measurements)} measurements"]
        for poi in self.pois:
            lines.append(f"  {poi} = {self.get_value(poi):.6g} +- {self.get_uncertainty(poi):.4g}")
        if self.nps:
            lines.append(f"  {len(self.nps)} combined NPs")
        if self.chi2 is not None:
            ptxt = f", p = {self.p_value:.3g}" if self.p_value is not None else ""
            lines.append(f"  chi2 / ndf = {self.chi2:.3f} / {self.ndf}{ptxt}")
        return "\n".join(lines)

    def pull_table(self, nps_only: bool = False, sort: bool = True) -> list[dict]:
        """One row per input entry used in the combination.

        Each row has: measurement, parameter, value and error (the input entry), combined and
        combined_error (the combined parameter, nan if it is not combined, e.g. a private NP),
        pull (residual over its standard deviation, see `pulls`).

        Args:
            nps_only: Keep only the entries of NPs (not of the POIs).
            sort: Sort by decreasing |pull| (nan last).
        """
        by_name = {m.name: m for m in self.measurements}
        rows = []
        for mname, block in (self.pulls or {}).items():
            m = by_name[mname]
            for x, pull in block.items():
                if nps_only and x in self.pois:
                    continue
                j = m.get_parameter_index(x)
                combined = x in self.names
                rows.append(dict(
                    measurement=mname, parameter=x, value=float(m.theta[j]),
                    error=float(np.sqrt(m.cov[j, j])),
                    combined=self.get_value(x) if combined else float("nan"),
                    combined_error=self.get_uncertainty(x) if combined else float("nan"),
                    pull=pull))
        if sort:
            rows.sort(key=lambda r: -abs(r["pull"]) if r["pull"] == r["pull"] else 1.0)
        return rows

    def format_pulls(self, n: int | None = None, nps_only: bool = False) -> str:
        """Text table of the pulls, largest first."""
        rows = self.pull_table(nps_only=nps_only)[:n]
        width_m = max([len("measurement")] + [len(r["measurement"]) for r in rows])
        width_p = max([len("parameter")] + [len(r["parameter"]) for r in rows])
        lines = [f"{'measurement':{width_m}s}  {'parameter':{width_p}s}  {'input':>21s}  "
                 f"{'combined':>21s}  {'pull':>6s}"]
        for r in rows:
            comb = (f"{r['combined']:10.4g} +- {r['combined_error']:<7.3g}"
                    if r["combined"] == r["combined"] else f"{'-':>21s}")
            lines.append(f"{r['measurement']:{width_m}s}  {r['parameter']:{width_p}s}  "
                         f"{r['value']:10.4g} +- {r['error']:<7.3g}  {comb}  {r['pull']:+6.2f}")
        return "\n".join(lines)

    def plot_pulls(self, nps: list[str] | None = None, n_max: int = 10, ax=None, **kwargs):
        """Post-fit NP values of each measurement and of the combination, one row per NP
        (combifit.plotting.plot_np_pulls); by default the n_max NPs in most tension."""
        from .plotting import plot_np_pulls
        return plot_np_pulls(self.measurements, self, nps=nps, n_max=n_max, ax=ax, **kwargs)
