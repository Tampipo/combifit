"""Compatibility and goodness of fit of a combination.

The chi2 at the minimum of a combination is the generalised least-squares chi2 of the stacked
measurements. If the model holds (Gaussian, correct covariances) it follows a chi2 law with
ndf = rank(V) - number of combined parameters, and it splits into two independent pieces by
nesting the model in a larger one where each measurement has its own POIs (Cochran's theorem):

    chi2_total = chi2_pois + chi2_nps

    chi2_pois = chi2_total - chi2_free   ~ chi2(ndf_total - ndf_free)
        do the measurements agree on the POIs? (free model: one copy of the POIs per
        measurement, NPs still common)
    chi2_nps  = chi2_free                ~ chi2(ndf_free)
        what remains once the POIs are free: agreement of the measurements on the shared NPs
        and, with the auxiliary measurement, of the NPs with their constraints.
"""

from dataclasses import dataclass, field

from scipy.stats import chi2 as chi2_distribution

from .combifit import CombinedMeasurement
from .combine import combine
from .measurement import Measurement


def _p_value(chi2: float, ndf: int) -> float | None:
    return float(chi2_distribution.sf(chi2, ndf)) if ndf > 0 else None


@dataclass
class Compatibility:
    """Goodness of fit of a combination, split into POIs and NPs.

    Attributes:
        method: Combination method.
        chi2, ndf, p_value: Total goodness of fit.
        chi2_pois, ndf_pois, p_value_pois: Compatibility of the measurements on the POIs.
        chi2_nps, ndf_nps, p_value_nps: Remainder, with one copy of the POIs per measurement.
        combined: The combination.
        free: The combination with one copy of the POIs per measurement.
    """

    method: str
    chi2: float
    ndf: int
    chi2_pois: float
    ndf_pois: int
    chi2_nps: float
    ndf_nps: int
    combined: CombinedMeasurement = field(repr=False)
    free: CombinedMeasurement = field(repr=False)

    @property
    def p_value(self) -> float | None:
        return _p_value(self.chi2, self.ndf)

    @property
    def p_value_pois(self) -> float | None:
        return _p_value(self.chi2_pois, self.ndf_pois)

    @property
    def p_value_nps(self) -> float | None:
        return _p_value(self.chi2_nps, self.ndf_nps)

    def worst_pulls(self, n: int = 5) -> list[tuple[str, str, float]]:
        """The n largest pulls in absolute value, as (measurement, parameter, pull)."""
        flat = [(m, x, v) for m, block in (self.combined.pulls or {}).items()
                for x, v in block.items() if v == v]   # v == v drops nan
        return sorted(flat, key=lambda t: -abs(t[2]))[:n]

    def summary(self) -> str:
        def line(label, chi2, ndf, p):
            ptxt = f"p = {p:.3g}" if p is not None else "p = -"
            return f"  {label:6s} chi2/ndf = {chi2:9.3f} / {ndf:<4d} {ptxt}"
        lines = [f"Compatibility ({self.method})",
                 line("total", self.chi2, self.ndf, self.p_value),
                 line("POIs", self.chi2_pois, self.ndf_pois, self.p_value_pois),
                 line("NPs", self.chi2_nps, self.ndf_nps, self.p_value_nps),
                 "  largest pulls: " + ", ".join(f"{m}:{x} {v:+.2f}"
                                                 for m, x, v in self.worst_pulls())]
        return "\n".join(lines)


def free_pois(measurements: list[Measurement], pois: list[str]):
    """Copies of the measurements where each one has its own POIs, renamed "poi[measurement]".

    Returns:
        The renamed measurements and the list of renamed POIs.
    """
    names = [m.name for m in measurements]
    if len(set(names)) != len(names):
        raise ValueError(f"Measurement names must be unique to free the POIs: {names}")
    renamed, free = [], []
    for m in measurements:
        mapping = {poi: f"{poi}[{m.name}]" for poi in pois if m.has_parameter(poi)}
        free += list(mapping.values())
        renamed.append(Measurement(m.name, m.theta, m.cov, [mapping.get(x, x) for x in m.names],
                                   type=m.type))
    return renamed, free


def compatibility(measurements: list[Measurement], pois: list[str],
                  type: str = "pois_nps_aux",
                  auxiliary: dict[str, float] | None = None) -> Compatibility:
    """Total goodness of fit of a combination and its split into POIs and NPs.

    Args:
        measurements: List of measurements to be combined.
        pois: List of parameters of interest (POIs).
        type: Combination method, as in combine().
        auxiliary: Observed values of the auxiliary measurement, as in combine().
    """
    combined = combine(measurements, pois, type=type, auxiliary=auxiliary)
    renamed, free_names = free_pois(measurements, pois)
    free = combine(renamed, free_names, type=type, auxiliary=auxiliary)
    return Compatibility(
        method=type,
        chi2=combined.chi2, ndf=combined.ndf,
        chi2_pois=combined.chi2 - free.chi2, ndf_pois=combined.ndf - free.ndf,
        chi2_nps=free.chi2, ndf_nps=free.ndf,
        combined=combined, free=free,
    )
