"""Plots of combinations (matplotlib, optional dependency: pip install combifit[plot])."""

from __future__ import annotations

import numpy as np

from .combifit import CombinedMeasurement
from .measurement import Measurement

# Categorical colours, assigned to the measurements in their order (validated palette)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
ROW_SHADE = "#efeee9"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
TOYS = SERIES[0]
OBSERVED = SERIES[1]


def _fitted(measurement: Measurement) -> bool:
    """Whether the measurement reports fitted NP values (type A, not the auxiliary)."""
    return measurement.type not in ("B", "C", "auxiliary")


def select_nps(measurements: list[Measurement], combined: CombinedMeasurement | None = None,
               n_max: int = 10) -> list[str]:
    """The n_max NPs in most tension: largest |pull| of the combination over all its entries
    (measurements and auxiliary). Without a combination: largest |fitted value|."""
    if combined is not None and combined.pulls:
        score = {}
        for block in combined.pulls.values():
            for x, v in block.items():
                if x in combined.nps and v == v:          # v == v drops nan
                    score[x] = max(score.get(x, 0.0), abs(v))
    else:
        score = {}
        for m in measurements:
            if _fitted(m):
                for x, v in zip(m.names, m.theta):
                    score[x] = max(score.get(x, 0.0), abs(float(v)))
    return sorted(score, key=lambda x: -score[x])[:n_max]


def plot_np_pulls(measurements: list[Measurement], combined: CombinedMeasurement | None = None,
                  nps: list[str] | None = None, n_max: int = 10, ax=None,
                  combined_style: str = "band", xlim=None, title: str | None = None):
    """Post-fit NP values of each measurement and of the combination, one row per NP.

    Each measurement that fitted the NPs is a point at its post-fit value with its post-fit
    uncertainty; the combination is a band (value +- uncertainty) across the row, or a point
    with combined_style="point". NPs are in units of their prior, so 0 is the prior value.
    Rows alternate shaded and white; the NP in most tension is at the top.

    Args:
        measurements: The measurements (types B and C, and the auxiliary, have no NP value).
        combined: The combination; its NP values are drawn, and its pulls choose the NPs.
        nps: NPs to show, top to bottom. Default: the n_max in most tension (select_nps).
        n_max: Number of NPs when `nps` is not given.
        ax: Matplotlib axes; a new figure if None.
        combined_style: "band" or "point".
        xlim: x range; default symmetric around 0.
        title: Axes title.

    Returns:
        The axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    nps = list(nps) if nps is not None else select_nps(measurements, combined, n_max)
    fitted = [m for m in measurements if _fitted(m)]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 0.45 * len(nps) + 1.2))

    n = len(nps)
    y_of = {x: n - 1 - i for i, x in enumerate(nps)}     # first NP at the top
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=ROW_SHADE, zorder=0, linewidth=0)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)

    extent = 1.0
    if combined is not None:
        for x in nps:
            if x not in combined.names:
                continue
            v, s = combined.get_value(x), combined.get_uncertainty(x)
            extent = max(extent, abs(v) + s)
            y = y_of[x]
            if combined_style == "band":
                ax.add_patch(Rectangle((v - s, y - 0.42), 2 * s, 0.84, facecolor=INK,
                                       alpha=0.13, linewidth=0, zorder=2))
                ax.plot([v, v], [y - 0.42, y + 0.42], color=INK, linewidth=1.6, zorder=2)
            else:
                ax.errorbar(v, y, xerr=s, fmt="D", ms=4.5, color=INK, elinewidth=1.4,
                            capsize=3, zorder=3)

    offsets = np.linspace(-0.22, 0.22, len(fitted)) if len(fitted) > 1 else [0.0]
    for k, (m, dy) in enumerate(zip(fitted, offsets)):
        color, marker = SERIES[k % len(SERIES)], MARKERS[k % len(MARKERS)]
        for x in nps:
            if not m.has_parameter(x):
                continue
            j = m.get_parameter_index(x)
            v, s = float(m.theta[j]), float(np.sqrt(m.cov[j, j]))
            extent = max(extent, abs(v) + s)
            ax.errorbar(v, y_of[x] - dy, xerr=s, fmt=marker, ms=4.5, color=color,
                        elinewidth=1.4, capsize=3, capthick=1.4, zorder=3)

    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(nps)))
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.5, n - 0.5)
    lim = xlim or (-1.1 * extent, 1.1 * extent)
    ax.set_xlim(*lim)
    ax.set_xlabel("post-fit NP value (units of the prior)")
    if title:
        ax.set_title(title)

    handles = [Line2D([], [], marker=MARKERS[k % len(MARKERS)], color=SERIES[k % len(SERIES)],
                      linestyle="-", label=m.name) for k, m in enumerate(fitted)]
    if combined is not None:
        handles.append(Patch(facecolor=INK, alpha=0.25, label="combination")
                       if combined_style == "band"
                       else Line2D([], [], marker="D", color=INK, label="combination"))
    ax.legend_handles_np_pulls = handles     # for a figure-level legend
    return ax


# --------------------------------------------------------------------------- toys

def _hist(ax, values, bins, headroom=1.0):
    heights, _, _ = ax.hist(values, bins=bins, density=True, color=TOYS, edgecolor=SURFACE,
                            linewidth=0.4)
    ax.set_ylim(0, headroom * heights.max())


def _chi2_panel(ax, toys, ndf, observed=None, observed_ndf=None):
    from scipy.stats import chi2 as chi2_distribution
    ndf = int(np.median(ndf))
    top = max(np.percentile(toys, 99.8), chi2_distribution.ppf(0.999, ndf))
    hi = top if observed is None else max(top, observed * 1.08)
    lo = 0.0 if ndf <= 6 else max(0.0, min([np.percentile(toys, 0.2)]
                                            + ([observed * 0.9] if observed is not None else [])))
    _hist(ax, toys, np.linspace(lo, top, 40), headroom=1.35)
    x = np.linspace(max(lo, 1e-3), hi, 400)
    ax.plot(x, chi2_distribution.pdf(x, ndf), color=INK2, linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    if observed is not None:
        p = chi2_distribution.sf(observed, observed_ndf) if observed_ndf > 0 else None
        ax.axvline(observed, color=OBSERVED, linewidth=1.5)
        right = (observed - lo) / (hi - lo) > 0.5      # keep the text inside the axes
        ax.annotate(f"observed {observed:.1f} / {observed_ndf}"
                    + (f"\np = {p:.2g}" if p is not None else ""),
                    xy=(observed, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(-4 if right else 4, -2), textcoords="offset points",
                    ha="right" if right else "left", va="top", fontsize="small", color=INK2)


def plot_toys(results: dict, parameter: str, observed: dict | None = None,
              parameter_label: str | None = None, unit: str = "", title: str = "",
              rows: tuple = ("parameter", "chi2", "chi2_pois"), figsize=None):
    """Toys of several combination methods: one column per method; rows: the combined
    parameter (with the Gaussian of the mean quoted uncertainty around the truth), the total chi2
    and, if the toys have it, the POI chi2 (each with its chi2 law and the observed value).

    Args:
        results: {method: ToyResults}, typically run_toys on the same toys for each method.
        parameter: The combined parameter shown in the first row.
        observed: {method: Compatibility} of the observed measurements, drawn as vertical lines.
        parameter_label: Its name on the axis (default: parameter).
        unit: Its unit.
        title: Figure title.
        rows: Rows to draw, among "parameter", "chi2" (total) and "chi2_pois" (dropped when the
            toys have no POI compatibility).
        figsize: Figure size in inches; default 3.25 x 2.75 per panel.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from scipy.stats import norm

    label = parameter_label or parameter
    with_pois = all(r.chi2_pois is not None for r in results.values())
    rows = [r for r in rows if r != "chi2_pois" or with_pois]
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, len(results), squeeze=False,
                             figsize=figsize or (3.25 * len(results), 2.75 * nrows))
    for j, (method, r) in enumerate(results.items()):
        obs = (observed or {}).get(method)
        for i, row in enumerate(rows):
            ax = axes[i, j]
            if row == "parameter":
                k = r.names.index(parameter)
                theta, err, truth = r.theta[:, k], r.errors[:, k], float(r.truth[k])
                rms, quoted = float(np.std(theta, ddof=1)), float(np.mean(err))
                lo, hi = truth - 4.5 * rms, truth + 4.5 * rms
                _hist(ax, theta, np.linspace(lo, hi, 40), headroom=1.85)
                x = np.linspace(lo, hi, 400)
                ax.plot(x, norm.pdf(x, truth, quoted), color=INK2, linewidth=1.5)
                ax.set_xlim(lo, hi)
                ax.set_yticks([])
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                u = f" {unit}" if unit else ""
                ax.text(0.02, 0.97, f"rms {rms:.3g}{u}\nquoted {quoted:.3g}{u}\n"
                        f"rms / quoted {rms / quoted:.3f}", transform=ax.transAxes, va="top",
                        fontsize="small", color=INK2)
                ax.set_xlabel(f"combined {label}" + (f" [{unit}]" if unit else ""))
            elif row == "chi2":
                _chi2_panel(ax, r.chi2, r.ndf, *(obs.chi2, obs.ndf) if obs else (None, None))
                ax.set_xlabel("total $\\chi^2$")
            else:
                _chi2_panel(ax, r.chi2_pois, r.ndf_pois,
                            *(obs.chi2_pois, obs.ndf_pois) if obs else (None, None))
                ax.set_xlabel("$\\chi^2$ of the POIs")
        axes[0, j].set_title(method.replace("_", " "), color=INK, fontweight="bold")

    names = {"parameter": label, "chi2": "goodness of fit", "chi2_pois": "POI compatibility"}
    for i, row in enumerate(rows):
        axes[i, 0].set_ylabel(names[row], color=INK2)
    n_toys = len(next(iter(results.values())).chi2)
    handles = [Patch(color=TOYS, label=f"toys ({n_toys})"),
               Line2D([], [], color=INK2, linewidth=1.5,
                      label="expected: Gaussian of the quoted uncertainty / chi2 law")]
    if observed:
        handles.append(Line2D([], [], color=OBSERVED, linewidth=1.5, label="observed data"))
    if title:
        fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize="large", fontweight="bold")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.955 if title else 0.995),
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_toy_overview(results: dict, parameter: str, parameter_label: str | None = None,
                      unit: str = "", title: str = "Toys: true spread against quoted uncertainty"):
    """True spread (rms over the toys, with its statistical uncertainty) against the mean quoted
    uncertainty of one parameter: one panel per method, one row per configuration.

    Args:
        results: {configuration label: {method: ToyResults}}.
        parameter: The combined parameter.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = list(results)
    methods = list(next(iter(results.values())))
    fig, axes = plt.subplots(1, len(methods), figsize=(3.25 * len(methods),
                                                       0.42 * len(labels) + 1.6),
                             sharey=True, sharex=True, squeeze=False)
    y = np.arange(len(labels))[::-1]
    label = parameter_label or parameter
    for j, method in enumerate(methods):
        ax = axes[0, j]
        for k, yk in enumerate(y):
            if k % 2 == 0:
                ax.axhspan(yk - 0.5, yk + 0.5, color=ROW_SHADE, linewidth=0, zorder=0)
        for yk, conf in zip(y, labels):
            r = results[conf][method]
            col = r.names.index(parameter)
            theta, err = r.theta[:, col], r.errors[:, col]
            rms, quoted, n = np.std(theta, ddof=1), np.mean(err), len(theta)
            ax.errorbar(rms, yk + 0.12, xerr=rms / np.sqrt(2 * (n - 1)), fmt="o", ms=5,
                        color=TOYS, elinewidth=1.2, capsize=0, zorder=3)
            ax.plot(quoted, yk - 0.12, marker="D", ms=5, color=OBSERVED, linestyle="none",
                    zorder=3)
        ax.set_title(method.replace("_", " "), color=INK, fontweight="bold")
        ax.set_xlabel(f"uncertainty on {label}" + (f" [{unit}]" if unit else ""))
        ax.set_ylim(-0.5, len(labels) - 0.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels(labels, color=INK)
    handles = [Line2D([], [], marker="o", color=TOYS, linestyle="none",
                      label="true spread in the toys (rms)"),
               Line2D([], [], marker="D", color=OBSERVED, linestyle="none",
                      label="quoted uncertainty (mean over toys)")]
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize="large", fontweight="bold")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=2,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def plot_toy_coverage(results: dict, parameter: str, ax=None, title: str | None = None,
                      xlim=None):
    """Coverage of the quoted uncertainty: true spread over the toys (rms) divided by the mean
    quoted uncertainty, one row per configuration, one marker per method. A method covers when
    its ratio is compatible with 1; above 1 it quotes too small an uncertainty.

    Args:
        results: {configuration label: {method: ToyResults}}, same methods in every row.
        parameter: The combined parameter.
        ax: Matplotlib axes; a new figure if None.
        title: Axes title.
        xlim: x range of the ratio.

    Returns:
        The axes; ax.legend_handles_toy_coverage holds one handle per method.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = list(results)
    methods = list(next(iter(results.values())))
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 0.55 * len(labels) + 1.2))
    y = np.arange(len(labels))[::-1]
    for k, yk in enumerate(y):
        if k % 2 == 0:
            ax.axhspan(yk - 0.5, yk + 0.5, color=ROW_SHADE, linewidth=0, zorder=0)
    ax.axvline(1.0, color=INK2, linewidth=1.0, zorder=1)
    offsets = np.linspace(0.27, -0.27, len(methods)) if len(methods) > 1 else [0.0]
    for m, (method, dy) in enumerate(zip(methods, offsets)):
        for yk, conf in zip(y, labels):
            c = results[conf][method].coverage(parameter)
            n = len(results[conf][method].theta)
            ratio = c["rms"] / c["quoted"]
            ax.errorbar(ratio, yk + dy, xerr=ratio / np.sqrt(2 * (n - 1)),
                        fmt=MARKERS[m % len(MARKERS)], ms=5, color=SERIES[m % len(SERIES)],
                        elinewidth=1.2, capsize=0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel("true spread / quoted uncertainty")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontweight="bold")
    ax.legend_handles_toy_coverage = [
        Line2D([], [], marker=MARKERS[m % len(MARKERS)], color=SERIES[m % len(SERIES)],
               linestyle="none", label=method.replace("_", " "))
        for m, method in enumerate(methods)]
    return ax


def style():
    """Matplotlib settings matching these plots (surface, ink, no top/right spines)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False,
    })
