"""LaTeX tables of combinations (booktabs).

    steps = combine_steps(cumulative(measurements), ["mu"])     # step-by-step, every method
    tex = latex_chain_table(steps, "mu", caption="...", label="tab:chain")
    tex = latex_summary_table({"set A": final_a, "set B": final_b}, "mu")
    tex = latex_pulls_table(combined, n=15)
"""

from __future__ import annotations

from .combifit import CombinedMeasurement
from .compatibility import Compatibility, compatibility
from .measurement import Measurement

METHODS = ["weighted_average", "pois_only", "pois_nps", "pois_nps_aux"]
METHOD_TITLES = {"weighted_average": "Weighted average, no correlation",
                 "pois_only": "Marginalised combination (POIs)",
                 "pois_nps": "Full combination (POIs and NPs)",
                 "pois_nps_aux": "Likelihood sum (POIs, NPs and auxiliary measurement)"}
METHOD_SHORT = {"weighted_average": "weighted average", "pois_only": "marginalised",
                "pois_nps": "full", "pois_nps_aux": "likelihood sum"}


# --------------------------------------------------------------------------- running

def cumulative(measurements: list[Measurement], start: int = 2) -> list[list[Measurement]]:
    """The first k measurements, for k = start, ..., N: the steps of a step-by-step chain."""
    return [list(measurements[:k]) for k in range(start, len(measurements) + 1)]


def combine_steps(steps: list[list[Measurement]], pois: list[str],
                  methods: list[str] = METHODS) -> dict[str, list[Compatibility]]:
    """Combine each list of measurements from scratch with each method.

    Returns:
        {method: [Compatibility of each step]}.
    """
    return {method: [compatibility(step, pois, type=method) for step in steps]
            for method in methods}


# --------------------------------------------------------------------------- formatting

def format_p(p) -> str:
    """p-value, switching to scientific notation before it rounds to zero."""
    if p is None or p != p:
        return "---"
    if p < 1e-12:
        return r"$<10^{-12}$"
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return rf"${mant}\times 10^{{{int(exp)}}}$"
    return f"{p:.3f}"


def format_chi2(chi2: float, ndf: int) -> str:
    return rf"{chi2:.2f}\,/\,{ndf}"


def _tex(name: str, tex_names: dict | None) -> str:
    return (tex_names or {}).get(name, name.replace("_", r"\_"))


def _cells(c: Compatibility, poi: str, decimals: int) -> list[str]:
    return [f"{c.combined.get_value(poi):.{decimals}f}",
            f"{c.combined.get_uncertainty(poi):.{decimals}f}",
            format_chi2(c.chi2, c.ndf), format_p(c.p_value),
            format_chi2(c.chi2_pois, c.ndf_pois), format_p(c.p_value_pois)]


def _header(first: list[str], poi_label: str) -> list[str]:
    lead = " & ".join([""] * len(first))
    return [rf"    {lead} & \multicolumn{{2}}{{c}}{{}} & \multicolumn{{2}}{{c}}{{Total}} & "
            r"\multicolumn{2}{c}{" + poi_label + r" compatibility} \\",
            rf"    \cmidrule(lr){{{len(first) + 3}-{len(first) + 4}}}"
            rf"\cmidrule(lr){{{len(first) + 5}-{len(first) + 6}}}"]


def _wrap(caption: str, label: str, columns: str, body: list[str]) -> str:
    lines = [r"\begin{table}[htbp]", r"  \centering"]
    if caption:
        lines.append(rf"  \caption{{{caption}}}")
    if label:
        lines.append(rf"  \label{{{label}}}")
    lines += [rf"  \begin{{tabular}}{{{columns}}}", r"    \toprule", *body,
              r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- tables

def latex_chain_table(steps: dict[str, list[Compatibility]], poi: str, caption: str = "",
                      label: str = "", poi_label: str | None = None, unit: str = "",
                      tex_names: dict | None = None, method_titles: dict | None = None,
                      decimals: int = 2) -> str:
    """Step-by-step table: one block per method, one row per step.

    Columns: step (first two measurements, then "+ the one added"), number of measurements,
    POI +- uncertainty, total chi2/ndf and p, POI compatibility chi2/ndf and p.

    Args:
        steps: Output of combine_steps.
        poi: The POI shown.
        caption, label: LaTeX caption and label.
        poi_label: LaTeX name of the POI (default: its name).
        unit: Unit shown in the header, e.g. "MeV".
        tex_names: LaTeX names of the measurements, e.g. {"D0": r"D\\O"}.
        method_titles: Titles of the method blocks (default METHOD_TITLES).
        decimals: Decimals of the POI and its uncertainty.
    """
    poi_label = poi_label or _tex(poi, None)
    titles = {**METHOD_TITLES, **(method_titles or {})}
    head = rf"{poi_label} [{unit}]" if unit else poi_label
    body = _header(["Step", "N"], poi_label)
    body += [r"    Step & $N$ & \multicolumn{2}{c}{" + head + r"} & $\chi^2/n_{\mathrm{dof}}$ & "
             r"$p$ & $\chi^2/n_{\mathrm{dof}}$ & $p$ \\", r"    \midrule"]
    for i, (method, rows) in enumerate(steps.items()):
        if i:
            body.append(r"    \addlinespace")
        body.append(rf"    \multicolumn{{8}}{{l}}{{\itshape {titles.get(method, method)}}} \\")
        for k, c in enumerate(rows):
            names = [m.name for m in c.combined.measurements if m.type != "auxiliary"]
            tex = [_tex(x, tex_names) for x in names]
            step = rf"{tex[0]}\,+\,{tex[1]}" if k == 0 else rf"\quad+\,{tex[-1]}"
            body.append("    " + " & ".join([step, str(len(names))] + _cells(c, poi, decimals))
                        + r" \\")
    return _wrap(caption, label, r"lrr@{\,$\pm$\,}lrrrr", body)


def latex_summary_table(groups: dict[str, dict[str, Compatibility]], poi: str,
                        caption: str = "", label: str = "", group_title: str = "",
                        poi_label: str | None = None, unit: str = "",
                        method_labels: dict | None = None, decimals: int = 2) -> str:
    """Several combinations side by side: one block per group (e.g. a PDF set), one row per
    method.

    Args:
        groups: {group name: {method: Compatibility}}, e.g. the last step of combine_steps
            for each group: {g: {m: steps[m][-1] for m in steps}}.
        poi, caption, label, poi_label, unit, decimals: As in latex_chain_table.
        group_title: Header of the group column.
        method_labels: Row labels of the methods (default METHOD_SHORT).
    """
    poi_label = poi_label or _tex(poi, None)
    labels = {**METHOD_SHORT, **(method_labels or {})}
    head = rf"{poi_label} [{unit}]" if unit else poi_label
    body = _header([group_title, "method"], poi_label)
    body += [rf"    {group_title} & method & \multicolumn{{2}}{{c}}{{{head}}} & "
             r"$\chi^2/n_{\mathrm{dof}}$ & $p$ & $\chi^2/n_{\mathrm{dof}}$ & $p$ \\",
             r"    \midrule"]
    for i, (group, by_method) in enumerate(groups.items()):
        if i:
            body.append(r"    \midrule")
        body.append(rf"    \multicolumn{{8}}{{l}}{{\itshape {group}}} \\")
        for method, c in by_method.items():
            body.append("    " + " & ".join(["", labels.get(method, method)]
                                           + _cells(c, poi, decimals)) + r" \\")
    return _wrap(caption, label, r"llr@{\,$\pm$\,}lrrrr", body)


def latex_pulls_table(combined: CombinedMeasurement, n: int | None = None,
                      nps_only: bool = False, caption: str = "", label: str = "",
                      tex_names: dict | None = None, decimals: int = 2) -> str:
    """Pulls of the combination, largest first (CombinedMeasurement.pull_table).

    Columns: measurement, parameter, input value +- error, combined value +- error, pull.
    """
    body = [r"    Measurement & Parameter & \multicolumn{2}{c}{Input} & "
            r"\multicolumn{2}{c}{Combined} & Pull \\", r"    \midrule"]
    for r in combined.pull_table(nps_only=nps_only)[:n]:
        comb = ([f"{r['combined']:.{decimals}f}", f"{r['combined_error']:.{decimals}f}"]
                if r["combined"] == r["combined"] else [r"\multicolumn{2}{c}{---}"])
        pull = f"{r['pull']:+.2f}" if r["pull"] == r["pull"] else "---"
        body.append("    " + " & ".join(
            [_tex(r["measurement"], tex_names), _tex(r["parameter"], None),
             f"{r['value']:.{decimals}f}", f"{r['error']:.{decimals}f}", *comb, pull]) + r" \\")
    return _wrap(caption, label, r"llr@{\,$\pm$\,}lr@{\,$\pm$\,}lr", body)
