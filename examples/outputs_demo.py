#!/usr/bin/env python3
"""The outputs of combifit on any set of measurements (here synthetic ones).

Combination summary, pulls (text, LaTeX, plot), step-by-step and summary LaTeX tables, toys and
their plots. Writes into examples/output/.

    python examples/outputs_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from combifit import (combine, combine_steps, compatibility, cumulative,  # noqa: E402
                      generate_toys, latex_chain_table, latex_pulls_table,
                      latex_summary_table, run_toys)
from combifit.plotting import plot_toys, style  # noqa: E402
from toys_and_compatibility import synthetic  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def main():
    OUT.mkdir(exist_ok=True)
    style()
    meas = synthetic()          # 3 of type A (post-fit NPs), 2 of type C (impacts only)

    # Combination, its compatibility and its pulls
    res = combine(meas, ["mu"], type="pois_nps_aux")
    print(res.summary(), "\n")
    print(compatibility(meas, ["mu"]).summary(), "\n")
    print(res.format_pulls(n=8), "\n")
    (OUT / "tab_pulls.tex").write_text(latex_pulls_table(res, n=10, caption="Largest pulls.",
                                                         label="tab:pulls"))
    ax = res.plot_pulls(n_max=5, title="NPs in most tension")
    ax.figure.legend(handles=ax.legend_handles_np_pulls, loc="upper right", frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / "pulls.png", dpi=130)
    plt.close(ax.figure)

    # Step-by-step table (M0+M1, +M2, ...) and summary of the last step
    steps = combine_steps(cumulative(meas), ["mu"])
    (OUT / "tab_chain.tex").write_text(latex_chain_table(
        steps, "mu", poi_label=r"$\mu$", caption="Step-by-step combination.", label="tab:chain"))
    (OUT / "tab_summary.tex").write_text(latex_summary_table(
        {"synthetic": {m: rows[-1] for m, rows in steps.items()}}, "mu", poi_label=r"$\mu$",
        group_title="inputs", caption="Final combination.", label="tab:summary"))

    # Toys around the combination, the same toys for every method
    toys = generate_toys(meas, ["mu"], res, n_toys=500, seed=1)
    results = {m: run_toys(meas, ["mu"], res, type=m, toys=toys, with_compatibility=True)
               for m in steps}
    print(results["pois_nps_aux"].summary(["mu"]))
    observed = {m: rows[-1] for m, rows in steps.items()}
    fig = plot_toys(results, "mu", observed, parameter_label=r"$\mu$", title="Synthetic toys")
    fig.savefig(OUT / "toys.png", dpi=130)
    plt.close(fig)
    print(f"\noutputs in {OUT}")


if __name__ == "__main__":
    main()
