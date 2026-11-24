# combifit

Combine several fit results from their best-fit values and covariance matrices, including
measurements that share nuisance parameters (NPs) constrained by a common auxiliary measurement.

```python
from combifit import Measurement, combine, compatibility

a = Measurement("M1", theta=[80360.0], cov=[[10.0**2]], names=["mW"])
b = Measurement("M2", theta=[80370.0], cov=[[12.0**2]], names=["mW"])
res = combine([a, b], ["mW"], type="weighted_average")
print(res.summary())
```

Conventions of the theory notes (`fitcombi.tex`, `heterogeneous.tex`): θ = (μ, α), POIs μ and
NPs α shared by the measurements, constrained once by the auxiliary measurement a ~ N(α, I) (NPs
in units of their prior).

Measurement types:

| type | published | built with |
|---|---|---|
| `"A"` | result and covariance on the POIs and the shared NPs, (θ̂_K, Σ_K) | `Measurement`, `linear_gaussian_fit` |
| `"B"` | POI result and covariance only, (μ̂_K, Σ_K^μμ) | `Measurement` over the POIs |
| `"C"` | POI result, covariance and impacts λ_K = Cov(μ̂_K, a) | `impacts_only` |

Methods (`type=`):

| method | note | stacked input |
|---|---|---|
| `weighted_average` | no correlation | POIs |
| `pois_only` | marginalised combination | Y_μ = (μ̂_1, …, μ̂_N) |
| `pois_nps` | full combination | Y = (θ̂_1, …, θ̂_N) |
| `pois_nps_aux` (default) | likelihood sum, efficient | Ỹ = (θ̂_1, …, θ̂_N, a) |

The correlations come from the master formula Cov(θ̂_K, θ̂_L) = Σ_K S Sᵀ Σ_L.

## Outputs

```python
res.summary()                      # combined POIs, chi2, p-value
res.format_pulls(n=10)             # text table of the largest pulls
res.pull_table()                   # the same as a list of dicts
res.plot_pulls(n_max=10)           # NP pull plot: every measurement + the combination
compatibility(meas, pois).summary()   # total / POIs / NPs goodness of fit

from combifit import combine_steps, cumulative, latex_chain_table, latex_summary_table, \
    latex_pulls_table
steps = combine_steps(cumulative(meas), pois)      # step-by-step, every method
latex_chain_table(steps, "mW", unit="MeV")         # LaTeX (booktabs)
latex_pulls_table(res, n=15)

from combifit import generate_toys, run_toys
from combifit.plotting import plot_toys, plot_toy_overview
toys = generate_toys(meas, pois, truth=res, n_toys=1000)
results = {m: run_toys(meas, pois, res, type=m, toys=toys, with_compatibility=True)
           for m in steps}
plot_toys(results, "mW", observed={m: rows[-1] for m, rows in steps.items()})
```

## Examples (`examples/`)

- `limit_independent.py`: nothing shared, every method gives the weighted average.
- `limit_common_systematic.py`: a common systematic the data do not constrain: the correlated
  methods give the textbook BLUE, type A and type C inputs agree, the weighted average does not.
- `limit_profiled_np.py`: the data constrain the common NP more and more; the methods meet at
  both ends and are ordered in between.
- `outputs_demo.py`: every output (pulls, LaTeX tables, toys and their plots).
- `toys_and_compatibility.py`: coverage and chi2 laws of every method with toys.

Building inputs: `impacts_only(...)` for a POI with its impacts (type C),
`linear_gaussian_fit(...)` for a toy measurement fitted in the linear Gaussian model (type A).

## Development

`pip install -e ".[dev,plot]"`, then `pytest` (tests in `tests/`: every method against an
independent stacked fit, pois_nps_aux against the joint fit of all the raw data, limiting cases,
compatibility, toys, outputs).
