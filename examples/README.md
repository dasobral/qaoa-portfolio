# Examples

Four runnable walkthroughs of the QAOA Portfolio Optimizer, from a four-asset toy
problem to a demo on the latest market data. Each example is a plain Python script
plus a Jupyter notebook that mirrors it with narration and inline figures.

| # | Script / notebook | Data | Network | What it shows | Typical time* |
|---|-------------------|------|---------|---------------|---------------|
| 01 | `01_basic_four_assets.py` / `.ipynb` | synthetic, seeded | no | prices → `build_qubo` → brute-force ground truth → QAOA → compare; composition, convergence, and probability charts | ~3 s |
| 02 | `02_crypto_portfolio.py` / `.ipynb` | `major_crypto` preset via `quick_portfolio_load` (BTC, ETH, BNB, SOL, ADA, 1 year) | first run of the day | five-asset QUBO, QAOA vs simulated annealing vs brute force; risk-return scatter | ~8 s |
| 03 | `03_qaoa_vs_classical.py` / `.ipynb` | synthetic, seeded | no | `run_quality_benchmark` at 6 assets, `summarize_quality`, Wilcoxon `significance_test`, exact `mcnemar_test`, `plot_solver_comparison` | ~50 s |
| 04 | `04_live_market_demo.py` / `.ipynb` | Yahoo Finance, trailing ~180 days, six large US stocks | yes | end-to-end run on today's data, in-sample metrics with caveat; correlation, risk-return, and top-solution charts | ~12 s |

\* Single CPU run with the `default.qubit` simulator; QAOA dominates the time.

All examples keep QAOA tiny (n ≤ 6 assets, p = 1 layer, COBYLA, a few dozen
iterations) and use fixed seeds, so the offline examples (01, 03) give the same
numbers on every run.

## Prerequisites

Set up the project environment from the repository root as described in the main
[README](../README.md):

```bash
export UV_PROJECT_ENVIRONMENT=qaoa-env
uv sync --extra dev          # builds the Rust extension qaoa_portfolio_core
source qaoa-env/bin/activate
```

To open the notebooks you also need Jupyter (not a project dependency), for example
`pip install jupyterlab` in the same environment, or any Jupyter install whose kernel
points at `qaoa-env/bin/python`.

## Running the scripts

Run from the repository root:

```bash
python examples/01_basic_four_assets.py
python examples/02_crypto_portfolio.py
python examples/03_qaoa_vs_classical.py
python examples/04_live_market_demo.py
```

Each script prints its results and saves PNG figures to `examples/output/`
(gitignored). Scripts never call `plt.show()` and force Matplotlib's `Agg` backend,
so they behave the same in a terminal, over SSH, and in CI.

Every script exposes `main(output_dir=None, ...)`, which returns a dictionary of
results plus the list of saved figure paths, so you can also drive them from Python:

```python
import importlib.util

spec = importlib.util.spec_from_file_location("ex01", "examples/01_basic_four_assets.py")
ex01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex01)
result = ex01.main(output_dir="/tmp/qaoa-figures")
print(result["qaoa_assets"], result["p_opt"])
```

Examples 02 and 04 also accept `price_data=` (a loader-shaped frame with
`(symbol, "close")` columns) to skip the download.

## Notes per example

- **01 and the decoding rule.** QAOA keeps its 64 most probable bitstrings
  (`max_stored_solutions`) and returns the one with the lowest QUBO objective. For
  n ≤ 6 every bitstring is kept, so the decoded answer equals the brute-force optimum
  by construction. The examples therefore report the circuit's real quality: the
  probability on the optimum (`p_opt`) and on feasible portfolios
  (`feasible_probability`).
- **02 network and cache.** The loader caches per-symbol CSVs under `data/cache/`
  relative to the working directory (gitignored). The cache key contains the
  requested window, which ends today, so the first run of each day downloads again.
  Crypto trades continuously, so numbers drift slightly between runs on the same day.
- **03 is an indicative mini-run.** Four paired instances are far too few for firm
  conclusions; the p-values are illustrative. It uses the benchmark preset (p = 1,
  COBYLA, 60 iterations, 2 restarts). Curated full-size results, methodology, and
  GPU measurements are in [docs/benchmarks.md](../docs/benchmarks.md).
- **04 "real-time" means the latest daily bars.** There is no intraday streaming:
  the demo downloads the most recent daily bars Yahoo Finance has published (today's
  bar may be partial while the market is open). The reported return, volatility,
  Sharpe ratio, and drawdown are **in-sample** - computed on the same window the
  optimizer saw - and are not a forecast or out-of-sample performance. For a
  train/test evaluation see `run_market_study` in
  [docs/benchmarks.md](../docs/benchmarks.md).

## Scripts and notebooks: the contract

- The **scripts are the tested source of truth.** `tests/test_examples.py` imports
  each script and runs its `main()` with a temporary output directory. Examples 01
  and 03, plus offline variants of 02 and 04 that inject synthetic prices, run in the
  default test gate; the live runs of 02 and 04 carry the `network` marker:

  ```bash
  pytest tests/test_examples.py -m "not network"   # offline
  pytest tests/test_examples.py -m network         # live downloads
  ```

- The **notebooks are presentation.** Each one mirrors its script block by block
  with markdown narration, and is committed with executed outputs. Notebooks are not
  executed by the test suite; whenever a script changes, update its notebook to
  match and re-execute it manually, for example:

  ```bash
  jupyter nbconvert --to notebook --execute --inplace examples/01_basic_four_assets.ipynb
  ```

  Run this from a Jupyter whose kernel uses `qaoa-env/bin/python`. Notebooks run
  with `examples/` as the working directory, so their figures go to
  `examples/output/` and the loader cache to `examples/data/cache/` (both
  gitignored).
- Notebooks 02 and 04 need network access; their committed outputs reflect the market
  data on the day they were last executed.
