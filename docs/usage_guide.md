# Usage Guide

Task-oriented walkthroughs: install the project, then load market data, build
a QUBO, solve it classically and with QAOA, plot the result, and run the
benchmark harness. Every Python block below was executed as written; sizes are
kept tiny (4 assets, a few optimizer iterations) so each chapter finishes in
seconds on a laptop.

See also: [algorithm](algorithm.md) (what each step computes),
[API reference](api_reference.md) (every public name), the runnable scripts in
[`examples/`](../examples/README.md), and the module guides
[data loader](dataloader.md), [Rust core](rust_core.md),
[quantum backend](quantum_backend.md), [visualization](visualization.md),
[benchmarks](benchmarks.md).

## 1. Install

Requirements: Python ≥ 3.10, a Rust toolchain (the QUBO core is a Rust
extension compiled by maturin during installation), and
[uv](https://docs.astral.sh/uv/).

```bash
# Rust toolchain (skip if `cargo --version` already works)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# uv
python -m pip install uv
```

The project environment lives in `qaoa-env/` rather than uv's default
`.venv/`. Export the variable in every shell where you run `uv`:

```bash
git clone <repository-url> qaoa-portfolio && cd qaoa-portfolio
export UV_PROJECT_ENVIRONMENT=qaoa-env
uv sync --extra dev          # creates qaoa-env/, installs deps, builds qaoa_portfolio_core
source qaoa-env/bin/activate # optional: puts python and qaoa-portfolio on PATH
qaoa-portfolio --help
```

On a Linux host with an NVIDIA GPU (Python ≥ 3.11), add the `gpu` extra to
install the `lightning.gpu` simulator, and keep adding it on every later sync:

```bash
uv sync --extra dev --extra gpu
```

Check the installation:

```bash
uv run python -c "import qaoa_portfolio, qaoa_portfolio_core; print(qaoa_portfolio.__version__)"
uv run pytest -m "not network"
```

The Python blocks below assume the environment is active (or run them with
`uv run python`). Each chapter is self-contained; blocks inside a chapter
continue the same Python session.

## 2. Load market data

`MarketDataLoader` downloads daily bars from Yahoo Finance (network required)
and caches them as CSV files under `data/cache/` in the working directory.

```python
import asyncio

import numpy as np
import pandas as pd

from qaoa_portfolio import MarketDataLoader

symbols = ["AAPL", "MSFT", "JNJ", "XOM"]
loader = MarketDataLoader()
price_data = asyncio.run(
    loader.load_portfolio_data(symbols, start_date="2024-01-01", end_date="2024-12-31")
)
# Columns are a (symbol, price_type) MultiIndex: open, high, low, close, volume, ...
closes = pd.DataFrame({s: price_data[(s, "close")] for s in symbols}).dropna()
prices = closes.to_numpy(dtype=np.float64)  # shape (periods, assets)
print(prices.shape)

returns = loader.calculate_returns(price_data)  # simple returns, one column per symbol
print(returns.columns.tolist())
```

```text
(252, 4)
['AAPL', 'MSFT', 'JNJ', 'XOM']
```

In a Jupyter notebook an event loop is already running: replace
`asyncio.run(...)` with `await loader.load_portfolio_data(...)`.

Presets and the one-call loader:

```python
import asyncio

from qaoa_portfolio import quick_portfolio_load
from qaoa_portfolio.portfolios import get_preset_portfolio, list_portfolio_presets

print(list_portfolio_presets())
print(get_preset_portfolio("major_crypto"))

# Last 120 calendar days of a preset; returns (price_data, simple returns)
price_data, returns = asyncio.run(quick_portfolio_load(preset="growth_stocks", days_back=120))
print(returns.shape[1])
```

```text
{'conservative_stocks': 'Large-cap defensive stocks', 'growth_stocks': 'High-growth technology stocks', 'major_crypto': 'Top 5 cryptocurrencies by market cap', 'defi_crypto': 'DeFi and smart contract platforms', 'balanced_mixed': 'Balanced mix of stocks, crypto, and index'}
['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD']
5
```

Offline alternative — the deterministic synthetic prices the benchmarks use
(no network, used by every following chapter):

```python
from qaoa_portfolio import generate_synthetic_prices

prices = generate_synthetic_prices(num_assets=4, periods=252, seed=42)
print(prices.shape, bool((prices > 0).all()))
```

```text
(252, 4) True
```

Mixed stock/crypto portfolios: the loader aligns both on calendar dates, and
weekend crypto rows appear as NaN for equities — the `.dropna()` above keeps
only common trading days. More in [dataloader.md](dataloader.md).

## 3. Build a QUBO

The Rust core turns a price matrix into the cardinality-constrained
mean-variance QUBO ([algorithm.md §2](algorithm.md#2-qubo-encoding)):

```python
import numpy as np
import qaoa_portfolio_core as core

from qaoa_portfolio import generate_synthetic_prices

symbols = ["A", "B", "C", "D"]
prices = generate_synthetic_prices(num_assets=4, periods=252, seed=42)

# risk_aversion q in [0, 1]; select exactly target_assets = 2
qubo = core.build_qubo(prices, symbols, 0.5, 2)
print(qubo.num_variables, round(qubo.offset, 4))
print(np.round(qubo.to_numpy(), 4))

# Energy of one selection (True = asset selected)
print(round(qubo.evaluate([True, False, True, False]), 4))

# The Rust-side statistics the QUBO is built from (annualized log returns)
log_returns = np.diff(np.log(prices), axis=0)
stats = core.PyReturnSeries(symbols, log_returns)
print(np.round(stats.mean_returns(), 4))
```

```text
4 0.6061
[[-0.3788  0.3073  0.3054  0.3075]
 [ 0.3073 -0.403   0.3068  0.3043]
 [ 0.3054  0.3068 -0.4968  0.3124]
 [ 0.3075  0.3043  0.3124 -0.4138]]
0.0359
[-0.1202 -0.0571  0.1393 -0.0199]
```

`build_qubo` raises `ValueError` for invalid input (risk aversion outside
[0, 1], `target_assets` of 0 or above the number of symbols, non-positive
prices, fewer than two rows).

## 4. Solve classically

Three Rust baselines work on the same QUBO object:

```python
import qaoa_portfolio_core as core

from qaoa_portfolio import generate_synthetic_prices

symbols = ["A", "B", "C", "D"]
prices = generate_synthetic_prices(num_assets=4, periods=252, seed=42)
qubo = core.build_qubo(prices, symbols, 0.5, 2)

exact = core.solve_brute_force(qubo)  # exact, n <= 20
print(exact.selected_assets, round(exact.objective_value, 6))

annealed = core.solve_simulated_annealing(qubo, max_iterations=2_000, seed=7)
print(annealed.selected_assets, round(annealed.objective_value, 6))

continuous = core.solve_markowitz(prices, symbols)  # max-Sharpe weights, not a QUBO solve
print([round(w, 3) for w in continuous["weights"]], round(continuous["sharpe_ratio"], 3))

print(exact.to_dict()["metadata"]["solutions_evaluated"])
```

```text
['C', 'D'] 0.007906
['C', 'D'] 0.007906
[1.234, 0.378, -0.865, 0.254] -1.051
16
```

For more than 20 assets use `exact_optimum` from the benchmark module, which
switches to a NumPy enumeration up to 28 assets in the harness:

```python
from qaoa_portfolio.benchmarks import exact_optimum

value, method, solution = exact_optimum(qubo)
print(method, round(value, 6), solution)
```

```text
rust_brute_force 0.007906 [False, False, True, True]
```

## 5. Solve with QAOA

`solve_qubo_qaoa` accepts the Rust QUBO (or any square symmetric array) and
returns a `QAOAResult` ([algorithm.md §3](algorithm.md#3-qaoa)):

```python
import qaoa_portfolio_core as core

from qaoa_portfolio import QAOAConfig, generate_synthetic_prices, solve_qubo_qaoa

symbols = ["A", "B", "C", "D"]
prices = generate_synthetic_prices(num_assets=4, periods=252, seed=42)
qubo = core.build_qubo(prices, symbols, 0.5, 2)

config = QAOAConfig(
    layers=1,              # circuit depth p
    optimizer="cobyla",    # adam | gradient_descent | cobyla | nelder_mead
    max_iterations=30,
    num_restarts=1,
    seed=42,
    backend="default.qubit",
    target_assets=2,       # report the probability mass on 2-asset states
)
result = solve_qubo_qaoa(qubo, labels=symbols, config=config)

print(result.best_bitstring, result.selected_assets, round(result.objective_value, 6))
print(round(result.metadata["feasible_probability"], 3))
print([round(m, 3) for m in result.metadata["hamming_weight_distribution"]])
for entry in result.top_solutions[:3]:
    print(entry["bitstring"], round(entry["objective_value"], 4), round(entry["probability"], 3))
```

```text
0011 ['C', 'D'] 0.007906
0.641
[0.002, 0.219, 0.641, 0.137, 0.001]
0011 0.0079 0.118
0110 0.0131 0.117
1010 0.0359 0.113
```

The answer is the lowest-energy state among the `max_stored_solutions` (64)
most probable states, so it can differ from the single most probable
bitstring. Two options help with QAOA's soft cardinality constraint:

```python
from dataclasses import replace

exact = core.solve_brute_force(qubo)
optimum = "".join("1" if bit else "0" for bit in exact.solution)

feasible = replace(config, feasible_decoding=True)  # decode among 2-asset states only
result = solve_qubo_qaoa(qubo, labels=symbols, config=feasible, reference_bitstrings=[optimum])
print(result.selected_assets, len(result.selected_assets) == 2)
print("p_opt =", round(result.metadata["reference_probabilities"][optimum], 3))
```

```text
['C', 'D'] True
p_opt = 0.118
```

`backend="lightning.qubit"` (CPU) and `backend="lightning.gpu"` (NVIDIA, needs
the `gpu` extra) give identical results faster for larger n; see
[quantum_backend.md](quantum_backend.md) and
[benchmarks.md §7](benchmarks.md#7-october-2026-campaign--rtx-3080-and-dgx-spark-gb10).

## 6. Visualize

Plot functions return figures and never open windows. On servers, CI, or any
machine without a display, select the `Agg` backend before importing
anything that imports matplotlib:

```python
import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pandas as pd
import qaoa_portfolio_core as core

from qaoa_portfolio import (
    QAOAConfig,
    generate_synthetic_prices,
    plot_portfolio_composition,
    plot_qaoa_convergence,
    plot_risk_return_scatter,
    plot_solution_probabilities,
    plot_solver_comparison,
    render_qaoa_circuit_summary,
    solve_qubo_qaoa,
)

symbols = ["A", "B", "C", "D"]
prices = generate_synthetic_prices(num_assets=4, periods=252, seed=42)
qubo = core.build_qubo(prices, symbols, 0.5, 2)
result = solve_qubo_qaoa(
    qubo,
    labels=symbols,
    config=QAOAConfig(layers=1, optimizer="cobyla", max_iterations=30, num_restarts=1),
)
exact = core.solve_brute_force(qubo)
returns = pd.DataFrame(prices, columns=symbols).pct_change().dropna()

out = Path("figures")
out.mkdir(exist_ok=True)
figures = {
    "composition": plot_portfolio_composition(result.selected_assets),
    "risk_return": plot_risk_return_scatter(returns, highlighted_assets=result.selected_assets),
    "convergence": plot_qaoa_convergence(result),
    "probabilities": plot_solution_probabilities(result),
    "solvers": plot_solver_comparison(
        [
            {"solver_name": "qaoa", "objective_value": result.objective_value},
            {"solver_name": "brute_force", "objective_value": exact.objective_value},
        ]
    ),
}
for name, figure in figures.items():
    figure.savefig(out / f"{name}.png", bbox_inches="tight")

print(sorted(p.name for p in out.glob("*.png")))
print(render_qaoa_circuit_summary(result).splitlines()[2])
```

```text
['composition.png', 'convergence.png', 'probabilities.png', 'risk_return.png', 'solvers.png']
Qubits (assets): 4
```

Pass `config=VisualizationConfig(backend="plotly")` to any plot function for
interactive Plotly figures. Full list in [visualization.md](visualization.md).

## 7. Benchmark

### 7.1 From Python

The harness generates paired synthetic instances, runs every solver on each,
and scores them against the exact optimum
([algorithm.md §4](algorithm.md#4-measuring-quality)):

```python
from qaoa_portfolio import (
    BenchmarkConfig,
    QAOAConfig,
    run_quality_benchmark,
    save_benchmark_results,
    significance_test,
    summarize_quality,
)
from qaoa_portfolio.benchmarks import SAConfig, mcnemar_test

config = BenchmarkConfig(
    num_assets=4,
    target_assets=2,
    repeats=3,
    periods=60,
    seed=42,
    qaoa=QAOAConfig(layers=1, optimizer="cobyla", max_iterations=20, num_restarts=1),
    sa=SAConfig(schedule="auto", sweeps=200, restarts=2),  # instance-scaled SA
)
records = run_quality_benchmark(config)  # all five solvers

summary = summarize_quality(records)
for solver, stats in summary.items():
    print(
        f"{solver:20s} ratio={stats['mean_approximation_ratio']:.3f} "
        f"hits={stats['optimal_hit_rate']:.2f} feasible={stats['feasibility_rate']:.2f}"
    )
print("QAOA mean p_opt:", round(summary["qaoa"]["mean_p_opt"], 3))

print(significance_test(records, "qaoa", "random")["p_value"])
print(mcnemar_test(records, "qaoa", "simulated_annealing")["p_value"])

path = save_benchmark_results(records, suite="quality", output_dir="bench-out", config=config)
print(path.parent.name, path.suffix)
```

```text
brute_force          ratio=1.000 hits=1.00 feasible=1.00
markowitz            ratio=0.742 hits=0.67 feasible=1.00
qaoa                 ratio=1.000 hits=1.00 feasible=1.00
random               ratio=0.595 hits=0.33 feasible=1.00
simulated_annealing  ratio=1.000 hits=1.00 feasible=1.00
QAOA mean p_opt: 0.11
0.5
1.0
bench-out .json
```

Three paired instances are far too few for meaningful p-values; the numbers
above only show the call pattern. With `config.qaoa=None` the harness uses
its preset: p = 1, COBYLA, 60 iterations, 2 restarts.

### 7.2 From the command line

```bash
# Quality suite: 8 assets, select 4, 10 paired instances, all solvers
uv run qaoa-portfolio benchmark --suite quality --assets 8 --target 4 --repeats 10 --seed 42 --plot

# Faster simulator, instance-scaled SA with restarts, feasible QAOA decoding
uv run qaoa-portfolio benchmark --suite quality --assets 12 --repeats 10 \
  --qaoa-backend lightning.qubit --sa-schedule auto --sa-sweeps 1000 --sa-restarts 4 \
  --qaoa-feasible-decoding

# Cost vs size and vs depth
uv run qaoa-portfolio benchmark --suite scaling --asset-counts 4,8,12 --repeats 3
uv run qaoa-portfolio benchmark --suite layers --layers 1,2,3 --assets 6 --target 3 --repeats 3

# Real data, 70/30 in-/out-of-sample split (network)
uv run qaoa-portfolio benchmark --suite market --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,JPM \
  --start-date 2022-01-01 --end-date 2024-12-31 --assets 6 --target 3
```

A tiny run (seconds) prints:

```text
$ qaoa-portfolio benchmark --suite quality --assets 4 --repeats 2 --qaoa-iterations 10 \
    --qaoa-restarts 1 --sa-schedule auto --sa-restarts 2 --qaoa-feasible-decoding --output bench-out
Benchmark suite 'quality' complete
Artifact: bench-out/quality-<timestamp>.json
- brute_force: ratio 1.0000 ± 0.0000, optimal 100%, feasible 100%, 0.0 ms
- markowitz: ratio 0.5300 ± 0.4700, optimal 50%, feasible 100%, 0.0 ms
- qaoa: ratio 1.0000 ± 0.0000, optimal 100%, feasible 100%, 345.6 ms, p_opt 0.099
- random: ratio 0.4961 ± 0.4308, optimal 0%, feasible 100%, 0.1 ms
- simulated_annealing: ratio 1.0000 ± 0.0000, optimal 100%, feasible 100%, 0.6 ms
```

Every flag is listed in the
[API reference § CLI](api_reference.md#13-command-line-interface-qaoa-portfolio);
artifacts default to `results/benchmarks/`. `--qaoa-optimizer adam`
reproduces artifacts made with the earlier Adam preset. Measured results and
methodology: [benchmarks.md](benchmarks.md).

## 8. Troubleshooting

**Yahoo Finance errors.** `MarketDataError: No data could be loaded for any
symbols` or `Failed to load <symbol>` usually means a wrong symbol (crypto
needs the `-USD` suffix, e.g. `BTC-USD`), a window without trading days, or a
temporary rate limit (HTTP 429). An end date in the future raises
`DataValidationError` before any download. Symbols that
fail are dropped with a warning when at least one other symbol loads. Wait a
minute and retry, request fewer symbols, or work offline with
`generate_synthetic_prices`.

**Cache behavior.** Each symbol/window is cached as
`data/cache/<SYMBOL>_yfinance_<start>_<end>.csv` relative to the working
directory and reused for 7 days (`data_sources.cache_duration_days`); expired
or unreadable files are deleted and re-downloaded. Fixed `start_date` /
`end_date` windows hit the cache; `quick_portfolio_load` and the CLI's
`--days-back` derive the window from the current time, so they download again
on a new day. Delete `data/cache/` to force a refresh, or disable caching with
`config.set("data_sources.cache_enabled", False)` before creating the loader.

**Plots on headless machines.** `TclError: no display name` or a hanging
`plt.show()` means matplotlib picked an interactive backend: call
`matplotlib.use("Agg")` before importing `qaoa_portfolio` plot functions (or
set `MPLBACKEND=Agg`) and save figures with `savefig`. The CLI's `--plot`
already uses `Agg`.

**`qaoa_portfolio_core` cannot be imported.** The Rust extension is built by
`uv sync`. Make sure `UV_PROJECT_ENVIRONMENT=qaoa-env` is exported (otherwise
uv builds into `.venv/` and warns that `VIRTUAL_ENV=qaoa-env` does not match),
that `cargo` is on `PATH`, and re-run `uv sync --extra dev`. After changing
Rust sources, sync again to rebuild.

**`lightning.gpu` is missing.** `QuantumBackendError: Unable to create
PennyLane backend 'lightning.gpu'` means the plugin is not installed or no
CUDA device is visible. It comes only with the `gpu` extra (Linux, Python
≥ 3.11, NVIDIA driver): run `uv sync --extra dev --extra gpu`. A later
`uv sync` **without** `--extra gpu` uninstalls the plugin again, so always
pass both extras on GPU hosts. `lightning.qubit` is the CPU fallback.

**Slow or memory-hungry QAOA.** Cost doubles per asset. Keep exploratory runs
at n ≤ 12 and few iterations, and prefer `lightning.qubit` over
`default.qubit`: it was faster at every measured size and needs far less
memory (`default.qubit` differentiates by backpropagation and stores
intermediate states). See
[benchmarks.md §7](benchmarks.md#7-october-2026-campaign--rtx-3080-and-dgx-spark-gb10)
for measured costs.

**`Error: risk_factor must be in (0, 1]`.** The Rust QUBO formulation
accepts risk aversion in [0, 1], and the benchmark config rejects values
outside (0, 1] before any work starts. Keep `--target` between 1 and
`--assets` as well.
