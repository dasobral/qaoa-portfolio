# API Reference

One page for the public surface of the QAOA Portfolio Optimizer: every name in
`qaoa_portfolio.__all__`, the public benchmark helpers that live outside
`__all__`, the `qaoa-portfolio` CLI, and the Rust bridge `qaoa_portfolio_core`.
Each entry gives the signature and one sentence; the linked module guides hold
the narrative detail.

See also: [usage guide](usage_guide.md) (task walkthroughs),
[algorithm](algorithm.md) (the mathematics behind the QUBO, QAOA and the
quality metrics), and the module guides
[data loader](dataloader.md), [Rust core](rust_core.md),
[quantum backend](quantum_backend.md), [visualization](visualization.md),
[benchmarks](benchmarks.md), [testing](testing_manual.md).

## Contents

1. [Package metadata](#1-package-metadata)
2. [Data loading](#2-data-loading-qaoa_portfoliodata_loader)
3. [Portfolios and presets](#3-portfolios-and-presets-qaoa_portfolioportfolios-qaoa_portfoliopresets)
4. [Configuration](#4-configuration-qaoa_portfolioconfig)
5. [Financial metrics](#5-financial-metrics-qaoa_portfoliometrics)
6. [Parameters](#6-parameters-qaoa_portfolioparams)
7. [Validation](#7-validation-qaoa_portfoliovalidation)
8. [Utilities and timing](#8-utilities-and-timing-qaoa_portfolioutils-qaoa_portfoliotiming)
9. [Exceptions](#9-exceptions-qaoa_portfolioexceptions)
10. [Quantum backend](#10-quantum-backend-qaoa_portfolioquantum_backend)
11. [Visualization](#11-visualization-qaoa_portfoliovisualization)
12. [Benchmarks](#12-benchmarks-qaoa_portfoliobenchmarks)
13. [Command-line interface](#13-command-line-interface-qaoa-portfolio)
14. [Rust bridge](#14-rust-bridge-qaoa_portfolio_core)

**Import behavior.** Everything below is importable from the package root
(`from qaoa_portfolio import X`) unless the entry names a submodule. The
quantum backend, visualization and benchmark names are resolved lazily
(PEP 562 `__getattr__`): `import qaoa_portfolio` does not import PennyLane,
SciPy or matplotlib until one of those names is first accessed. `dir()` and
`__all__` list the lazy names as well.

---

## 1. Package metadata

| Name | Value |
|---|---|
| `__version__` | Package version string (`"0.1.0"`). |
| `__author__` | Author name. |
| `__email__` | Author contact address. |
| `__license__` | License identifier (`"CC BY-NC-ND 4.0"`). |

---

## 2. Data loading (`qaoa_portfolio.data_loader`)

Narrative: [dataloader.md](dataloader.md).

### `MarketDataLoader()`

Yahoo Finance loader configured from `config`: per-symbol CSV cache under
`data/cache/`, conservative rate limiting, and two-level validation.

| Method | Description |
|---|---|
| `async load_portfolio_data(symbols, start_date, end_date, include_volume=True, validate_data=True) -> pd.DataFrame` | Load daily bars for all symbols concurrently and return one DataFrame with `(symbol, price_type)` column MultiIndex (lower-cased Yahoo fields: `open`, `high`, `low`, `close`, `volume` unless `include_volume=False`, plus `dividends`/`stock_splits` when Yahoo returns them) on a timezone-naive date index; symbols that fail are logged and dropped, and `MarketDataError` is raised only when none loads. |
| `calculate_returns(price_data, return_type="simple", price_column="close") -> pd.DataFrame` | Return per-symbol simple (`pct_change`) or `"log"` returns from a loaded price frame, one column per symbol. |
| `get_market_data_summary(data) -> dict` | Summarize symbols, date range and data-quality figures of a loaded frame. |

### `get_free_tier_recommendations() -> dict`

Return the active data-source, cache and rate-limit settings plus free-tier best practices.

### `setup_free_tier_environment() -> None`

Print a summary of the free-tier configuration (not called automatically).

---

## 3. Portfolios and presets (`qaoa_portfolio.portfolios`, `qaoa_portfolio.presets`)

Narrative: [dataloader.md § Portfolio Utilities](dataloader.md#portfolio-utilities).

| Name | Signature | Description |
|---|---|---|
| `load_sp500_symbols` | `load_sp500_symbols() -> list[str]` | Read the S&P 500 symbol list from Wikipedia (dots mapped to dashes), falling back to a built-in large-cap list on any error. |
| `create_sample_portfolio` | `create_sample_portfolio(size=None) -> list[str]` | Return the first `size` large-cap stocks (default `config["portfolio.default_size"]`, 5). |
| `quick_portfolio_load` | `async quick_portfolio_load(symbols=None, portfolio_type="stock", days_back=252, preset=None) -> (price_data, returns_data)` | Load the last `days_back` calendar days for explicit symbols, a preset, or a sample `stock`/`crypto`/`mixed` portfolio, and return prices plus simple returns; failures raise `MarketDataError`. |

Public helpers importable from the submodules (not in `__all__`):

| Name | Module | Description |
|---|---|---|
| `PORTFOLIO_PRESETS` | `presets` (re-exported by `portfolios`) | Dict of named presets: `conservative_stocks`, `growth_stocks`, `major_crypto`, `defi_crypto`, `balanced_mixed`. |
| `LARGE_CAP_STOCKS`, `CRYPTO_SYMBOLS` | `presets` | Ordered symbol lists the sample creators slice from. |
| `get_preset_portfolio(preset_name) -> list[str]` | `presets`, `portfolios` | Return a preset's symbols; unknown names raise `ValueError`. |
| `list_portfolio_presets() -> dict[str, str]` | `presets`, `portfolios` | Map preset names to descriptions. |
| `load_crypto_symbols() -> list[str]` | `presets`, `portfolios` | Return the crypto symbol list in Yahoo format (`BTC-USD`, ...). |
| `create_sample_crypto_portfolio(size=None) -> list[str]` | `presets`, `portfolios` | Return the first `size` crypto symbols. |
| `create_mixed_portfolio(stocks=3, crypto=2) -> list[str]` | `presets`, `portfolios` | Concatenate sample stocks and sample crypto. |
| `classify_asset_type(symbol) -> str` | `portfolios` | Classify a symbol as `crypto` (`-USD` suffix), `forex`, `index` (`^` prefix) or `stock`. |
| `analyze_portfolio_composition(symbols) -> dict[str, int]` | `portfolios` | Count symbols per asset type. |

---

## 4. Configuration (`qaoa_portfolio.config`)

### `ConfigManager(config_path=None)`

Dot-notation configuration store with free-tier defaults, overlaid by
`config/settings.json` (or `config_path`) when that file exists.

| Method | Description |
|---|---|
| `get(key, default=None)` | Read a value by dotted key, e.g. `"data_sources.cache_enabled"`. |
| `set(key, value)` | Set a value by dotted key, creating intermediate sections. |
| `load_config()` / `save_config()` | Deep-merge the JSON file into the defaults / write the current configuration to it; I/O errors raise `ConfigurationError`. |
| `setup_logging()` | Configure root logging from the `logging` section (the CLI calls this; importing the package does not). |

### `config`

The process-wide `ConfigManager` instance used by the loader, presets and CLI.

---

## 5. Financial metrics (`qaoa_portfolio.metrics`)

### `FinancialMetrics`

Static reporting metrics on **simple** returns with compound annualization
(the Python reporting convention; the Rust QUBO layer uses log returns — see
[rust_core.md § Return and Annualization Conventions](rust_core.md#return-and-annualization-conventions)).

| Static method | Description |
|---|---|
| `simple_return(start_price, end_price)` | `end / start − 1`. |
| `log_return(start_price, end_price)` | `ln(end / start)`. |
| `annualized_return(returns, periods_per_year=252)` | `(1 + mean)^periods − 1`. |
| `annualized_volatility(returns, periods_per_year=252)` | `std · √periods`. |
| `sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)` | Annualized excess return over annualized volatility. |
| `sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252)` | Sharpe variant using downside deviation. |
| `max_drawdown(returns)` | Largest peak-to-trough drop of the compounded return path (a value ≤ 0). |
| `value_at_risk(returns, confidence_level=0.05)` | Historical VaR: the `confidence_level` percentile of the returns. |
| `conditional_var(returns, confidence_level=0.05)` | Expected shortfall beyond the VaR quantile. |
| `beta(asset_returns, market_returns)` | Covariance with the market over market variance. |
| `correlation(returns1, returns2)` | Pearson correlation of two return series. |
| `calculate_returns(price_data, return_type="simple", price_column="close")` | Per-symbol returns from a `(symbol, price_type)` price frame. |

---

## 6. Parameters (`qaoa_portfolio.params`)

| Class | Contents |
|---|---|
| `PortfolioParams` | `RISK_LEVELS`, `OBJECTIVES`, `REBALANCING` tables and `get_risk_params(risk_level)` (unknown levels fall back to `moderate`). |
| `MarketDataParams` | Trading-day constants (`TRADING_DAYS_PER_YEAR = 252`, ...) and data-quality thresholds (`MAX_MISSING_DATA_PCT`, `MAX_DAILY_RETURN_THRESHOLD`, `MIN_TRADING_DAYS_REQUIRED = 60`). |
| `QAOAParams` | QAOA defaults (`DEFAULT_LAYERS = 3`, `DEFAULT_MAX_ITERATIONS = 100`, `DEFAULT_CONVERGENCE_THRESHOLD = 1e-6`) and the single source of truth for `SUPPORTED_OPTIMIZERS` (`adam`, `gradient_descent`, `cobyla`, `nelder_mead`) and `SUPPORTED_BACKENDS` (`default.qubit`, `lightning.qubit`, `lightning.gpu`). |

---

## 7. Validation (`qaoa_portfolio.validation`)

### `DataValidator`

Static validators used by the loader (also re-exported from `qaoa_portfolio.utils`).

| Static method | Description |
|---|---|
| `val_symbols(symbols) -> list[str]` | Strip and upper-case symbols, dropping blanks (unusual characters only log a warning); an empty result raises `DataValidationError`. |
| `val_date_range(start_date, end_date) -> (start, end)` | Require `start < end` and an end no later than tomorrow (warns above 10 years). |
| `val_price_data(data, symbol="Unknown") -> bool` | Require a non-empty single-asset frame with a `close` column, strictly positive prices and at most 20 % missing cells (OHLC inconsistencies only warn). |
| `val_returns_data(returns, symbol="Unknown") -> bool` | Reject empty or infinite return data. |

---

## 8. Utilities and timing (`qaoa_portfolio.utils`, `qaoa_portfolio.timing`)

| Name | Signature | Description |
|---|---|---|
| `PerformanceTimer` | `PerformanceTimer(name="Operation", log_result=True)` | Context manager that times a block; `get_duration()` returns seconds. |
| `performance_monitor` | `@performance_monitor` | Decorator that logs a function's execution time. |
| `ensure_directory` | `ensure_directory(path) -> Path` | Create a directory (with parents) if needed and return it. |
| `safe_divide` | `safe_divide(numerator, denominator, default=0.0) -> float` | Divide, returning `default` when the denominator is zero. |
| `format_percentage` | `format_percentage(value, decimals=2) -> str` | Format a fraction as a percentage string (`0.123 → "12.30%"`). |
| `validate_weights` | `validate_weights(weights, tolerance=1e-6) -> bool` | True when weights are non-negative and sum to one. |
| `normalize_weights` | `normalize_weights(weights) -> list[float]` | Scale weights to sum to one (equal weights when the sum is zero). |

---

## 9. Exceptions (`qaoa_portfolio.exceptions`)

All package exceptions derive from `QAOAPortfolioError(Exception)`.

| Exception | Raised for |
|---|---|
| `QAOAPortfolioError` | Base class; catch it to handle any package error. |
| `MarketDataError` | Market data loading or processing failures. |
| `DataValidationError` | Invalid symbols, date ranges or price/return frames. |
| `OptimizationError` | Optimization algorithm errors (Python side; the Rust bridge has its own `qaoa_portfolio_core.OptimizationError`). |
| `QuantumBackendError` | Invalid `QAOAConfig`, malformed QUBO matrices, labels or bitstrings, or device creation failures. |
| `RateLimitError` | Data-provider rate limits. |
| `ConfigurationError` | Configuration loading or saving errors. |
| `VisualizationError` | Invalid visualization configuration or chart data. |
| `BenchmarkError` | Invalid benchmark configuration, solver names, or unpaired statistics. |

---

## 10. Quantum backend (`qaoa_portfolio.quantum_backend`)

Narrative: [quantum_backend.md](quantum_backend.md); mathematics:
[algorithm.md §3](algorithm.md#3-qaoa).

### `QAOAConfig`

```python
QAOAConfig(
    layers=3, optimizer="adam", max_iterations=100, convergence_threshold=1e-6,
    shots=None, seed=42, backend="default.qubit", num_restarts=3,
    max_stored_solutions=64, target_assets=None, feasible_decoding=False,
)
```

Frozen, validated QAOA settings: depth $p$ (`layers`), optimizer
(`adam`, `gradient_descent`, `cobyla`, `nelder_mead`), PennyLane device
(`default.qubit`, `lightning.qubit`, `lightning.gpu`), restarts, the number of
most-probable states kept for decoding, the cardinality `target_assets` used
for `feasible_probability`, and `feasible_decoding` (requires
`target_assets`). Invalid values raise `QuantumBackendError`.

### `QAOAResult`

Dataclass returned by every solve: `best_bitstring`, `best_solution`,
`selected_indices`, `selected_assets`, `objective_value`, `probabilities`
(top-`max_stored_solutions` states), `top_solutions` (at most 10, ranked by
objective), `optimal_parameters` (`gammas`, `betas`), `convergence_history`,
`iterations`, `elapsed_ms`, `metadata`. `to_dict()` returns a JSON-safe dict.
`metadata` keys: `backend`, `optimizer`, `layers`, `shots`, `num_restarts`,
`max_stored_solutions`, `num_variables`, `offset`, `source`, `expected_cost`,
`hamming_weight_distribution`, plus `feasible_probability` (when
`target_assets` is set) and `reference_probabilities` (when
`reference_bitstrings` is passed).

### `QAOAQuantumBackend(config=None)`

| Method | Description |
|---|---|
| `solve(qubo, labels=None, reference_bitstrings=None) -> QAOAResult` | Optimize the QAOA angles for a `PyQUBOMatrix` or square symmetric array and decode ranked selections; the final probabilities of `reference_bitstrings` are reported in `metadata["reference_probabilities"]`. |
| `build_cost_hamiltonian(qubo, offset=0.0)` | Instance wrapper that also accepts a `PyQUBOMatrix` (its offset is added). |
| `build_mixer_hamiltonian(num_wires)` | Instance wrapper for the X-mixer. |

### Functions

| Name | Signature | Description |
|---|---|---|
| `solve_qubo_qaoa` | `solve_qubo_qaoa(qubo, labels=None, config=None, reference_bitstrings=None) -> QAOAResult` | Convenience wrapper: `QAOAQuantumBackend(config).solve(...)`. |
| `build_cost_hamiltonian` | `build_cost_hamiltonian(qubo, offset=0.0) -> qml.Hamiltonian` | Map an upper-triangle-convention QUBO onto identity, $Z_i$ and $Z_iZ_j$ terms via $x_i = (1 - Z_i)/2$. |
| `build_mixer_hamiltonian` | `build_mixer_hamiltonian(num_wires) -> qml.Hamiltonian` | Return $\sum_i X_i$. |
| `evaluate_qubo_bitstring` | `evaluate_qubo_bitstring(qubo, bitstring, offset=0.0) -> float` | Evaluate $c + \sum_i Q_{ii}x_i + \sum_{i<j} Q_{ij}x_ix_j$ for a bitstring (character $i$ = variable $i$). |
| `bitstring_to_solution` | `bitstring_to_solution(bitstring) -> list[bool]` | Convert `"0110"` into a boolean selection mask. |
| `decode_solution` | `decode_solution(bitstring, labels=None) -> dict` | Return `bitstring`, `solution`, `selected_indices` and `selected_assets` (labels default to `x0, x1, ...`). |

Module-level, not in `__all__`: `hamming_weights(num_wires) -> np.ndarray`
(Hamming weight of every basis index, `uint8`), and the sets
`SUPPORTED_OPTIMIZERS` / `SUPPORTED_BACKENDS` (from `QAOAParams`).

---

## 11. Visualization (`qaoa_portfolio.visualization`)

Narrative: [visualization.md](visualization.md). Plot functions return a
matplotlib or plotly figure and never call `show()`; QAOA-facing functions
accept a `QAOAResult` or its `to_dict()` payload.

| Name | Signature | Description |
|---|---|---|
| `VisualizationConfig` | `VisualizationConfig(backend="matplotlib", style="default", figure_size=(10.0, 6.0), color_palette="tab10", max_solutions=10)` | Validated rendering settings (`backend` is `"matplotlib"` or `"plotly"`). |
| `normalize_qaoa_result` | `normalize_qaoa_result(result) -> dict` | Validate a QAOA result or dict and return a plain dict with the required fields. |
| `prepare_composition_data` | `prepare_composition_data(selected_assets, weights=None) -> (labels, weights)` | Labels and normalized weights (equal by default). |
| `prepare_risk_return_data` | `prepare_risk_return_data(returns, periods_per_year=252) -> pd.DataFrame` | Annualized return and volatility per asset via `FinancialMetrics`. |
| `prepare_probability_data` | `prepare_probability_data(result, max_solutions=10) -> list[(str, float)]` | Bitstring probabilities sorted descending. |
| `prepare_top_solutions_data` | `prepare_top_solutions_data(result, max_solutions=10) -> list[dict]` | Top solutions sorted by objective ascending. |
| `prepare_solver_comparison_data` | `prepare_solver_comparison_data(results, metric="objective_value") -> list[(str, float)]` | `(solver_name, value)` pairs from record mappings. |
| `plot_portfolio_composition` | `plot_portfolio_composition(selected_assets, weights=None, config=None)` | Pie chart of the selection. |
| `plot_risk_return_scatter` | `plot_risk_return_scatter(returns, highlighted_assets=None, config=None)` | Annualized volatility vs return per asset, selected assets starred. |
| `plot_correlation_heatmap` | `plot_correlation_heatmap(returns, config=None)` | Correlation matrix heatmap (annotated for ≤ 12 assets). |
| `plot_efficient_frontier` | `plot_efficient_frontier(frontier_points, selected_portfolio=None, config=None)` | Frontier points (`return`, `volatility`, optional `sharpe_ratio`) with an optional marked portfolio. |
| `plot_qaoa_convergence` | `plot_qaoa_convergence(result, config=None)` | Best-so-far cost per optimizer iteration. |
| `plot_solution_probabilities` | `plot_solution_probabilities(result, config=None)` | Most probable bitstrings. |
| `plot_top_solutions` | `plot_top_solutions(result, config=None)` | Best decoded solutions by objective, labeled with assets. |
| `render_qaoa_circuit_summary` | `render_qaoa_circuit_summary(result) -> str` | Text summary of qubits, depth, optimizer, device, angles and best solution. |
| `plot_solver_comparison` | `plot_solver_comparison(results, metric="objective_value", config=None)` | Bar chart of one metric across solver records. |

---

## 12. Benchmarks (`qaoa_portfolio.benchmarks`)

Narrative and results: [benchmarks.md](benchmarks.md); metric definitions:
[algorithm.md §4](algorithm.md#4-measuring-quality).

### Configuration and records

| Name | Signature | Description |
|---|---|---|
| `BenchmarkConfig` | `BenchmarkConfig(num_assets=8, risk_factor=0.5, target_assets=4, repeats=10, seed=42, qaoa=None, periods=252, sa=SAConfig())` | Frozen suite settings; `2 ≤ num_assets ≤ MAX_EXACT_ASSETS`, `1 ≤ target_assets ≤ num_assets`, `periods ≥ 60`; `qaoa=None` selects the benchmark preset (COBYLA, p = 1, 60 iterations, 2 restarts). |
| `BenchmarkRecord` | `BenchmarkRecord(solver_name, num_assets, objective_value, approximation_ratio, selected_assets, elapsed_ms, peak_memory_kb, seed, run_index, metadata={})` | One solver run on one instance; `to_dict()` is JSON-safe. `metadata` always has `feasible` (selection has exactly `target_assets` assets) and `optimum_reference` (`rust_brute_force` or `exact_enumeration`); QAOA records add `p_opt` and `feasible_probability`, SA records the schedule actually used. |
| `DEFAULT_SOLVERS` | `("brute_force", "simulated_annealing", "markowitz", "random", "qaoa")` | Solver names the harness accepts. |
| `MAX_EXACT_ASSETS` | `28` | Largest portfolio the harness accepts (exact statevector QAOA, exact reference optimum). |

Public, not in `__all__` (import from `qaoa_portfolio.benchmarks`):

| Name | Signature | Description |
|---|---|---|
| `SAConfig` | `SAConfig(schedule="default", sweeps=1000, restarts=1, initial_temperature=None, cooling_rate=None, max_iterations=None)` | Simulated-annealing settings: `default` is the Rust schedule (T0 = 100, cooling 0.995, 10 000 moves), `auto` scales it to the instance; explicit values override either; `restarts > 1` keeps the best of seeds `seed·1000 + r`. |
| `SA_SCHEDULES` | `frozenset({"default", "auto"})` | Valid `SAConfig.schedule` values. |
| `MAX_RUST_BRUTE_FORCE_ASSETS` | `20` | Above this size the reference optimum comes from `exact_enumeration`. |
| `MIN_BENCHMARK_PERIODS` | `60` | Minimum price history (rows) per instance and per in-sample window. |
| `auto_sa_schedule` | `auto_sa_schedule(qubo, target_assets, sweeps, seed) -> dict` | Instance-scaled schedule: returns `initial_temperature`, `cooling_rate`, `max_iterations` and the calibrating `median_uphill_delta`. |
| `exact_enumeration` | `exact_enumeration(qubo) -> (float, list[bool])` | Exact minimum over all $2^n$ bitstrings in chunked NumPy (≤ 40 variables). |
| `exact_optimum` | `exact_optimum(qubo) -> (float, str, list[bool])` | `(optimum, reference, solution)` from the Rust brute force for n ≤ 20, otherwise `exact_enumeration`. |
| `mcnemar_test` | `mcnemar_test(records, solver_a, solver_b) -> dict` | Exact two-sided McNemar test on paired optimal-hit outcomes. |
| `with_qaoa_overrides` | `with_qaoa_overrides(overrides) -> QAOAConfig \| None` | Return the benchmark QAOA preset with the non-`None` overrides applied, or `None` when nothing is overridden. |

### Functions in `__all__`

| Name | Signature | Description |
|---|---|---|
| `approximation_ratio` | `approximation_ratio(achieved, optimum, tolerance=1e-9) -> float` | Sign-safe quality in (0, 1]: 1 at the optimum, else $1/(1+\text{gap})$. |
| `generate_synthetic_prices` | `generate_synthetic_prices(num_assets, periods, seed) -> np.ndarray` | Deterministic positive `(periods, num_assets)` price matrix with a shared market factor. |
| `run_solver` | `run_solver(name, qubo, labels, *, prices, config, seed, run_index=0, optimum=None, optimum_reference=None, optimum_solution=None) -> BenchmarkRecord` | Run one solver on one QUBO; computes the optimum with `exact_optimum` when not given. |
| `run_quality_benchmark` | `run_quality_benchmark(config, solvers=DEFAULT_SOLVERS) -> list[BenchmarkRecord]` | Run every solver on `repeats` paired synthetic instances (seed = `config.seed + run_index`). |
| `run_scaling_benchmark` | `run_scaling_benchmark(asset_counts=(4, 8, 12, 16, 20), *, config, solvers=DEFAULT_SOLVERS)` | Quality benchmark per size with `target_assets = max(1, n // 2)`. |
| `run_layer_benchmark` | `run_layer_benchmark(layers=(1, 2, 3, 5, 10), *, config)` | QAOA-only quality benchmark per circuit depth. |
| `run_market_study` | `run_market_study(symbols, start_date, end_date, *, split=0.7, config, solvers=DEFAULT_SOLVERS) -> dict` | Optimize on the in-sample share of real prices and score each selection out of sample with `FinancialMetrics` (needs network). |
| `summarize_quality` | `summarize_quality(records) -> dict` | Per-solver `runs`, `mean_approximation_ratio`, `std_approximation_ratio`, `optimal_hit_rate`, `mean_elapsed_ms`, and when available `feasibility_rate`, `mean_p_opt`, `mean_feasible_probability`. |
| `significance_test` | `significance_test(records, solver_a, solver_b) -> dict` | Paired Wilcoxon signed-rank test on approximation ratios. |
| `save_benchmark_results` | `save_benchmark_results(records, *, suite, output_dir="results/benchmarks", config=None, extra=None) -> Path` | Write `<suite>-<UTC timestamp>.json` with config, records and extras. |

---

## 13. Command-line interface (`qaoa-portfolio`)

Entry point `qaoa_portfolio.cli:main`; `build_parser()` returns the
`argparse` parser. Without a subcommand the CLI loads market data and prints
a shape summary.

| Global option | Default | Description |
|---|---|---|
| `--symbols` | none | Comma-separated symbols, e.g. `AAPL,MSFT,BTC-USD`. |
| `--portfolio-type {stock,crypto,mixed}` | `stock` | Sample portfolio when neither symbols nor preset are given. |
| `--days-back` | `252` | Calendar days to load. |
| `--preset` | none | Preset name (`conservative_stocks`, `growth_stocks`, `major_crypto`, `defi_crypto`, `balanced_mixed`). |

`qaoa-portfolio benchmark` runs one suite, writes a JSON artifact and prints
a per-solver summary (ratio ± std, optimal and feasible rates, time, and
`p_opt` for QAOA). Exit code 1 on `BenchmarkError`/`QuantumBackendError`.

| Benchmark flag | Default | Description |
|---|---|---|
| `--suite {quality,scaling,layers,market}` | `quality` | Suite to run. |
| `--assets` | `8` | Portfolio size n (2–28). |
| `--target` | `n // 2` (at least 1) | Assets to select, k. |
| `--repeats` | `5` | Paired instances per size or depth. |
| `--seed` | `42` | Base seed; instance i uses `seed + i`. |
| `--periods` | `252` | Synthetic price rows per instance (≥ 60). |
| `--risk-factor` | `0.5` | Risk-aversion weight $q$ passed to `build_qubo` (the Rust core accepts 0–1). |
| `--solvers` | all five | Comma-separated subset of `DEFAULT_SOLVERS`. |
| `--asset-counts` | `4,8,12,16,20` | Sizes for the `scaling` suite. |
| `--layers` | `1,2,3,5,10` | Depths for the `layers` suite. |
| `--qaoa-layers` | preset (1) | QAOA depth p. |
| `--qaoa-iterations` | preset (60) | Optimizer iterations per restart. |
| `--qaoa-restarts` | preset (2) | Random restarts. |
| `--qaoa-optimizer` | preset (`cobyla`) | `adam`, `gradient_descent`, `cobyla` or `nelder_mead`; `adam` reproduces artifacts made before the preset switched to COBYLA. |
| `--qaoa-backend` | preset (`default.qubit`) | `default.qubit`, `lightning.qubit`, or `lightning.gpu` (needs the `gpu` extra). |
| `--qaoa-feasible-decoding` | off | Decode among the most probable states with exactly `--target` assets. |
| `--sa-schedule {default,auto}` | `default` | Rust default schedule or the instance-scaled schedule. |
| `--sa-sweeps` | `1000` | Moves per variable for `--sa-schedule auto`. |
| `--sa-restarts` | `1` | Independent SA runs per instance; the best is kept. |
| `--sa-iterations` | none | Override the SA move count of either schedule. |
| `--symbols`, `--start-date`, `--end-date` | none | Required by the `market` suite. |
| `--split` | `0.7` | In-sample fraction for the `market` suite. |
| `--output` | `results/benchmarks` | Artifact directory. |
| `--plot` | off | Also save a solver-comparison PNG next to the artifact (not for `market`). |

QAOA overrides merge into the benchmark preset field by field
(`with_qaoa_overrides`), so passing one flag never resets the others.

---

## 14. Rust bridge (`qaoa_portfolio_core`)

Narrative: [rust_core.md § Python Bridge](rust_core.md#python-bridge). Built
by maturin with the `python-bindings` feature (PyO3 0.29). Rust errors map to
`ValueError` (invalid input, dimension mismatch), `RuntimeError` (numerical,
serialization) or `qaoa_portfolio_core.OptimizationError` (solver failure).

| Name | Signature | Description |
|---|---|---|
| `build_qubo` | `build_qubo(prices, symbols, risk_aversion, target_assets) -> PyQUBOMatrix` | Build the cardinality-constrained Markowitz QUBO from a `(periods, assets)` float64 price array; `risk_aversion` ∈ [0, 1], `1 ≤ target_assets ≤ len(symbols)`. |
| `solve_brute_force` | `solve_brute_force(qubo) -> PyOptimizationResult` | Exact minimum over all $2^n$ bitstrings (n ≤ 20, parallel above 12). |
| `solve_simulated_annealing` | `solve_simulated_annealing(qubo, initial_temperature=None, cooling_rate=None, max_iterations=None, seed=None) -> PyOptimizationResult` | Single-bit-flip annealing from a random start; defaults T0 = 100, cooling 0.995, 10 000 moves, unseeded. |
| `solve_markowitz` | `solve_markowitz(prices, symbols) -> dict` | Continuous max-Sharpe weights (risk-free rate 0.02) with `weights`, `expected_return`, `volatility`, `sharpe_ratio`, `symbols`. |
| `PyQUBOMatrix` | `PyQUBOMatrix(num_variables)` | Symmetric QUBO with `num_variables`, `offset`, `evaluate(solution: list[bool])`, `to_numpy()`, `to_list()`. |
| `PyOptimizationResult` | returned by the solvers | `solution`, `objective_value`, `selected_assets`, `solver_name`, `iterations`, and `to_dict()` (adds `elapsed_ms` and `metadata` with `solutions_evaluated`, `best_found_at_iteration`, `convergence_history`). |
| `PyReturnSeries` | `PyReturnSeries(symbols, returns)` | Rust return statistics from a `(periods, assets)` log-return array: `num_periods`, `num_assets`, `mean_returns()` (daily mean × 252), `covariance_matrix()` (sample covariance × 252). |
| `OptimizationError` | exception | Raised when a Rust solver fails (e.g. brute force above 20 variables). |
