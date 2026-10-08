# Visualization

The Phase 4 visualization layer provides reusable plotting and chart-data helpers for portfolio composition, risk-return structure, QAOA optimization behavior, and solver comparisons. It lives in `qaoa_portfolio/visualization.py` and is exported from the package root.

Plotting functions return figure objects and never call `show()`. Chart-data helpers are independent from any rendering backend, so they can be tested and reused without a display server.

## Public API

```python
from qaoa_portfolio import (
    VisualizationConfig,
    VisualizationError,
    normalize_qaoa_result,
    # Portfolio visualizations
    plot_portfolio_composition,
    plot_risk_return_scatter,
    plot_correlation_heatmap,
    plot_efficient_frontier,
    # QAOA visualizations
    plot_qaoa_convergence,
    plot_solution_probabilities,
    plot_top_solutions,
    render_qaoa_circuit_summary,
    # Solver comparison
    plot_solver_comparison,
    # Chart-data helpers (rendering-free)
    prepare_composition_data,
    prepare_risk_return_data,
    prepare_probability_data,
    prepare_top_solutions_data,
    prepare_solver_comparison_data,
)
```

All validation failures raise `VisualizationError`, a subclass of `QAOAPortfolioError`.

## Configuration

```python
config = VisualizationConfig(
    backend="matplotlib",      # "matplotlib" (default) or "plotly"
    style="default",           # any registered Matplotlib style
    figure_size=(10.0, 6.0),   # two positive numbers (inches)
    color_palette="tab10",     # any registered Matplotlib colormap
    max_solutions=10,          # cap for probability/top-solution charts
)
```

Every plotting function accepts an optional `config` argument; when omitted, the defaults above apply. With `backend="matplotlib"` functions return `matplotlib.figure.Figure`; with `backend="plotly"` they return `plotly.graph_objects.Figure`.

## QAOA Result Inputs

All QAOA-facing functions accept either a Phase 3 `QAOAResult` instance or the plain dictionary produced by `QAOAResult.to_dict()`. Inputs are validated by `normalize_qaoa_result()`, which requires:

- `best_bitstring`, `selected_assets`, `objective_value`
- `probabilities`: non-empty mapping of bitstrings to finite probabilities
- `top_solutions`: records with `bitstring`, `selected_assets`, `objective_value`, and `probability`
- `convergence_history`: sequence of objective values per iteration

## Portfolio Visualizations

```python
plot_portfolio_composition(selected_assets, weights=None, config=None)
plot_risk_return_scatter(returns, highlighted_assets=None, config=None)
plot_correlation_heatmap(returns, config=None)
plot_efficient_frontier(frontier_points, selected_portfolio=None, config=None)
```

- **Composition** renders a pie chart. Weights default to equal allocation; provided weights must be positive, match the asset count, and are normalized to sum to one.
- **Risk-return scatter** plots annualized volatility against annualized return per asset, computed with `FinancialMetrics` using 252 trading periods per year (simple returns, compound annualization — the reporting convention; see "Return and Annualization Conventions" in `docs/rust_core.md` for how this differs from the Rust QUBO layer's log-return convention). `highlighted_assets` must be a subset of the return columns and are starred in red.
- **Correlation heatmap** renders `returns.corr()` with asset labels on both axes (Seaborn heatmap with the matplotlib backend, cells annotated with values for 12 or fewer assets; a Plotly heatmap with `backend="plotly"`).
- **Efficient frontier** requires `return` and `volatility` columns in `frontier_points`; an optional `sharpe_ratio` column colors the points. `selected_portfolio` must provide `return` and `volatility` keys and is marked with a star.

`returns` is a DataFrame with one column per asset symbol containing periodic returns, e.g. from `FinancialMetrics.calculate_returns(price_data)`.

## QAOA Visualizations

```python
plot_qaoa_convergence(result, config=None)
plot_solution_probabilities(result, config=None)
plot_top_solutions(result, config=None)
render_qaoa_circuit_summary(result)
```

- **Convergence** plots the full `convergence_history` (best-so-far expected cost of the winning restart) against the iteration index.
- **Solution probabilities** sorts bitstrings by probability descending and shows the top `config.max_solutions` entries.
- **Top solutions** sorts records by objective value ascending, caps them at `config.max_solutions`, and labels each bar with the selected assets.
- **Circuit summary** returns a text description (qubits, layers, optimizer, device backend, optimal parameters, structure, and best solution). Rendered circuit diagrams are not part of the library.

## Solver Comparison

```python
plot_solver_comparison(results, metric="objective_value", config=None)
```

`results` is a sequence of mappings; each record must include `solver_name` and the selected `metric` as a finite number. This supports QAOA, brute force, simulated annealing, and future benchmark records without depending on a single result class:

```python
records = [
    {"solver_name": "qaoa", "objective_value": result.objective_value},
    {"solver_name": "brute_force", "objective_value": brute.objective_value},
    {"solver_name": "simulated_annealing", "objective_value": annealed.objective_value},
]
fig = plot_solver_comparison(records)
```

## Chart-Data Helpers

The rendering-free helpers behind the plots are public and deterministic:

- `prepare_composition_data(selected_assets, weights=None)` → `(labels, normalized_weights)`
- `prepare_risk_return_data(returns, periods_per_year=252)` → DataFrame indexed by symbol with `annualized_return` and `annualized_volatility` columns
- `prepare_probability_data(result, max_solutions=10)` → `(bitstring, probability)` pairs sorted descending
- `prepare_top_solutions_data(result, max_solutions=10)` → solution records sorted by objective ascending
- `prepare_solver_comparison_data(results, metric)` → `(solver_name, value)` pairs

## End-to-End Example

```python
import numpy as np
import qaoa_portfolio_core
from qaoa_portfolio import (
    QAOAConfig,
    solve_qubo_qaoa,
    plot_qaoa_convergence,
    plot_solution_probabilities,
    render_qaoa_circuit_summary,
)

prices = np.full((70, 4), 100.0)
symbols = ["A", "B", "C", "D"]

qubo = qaoa_portfolio_core.build_qubo(prices, symbols, 0.5, 2)
result = solve_qubo_qaoa(
    qubo,
    labels=symbols,
    config=QAOAConfig(layers=1, max_iterations=10, num_restarts=1),
)

convergence_fig = plot_qaoa_convergence(result)
probability_fig = plot_solution_probabilities(result)
print(render_qaoa_circuit_summary(result))

convergence_fig.savefig("qaoa_convergence.png")
```

## Headless Usage

Tests and CI render without a display server by selecting the Matplotlib `Agg` backend before importing plotting code:

```python
import matplotlib
matplotlib.use("Agg")
```

## Verification

Run the Phase 4 tests:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run pytest tests/test_visualization.py
```

Run the full project checks:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run pytest
cargo test
cargo clippy -- -D warnings
python -m maturin build --features python-bindings
```

## Current Limits

The module provides reusable plotting functions only; it renders no circuit diagrams. Benchmark studies live in [benchmarks.md](benchmarks.md) (the CLI's `benchmark --plot` uses `plot_solver_comparison`), and end-to-end walkthroughs with saved figures are in the [usage guide](usage_guide.md#6-visualize) and [`examples/`](../examples/README.md).

## See Also

- [Usage guide §6](usage_guide.md#6-visualize) — runnable plotting walkthrough.
- [API reference](api_reference.md#11-visualization-qaoa_portfoliovisualization) — signatures.
- [Quantum backend](quantum_backend.md) — the `QAOAResult` fields these plots read.
- [Algorithm](algorithm.md) — what the probabilities and convergence history mean.
