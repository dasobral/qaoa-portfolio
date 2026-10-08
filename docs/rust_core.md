# Rust Core API

Phase 2 provides the Rust computational core for portfolio-to-QUBO conversion and classical validation solvers. The crate name is `qaoa_portfolio`; the optional Python extension module is `qaoa_portfolio_core`.

## Modules

- `qaoa_portfolio::portfolio`: asset metadata, validated portfolios, and return statistics.
- `qaoa_portfolio::qubo`: symmetric QUBO matrix storage, Markowitz formulation, and constraint penalties.
- `qaoa_portfolio::optimization`: brute-force, simulated annealing, and continuous Markowitz baselines.
- `qaoa_portfolio::python`: PyO3 bindings enabled with `--features python-bindings` (PyO3 0.29 `Bound` API with the matching `numpy` 0.29 crate; build with maturin ≥ 1.0).

## Portfolio Data

Use `Asset::new(symbol, AssetClass)` to construct assets, then add expected return and volatility with builder methods:

```rust
use qaoa_portfolio::{Asset, AssetClass, Portfolio};

let aapl = Asset::new("AAPL", AssetClass::Stock)
    .with_return(0.12)
    .with_volatility(0.20);
let msft = Asset::new("MSFT", AssetClass::Stock);
let portfolio = Portfolio::new(vec![aapl, msft])?.with_weights(vec![0.6, 0.4])?;
```

`Portfolio::new` rejects empty portfolios, duplicate symbols, invalid assets, and malformed weights. `ReturnSeries::from_prices` accepts a row-major `DMatrix<f64>` of positive prices, computes log returns, and exposes annualized mean returns, covariance, and correlation matrices.

## Return and Annualization Conventions

The two layers intentionally use different conventions, each standard for its role:

| Layer | Returns | Annualized return | Annualized risk |
|---|---|---|---|
| Rust `ReturnSeries` (QUBO construction, solver baselines) | log returns | daily mean × 252 (exact for time-additive log returns) | covariance × 252 |
| Python `FinancialMetrics` (reporting, Phase 4 charts) | simple returns (`pct_change`) | `(1 + daily mean)^252 − 1` (compound) | daily std × √252 |

The conventions agree to first order (`exp(μ_log · 252) − 1 ≈ (1 + μ_simple)^252 − 1`) but diverge materially for volatile assets. **Rule for comparisons:** never mix layers in one table or chart. Rust conventions are internal to QUBO/solver construction; any reported or plotted metrics — including Phase 5 QAOA-vs-classical comparisons — must be computed for *all* candidates with Python `FinancialMetrics` on the same simple-returns series. A cross-layer consistency test in `tests/test_rust_bridge.py` verifies the first-order agreement.

## QUBO Formulation

`QUBOMatrix` stores a symmetric matrix plus an offset and variable labels. Off-diagonal entries are evaluated once using the upper triangle, which matches the binary QUBO coefficient convention used by the budget penalty expansion.

```rust
use qaoa_portfolio::{QUBOFormulation, ReturnSeries};

let qubo = QUBOFormulation::new(0.5, 3)?.build(&returns)?;
let objective = qubo.evaluate(&[true, false, true, true])?;
```

`QUBOFormulation::build_from_params` accepts precomputed covariance, expected returns, and labels. `PenaltyBuilder::budget` adds the exact-cardinality constraint `(sum(x) - k)^2`; `position_limit` and `diversity` provide optional concentration and diversification shaping terms.

## Solvers

`BruteForceSolver::solve` enumerates all solutions up to 20 variables. `solve_constrained` evaluates only bitstrings with exactly `k` selected assets. `SimulatedAnnealing` provides a seeded heuristic solver with configurable temperature, cooling rate, and iteration count. `MarkowitzSolver` computes continuous min-variance and max-Sharpe baselines from `ReturnSeries`.

All solvers return `OptimizationResult` or `ContinuousResult` with selected assets, objective value, solver metadata, and JSON serialization where applicable.

## Python Bridge

For normal development, let `uv` build and install the extension into `qaoa-env/`:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv sync --extra dev
UV_PROJECT_ENVIRONMENT=qaoa-env uv run python -c "import qaoa_portfolio_core"
```

For manual wheel verification, build the extension directly:

```bash
python -m maturin build --features python-bindings
```

The extension exposes `build_qubo(prices, symbols, risk_aversion, target_assets)`, `solve_brute_force(qubo)`, `solve_simulated_annealing(qubo, ...)`, and `solve_markowitz(prices, symbols)`. Errors map to Python `ValueError`, `RuntimeError`, or `qaoa_portfolio_core.OptimizationError`.
