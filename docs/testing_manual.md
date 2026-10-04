# Testing Manual

This manual describes how to test the current QAOA Portfolio Optimizer implementation. As of Phase 4, the core path and the visualization layer are implemented: mock or market price data can flow into the Rust QUBO builder, classical Rust baselines, the PennyLane QAOA backend, and the Phase 4 plotting helpers. Phase 5 and later are mainly benchmarking and presentation layers, so they should add coverage without weakening the core gates below.

## 1. Test Scope

The test suite is modular:

- Phase 1 Python foundation: market data loading, presets, configuration, validation, metrics, and CLI infrastructure.
- Phase 2 Rust core: portfolio data structures, QUBO construction, classical solvers, and PyO3 bridge.
- Phase 3 quantum backend: QUBO normalization, Hamiltonian construction, QAOA optimization, solution ranking, and Rust-to-QAOA integration.
- Phase 4 visualization: config validation, result normalization, chart-data helpers, headless Matplotlib/Plotly figure creation, and QAOA-result-to-figure integration.
- Phase 5 benchmarks: config/record validation, solver adapters, paired quality/scaling/depth suites, Wilcoxon significance testing, market studies (network-marked), and the `qaoa-portfolio benchmark` CLI artifact flow.
- Full core integration: mock market data -> Rust QUBO -> QAOA result -> figures -> benchmark records.

Current non-core gaps:

- The default CLI command validates market-data loading; the `benchmark` subcommand covers the full QUBO-to-QAOA solve path.
- Shot-based sampling for portfolios beyond 20 assets is an unimplemented stretch goal (spec §3.5).

## 2. Environment Setup

Use `uv` as the default Python environment manager, with `qaoa-env/` as this repository's project environment:

```bash
export UV_PROJECT_ENVIRONMENT=qaoa-env
uv sync --extra dev
source qaoa-env/bin/activate
qaoa-portfolio --help
```

`UV_PROJECT_ENVIRONMENT=qaoa-env` prevents uv from falling back to `.venv/`. Keep it exported in shells where you run `uv sync` or `uv run`; otherwise an active `qaoa-env` shell can still produce a `VIRTUAL_ENV ... does not match .venv` warning.

On an NVIDIA GPU host (Linux, Python ≥ 3.11), add the `gpu` extra to install the `lightning.gpu` simulator used by `--qaoa-backend lightning.gpu`. Always pass it on later syncs too: a sync without `--extra gpu` uninstalls the plugin.

```bash
uv sync --extra dev --extra gpu
```

For one-off commands without exporting first, prefix the command:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run pytest
```

For Codex-agent shell work, prefix commands with `rtk`, for example `rtk env UV_PROJECT_ENVIRONMENT=qaoa-env uv run pytest`.

## 3. Pytest Configuration

`pytest.ini` is the authoritative pytest configuration. It defines:

- Test discovery under `tests/`
- File pattern `test_*.py`
- Strict markers: `unit`, `integration`, `slow`, `network`, and `performance`. Every suite carries a module-level `pytestmark`, so `uv run pytest -m unit` and `-m integration` select real subsets; `slow`/`network`/`performance` are reserved for Phase 5 benchmarks and currently select nothing.
- Verbose output, short tracebacks, disabled warnings, and slowest-test durations

If pytest prints a warning about ignoring pytest config in `pyproject.toml`, that is expected because `pytest.ini` takes precedence.

## 4. Modular Test Commands

Run the complete Python suite:

```bash
uv run pytest
```

Run Phase 1 foundation tests:

```bash
uv run pytest tests/test_data_loader.py tests/test_portfolios.py tests/test_data_integration.py tests/test_simple_integration.py
```

Run Phase 2 Rust core tests:

```bash
cargo test
cargo test --features python-bindings
cargo clippy -- -D warnings
cargo clippy --features python-bindings -- -D warnings
```

Run Phase 2 Python bridge tests:

```bash
uv run pytest tests/test_rust_bridge.py
```

Run Phase 3 quantum backend tests:

```bash
uv run pytest tests/test_quantum_backend.py
uv run pytest tests/test_qaoa_integration.py
```

Run Phase 4 visualization tests (headless, Matplotlib `Agg`):

```bash
uv run pytest tests/test_visualization.py
```

Run formatting checks:

```bash
cargo fmt --check
uv run black --check qaoa_portfolio tests
```

Optional checks when changing public Python interfaces:

```bash
uv run flake8 qaoa_portfolio tests
uv run mypy qaoa_portfolio
```

These tools are declared as development dependencies, but the current required gate is the uv pytest, cargo, clippy, format, and maturin build sequence.

## 5. Full Core End-to-End Test

The canonical full-core automated test is:

```bash
uv run pytest tests/test_qaoa_integration.py::test_mock_market_data_to_rust_qubo_to_qaoa_workflow
```

It verifies:

1. Mock OHLCV market data is generated with realistic price columns.
2. Close prices are converted into a NumPy price matrix.
3. `qaoa_portfolio_core.build_qubo()` builds a Rust-backed QUBO.
4. `solve_qubo_qaoa()` solves the QUBO with PennyLane.
5. The result has a finite objective, valid bitstring, valid selected assets, probabilities, and metadata.

Manual smoke equivalent:

```bash
uv run python - <<'PY'
import numpy as np
import qaoa_portfolio_core

from qaoa_portfolio import QAOAConfig, solve_qubo_qaoa
from tests.utils import MockDataGenerator

symbols = ["AAPL", "MSFT", "NVDA", "AMZN"]
market_data = MockDataGenerator().create_realistic_price_data(symbols, days=75, seed=42)
prices = np.column_stack([
    market_data[(symbol, "close")].to_numpy(dtype=np.float64)
    for symbol in symbols
])

qubo = qaoa_portfolio_core.build_qubo(prices, symbols, 0.6, 2)
result = solve_qubo_qaoa(
    qubo,
    labels=symbols,
    config=QAOAConfig(
        layers=1,
        optimizer="gradient_descent",
        max_iterations=6,
        convergence_threshold=1e-12,
        seed=11,
        num_restarts=1,
    ),
)

assert len(result.best_solution) == len(symbols)
assert len(result.best_bitstring) == len(symbols)
assert np.isfinite(result.objective_value)
assert set(result.selected_assets).issubset(symbols)
assert result.probabilities
assert result.metadata["num_variables"] == len(symbols)

print(result.best_bitstring)
print(result.selected_assets)
print(result.objective_value)
PY
```

This is the best current approximation of a full application test because it exercises all implemented core layers without relying on live market data or future visualization modules.

## 6. Full Verification Gate

Use this before marking a core change complete:

```bash
cargo fmt --check
cargo test
cargo test --features python-bindings
cargo clippy -- -D warnings
cargo clippy --features python-bindings -- -D warnings
python3 -m maturin build --features python-bindings
uv run black --check qaoa_portfolio tests
uv run pytest
```

Expected current baseline after Phase 5 (June 2026):

- `uv run pytest`: 243 tests passing with `qaoa_portfolio_core` installed by uv (242 with `-m "not network"`; selectable via `-m unit` / `-m integration`).
- `uv run pytest --cov=qaoa_portfolio -m "not network"`: 88% total coverage (network-marked market-study paths excluded).
- `uv run flake8 qaoa_portfolio tests`: no findings (configured by `.flake8`, 88 columns, black-compatible).
- `cargo test`: 13 Rust tests passing.
- `cargo test --features python-bindings`: 13 Rust tests passing.
- `cargo clippy -- -D warnings`: no issues.
- `cargo clippy --features python-bindings -- -D warnings`: no issues.
- `python3 -m maturin build --features python-bindings`: produces a CPython wheel under `target/wheels/` for manual packaging checks.

## 7. Network and CLI Testing

Most tests avoid live network calls. Use mocked market data for routine development.

The CLI currently validates portfolio loading and data-return shapes:

```bash
uv run qaoa-portfolio --help
uv run qaoa-portfolio --preset growth_stocks --days-back 180
```

The second command may use Yahoo Finance and should be treated as an optional manual/network smoke test. Do not rely on it as the primary CI gate.

## 8. Troubleshooting

If `tests/test_rust_bridge.py` or `tests/test_qaoa_integration.py` skips or fails to import `qaoa_portfolio_core`, refresh the uv environment:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv sync --extra dev
```

For manual wheel verification, build the wheel directly:

```bash
python3 -m maturin build --features python-bindings
```

If PennyLane tests fail nondeterministically, reduce scope to the focused Phase 3 tests and confirm they use seeded `QAOAConfig` settings:

```bash
uv run pytest tests/test_quantum_backend.py tests/test_qaoa_integration.py
```

If market-data tests fail because of network behavior, prefer mock-based tests first:

```bash
uv run pytest tests/test_data_loader.py tests/test_portfolios.py
```

If a generated or ignored directory appears in status, do not treat it as source:

- `target/`
- `.pytest_cache/`
- `qaoa-env/`
- `__pycache__/`

## 9. Testing New Phases

Phase 4 visualization tests live in `tests/test_visualization.py` and use the Matplotlib `Agg` backend. They verify chart data and returned figure objects rather than pixel-perfect images; follow the same pattern when extending them.

Phase 5 benchmark tests should separate correctness from performance. Correctness can run in normal pytest; long-running timing studies should use `slow` or `performance` markers and should not block quick development loops unless explicitly requested.

Any phase that changes public behavior should update this manual, the relevant feature docs, and the project roadmap if phase status or deliverables change.
