# QAOA Portfolio Optimizer (QOPO)

A high-performance implementation of the Quantum Approximate Optimization Algorithm (QAOA) for portfolio optimization problems, demonstrating quantum-inspired solutions for real-world financial applications.

## 🎯 Overview

This project showcases how quantum-inspired algorithms can solve complex portfolio optimization problems that are challenging for classical methods. By implementing QAOA with classical simulation, we bridge the gap between current optimization capabilities and future quantum computing advantages.

## Quick Start

### Prerequisites

Install Rust and `uv` before setting up the project:

```bash
# Rust toolchain for the QUBO core and PyO3 extension
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# uv manages the Python environment and builds the maturin extension
python -m pip install uv
```

### Installation

```bash
git clone https://github.com/dasobral/qaoa-portfolio.git
cd qaoa-portfolio
export UV_PROJECT_ENVIRONMENT=qaoa-env
uv sync --extra dev
source qaoa-env/bin/activate
qaoa-portfolio --help
```

`UV_PROJECT_ENVIRONMENT=qaoa-env` makes uv use `qaoa-env/` as the project environment instead of `.venv/`. Keep that variable exported in shells where you run `uv sync` or `uv run`; otherwise uv will fall back to `.venv` and may warn that `VIRTUAL_ENV=qaoa-env` does not match the project environment.

On an NVIDIA GPU host (Linux, Python ≥ 3.11), add the `gpu` extra to install the `lightning.gpu` simulator used by `--qaoa-backend lightning.gpu`. Always pass it on later syncs too: a sync without `--extra gpu` uninstalls the plugin.

```bash
uv sync --extra dev --extra gpu
```

Build the Rust core directly when working on Rust internals:

```bash
cargo build --release
```

## Current Implementation Status

The QAOA Portfolio Optimizer is currently in active development with the following components implemented:

### ✅ Market Data Loader (Completed)

**Status:** Production-ready

- Yahoo Finance integration (100% free, no API keys required)
- Async data loading with performance monitoring
- Comprehensive data validation (dual-level: per-asset + portfolio-wide)
- Smart caching system with configurable duration
- Support for stocks, cryptocurrencies, and mixed portfolios
- Configuration-driven behavior with free-tier optimization

**Features:**

- `MarketDataLoader` class for async data loading
- Portfolio utilities for stocks, crypto, and mixed assets
- Predefined portfolio presets (conservative, growth, DeFi, etc.)
- Quick-start functions for rapid prototyping
- Professional error handling and logging

For detailed API documentation and usage examples, see [Market Data Loader Documentation](docs/dataloader.md).

### ✅ Rust QUBO Core (Completed)

- Portfolio and return-series data structures in Rust
- Covariance/correlation statistics and Markowitz-to-QUBO formulation
- Budget, position, and diversification penalty builders
- Brute-force, simulated annealing, and Markowitz baseline solvers
- PyO3 bridge module: `qaoa_portfolio_core`

See [Rust Core API](docs/rust_core.md) for usage and build details.

### ✅ PennyLane QAOA Quantum Backend (Completed)

- Cost Hamiltonian construction from Phase 2 QUBO matrices
- Standard X-mixer and configurable QAOA layer count
- Variational optimization with Adam, gradient descent, COBYLA, and Nelder-Mead
- Deterministic statevector runs for tests and optional shot-based sampling
- Ranked bitstring decoding into selected portfolio assets
- Rust QUBO bridge integration through `qaoa_portfolio_core.PyQUBOMatrix`

See [Quantum Backend API](docs/quantum_backend.md) for usage and configuration details.

### ✅ Visualization & Analysis (Completed)

- Portfolio composition, risk-return scatter, correlation heatmap, and efficient frontier plots
- QAOA convergence, solution probability, and top-solution charts from `QAOAResult` payloads
- Text-based QAOA circuit summaries and solver comparison plots (QAOA vs classical baselines)
- Matplotlib static figures by default with optional Plotly interactive backend
- Rendering-free chart-data helpers validated headlessly in tests

See [Visualization API](docs/visualization.md) for usage and configuration details.

### ✅ Benchmarking & Performance (Completed)

- Seeded, paired benchmark harness comparing QAOA against brute force, simulated annealing, Markowitz top-k, and random selection
- Approximation-ratio quality metric with paired Wilcoxon significance testing
- Time/memory scaling studies across 4–20 assets (up to 28 outside the harness) and QAOA depths 1–10
- Selectable simulator backend (`--qaoa-backend`: `default.qubit`, `lightning.qubit`, `lightning.gpu`), benchmarked on an RTX 3080 and an NVIDIA DGX Spark (GB10)
- Real market data studies (S&P 500 subset, crypto, mixed) with out-of-sample evaluation
- `qaoa-portfolio benchmark` CLI subcommand writing reproducible JSON artifacts

**Headline results** (8 assets, select 4, 10 paired instances; full tables and methodology in [Benchmarks](docs/benchmarks.md)):

| Solver | Mean quality ratio | Optimal runs | Median time |
|--------|-------------------:|-------------:|------------:|
| Brute force (Rust) | 1.000 | 10/10 | < 1 ms |
| Simulated annealing (Rust) | 1.000 | 10/10 | 0.7 ms |
| Markowitz top-k (Rust) | 0.886 | 6/10 | 0.1 ms |
| QAOA (PennyLane, 1 layer) | 0.825 | 3/10 | 23.2 s |
| Random selection | 0.558 | 0/10 | 0.1 ms |

QAOA beats random selection by +48 % relative quality (Wilcoxon p ≈ 0.002, exceeding the 15–25 % roadmap target) and is statistically indistinguishable from the classical Markowitz baseline (p ≈ 0.30). On real 2022–2024 data QAOA found the exact QUBO optimum for the crypto and mixed-asset studies.

**October 2026 re-measurement** (RTX 3080 + DGX Spark, [Benchmarks §7](docs/benchmarks.md)):

- Simulator backends change speed, never results: every backend reproduces the table above exactly.
- n = 20 QAOA solve: 396 s → 12.7 s on `lightning.gpu` (31×); the June 16 GB peak was `default.qubit` backprop memory, ~17 MB with the lightning backends.
- Exact simulation now reaches 26 assets on the RTX 3080 and 28 on the DGX Spark (measured outside the harness, which still caps n at 20). The RTX is ~2.8× faster per solve (memory bandwidth); the Spark goes larger (capacity).
- QAOA reaches the exact optimum at 24–26 assets where the default, untuned simulated annealing does not, but with a lower mean ratio and minutes-to-hours versus milliseconds — not yet a fair comparison.
- COBYLA beats Adam for this problem: better or equal hit rates at every size except 24 assets (2/5 vs 3/5), at 2.5–12× less time on the same backend.

**Results dashboard:** `python front/build_data.py` bundles every artifact under `results/benchmarks/` (including datasets copied from other hosts) and `front/index.html` displays it — quality, scaling, optimizer budget, depth, market studies, and a run log. Static page, no server needed; see [front/README.md](front/README.md).

### 🚧 In Development

- **Polish & Presentation:** Documentation, examples, and demos (Phase 6)

### Current CLI

After installing with `UV_PROJECT_ENVIRONMENT=qaoa-env uv sync --extra dev`:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run qaoa-portfolio --help
```

Example with a preset portfolio:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run qaoa-portfolio --preset growth_stocks --days-back 180
```

Run a benchmark suite:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run qaoa-portfolio benchmark --suite quality --assets 8 --repeats 10 --plot
```

### Current Limits

- The benchmark harness caps portfolios at 20 assets (`MAX_EXACT_ASSETS`; the Rust brute force that supplies the reference optimum accepts n ≤ 20). Exact simulation itself reaches 26 assets on a 10 GB RTX 3080 and 28 on a DGX Spark; beyond ~30 assets a shot-based sampling mode is required.
- Visualization covers reusable plotting functions and the static results dashboard in `front/`; notebook walkthroughs are deferred to later phases.
- Rendered quantum circuit diagrams are text-only summaries until Phase 6.

### 📋 Planned Components

- Advanced portfolio optimization algorithms
- Risk analysis and stress testing
- Backtesting framework
- Web-based dashboard
- Quantum readiness consulting tools

## Related Work

- Quantum machine learning for finance
- Variational quantum algorithms
- Portfolio optimization with quantum computing

## 📄 License

This project is licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives) - see the [LICENSE](LICENSE) file for details.
For commercial use, please contact the author to discuss licensing terms.

## 🤝 Acknowledgments

PennyLane Team for excellent quantum computing framework
Yahoo Finance for free market data access
