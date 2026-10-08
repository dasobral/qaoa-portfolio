# Quantum Backend

The Phase 3 quantum backend implements a PennyLane QAOA solver for QUBO matrices produced by the Rust core or supplied as NumPy-compatible arrays. It lives in `qaoa_portfolio/quantum_backend.py` and is exported from the package root.

## Public API

```python
from qaoa_portfolio import QAOAConfig, QAOAQuantumBackend, solve_qubo_qaoa
```

Primary objects:

- `QAOAConfig` validates QAOA layers, optimizer, backend, iteration limits, shots, seed, restart count, the decoding cap, and the optional cardinality settings (`target_assets`, `feasible_decoding`).
- `QAOAQuantumBackend.solve(qubo, labels=None, reference_bitstrings=None)` runs QAOA and returns a `QAOAResult`; the final probabilities of any `reference_bitstrings` (e.g. the known optimum) are reported in `metadata["reference_probabilities"]`.
- `solve_qubo_qaoa(qubo, labels=None, config=None, reference_bitstrings=None)` is a convenience wrapper around `QAOAQuantumBackend`.
- `QAOAResult.to_dict()` returns JSON-safe values for reporting or later visualization.

Helper functions:

- `build_cost_hamiltonian(qubo, offset=0.0)`
- `build_mixer_hamiltonian(num_wires)`
- `evaluate_qubo_bitstring(qubo, bitstring, offset=0.0)`
- `bitstring_to_solution(bitstring)`
- `decode_solution(bitstring, labels=None)`

## Configuration

```python
config = QAOAConfig(
    layers=1,
    optimizer="gradient_descent",
    max_iterations=20,
    convergence_threshold=1e-8,
    shots=None,
    seed=42,
    backend="default.qubit",
    num_restarts=2,
    max_stored_solutions=64,
    target_assets=None,        # k: report probability mass on weight-k states
    feasible_decoding=False,   # decode among weight-k states only (needs target_assets)
)
```

`QAOAConfig()` defaults: `layers=3`, `optimizer="adam"`, `max_iterations=100`, `convergence_threshold=1e-6`, `shots=None`, `seed=42`, `backend="default.qubit"`, `num_restarts=3`, `max_stored_solutions=64`. The benchmark harness uses its own preset (p = 1, COBYLA, 60 iterations, 2 restarts; see [benchmarks.md](benchmarks.md)).

Supported optimizers are `adam`, `gradient_descent` (PennyLane, step size 0.01, stop after 5 consecutive steps with a cost change below `convergence_threshold`), `cobyla`, and `nelder_mead` (`scipy.optimize.minimize` with `maxiter=max_iterations`). Supported PennyLane devices are `default.qubit`, `lightning.qubit` (CPU, installed with the dev extra), and `lightning.gpu` (NVIDIA CUDA; install with `uv sync --extra dev --extra gpu` on Linux with Python ≥ 3.11). All three return identical results; the `lightning.*` devices are faster and use adjoint differentiation instead of backpropagation, which cuts memory sharply (measurements in [benchmarks.md §7](benchmarks.md#7-october-2026-campaign--rtx-3080-and-dgx-spark-gb10)). Use `shots=None` for exact statevector probabilities; a finite `shots` value makes the device estimate them from samples (the full statevector is still simulated).

`max_stored_solutions` (default 64) caps how many of the most probable basis states are kept in `QAOAResult.probabilities` and considered for ranking — the full 2ⁿ distribution grows exponentially and is never needed downstream. For n ≤ 6 the default keeps every state, so small-instance behavior is exact.

## QUBO Input

The solver accepts either:

1. `qaoa_portfolio_core.PyQUBOMatrix`
2. A square symmetric NumPy-compatible matrix

`PyQUBOMatrix` inputs use `.to_numpy()` and `.offset`. Labels are not stored by the Rust QUBO object, so pass the original symbols when asset names are needed:

```python
result = solve_qubo_qaoa(rust_qubo, labels=["AAPL", "MSFT", "NVDA"], config=config)
```

Matrices must be non-empty, square, finite, and symmetric within `1e-9`.

## Hamiltonian Mapping

The backend maps QUBO variables with `x_i = (1 - Z_i) / 2`. QUBO evaluation follows the Phase 2 upper-triangle convention:

```text
offset + sum_i Q[i, i] x_i + sum_i<j Q[i, j] x_i x_j
```

The mixer Hamiltonian is the standard X-mixer with one Pauli-X term per wire.

## Result Fields

`QAOAResult` includes:

- `best_bitstring` and `best_solution`
- `selected_indices` and `selected_assets`
- `objective_value`
- `probabilities`
- `top_solutions`
- `optimal_parameters`
- `convergence_history`
- `iterations`, `elapsed_ms`, and `metadata`

Top solutions are ranked by original QUBO objective value ascending, among the `max_stored_solutions` most probable basis states (`probabilities` carries exactly that capped set; the cap is echoed in `metadata["max_stored_solutions"]`). `top_solutions` holds the first 10 ranked entries and `best_*` the first one. With `feasible_decoding=True` the candidate set is restricted to states with exactly `target_assets` selected assets before the cap is applied, so the answer always selects `target_assets` assets.

`metadata` keys: `backend`, `optimizer`, `layers`, `shots`, `num_restarts`, `max_stored_solutions`, `num_variables`, `offset`, `source` (input type), `expected_cost` (final ⟨H_C⟩), and `hamming_weight_distribution` (probability mass per number of selected assets, index 0..n). With `target_assets` set it adds `feasible_probability` (mass on weight-k states); with `reference_bitstrings` it adds `reference_probabilities`.

## End-to-End Example

```python
import numpy as np
import qaoa_portfolio_core
from qaoa_portfolio import QAOAConfig, solve_qubo_qaoa

prices = np.full((70, 4), 100.0)
symbols = ["A", "B", "C", "D"]

qubo = qaoa_portfolio_core.build_qubo(prices, symbols, 0.5, 2)
result = solve_qubo_qaoa(
    qubo,
    labels=symbols,
    config=QAOAConfig(layers=1, max_iterations=10, num_restarts=1),
)

print(result.best_bitstring)
print(result.selected_assets)
print(result.objective_value)
```

## Verification

Run the Phase 3 tests:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run pytest tests/test_quantum_backend.py
UV_PROJECT_ENVIRONMENT=qaoa-env uv run pytest tests/test_qaoa_integration.py
```

Run the full project checks before merging Phase 3:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run pytest
cargo test
cargo clippy -- -D warnings
python -m maturin build --features python-bindings
```

## Current Limits

The backend targets simulator-backed QAOA and binary include/exclude portfolio selection with the standard X-mixer. Exact statevector simulation bounds the size: the benchmark harness accepts at most 28 assets, and a sampling-based pipeline beyond that is not implemented. Constraint-preserving mixers are not implemented; the cardinality constraint is a QUBO penalty, and feasible decoding only filters the decoded candidates.

## See Also

- [Algorithm](algorithm.md) — the Hamiltonian mapping, ansatz, optimizer loop, and decoding rule as equations.
- [API reference](api_reference.md#10-quantum-backend-qaoa_portfolioquantum_backend) — signatures.
- [Usage guide §5](usage_guide.md#5-solve-with-qaoa) — runnable walkthrough.
- [Visualization](visualization.md) and [benchmarks](benchmarks.md) — consumers of `QAOAResult`.
