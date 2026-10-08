"""
Benchmark harness for the QAOA Portfolio Optimizer (QOPO).

Phase 5 tooling: solution-quality comparisons against classical baselines,
execution-time and memory scaling studies, and real market data evaluation.
Every run is seeded and reproducible; one record schema serves all solvers
and feeds the Phase 4 solver-comparison charts directly.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
import tracemalloc
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .exceptions import BenchmarkError
from .metrics import FinancialMetrics
from .quantum_backend import QAOAConfig, solve_qubo_qaoa
from .utils import ensure_directory

# The Rust extension ships no type stubs, so the handle is typed Any.
_core: Any
try:
    import qaoa_portfolio_core as _core  # type: ignore[no-redef]
except ImportError:  # pragma: no cover - exercised only without the extension
    _core = None

logger = logging.getLogger(__name__)

#: Solvers supported by the harness; brute force defines the optimum.
DEFAULT_SOLVERS: Tuple[str, ...] = (
    "brute_force",
    "simulated_annealing",
    "markowitz",
    "random",
    "qaoa",
)

#: Largest portfolio the harness accepts. Exact statevector QAOA was measured
#: up to n = 28 (DGX Spark GB10, October 2026); the reference optimum is exact
#: at every accepted size. Larger sizes need shot-based sampling.
MAX_EXACT_ASSETS = 28

#: The Rust brute-force solver refuses larger inputs; above this size the
#: reference optimum comes from a chunked NumPy enumeration instead.
MAX_RUST_BRUTE_FORCE_ASSETS = 20

#: Bitstrings scored per enumeration chunk (bounds memory to ~n * 8 MiB).
_ENUMERATION_CHUNK = 1 << 20

#: Minimum price history for stable covariance estimates.
MIN_BENCHMARK_PERIODS = 60

_RATIO_TOLERANCE = 1e-9

#: Light QAOA settings used when a study does not provide its own config:
#: benchmarking favors many repeats over deep single optimizations. COBYLA
#: replaced Adam in October 2026 (better or equal hit rates except n = 24, at
#: 2.5-12x less time, no gradient memory); pass optimizer="adam" to reproduce
#: earlier artifacts.
_DEFAULT_BENCH_QAOA = QAOAConfig(
    layers=1,
    optimizer="cobyla",
    max_iterations=60,
    num_restarts=2,
)


def with_qaoa_overrides(overrides: Dict[str, Any]) -> Optional[QAOAConfig]:
    """Return the benchmark QAOA preset with the given overrides applied.

    Unspecified fields keep the preset's values, so partial CLI overrides
    never silently replace the preset with full-parameter defaults.
    Returns ``None`` when nothing is overridden, so callers can keep the
    shared preset untouched.
    """
    changes = {key: value for key, value in overrides.items() if value is not None}
    if not changes:
        return None
    return replace(_DEFAULT_BENCH_QAOA, **changes)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration shared by all benchmark suites."""

    num_assets: int = 8
    risk_factor: float = 0.5
    target_assets: int = 4
    repeats: int = 10
    seed: int = 42
    qaoa: Optional[QAOAConfig] = None
    periods: int = 252

    def __post_init__(self) -> None:
        if self.num_assets < 2:
            raise BenchmarkError("num_assets must be at least 2")
        if self.num_assets > MAX_EXACT_ASSETS:
            raise BenchmarkError(
                f"num_assets={self.num_assets} exceeds the exact-simulation limit "
                f"of {MAX_EXACT_ASSETS}; larger sizes require shot-based sampling"
            )
        if not 0 < self.target_assets <= self.num_assets:
            raise BenchmarkError(
                "target_assets must be between 1 and num_assets inclusive"
            )
        if self.repeats < 1:
            raise BenchmarkError("repeats must be at least 1")
        if self.periods < MIN_BENCHMARK_PERIODS:
            raise BenchmarkError(
                f"periods must be at least {MIN_BENCHMARK_PERIODS} "
                "for stable covariance estimates"
            )
        if not math.isfinite(self.risk_factor) or self.risk_factor <= 0:
            raise BenchmarkError("risk_factor must be a positive finite number")


@dataclass(frozen=True)
class BenchmarkRecord:
    """One solver run on one problem instance."""

    solver_name: str
    num_assets: int
    objective_value: float
    approximation_ratio: float
    selected_assets: List[str]
    elapsed_ms: float
    peak_memory_kb: Optional[float]
    seed: int
    run_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the record."""

        return _json_safe(asdict(self))


def generate_synthetic_prices(num_assets: int, periods: int, seed: int) -> np.ndarray:
    """Generate a deterministic positive price matrix of shape (periods, assets)."""

    if num_assets < 1 or periods < 2:
        raise BenchmarkError("synthetic prices need num_assets >= 1, periods >= 2")

    rng = np.random.default_rng(seed)
    drifts = 0.0002 * (1.0 + np.arange(num_assets))
    volatilities = 0.01 + 0.002 * np.arange(num_assets)
    shocks = rng.normal(loc=drifts, scale=volatilities, size=(periods, num_assets))
    # A shared market factor keeps the covariance matrix non-trivial.
    market = rng.normal(loc=0.0, scale=0.005, size=(periods, 1))
    returns = np.clip(shocks + market, -0.5, 0.5)
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    return np.asarray(prices, dtype=np.float64)


def approximation_ratio(
    achieved: float, optimum: float, tolerance: float = _RATIO_TOLERANCE
) -> float:
    """Map an achieved minimization objective onto (0, 1] quality.

    1.0 means the brute-force optimum was reached; the ratio decays as
    ``1 / (1 + gap)`` with the relative optimality gap, which stays
    well-defined when objectives are negative or cross zero (exact formula
    documented in docs/benchmarks.md).
    """

    if not (math.isfinite(achieved) and math.isfinite(optimum)):
        raise BenchmarkError("approximation_ratio requires finite objective values")
    if achieved < optimum - tolerance:
        raise BenchmarkError(
            "achieved objective is below the optimum; check solver/optimum pairing"
        )
    if abs(achieved - optimum) <= tolerance:
        return 1.0

    scale = max(abs(optimum), tolerance)
    gap = (achieved - optimum) / scale
    return 1.0 / (1.0 + gap)


def exact_enumeration(qubo: Any) -> Tuple[float, List[bool]]:
    """Return the exact QUBO minimum and its bitstring by full enumeration.

    Scores all 2^n bitstrings in chunks with NumPy, using the upper-triangle
    convention of ``PyQUBOMatrix.evaluate``; bit ``j`` of the solution is
    variable ``j``. Used as the reference optimum above
    ``MAX_RUST_BRUTE_FORCE_ASSETS``; validated bit-identical to the Rust brute
    force at n = 12-20 during the October 2026 campaign.
    """

    upper = np.triu(np.asarray(qubo.to_numpy(), dtype=np.float64))
    num_variables = upper.shape[0]
    if num_variables > 40:
        raise BenchmarkError("exact enumeration is limited to 40 variables")

    shifts = np.uint64(num_variables - 1) - np.arange(num_variables, dtype=np.uint64)
    best_value = math.inf
    best_index = 0
    total = 1 << num_variables
    for start in range(0, total, _ENUMERATION_CHUNK):
        indices = np.arange(
            start, min(start + _ENUMERATION_CHUNK, total), dtype=np.uint64
        )
        bits = ((indices[:, None] >> shifts) & np.uint64(1)).astype(np.float64)
        values = np.einsum("ij,ij->i", bits @ upper, bits)
        position = int(np.argmin(values))
        if values[position] < best_value:
            best_value = float(values[position])
            best_index = start + position

    solution = [
        bool((best_index >> (num_variables - 1 - j)) & 1) for j in range(num_variables)
    ]
    return best_value + float(qubo.offset), solution


def exact_optimum(qubo: Any) -> Tuple[float, str]:
    """Return ``(optimum, reference)`` for a QUBO instance.

    The reference names the method: ``"rust_brute_force"`` up to
    ``MAX_RUST_BRUTE_FORCE_ASSETS`` variables, ``"exact_enumeration"`` above.
    """

    _require_core()
    if qubo.num_variables <= MAX_RUST_BRUTE_FORCE_ASSETS:
        return float(_core.solve_brute_force(qubo).objective_value), "rust_brute_force"
    value, _ = exact_enumeration(qubo)
    return value, "exact_enumeration"


def run_solver(
    name: str,
    qubo: Any,
    labels: Sequence[str],
    *,
    prices: np.ndarray,
    config: BenchmarkConfig,
    seed: int,
    run_index: int = 0,
    optimum: Optional[float] = None,
    optimum_reference: Optional[str] = None,
) -> BenchmarkRecord:
    """Run one solver on one QUBO instance and return its benchmark record.

    When ``optimum`` is not provided it is computed with `exact_optimum`, so
    the approximation ratio is always defined. The record's metadata names
    the reference method under ``optimum_reference``.
    """

    _require_core()
    if name not in DEFAULT_SOLVERS:
        raise BenchmarkError(
            f"Unknown solver '{name}'. Expected one of {sorted(DEFAULT_SOLVERS)}"
        )

    if optimum is None:
        optimum, optimum_reference = exact_optimum(qubo)

    track_memory = name in {"qaoa", "random"} and not tracemalloc.is_tracing()
    peak_memory_kb: Optional[float] = None

    if track_memory:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        objective, selected, metadata = _dispatch_solver(
            name, qubo, labels, prices=prices, config=config, seed=seed
        )
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if track_memory:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_kb = peak_bytes / 1024.0

    if optimum_reference is not None:
        metadata = {**metadata, "optimum_reference": optimum_reference}

    return BenchmarkRecord(
        solver_name=name,
        num_assets=len(labels),
        objective_value=float(objective),
        approximation_ratio=approximation_ratio(float(objective), optimum),
        selected_assets=list(selected),
        elapsed_ms=elapsed_ms,
        peak_memory_kb=peak_memory_kb,
        seed=seed,
        run_index=run_index,
        metadata=metadata,
    )


def run_quality_benchmark(
    config: BenchmarkConfig,
    solvers: Sequence[str] = DEFAULT_SOLVERS,
) -> List[BenchmarkRecord]:
    """Run all requested solvers on `repeats` paired synthetic instances (5A)."""

    _require_core()
    _validate_solvers(solvers)

    records: List[BenchmarkRecord] = []
    for run_index in range(config.repeats):
        run_seed = config.seed + run_index
        prices = generate_synthetic_prices(config.num_assets, config.periods, run_seed)
        labels = [f"A{index}" for index in range(config.num_assets)]
        qubo = _core.build_qubo(
            prices, labels, config.risk_factor, config.target_assets
        )
        optimum, reference = exact_optimum(qubo)

        for name in solvers:
            records.append(
                run_solver(
                    name,
                    qubo,
                    labels,
                    prices=prices,
                    config=config,
                    seed=run_seed,
                    run_index=run_index,
                    optimum=optimum,
                    optimum_reference=reference,
                )
            )

    return records


def summarize_quality(records: Sequence[BenchmarkRecord]) -> Dict[str, Any]:
    """Aggregate per-solver quality statistics from benchmark records."""

    if not records:
        raise BenchmarkError("summarize_quality requires at least one record")

    summary: Dict[str, Any] = {}
    for name in sorted({record.solver_name for record in records}):
        ratios = [
            record.approximation_ratio
            for record in records
            if record.solver_name == name
        ]
        timings = [
            record.elapsed_ms for record in records if record.solver_name == name
        ]
        summary[name] = {
            "runs": len(ratios),
            "mean_approximation_ratio": float(np.mean(ratios)),
            "std_approximation_ratio": float(np.std(ratios)),
            "optimal_hit_rate": float(
                np.mean([ratio >= 1.0 - _RATIO_TOLERANCE for ratio in ratios])
            ),
            "mean_elapsed_ms": float(np.mean(timings)),
        }
    return summary


def significance_test(
    records: Sequence[BenchmarkRecord], solver_a: str, solver_b: str
) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank test on approximation ratios.

    Records are paired by (num_assets, run_index), i.e. identical problem
    instances. With fewer than ~10 pairs the p-value is indicative only.
    """

    pairs: Dict[Tuple[int, int], Dict[str, float]] = {}
    for record in records:
        if record.solver_name in (solver_a, solver_b):
            key = (record.num_assets, record.run_index)
            pairs.setdefault(key, {})[record.solver_name] = record.approximation_ratio

    ratios_a = []
    ratios_b = []
    for values in pairs.values():
        if solver_a in values and solver_b in values:
            ratios_a.append(values[solver_a])
            ratios_b.append(values[solver_b])

    if not ratios_a:
        raise BenchmarkError(
            f"No paired records found for '{solver_a}' vs '{solver_b}'"
        )

    differences = np.asarray(ratios_a) - np.asarray(ratios_b)
    if np.allclose(differences, 0.0, atol=_RATIO_TOLERANCE):
        return {
            "solver_a": solver_a,
            "solver_b": solver_b,
            "statistic": 0.0,
            "p_value": 1.0,
            "num_pairs": len(ratios_a),
            "identical": True,
        }

    from scipy.stats import wilcoxon  # type: ignore[import-untyped]

    outcome = wilcoxon(ratios_a, ratios_b)
    return {
        "solver_a": solver_a,
        "solver_b": solver_b,
        "statistic": float(outcome.statistic),
        "p_value": float(outcome.pvalue),
        "num_pairs": len(ratios_a),
        "identical": False,
    }


def run_scaling_benchmark(
    asset_counts: Sequence[int] = (4, 8, 12, 16, 20),
    *,
    config: BenchmarkConfig,
    solvers: Sequence[str] = DEFAULT_SOLVERS,
) -> List[BenchmarkRecord]:
    """Measure solver behavior across portfolio sizes (5B).

    Each size selects half of the assets (at least one), so the cardinality
    constraint scales with the problem rather than degenerating to
    select-everything for small portfolios.
    """

    records: List[BenchmarkRecord] = []
    for count in asset_counts:
        sized = replace(
            config,
            num_assets=count,
            target_assets=max(1, count // 2),
        )
        records.extend(run_quality_benchmark(sized, solvers=solvers))
    return records


def run_layer_benchmark(
    layers: Sequence[int] = (1, 2, 3, 5, 10),
    *,
    config: BenchmarkConfig,
) -> List[BenchmarkRecord]:
    """Measure QAOA cost/quality as circuit depth grows (5B)."""

    base = config.qaoa if config.qaoa is not None else _DEFAULT_BENCH_QAOA

    records: List[BenchmarkRecord] = []
    for depth in layers:
        if depth < 1:
            raise BenchmarkError("QAOA layer counts must be positive")
        layered = replace(config, qaoa=replace(base, layers=depth))
        records.extend(run_quality_benchmark(layered, solvers=("qaoa",)))
    return records


def run_market_study(
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    *,
    split: float = 0.7,
    config: BenchmarkConfig,
    solvers: Sequence[str] = DEFAULT_SOLVERS,
) -> Dict[str, Any]:
    """Optimize on in-sample history, evaluate out-of-sample (5C).

    The first ``split`` fraction of the window feeds the QUBO; the held-out
    remainder scores each solver's equal-weighted selection with
    `FinancialMetrics`.
    """

    _require_core()
    _validate_solvers(solvers)
    if not symbols:
        raise BenchmarkError("market studies require at least one symbol")
    if not 0.0 < split < 1.0:
        raise BenchmarkError("split must be strictly between 0 and 1")
    if config.target_assets > len(symbols):
        raise BenchmarkError("target_assets cannot exceed the number of symbols")

    closes = _load_close_prices(list(symbols), start_date, end_date)
    cutoff = int(len(closes) * split)
    if cutoff < MIN_BENCHMARK_PERIODS:
        raise BenchmarkError(
            f"in-sample window has {cutoff} rows; at least "
            f"{MIN_BENCHMARK_PERIODS} are required"
        )
    if len(closes) - cutoff < 5:
        raise BenchmarkError("out-of-sample window must contain at least 5 rows")

    labels = list(closes.columns)
    in_sample = closes.iloc[:cutoff]
    out_sample = closes.iloc[cutoff:]
    prices = in_sample.to_numpy(dtype=np.float64)
    qubo = _core.build_qubo(prices, labels, config.risk_factor, config.target_assets)
    optimum, reference = exact_optimum(qubo)

    out_returns = out_sample.pct_change().dropna()
    study: Dict[str, Any] = {
        "symbols": labels,
        "start_date": start_date,
        "end_date": end_date,
        "split": split,
        "in_sample_rows": int(cutoff),
        "out_of_sample_rows": int(len(out_sample)),
        "solvers": {},
    }

    for name in solvers:
        record = run_solver(
            name,
            qubo,
            labels,
            prices=prices,
            config=config,
            seed=config.seed,
            optimum=optimum,
            optimum_reference=reference,
        )
        study["solvers"][name] = {
            "record": record.to_dict(),
            "out_of_sample": _out_of_sample_metrics(
                out_returns, record.selected_assets
            ),
        }

    return study


def save_benchmark_results(
    records: Sequence[Union[BenchmarkRecord, Dict[str, Any]]],
    *,
    suite: str,
    output_dir: Union[str, Path] = "results/benchmarks",
    config: Optional[BenchmarkConfig] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write one benchmark run to a timestamped JSON artifact and return its path."""

    if not suite or not suite.strip():
        raise BenchmarkError("suite must be a non-empty name")

    directory = ensure_directory(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{suite}-{timestamp}.json"

    payload: Dict[str, Any] = {
        "suite": suite,
        "created_utc": timestamp,
        "config": _json_safe(asdict(config)) if config is not None else None,
        "records": [
            record.to_dict() if isinstance(record, BenchmarkRecord) else record
            for record in records
        ],
    }
    if extra:
        payload.update(_json_safe(extra))

    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Saved {suite} benchmark artifact to {path}")
    return path


# ============================================================================
# Internal helpers
# ============================================================================


def _require_core() -> None:
    if _core is None:
        raise BenchmarkError(
            "qaoa_portfolio_core is required for benchmarking. Install it "
            "with: uv sync --extra dev"
        )


def _validate_solvers(solvers: Sequence[str]) -> None:
    if not solvers:
        raise BenchmarkError("at least one solver must be requested")
    for name in solvers:
        if name not in DEFAULT_SOLVERS:
            raise BenchmarkError(
                f"Unknown solver '{name}'. Expected one of {sorted(DEFAULT_SOLVERS)}"
            )


def _dispatch_solver(
    name: str,
    qubo: Any,
    labels: Sequence[str],
    *,
    prices: np.ndarray,
    config: BenchmarkConfig,
    seed: int,
) -> Tuple[float, List[str], Dict[str, Any]]:
    """Run one solver and return (objective, selected assets, metadata)."""

    if name == "brute_force":
        if len(labels) > MAX_RUST_BRUTE_FORCE_ASSETS:
            objective, solution = exact_enumeration(qubo)
            return (
                objective,
                [label for label, bit in zip(labels, solution) if bit],
                {"iterations": 1 << len(labels), "method": "exact_enumeration"},
            )
        result = _core.solve_brute_force(qubo)
        return (
            result.objective_value,
            list(result.selected_assets),
            {"iterations": result.iterations},
        )

    if name == "simulated_annealing":
        result = _core.solve_simulated_annealing(qubo, seed=seed)
        return (
            result.objective_value,
            list(result.selected_assets),
            {"iterations": result.iterations},
        )

    if name == "markowitz":
        continuous = _core.solve_markowitz(prices, list(labels))
        weights = np.asarray(continuous["weights"], dtype=float)
        ranked = np.argsort(weights)[::-1][: config.target_assets]
        solution = [index in set(ranked.tolist()) for index in range(len(labels))]
        objective = qubo.evaluate(solution)
        selected = [labels[index] for index in sorted(ranked.tolist())]
        return (
            objective,
            selected,
            {
                "selection": "top weights from continuous Markowitz solution",
                "sharpe_ratio": float(continuous["sharpe_ratio"]),
            },
        )

    if name == "random":
        rng = random.Random(seed)
        chosen = sorted(rng.sample(range(len(labels)), config.target_assets))
        solution = [index in set(chosen) for index in range(len(labels))]
        objective = qubo.evaluate(solution)
        return (
            objective,
            [labels[index] for index in chosen],
            {"selection": "uniform cardinality-constrained sample"},
        )

    # name == "qaoa" (guarded by run_solver)
    base = config.qaoa if config.qaoa is not None else _DEFAULT_BENCH_QAOA
    qaoa_config = replace(base, seed=seed)
    result = solve_qubo_qaoa(qubo, labels=list(labels), config=qaoa_config)
    return (
        result.objective_value,
        list(result.selected_assets),
        {
            "layers": qaoa_config.layers,
            "optimizer": qaoa_config.optimizer,
            "max_iterations": qaoa_config.max_iterations,
            "num_restarts": qaoa_config.num_restarts,
            "iterations": result.iterations,
        },
    )


def _load_close_prices(
    symbols: List[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """Load close prices per symbol as a plain (periods, assets) DataFrame."""

    from .data_loader import MarketDataLoader

    loader = MarketDataLoader()
    price_data = asyncio.run(loader.load_portfolio_data(symbols, start_date, end_date))

    closes = pd.DataFrame(
        {symbol: price_data[(symbol, "close")] for symbol in symbols}
    ).dropna()
    if closes.empty:
        raise BenchmarkError("no overlapping close prices for the requested window")
    return closes


def _out_of_sample_metrics(
    out_returns: pd.DataFrame, selected_assets: Sequence[str]
) -> Dict[str, float]:
    """Equal-weighted out-of-sample performance of one selection."""

    if not selected_assets:
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    portfolio = out_returns[list(selected_assets)].mean(axis=1)
    return {
        "annualized_return": float(FinancialMetrics.annualized_return(portfolio)),
        "annualized_volatility": float(
            FinancialMetrics.annualized_volatility(portfolio)
        ),
        "sharpe_ratio": float(FinancialMetrics.sharpe_ratio(portfolio)),
        "max_drawdown": float(FinancialMetrics.max_drawdown(portfolio)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, QAOAConfig):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    return value
