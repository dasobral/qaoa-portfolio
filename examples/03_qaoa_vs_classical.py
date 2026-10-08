"""
Example 03 - QAOA vs classical solvers: an indicative mini-benchmark.

Runs the project's benchmark harness on a handful of seeded synthetic
six-asset instances (select three), pairing QAOA against brute force,
simulated annealing, a Markowitz top-k heuristic, and random selection:

1. `run_quality_benchmark` - every solver on the same instances;
2. `summarize_quality` - mean approximation ratio, optimal-hit rate,
   feasibility rate, and QAOA's mean probability on the optimum (p_opt);
3. `significance_test` (paired Wilcoxon on ratios) and `mcnemar_test`
   (exact McNemar on optimal hits);
4. `plot_solver_comparison` - bar charts of quality and p_opt.

This is a *mini-run* sized to finish in well under a minute on a laptop CPU.
With so few paired instances the p-values are indicative only. The curated,
full-size numbers (8-28 assets, many instances, GPU backends) live in
``docs/benchmarks.md``.

Run it from the repository root::

    python examples/03_qaoa_vs_classical.py

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

from qaoa_portfolio import (
    BenchmarkConfig,
    plot_solver_comparison,
    run_quality_benchmark,
    significance_test,
    summarize_quality,
)

# `mcnemar_test` is not re-exported at package level; import it directly.
from qaoa_portfolio.benchmarks import mcnemar_test

# Headless rendering: figures are saved to disk, never shown.
matplotlib.use("Agg")

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main(
    output_dir: Optional[Path] = None,
    repeats: int = 4,
    num_assets: int = 6,
    target_assets: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the mini-benchmark and return summary, tests, and figure paths."""

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Configure ------------------------------------------------------
    # Instance r uses seed `seed + r` for its synthetic prices, and every
    # solver sees the same instance, so results are paired. Leaving
    # `qaoa=None` uses the harness preset: p = 1, COBYLA, 60 iterations,
    # 2 restarts, on the CPU `default.qubit` simulator.
    config = BenchmarkConfig(
        num_assets=num_assets,
        target_assets=target_assets,
        repeats=repeats,
        seed=seed,
    )

    # --- 2. Run ------------------------------------------------------------
    records = run_quality_benchmark(config)
    print(f"{len(records)} records ({repeats} instances x 5 solvers)")

    # Every record carries `metadata["feasible"]` (exactly `target_assets`
    # selected); QAOA records also carry `p_opt` (final probability on the
    # exact optimum) and `feasible_probability` (mass on exactly-k states).
    for record in records:
        if record.solver_name == "qaoa":
            print(
                f"  instance {record.run_index}: QAOA ratio "
                f"{record.approximation_ratio:.3f}, "
                f"p_opt {record.metadata['p_opt']:.3f}, "
                f"P(feasible) {record.metadata['feasible_probability']:.3f}, "
                f"{record.elapsed_ms / 1000:.1f} s"
            )

    # --- 3. Summarize ------------------------------------------------------
    summary = summarize_quality(records)
    print()
    print(f"{'solver':<20} {'ratio':>7} {'hits':>6} {'feasible':>9} {'ms':>9}")
    for name, stats in summary.items():
        print(
            f"{name:<20} {stats['mean_approximation_ratio']:>7.3f} "
            f"{stats['optimal_hit_rate']:>6.0%} "
            f"{stats.get('feasibility_rate', float('nan')):>9.0%} "
            f"{stats['mean_elapsed_ms']:>9.1f}"
        )
    qaoa_stats = summary["qaoa"]
    print(
        f"QAOA mean p_opt {qaoa_stats['mean_p_opt']:.3f} "
        f"(uniform guess {1 / 2 ** num_assets:.3f}), mean P(feasible) "
        f"{qaoa_stats['mean_feasible_probability']:.3f}"
    )

    # Caveat that matters at this size: QAOA decodes the lowest-objective
    # bitstring among its `max_stored_solutions` (default 64) most probable
    # states. With 6 assets all 2^6 = 64 states are kept, so the decoded
    # answer is always the optimum and the QAOA ratio / hit rate are 1.0 by
    # construction. p_opt is the honest measure of circuit quality here; the
    # decoded ratio only becomes informative above 6 assets.

    # --- 4. Statistical tests ----------------------------------------------
    # Wilcoxon compares approximation ratios pair by pair; McNemar compares
    # who hit the optimum. Both are paired on identical instances.
    tests = {}
    for baseline in ("random", "markowitz", "simulated_annealing"):
        wilcoxon = significance_test(records, "qaoa", baseline)
        mcnemar = mcnemar_test(records, "qaoa", baseline)
        tests[baseline] = {"wilcoxon": wilcoxon, "mcnemar": mcnemar}
        print(
            f"QAOA vs {baseline:<20} Wilcoxon p = {wilcoxon['p_value']:.3f}   "
            f"McNemar p = {mcnemar['p_value']:.3f} "
            f"(hits {mcnemar['hits_a']}/{mcnemar['num_pairs']} vs "
            f"{mcnemar['hits_b']}/{mcnemar['num_pairs']})"
        )
    print(
        f"Only {repeats} pairs: p-values are indicative only. "
        "See docs/benchmarks.md for the curated results."
    )

    # --- 5. Figures --------------------------------------------------------
    quality = plot_solver_comparison(
        [
            {
                "solver_name": name,
                "approximation_ratio": stats["mean_approximation_ratio"],
            }
            for name, stats in summary.items()
        ],
        metric="approximation_ratio",
    )
    hit_rate = plot_solver_comparison(
        [
            {"solver_name": name, "optimal_hit_rate": stats["optimal_hit_rate"]}
            for name, stats in summary.items()
        ],
        metric="optimal_hit_rate",
    )
    paths = []
    for name, figure in (("quality", quality), ("hit_rate", hit_rate)):
        path = output_dir / f"03_{name}.png"
        figure.savefig(path, bbox_inches="tight", dpi=100)
        paths.append(path)
    print(f"Saved {len(paths)} figures to {output_dir}")

    return {"records": records, "summary": summary, "tests": tests, "figures": paths}


if __name__ == "__main__":
    main()
