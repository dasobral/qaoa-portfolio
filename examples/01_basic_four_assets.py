"""
Example 01 - Basic four-asset portfolio, end to end.

The smallest complete tour of the pipeline, fully offline and deterministic:

1. generate a seeded synthetic price history for four assets;
2. turn it into a QUBO with the Rust core (`qaoa_portfolio_core.build_qubo`);
3. find the exact optimum by brute force (the ground truth);
4. solve the same QUBO with QAOA (PennyLane statevector simulator);
5. compare the two and save composition, convergence, and probability charts.

Run it from the repository root::

    python examples/01_basic_four_assets.py

Figures are written to ``examples/output/`` (never shown interactively).

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

import qaoa_portfolio_core as core
from qaoa_portfolio import (
    QAOAConfig,
    generate_synthetic_prices,
    plot_portfolio_composition,
    plot_qaoa_convergence,
    plot_solution_probabilities,
    render_qaoa_circuit_summary,
    solve_qubo_qaoa,
)

# Headless rendering: figures are saved to disk, never shown, so the script
# behaves the same in a terminal, over SSH, in CI, and under pytest.
matplotlib.use("Agg")

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

#: Four fictitious assets; the synthetic generator gives each a slightly
#: higher drift and volatility than the previous one.
LABELS = ["ALPHA", "BETA", "GAMMA", "DELTA"]


def main(
    output_dir: Optional[Path] = None,
    seed: int = 7,
    risk_factor: float = 0.5,
    target_assets: int = 2,
) -> Dict[str, Any]:
    """Run the four-asset walkthrough and return results plus figure paths."""

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Data -----------------------------------------------------------
    # One year (252 trading days) of seeded synthetic prices, shape
    # (periods, assets). The same seed always yields the same matrix.
    prices = generate_synthetic_prices(len(LABELS), periods=252, seed=seed)
    print(f"Price history: {prices.shape[0]} days x {prices.shape[1]} assets")

    # --- 2. QUBO -----------------------------------------------------------
    # The Rust core estimates mean returns and covariance from the prices and
    # encodes "minimize risk - return, select exactly `target_assets`" as a
    # QUBO: binary x_i = 1 means asset i is held (equal-weighted).
    qubo = core.build_qubo(prices, LABELS, risk_factor, target_assets)
    print(f"QUBO: {qubo.num_variables} binary variables, offset {qubo.offset:.4f}")

    # --- 3. Ground truth by brute force ------------------------------------
    # 2^4 = 16 candidate portfolios: trivial to enumerate exactly.
    exact = core.solve_brute_force(qubo)
    optimum_bitstring = "".join("1" if bit else "0" for bit in exact.solution)
    print(
        f"Brute force optimum: {list(exact.selected_assets)} "
        f"(bitstring {optimum_bitstring}, objective {exact.objective_value:.6f})"
    )

    # --- 4. QAOA -----------------------------------------------------------
    # One QAOA layer (p = 1) optimized with COBYLA, two restarts, on the
    # exact statevector simulator. `target_assets` lets the result report how
    # much probability lands on feasible (exactly-k) portfolios; passing the
    # known optimum as a reference bitstring reports its final probability.
    config = QAOAConfig(
        layers=1,
        optimizer="cobyla",
        max_iterations=60,
        num_restarts=2,
        seed=seed,
        backend="default.qubit",
        target_assets=target_assets,
    )
    result = solve_qubo_qaoa(
        qubo,
        labels=LABELS,
        config=config,
        reference_bitstrings=[optimum_bitstring],
    )
    p_opt = result.metadata["reference_probabilities"][optimum_bitstring]
    print()
    print(render_qaoa_circuit_summary(result))
    print()

    # --- 5. Compare --------------------------------------------------------
    # How QAOA decodes: it keeps the `max_stored_solutions` (default 64) most
    # probable bitstrings and returns the one with the lowest QUBO objective.
    # With 4 assets all 16 bitstrings fit, so the decoded answer always equals
    # the brute-force optimum here - by construction, not by merit. The
    # circuit's real quality is how much probability it puts on the optimum
    # (P(optimum), and its rank in the distribution) and on feasible
    # portfolios with exactly `target_assets` holdings.
    matches = result.best_bitstring == optimum_bitstring
    ranked = sorted(result.probabilities, key=result.probabilities.get, reverse=True)
    optimum_rank = ranked.index(optimum_bitstring) + 1
    print(f"QAOA selection:    {result.selected_assets}")
    print(f"Matches optimum:   {matches}")
    uniform = 1 / 2 ** len(LABELS)
    print(f"P(optimum):        {p_opt:.3f}  (uniform guess: {uniform:.3f})")
    print(f"Optimum rank:      {optimum_rank} of {len(ranked)} by probability")
    print(
        f"P(feasible, k={target_assets}): "
        f"{result.metadata['feasible_probability']:.3f}"
    )
    print(f"Optimizer steps:   {result.iterations}, wall time {result.elapsed_ms} ms")

    # --- 6. Figures --------------------------------------------------------
    figures = {
        "composition": plot_portfolio_composition(result.selected_assets),
        "convergence": plot_qaoa_convergence(result),
        "probabilities": plot_solution_probabilities(result),
    }
    paths = []
    for name, figure in figures.items():
        path = output_dir / f"01_{name}.png"
        figure.savefig(path, bbox_inches="tight", dpi=100)
        paths.append(path)
    print(f"Saved {len(paths)} figures to {output_dir}")

    return {
        "optimum_bitstring": optimum_bitstring,
        "optimum_assets": list(exact.selected_assets),
        "optimum_objective": float(exact.objective_value),
        "qaoa_bitstring": result.best_bitstring,
        "qaoa_assets": list(result.selected_assets),
        "qaoa_objective": float(result.objective_value),
        "matches_optimum": matches,
        "p_opt": float(p_opt),
        "optimum_probability_rank": optimum_rank,
        "feasible_probability": float(result.metadata["feasible_probability"]),
        "figures": paths,
    }


if __name__ == "__main__":
    main()
