"""
Example 02 - Crypto portfolio from the ``major_crypto`` preset.

Loads one year of daily prices for the five assets of the ``major_crypto``
preset (BTC, ETH, BNB, SOL, ADA) through `quick_portfolio_load`, builds a
five-variable QUBO that selects two of them, and compares QAOA with the Rust
simulated-annealing solver against the brute-force optimum. A risk-return
scatter highlights the QAOA selection.

Network: the first run of the day downloads from Yahoo Finance (free, no API
key); the loader then caches per-symbol CSVs under ``data/cache/`` relative
to the working directory. The cache key includes the requested window, which
ends today, so a new day means a new download.

For offline use, `main` also accepts an already-loaded multi-index price
frame (columns ``(symbol, "close")``, as returned by the loader); the test
suite uses this with synthetic prices.

Run it from the repository root::

    python examples/02_crypto_portfolio.py

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import numpy as np
import pandas as pd

import qaoa_portfolio_core as core
from qaoa_portfolio import (
    QAOAConfig,
    plot_portfolio_composition,
    plot_risk_return_scatter,
    quick_portfolio_load,
    solve_qubo_qaoa,
)

# Headless rendering: figures are saved to disk, never shown.
matplotlib.use("Agg")

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PRESET = "major_crypto"


def close_prices(price_data: pd.DataFrame) -> pd.DataFrame:
    """Return a (days, symbols) frame of close prices with complete rows only."""

    symbols = list(price_data.columns.get_level_values(0).unique())
    closes = pd.DataFrame({s: price_data[(s, "close")] for s in symbols})
    return closes.dropna()


def main(
    output_dir: Optional[Path] = None,
    price_data: Optional[pd.DataFrame] = None,
    days_back: int = 365,
    risk_factor: float = 0.5,
    target_assets: int = 2,
    seed: int = 11,
) -> Dict[str, Any]:
    """Run the crypto example; pass `price_data` to skip the download."""

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Data -----------------------------------------------------------
    # `quick_portfolio_load` is a coroutine: scripts drive it with
    # asyncio.run(); in a notebook, `await` it directly instead.
    if price_data is None:
        price_data, _ = asyncio.run(
            quick_portfolio_load(preset=PRESET, days_back=days_back)
        )
    closes = close_prices(price_data)
    labels = list(closes.columns)
    print(
        f"Loaded {len(labels)} assets x {len(closes)} days "
        f"({closes.index[0].date()} -> {closes.index[-1].date()})"
    )
    # A symbol can fail to download and be dropped; never ask for more
    # holdings than there are assets.
    target_assets = min(target_assets, len(labels))

    # --- 2. QUBO -----------------------------------------------------------
    prices = closes.to_numpy(dtype=np.float64)
    qubo = core.build_qubo(prices, labels, risk_factor, target_assets)

    # --- 3. Solve three ways -----------------------------------------------
    # Brute force (2^5 = 32 candidates) is the reference; simulated annealing
    # is the fast classical heuristic; QAOA is p = 1 with COBYLA.
    exact = core.solve_brute_force(qubo)
    annealed = core.solve_simulated_annealing(qubo, seed=seed)
    optimum_bitstring = "".join("1" if bit else "0" for bit in exact.solution)
    result = solve_qubo_qaoa(
        qubo,
        labels=labels,
        config=QAOAConfig(
            layers=1,
            optimizer="cobyla",
            max_iterations=60,
            num_restarts=2,
            seed=seed,
            target_assets=target_assets,
        ),
        reference_bitstrings=[optimum_bitstring],
    )

    solutions = {
        "brute_force": (list(exact.selected_assets), exact.objective_value),
        "simulated_annealing": (
            list(annealed.selected_assets),
            annealed.objective_value,
        ),
        "qaoa": (list(result.selected_assets), result.objective_value),
    }
    for name, (assets, objective) in solutions.items():
        print(f"{name:<20} {', '.join(assets):<20} objective {objective:.6f}")
    # As in example 01: with 5 assets every bitstring is kept for decoding,
    # so the decoded QAOA answer is the optimum by construction; P(optimum)
    # measures the circuit itself.
    p_opt = result.metadata["reference_probabilities"][optimum_bitstring]
    print(
        f"QAOA P(optimum) {p_opt:.3f} (uniform {1 / 2 ** len(labels):.3f}), "
        f"P(feasible) {result.metadata['feasible_probability']:.3f}"
    )

    # --- 4. Figures --------------------------------------------------------
    # Risk-return uses daily simple returns of the same close prices.
    returns = closes.pct_change().dropna()
    figures = {
        "risk_return": plot_risk_return_scatter(
            returns, highlighted_assets=result.selected_assets
        ),
        "composition": plot_portfolio_composition(result.selected_assets),
    }
    paths = []
    for name, figure in figures.items():
        path = output_dir / f"02_{name}.png"
        figure.savefig(path, bbox_inches="tight", dpi=100)
        paths.append(path)
    print(f"Saved {len(paths)} figures to {output_dir}")

    return {
        "symbols": labels,
        "rows": len(closes),
        "solutions": solutions,
        "p_opt": float(p_opt),
        "feasible_probability": float(result.metadata["feasible_probability"]),
        "figures": paths,
    }


if __name__ == "__main__":
    main()
