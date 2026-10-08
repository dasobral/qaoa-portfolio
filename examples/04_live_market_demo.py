"""
Example 04 - Live market demo on the latest ~180 days.

End to end on today's data: download roughly the last six months of daily
bars for six large US stocks from Yahoo Finance, build a QUBO that picks
three of them, solve it with QAOA, cross-check against brute force, and
report the selected portfolio's metrics over the same window.

What "live" means here: the latest *daily* bars Yahoo Finance has published
(today's bar may be partial while markets are open). There is no intraday
streaming - the pipeline is a daily-resolution batch optimization.

Caveat on the metrics: they are computed on the same window the optimizer
saw (in-sample). They describe the past six months; they are not a forecast
and not out-of-sample performance. The benchmark harness
(`run_market_study`) does a proper train/test split - see
``docs/benchmarks.md``.

Network: required (``MarketDataLoader`` + Yahoo Finance, free, no API key).

Run it from the repository root::

    python examples/04_live_market_demo.py

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd

import qaoa_portfolio_core as core
from qaoa_portfolio import (
    FinancialMetrics,
    MarketDataLoader,
    QAOAConfig,
    plot_correlation_heatmap,
    plot_risk_return_scatter,
    plot_top_solutions,
    solve_qubo_qaoa,
)

# Headless rendering: figures are saved to disk, never shown.
matplotlib.use("Agg")

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

#: Six large caps from different sectors (tech, semis, banks, staples,
#: energy): enough spread for the covariance term to matter.
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "JPM", "KO", "XOM"]


async def load_window(symbols: Sequence[str], days_back: int) -> pd.DataFrame:
    """Download the trailing `days_back` calendar days of daily bars."""

    end = datetime.now()
    start = end - timedelta(days=days_back)
    loader = MarketDataLoader()
    return await loader.load_portfolio_data(list(symbols), start, end)


def portfolio_metrics(returns: pd.DataFrame, selected: List[str]) -> Dict[str, float]:
    """In-sample metrics of the equal-weighted selection over `returns`."""

    daily = returns[selected].mean(axis=1)
    return {
        "annualized_return": float(FinancialMetrics.annualized_return(daily)),
        "annualized_volatility": float(FinancialMetrics.annualized_volatility(daily)),
        "sharpe_ratio": float(FinancialMetrics.sharpe_ratio(daily)),
        "max_drawdown": float(FinancialMetrics.max_drawdown(daily)),
    }


def main(
    output_dir: Optional[Path] = None,
    price_data: Optional[pd.DataFrame] = None,
    symbols: Sequence[str] = tuple(DEFAULT_SYMBOLS),
    days_back: int = 180,
    risk_factor: float = 0.5,
    target_assets: int = 3,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Run the live demo; pass `price_data` to reuse an already-loaded frame."""

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Today's data ---------------------------------------------------
    # 180 calendar days is about 125 trading days, comfortably above the 60
    # the loader's validation requires. In a notebook, `await load_window(...)`
    # instead of asyncio.run().
    if price_data is None:
        price_data = asyncio.run(load_window(symbols, days_back))
    loaded = list(price_data.columns.get_level_values(0).unique())
    closes = pd.DataFrame({s: price_data[(s, "close")] for s in loaded}).dropna()
    labels = list(closes.columns)
    target_assets = min(target_assets, len(labels))
    print(
        f"{len(labels)} assets, {len(closes)} daily bars, "
        f"{closes.index[0].date()} -> {closes.index[-1].date()} (latest bar)"
    )

    # --- 2. QUBO + solvers -------------------------------------------------
    # Brute force (2^6 = 64 candidates) cross-checks QAOA. At 6 assets QAOA
    # keeps every bitstring for decoding, so its answer equals the optimum by
    # construction; P(optimum) is what measures the circuit (see example 01).
    prices = closes.to_numpy(dtype=np.float64)
    qubo = core.build_qubo(prices, labels, risk_factor, target_assets)
    exact = core.solve_brute_force(qubo)
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
    p_opt = result.metadata["reference_probabilities"][optimum_bitstring]
    print(f"QAOA selection:        {', '.join(result.selected_assets)}")
    print(f"Brute-force optimum:   {', '.join(exact.selected_assets)}")
    print(
        f"QAOA P(optimum) {p_opt:.3f} (uniform {1 / 2 ** len(labels):.3f}), "
        f"P(feasible) {result.metadata['feasible_probability']:.3f}"
    )

    # --- 3. In-sample metrics ----------------------------------------------
    returns = closes.pct_change().dropna()
    selected = list(result.selected_assets)
    metrics = portfolio_metrics(returns, selected)
    print("Equal-weighted selection over the same window (IN-SAMPLE):")
    print(f"  annualized return     {metrics['annualized_return']:+.1%}")
    print(f"  annualized volatility {metrics['annualized_volatility']:.1%}")
    print(f"  Sharpe ratio (rf 2%)  {metrics['sharpe_ratio']:.2f}")
    print(f"  max drawdown          {metrics['max_drawdown']:.1%}")
    print(
        "Caveat: the optimizer saw this window. These numbers describe the "
        "past, not future or out-of-sample performance."
    )

    # --- 4. Figures --------------------------------------------------------
    figures = {
        "correlation": plot_correlation_heatmap(returns),
        "risk_return": plot_risk_return_scatter(returns, highlighted_assets=selected),
        "top_solutions": plot_top_solutions(result),
    }
    paths = []
    for name, figure in figures.items():
        path = output_dir / f"04_{name}.png"
        figure.savefig(path, bbox_inches="tight", dpi=100)
        paths.append(path)
    print(f"Saved {len(paths)} figures to {output_dir}")

    return {
        "symbols": labels,
        "first_date": str(closes.index[0].date()),
        "last_date": str(closes.index[-1].date()),
        "selected_assets": selected,
        "optimum_assets": list(exact.selected_assets),
        "p_opt": float(p_opt),
        "in_sample_metrics": metrics,
        "figures": paths,
    }


if __name__ == "__main__":
    main()
