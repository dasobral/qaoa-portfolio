"""
Smoke tests for the scripts under ``examples/``.

Each test imports an example script by path and runs its ``main()`` with a
temporary output directory, asserting that it completes and writes its
figures. Figure contents are not checked. The notebooks mirror these scripts
and are not executed here (they are re-run manually when a script changes).
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(filename: str) -> ModuleType:
    path = EXAMPLES_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_figures(result: dict, output_dir: Path, expected: int) -> None:
    figures = result["figures"]
    assert len(figures) == expected
    for path in figures:
        path = Path(path)
        assert path.parent == output_dir
        assert path.is_file() and path.stat().st_size > 0


def _synthetic_price_frame(symbols, periods: int = 252, seed: int = 3):
    """Loader-shaped frame: MultiIndex columns (symbol, price_type)."""

    from qaoa_portfolio import generate_synthetic_prices

    prices = generate_synthetic_prices(len(symbols), periods, seed)
    index = pd.date_range("2025-01-01", periods=periods, freq="D")
    columns = pd.MultiIndex.from_tuples(
        [(symbol, "close") for symbol in symbols], names=["symbol", "price_type"]
    )
    return pd.DataFrame(prices, index=index, columns=columns)


@pytest.mark.integration
def test_example_01_basic_four_assets(tmp_path):
    module = _load_example("01_basic_four_assets.py")
    result = module.main(output_dir=tmp_path)

    _assert_figures(result, tmp_path, expected=3)
    # At four assets QAOA decodes among all 16 states, so it must agree
    # with the brute-force optimum.
    assert result["matches_optimum"]
    assert result["qaoa_assets"] == result["optimum_assets"]
    assert 0.0 < result["p_opt"] <= 1.0
    assert 0.0 < result["feasible_probability"] <= 1.0


@pytest.mark.integration
def test_example_02_crypto_portfolio_offline(tmp_path):
    module = _load_example("02_crypto_portfolio.py")
    symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]
    result = module.main(
        output_dir=tmp_path, price_data=_synthetic_price_frame(symbols)
    )

    _assert_figures(result, tmp_path, expected=2)
    assert result["symbols"] == symbols
    optimum = result["solutions"]["brute_force"][1]
    for _assets, objective in result["solutions"].values():
        assert objective >= optimum - 1e-9


@pytest.mark.network
def test_example_02_crypto_portfolio_live(tmp_path):
    module = _load_example("02_crypto_portfolio.py")
    result = module.main(output_dir=tmp_path)

    _assert_figures(result, tmp_path, expected=2)
    assert len(result["symbols"]) >= 2


@pytest.mark.integration
def test_example_03_qaoa_vs_classical(tmp_path):
    module = _load_example("03_qaoa_vs_classical.py")
    # Two instances keep the default gate fast; the script itself uses four.
    result = module.main(output_dir=tmp_path, repeats=2)

    _assert_figures(result, tmp_path, expected=2)
    summary = result["summary"]
    assert set(summary) == {
        "brute_force",
        "markowitz",
        "qaoa",
        "random",
        "simulated_annealing",
    }
    assert summary["brute_force"]["optimal_hit_rate"] == 1.0
    assert "mean_p_opt" in summary["qaoa"]
    assert all("feasibility_rate" in stats for stats in summary.values())
    for tests in result["tests"].values():
        assert 0.0 <= tests["wilcoxon"]["p_value"] <= 1.0
        assert 0.0 <= tests["mcnemar"]["p_value"] <= 1.0


@pytest.mark.network
def test_example_04_live_market_demo(tmp_path):
    module = _load_example("04_live_market_demo.py")
    result = module.main(output_dir=tmp_path)

    _assert_figures(result, tmp_path, expected=3)
    assert set(result["selected_assets"]) <= set(result["symbols"])
    assert np.isfinite(result["in_sample_metrics"]["annualized_volatility"])


@pytest.mark.integration
def test_example_04_live_market_demo_offline(tmp_path):
    module = _load_example("04_live_market_demo.py")
    symbols = module.DEFAULT_SYMBOLS
    frame = _synthetic_price_frame(symbols, periods=125)
    result = module.main(output_dir=tmp_path, price_data=frame)

    _assert_figures(result, tmp_path, expected=3)
    assert result["symbols"] == symbols
    assert result["selected_assets"] == result["optimum_assets"]
