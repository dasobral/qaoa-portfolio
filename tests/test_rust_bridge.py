import numpy as np
import pytest

qaoa_portfolio_core = pytest.importorskip("qaoa_portfolio_core")

pytestmark = pytest.mark.integration


def constant_prices(periods=70, assets=4):
    return np.full((periods, assets), 100.0, dtype=np.float64)


def trend_prices(periods=70):
    base = np.arange(periods, dtype=np.float64)
    return np.column_stack(
        [
            100.0 + base * 0.10,
            120.0 + base * 0.08,
            80.0 + base * 0.05,
        ]
    )


def test_build_qubo_returns_numpy_compatible_matrix():
    qubo = qaoa_portfolio_core.build_qubo(
        constant_prices(),
        ["A", "B", "C", "D"],
        0.5,
        2,
    )

    matrix = qubo.to_numpy()
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)
    assert np.isfinite(qubo.evaluate([True, True, False, False]))


def test_brute_force_bridge_returns_feasible_result_dict():
    qubo = qaoa_portfolio_core.build_qubo(
        constant_prices(),
        ["A", "B", "C", "D"],
        0.5,
        2,
    )

    result = qaoa_portfolio_core.solve_brute_force(qubo)
    payload = result.to_dict()

    assert payload["solver_name"] == "brute-force"
    assert len(payload["solution"]) == 4
    assert sum(payload["solution"]) == 2
    assert len(payload["selected_assets"]) == 2
    assert np.isfinite(payload["objective_value"])


def test_simulated_annealing_bridge_accepts_seeded_parameters():
    qubo = qaoa_portfolio_core.build_qubo(
        constant_prices(),
        ["A", "B", "C", "D"],
        0.5,
        2,
    )

    result = qaoa_portfolio_core.solve_simulated_annealing(
        qubo,
        initial_temperature=10.0,
        cooling_rate=0.98,
        max_iterations=1_000,
        seed=11,
    )

    assert len(result.solution) == 4
    assert sum(result.solution) == 2
    assert result.solver_name == "simulated-annealing"


def test_markowitz_bridge_weights_sum_to_one():
    result = qaoa_portfolio_core.solve_markowitz(
        trend_prices(),
        ["A", "B", "C"],
    )

    assert set(result) >= {
        "weights",
        "expected_return",
        "volatility",
        "sharpe_ratio",
        "symbols",
    }
    assert np.isclose(sum(result["weights"]), 1.0)
    assert result["symbols"] == ["A", "B", "C"]


def test_py_return_series_statistics():
    returns = np.full((70, 2), 0.001)
    series = qaoa_portfolio_core.PyReturnSeries(["A", "B"], returns)

    assert series.num_periods == 70
    assert series.num_assets == 2
    mean_returns = np.asarray(series.mean_returns())
    assert mean_returns == pytest.approx([0.001 * 252] * 2)
    covariance = np.asarray(series.covariance_matrix())
    assert covariance.shape == (2, 2)
    assert covariance == pytest.approx(np.zeros((2, 2)))


def test_annualization_conventions_agree_to_first_order():
    """Cross-layer consistency (docs/rust_core.md): Rust annualizes log
    returns by mean*252; Python compounds simple returns by (1+mean)^252-1.
    For a low-volatility series the two must agree to first order."""
    import pandas as pd

    from qaoa_portfolio.metrics import FinancialMetrics

    rng = np.random.default_rng(7)
    periods = 253
    daily_drift = 0.0004
    prices = 100.0 * np.exp(
        np.cumsum(rng.normal(daily_drift, 0.002, size=(periods, 2)), axis=0)
    )

    series = qaoa_portfolio_core.PyReturnSeries(
        ["A", "B"],
        np.diff(np.log(prices), axis=0),
    )
    rust_annualized = np.asarray(series.mean_returns())

    for column, rust_value in zip(range(2), rust_annualized):
        simple_returns = pd.Series(prices[:, column]).pct_change().dropna()
        python_annualized = FinancialMetrics.annualized_return(simple_returns)
        # exp(mu_log * 252) - 1 ~ (1 + mu_simple)^252 - 1 for small daily moves
        assert np.exp(rust_value) - 1.0 == pytest.approx(python_annualized, abs=5e-3)


def test_bridge_maps_invalid_inputs_to_python_exceptions():
    prices = constant_prices()
    prices[0, 0] = -1.0

    with pytest.raises(ValueError):
        qaoa_portfolio_core.build_qubo(prices, ["A", "B", "C", "D"], 0.5, 2)
