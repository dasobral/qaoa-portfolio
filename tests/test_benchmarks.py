"""
Unit tests for the Phase 5 benchmark harness.
"""

import json
import math

import numpy as np
import pytest

from qaoa_portfolio.benchmarks import (
    DEFAULT_SOLVERS,
    MAX_EXACT_ASSETS,
    MAX_RUST_BRUTE_FORCE_ASSETS,
    BenchmarkConfig,
    BenchmarkRecord,
    _DEFAULT_BENCH_QAOA,
    SAConfig,
    approximation_ratio,
    auto_sa_schedule,
    mcnemar_test,
    exact_enumeration,
    exact_optimum,
    generate_synthetic_prices,
    run_quality_benchmark,
    run_solver,
    save_benchmark_results,
    significance_test,
    summarize_quality,
)
from qaoa_portfolio.exceptions import BenchmarkError
from qaoa_portfolio.quantum_backend import QAOAConfig

qaoa_portfolio_core = pytest.importorskip("qaoa_portfolio_core")

pytestmark = pytest.mark.unit

FAST_QAOA = QAOAConfig(
    layers=1,
    optimizer="gradient_descent",
    max_iterations=5,
    convergence_threshold=1e-12,
    num_restarts=1,
)


def tiny_config(**overrides):
    defaults = dict(
        num_assets=4,
        target_assets=2,
        repeats=1,
        periods=70,
        seed=7,
        qaoa=FAST_QAOA,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def tiny_instance(config=None):
    config = config or tiny_config()
    prices = generate_synthetic_prices(config.num_assets, config.periods, config.seed)
    labels = [f"A{i}" for i in range(config.num_assets)]
    qubo = qaoa_portfolio_core.build_qubo(
        prices, labels, config.risk_factor, config.target_assets
    )
    return prices, labels, qubo


class TestBenchmarkConfig:
    def test_defaults_are_valid(self):
        config = BenchmarkConfig()
        assert config.num_assets == 8
        assert config.repeats == 10
        assert config.seed == 42

    def test_invalid_configs_raise_benchmark_error(self):
        invalid = [
            {"num_assets": 1},
            {"num_assets": MAX_EXACT_ASSETS + 1},
            {"target_assets": 0},
            {"target_assets": 9, "num_assets": 8},
            {"repeats": 0},
            {"periods": 59},
            {"risk_factor": 0.0},
            {"risk_factor": float("nan")},
        ]
        for overrides in invalid:
            with pytest.raises(BenchmarkError):
                BenchmarkConfig(**overrides)

    def test_oversized_problem_error_names_shot_based_sampling(self):
        with pytest.raises(BenchmarkError, match="shot-based"):
            BenchmarkConfig(num_assets=MAX_EXACT_ASSETS + 4)

    def test_sizes_above_twenty_are_accepted(self):
        config = BenchmarkConfig(num_assets=MAX_RUST_BRUTE_FORCE_ASSETS + 4)
        assert config.num_assets == 24

    def test_benchmark_preset_uses_cobyla(self):
        assert _DEFAULT_BENCH_QAOA.optimizer == "cobyla"
        assert _DEFAULT_BENCH_QAOA.layers == 1
        assert _DEFAULT_BENCH_QAOA.max_iterations == 60
        assert _DEFAULT_BENCH_QAOA.num_restarts == 2


class TestExactOptimum:
    @pytest.mark.parametrize("num_assets", [4, 9, 12])
    def test_enumeration_matches_rust_brute_force(self, num_assets):
        config = tiny_config(num_assets=num_assets, target_assets=num_assets // 2)
        _, labels, qubo = tiny_instance(config)

        value, solution = exact_enumeration(qubo)
        reference = qaoa_portfolio_core.solve_brute_force(qubo)

        assert value == pytest.approx(reference.objective_value, abs=1e-12)
        assert [label for label, bit in zip(labels, solution) if bit] == list(
            reference.selected_assets
        )
        assert qubo.evaluate(solution) == pytest.approx(value, abs=1e-12)

    def test_reference_switches_above_rust_limit(self):
        small = tiny_config(num_assets=6, target_assets=3)
        assert exact_optimum(tiny_instance(small)[2])[1] == "rust_brute_force"

        large = tiny_config(
            num_assets=MAX_RUST_BRUTE_FORCE_ASSETS + 1, target_assets=10
        )
        _, labels, qubo = tiny_instance(large)
        value, reference, solution = exact_optimum(qubo)
        assert qubo.evaluate(solution) == pytest.approx(value, abs=1e-12)
        assert reference == "exact_enumeration"

        record = run_solver(
            "brute_force", qubo, labels, prices=None, config=large, seed=7
        )
        assert record.objective_value == pytest.approx(value)
        assert record.metadata["method"] == "exact_enumeration"
        assert record.metadata["optimum_reference"] == "exact_enumeration"
        assert len(record.selected_assets) == 10


class TestSimulatedAnnealingConfig:
    def test_default_config_reproduces_plain_rust_call(self):
        config = tiny_config(num_assets=8, target_assets=4)
        prices, labels, qubo = tiny_instance(config)

        record = run_solver(
            "simulated_annealing", qubo, labels, prices=prices, config=config, seed=7
        )
        direct = qaoa_portfolio_core.solve_simulated_annealing(qubo, seed=7)

        assert record.objective_value == direct.objective_value
        assert record.metadata["schedule"] == "default"
        assert record.metadata["restarts"] == 1

    def test_auto_schedule_scales_with_instance(self):
        config = tiny_config(num_assets=8, target_assets=4)
        _, _, qubo = tiny_instance(config)

        schedule = auto_sa_schedule(qubo, 4, sweeps=100, seed=7)

        assert schedule["max_iterations"] == 800
        assert schedule["initial_temperature"] > 0
        assert 0 < schedule["cooling_rate"] < 1
        final = schedule["initial_temperature"] * schedule["cooling_rate"] ** 800
        # median uphill move accepted with ~1e-3 at the end of the schedule
        assert math.exp(-schedule["median_uphill_delta"] / final) == pytest.approx(
            1e-3, rel=1e-6
        )

    def test_auto_schedule_with_restarts_records_settings(self):
        config = tiny_config(
            num_assets=8,
            target_assets=4,
            sa=SAConfig(schedule="auto", sweeps=50, restarts=3),
        )
        prices, labels, qubo = tiny_instance(config)

        record = run_solver(
            "simulated_annealing", qubo, labels, prices=prices, config=config, seed=7
        )

        assert record.metadata["schedule"] == "auto"
        assert record.metadata["restarts"] == 3
        assert record.metadata["max_iterations"] == 400
        assert record.metadata["iterations"] == 1200

    @pytest.mark.parametrize(
        "overrides",
        [
            {"schedule": "fast"},
            {"sweeps": 0},
            {"restarts": 0},
            {"cooling_rate": 1.0},
            {"initial_temperature": -1.0},
            {"max_iterations": 0},
        ],
    )
    def test_invalid_sa_configs_rejected(self, overrides):
        with pytest.raises(BenchmarkError):
            SAConfig(**overrides)


class TestFeasibilityMetrics:
    def test_records_flag_feasibility_and_qaoa_reports_p_opt(self):
        config = tiny_config()
        records = run_quality_benchmark(config, solvers=("brute_force", "qaoa"))

        for record in records:
            assert record.metadata["feasible"] == (
                len(record.selected_assets) == config.target_assets
            )
        qaoa = next(record for record in records if record.solver_name == "qaoa")
        assert 0.0 <= qaoa.metadata["p_opt"] <= 1.0
        assert 0.0 <= qaoa.metadata["feasible_probability"] <= 1.0

        summary = summarize_quality(records)
        assert summary["brute_force"]["feasibility_rate"] == 1.0
        assert "mean_p_opt" in summary["qaoa"]

    def test_mcnemar_counts_discordant_pairs(self):
        def hits(name, outcomes):
            return [
                BenchmarkRecord(
                    solver_name=name,
                    num_assets=4,
                    objective_value=0.0,
                    approximation_ratio=1.0 if hit else 0.5,
                    selected_assets=[],
                    elapsed_ms=0.0,
                    peak_memory_kb=None,
                    seed=seed,
                    run_index=seed,
                )
                for seed, hit in enumerate(outcomes)
            ]

        records = hits("a", [True] * 8 + [False] * 2) + hits("b", [False] * 10)
        outcome = mcnemar_test(records, "a", "b")

        assert outcome["only_a"] == 8
        assert outcome["only_b"] == 0
        assert outcome["p_value"] == pytest.approx(2 / 2**8)
        assert mcnemar_test(records, "a", "a")["p_value"] == 1.0


class TestSyntheticPrices:
    def test_shape_positivity_and_determinism(self):
        first = generate_synthetic_prices(4, 70, seed=11)
        second = generate_synthetic_prices(4, 70, seed=11)
        other = generate_synthetic_prices(4, 70, seed=12)

        assert first.shape == (70, 4)
        assert np.all(first > 0)
        assert np.array_equal(first, second)
        assert not np.array_equal(first, other)


class TestApproximationRatio:
    def test_exact_optimum_gives_one(self):
        assert approximation_ratio(-1.5, -1.5) == pytest.approx(1.0)

    def test_worse_solutions_get_lower_ratio(self):
        near = approximation_ratio(-1.4, -1.5)
        far = approximation_ratio(-0.5, -1.5)
        assert 0.0 < far < near < 1.0

    def test_sign_handling_for_negative_objectives(self):
        assert approximation_ratio(-2.0, -2.0) == pytest.approx(1.0)
        assert approximation_ratio(-1.0, -2.0) < 1.0
        assert approximation_ratio(1.0, 0.0) < 1.0

    def test_achieved_below_optimum_rejected(self):
        with pytest.raises(BenchmarkError):
            approximation_ratio(-2.0, -1.0)


class TestBenchmarkRecord:
    def test_to_dict_is_json_safe_and_feeds_solver_comparison(self):
        from qaoa_portfolio.visualization import prepare_solver_comparison_data

        record = BenchmarkRecord(
            solver_name="brute_force",
            num_assets=4,
            objective_value=-1.0,
            approximation_ratio=1.0,
            selected_assets=["A0", "A2"],
            elapsed_ms=1.25,
            peak_memory_kb=None,
            seed=7,
            run_index=0,
            metadata={"iterations": 16},
        )

        payload = record.to_dict()
        json.dumps(payload)
        pairs = prepare_solver_comparison_data([payload], metric="objective_value")
        assert pairs == [("brute_force", -1.0)]


class TestSolverAdapters:
    @pytest.mark.parametrize("name", DEFAULT_SOLVERS)
    def test_each_adapter_returns_well_formed_record(self, name):
        config = tiny_config()
        prices, labels, qubo = tiny_instance(config)

        record = run_solver(
            name, qubo, labels, prices=prices, config=config, seed=config.seed
        )

        assert record.solver_name == name
        assert record.num_assets == 4
        assert math.isfinite(record.objective_value)
        assert 0.0 < record.approximation_ratio <= 1.0
        assert all(asset in labels for asset in record.selected_assets)
        assert record.elapsed_ms >= 0.0
        json.dumps(record.to_dict())

    def test_brute_force_record_is_optimal(self):
        config = tiny_config()
        prices, labels, qubo = tiny_instance(config)

        record = run_solver(
            "brute_force", qubo, labels, prices=prices, config=config, seed=7
        )

        assert record.approximation_ratio == pytest.approx(1.0)
        assert record.metadata["optimum_reference"] == "rust_brute_force"

    def test_random_baseline_respects_target_and_seed(self):
        config = tiny_config()
        prices, labels, qubo = tiny_instance(config)

        first = run_solver(
            "random", qubo, labels, prices=prices, config=config, seed=99
        )
        second = run_solver(
            "random", qubo, labels, prices=prices, config=config, seed=99
        )

        assert len(first.selected_assets) == config.target_assets
        assert first.selected_assets == second.selected_assets
        assert first.objective_value == pytest.approx(second.objective_value)

    def test_unknown_solver_rejected(self):
        config = tiny_config()
        prices, labels, qubo = tiny_instance(config)

        with pytest.raises(BenchmarkError, match="Unknown solver"):
            run_solver("annealer", qubo, labels, prices=prices, config=config, seed=7)


class TestQualityBenchmark:
    def test_deterministic_for_fixed_seed(self):
        config = tiny_config(repeats=2)
        solvers = ("brute_force", "simulated_annealing", "random", "qaoa")

        first = run_quality_benchmark(config, solvers=solvers)
        second = run_quality_benchmark(config, solvers=solvers)

        assert len(first) == len(solvers) * config.repeats
        for left, right in zip(first, second):
            assert left.solver_name == right.solver_name
            assert left.objective_value == pytest.approx(right.objective_value)
            assert left.approximation_ratio == pytest.approx(right.approximation_ratio)
            assert left.selected_assets == right.selected_assets

    def test_rejects_unknown_solver_names(self):
        with pytest.raises(BenchmarkError, match="Unknown solver"):
            run_quality_benchmark(tiny_config(), solvers=("brute_force", "magic"))


def make_records(solver_name, ratios, *, num_assets=4):
    return [
        BenchmarkRecord(
            solver_name=solver_name,
            num_assets=num_assets,
            objective_value=-ratio,
            approximation_ratio=ratio,
            selected_assets=["A0"],
            elapsed_ms=1.0,
            peak_memory_kb=None,
            seed=7 + index,
            run_index=index,
            metadata={},
        )
        for index, ratio in enumerate(ratios)
    ]


class TestSummaryAndSignificance:
    def test_summarize_quality_reports_expected_values(self):
        records = make_records("qaoa", [1.0, 0.5]) + make_records("random", [0.5, 0.5])

        summary = summarize_quality(records)

        assert summary["qaoa"]["mean_approximation_ratio"] == pytest.approx(0.75)
        assert summary["qaoa"]["optimal_hit_rate"] == pytest.approx(0.5)
        assert summary["qaoa"]["runs"] == 2
        assert summary["random"]["mean_approximation_ratio"] == pytest.approx(0.5)
        assert summary["random"]["optimal_hit_rate"] == pytest.approx(0.0)

    def test_significance_test_reports_wilcoxon_results(self):
        records = make_records("qaoa", [1.0, 0.9, 1.0, 0.8]) + make_records(
            "random", [0.6, 0.5, 0.7, 0.4]
        )

        outcome = significance_test(records, "qaoa", "random")

        assert outcome["num_pairs"] == 4
        assert 0.0 <= outcome["p_value"] <= 1.0
        assert not outcome["identical"]

    def test_significance_test_handles_identical_solvers(self):
        records = make_records("qaoa", [1.0, 0.9]) + make_records("random", [1.0, 0.9])

        outcome = significance_test(records, "qaoa", "random")

        assert outcome["identical"]
        assert outcome["p_value"] == pytest.approx(1.0)

    def test_significance_test_requires_pairs(self):
        records = make_records("qaoa", [1.0])
        with pytest.raises(BenchmarkError):
            significance_test(records, "qaoa", "random")


class TestArtifacts:
    def test_save_benchmark_results_writes_json(self, tmp_path):
        records = make_records("brute_force", [1.0])
        path = save_benchmark_results(
            [record.to_dict() for record in records],
            suite="quality",
            output_dir=tmp_path,
            config=tiny_config(),
        )

        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["suite"] == "quality"
        assert payload["config"]["num_assets"] == 4
        assert payload["records"][0]["solver_name"] == "brute_force"
