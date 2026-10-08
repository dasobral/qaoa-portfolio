"""
Integration tests for the Phase 5 benchmark harness.
"""

import json

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from qaoa_portfolio.benchmarks import (  # noqa: E402
    BenchmarkConfig,
    run_layer_benchmark,
    run_market_study,
    run_quality_benchmark,
    run_scaling_benchmark,
    summarize_quality,
)
from qaoa_portfolio.quantum_backend import QAOAConfig  # noqa: E402
from qaoa_portfolio.visualization import plot_solver_comparison  # noqa: E402

qaoa_portfolio_core = pytest.importorskip("qaoa_portfolio_core")

pytestmark = pytest.mark.integration

FAST_QAOA = QAOAConfig(
    layers=1,
    optimizer="gradient_descent",
    max_iterations=8,
    convergence_threshold=1e-12,
    num_restarts=1,
)


def fast_config(**overrides):
    defaults = dict(
        num_assets=4,
        target_assets=2,
        repeats=3,
        periods=70,
        seed=11,
        qaoa=FAST_QAOA,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


class TestQualitySuite:
    def test_qaoa_at_least_matches_random_on_paired_instances(self):
        records = run_quality_benchmark(
            fast_config(), solvers=("brute_force", "random", "qaoa")
        )
        summary = summarize_quality(records)

        assert summary["brute_force"]["mean_approximation_ratio"] == pytest.approx(1.0)
        assert (
            summary["qaoa"]["mean_approximation_ratio"]
            >= summary["random"]["mean_approximation_ratio"]
        )

    def test_records_render_through_solver_comparison_figure(self):
        records = run_quality_benchmark(
            fast_config(repeats=1), solvers=("brute_force", "simulated_annealing")
        )

        figure = plot_solver_comparison(
            [record.to_dict() for record in records],
            metric="approximation_ratio",
        )
        assert figure is not None


class TestScalingAndLayerSuites:
    def test_scaling_benchmark_covers_each_size(self):
        records = run_scaling_benchmark(
            (4, 6),
            config=fast_config(repeats=1),
            solvers=("brute_force", "simulated_annealing", "qaoa"),
        )

        sizes = {record.num_assets for record in records}
        assert sizes == {4, 6}
        assert all(record.elapsed_ms >= 0.0 for record in records)
        # Cardinality scales with the instance: half the assets each time.
        for record in records:
            if record.solver_name == "brute_force":
                assert len(record.selected_assets) == record.num_assets // 2

    def test_layer_benchmark_tags_depth_in_metadata(self):
        records = run_layer_benchmark((1, 2), config=fast_config(repeats=1))

        depths = {record.metadata["layers"] for record in records}
        assert depths == {1, 2}
        assert all(record.solver_name == "qaoa" for record in records)

    @pytest.mark.slow
    def test_mid_size_instance_runs_end_to_end(self):
        records = run_quality_benchmark(
            fast_config(num_assets=12, target_assets=6, repeats=1),
            solvers=("brute_force", "simulated_annealing", "qaoa"),
        )
        summary = summarize_quality(records)
        assert summary["brute_force"]["mean_approximation_ratio"] == pytest.approx(1.0)


class TestBenchmarkCLI:
    def test_cli_writes_parseable_artifact(self, tmp_path, monkeypatch, capsys):
        from qaoa_portfolio.cli import main

        monkeypatch.setattr(
            "sys.argv",
            [
                "qaoa-portfolio",
                "benchmark",
                "--suite",
                "quality",
                "--assets",
                "4",
                "--target",
                "2",
                "--repeats",
                "2",
                "--periods",
                "70",
                "--solvers",
                "brute_force,random",
                "--output",
                str(tmp_path),
            ],
        )

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "Benchmark suite 'quality' complete" in out

        artifacts = list(tmp_path.glob("quality-*.json"))
        assert len(artifacts) == 1
        payload = json.loads(artifacts[0].read_text())
        assert payload["config"]["num_assets"] == 4
        assert payload["summary"]["brute_force"]["mean_approximation_ratio"] == 1.0
        assert len(payload["records"]) == 4

    def test_cli_sa_and_feasible_decoding_flags(self, tmp_path, monkeypatch, capsys):
        from qaoa_portfolio.cli import main

        monkeypatch.setattr(
            "sys.argv",
            [
                "qaoa-portfolio",
                "benchmark",
                "--assets",
                "4",
                "--repeats",
                "1",
                "--periods",
                "70",
                "--solvers",
                "brute_force,simulated_annealing,qaoa",
                "--qaoa-iterations",
                "5",
                "--qaoa-feasible-decoding",
                "--sa-schedule",
                "auto",
                "--sa-sweeps",
                "20",
                "--sa-restarts",
                "2",
                "--output",
                str(tmp_path),
            ],
        )

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 0
        assert "feasible 100%" in capsys.readouterr().out
        payload = json.loads(next(tmp_path.glob("quality-*.json")).read_text())
        assert payload["config"]["sa"] == {
            "schedule": "auto",
            "sweeps": 20,
            "restarts": 2,
            "initial_temperature": None,
            "cooling_rate": None,
            "max_iterations": None,
        }
        assert payload["config"]["qaoa"]["feasible_decoding"] is True
        by_solver = {record["solver_name"]: record for record in payload["records"]}
        assert by_solver["simulated_annealing"]["metadata"]["max_iterations"] == 80
        assert by_solver["qaoa"]["metadata"]["feasible"] is True

    def test_cli_market_suite_requires_window_arguments(self, monkeypatch, capsys):
        from qaoa_portfolio.cli import main

        monkeypatch.setattr(
            "sys.argv",
            ["qaoa-portfolio", "benchmark", "--suite", "market"],
        )

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 1
        assert "--symbols" in capsys.readouterr().err


@pytest.mark.network
class TestMarketStudy:
    def test_two_symbol_historical_window_smoke(self, tmp_path):
        study = run_market_study(
            ["AAPL", "MSFT"],
            "2023-01-01",
            "2023-12-31",
            config=fast_config(num_assets=2, target_assets=1, repeats=1),
            solvers=("brute_force", "random"),
        )

        assert study["symbols"] == ["AAPL", "MSFT"]
        assert study["in_sample_rows"] >= 60
        for payload in study["solvers"].values():
            assert "annualized_return" in payload["out_of_sample"]
