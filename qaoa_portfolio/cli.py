"""
Command-line interface for QAOA Portfolio Optimizer.
"""

import argparse
import asyncio
import sys
from typing import List, Optional

from .config import config
from .exceptions import MarketDataError
from .portfolios import quick_portfolio_load, list_portfolio_presets


def _parse_symbols(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    symbols = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return symbols or None


async def _run(args: argparse.Namespace) -> int:
    symbols = _parse_symbols(args.symbols)

    try:
        price_data, returns_data = await quick_portfolio_load(
            symbols=symbols,
            portfolio_type=args.portfolio_type,
            days_back=args.days_back,
            preset=args.preset,
        )
    except (ValueError, MarketDataError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    summary = {
        "rows": len(price_data),
        "price_columns": price_data.shape[1] if hasattr(price_data, "shape") else 0,
        "return_rows": len(returns_data),
        "return_columns": (
            returns_data.shape[1] if hasattr(returns_data, "shape") else 0
        ),
    }

    print("QAOA Portfolio load summary")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    return 0


def _parse_int_list(raw: str) -> List[int]:
    try:
        values = [int(item) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers: {exc}")
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _run_benchmark(args: argparse.Namespace) -> int:
    # Imported lazily: the benchmark harness pulls in PennyLane and the Rust
    # core, which plain data-loading CLI invocations never need.
    from . import benchmarks
    from .exceptions import BenchmarkError, QuantumBackendError

    solvers = tuple(item.strip() for item in args.solvers.split(",") if item.strip())
    target = args.target or max(1, args.assets // 2)

    try:
        qaoa_config = benchmarks.with_qaoa_overrides(
            {
                "layers": args.qaoa_layers,
                "max_iterations": args.qaoa_iterations,
                "num_restarts": args.qaoa_restarts,
                "optimizer": args.qaoa_optimizer,
                "backend": args.qaoa_backend,
                # feasible decoding needs k inside the QAOA config itself
                "feasible_decoding": True if args.qaoa_feasible_decoding else None,
                "target_assets": target if args.qaoa_feasible_decoding else None,
            }
        )
        sa_config = benchmarks.SAConfig(
            schedule=args.sa_schedule,
            sweeps=args.sa_sweeps,
            restarts=args.sa_restarts,
            max_iterations=args.sa_iterations,
        )
        config = benchmarks.BenchmarkConfig(
            num_assets=args.assets,
            target_assets=target,
            repeats=args.repeats,
            seed=args.seed,
            periods=args.periods,
            risk_factor=args.risk_factor,
            qaoa=qaoa_config,
            sa=sa_config,
        )

        extra = {}
        if args.suite == "quality":
            records = benchmarks.run_quality_benchmark(config, solvers=solvers)
            extra["summary"] = benchmarks.summarize_quality(records)
        elif args.suite == "scaling":
            records = benchmarks.run_scaling_benchmark(
                tuple(args.asset_counts), config=config, solvers=solvers
            )
            extra["summary"] = benchmarks.summarize_quality(records)
        elif args.suite == "layers":
            records = benchmarks.run_layer_benchmark(tuple(args.layers), config=config)
            extra["summary"] = benchmarks.summarize_quality(records)
        else:  # market
            if not args.symbols or not args.start_date or not args.end_date:
                print(
                    "Error: the market suite requires --symbols, --start-date, "
                    "and --end-date",
                    file=sys.stderr,
                )
                return 1
            records = []
            extra["study"] = benchmarks.run_market_study(
                _parse_symbols(args.symbols) or [],
                args.start_date,
                args.end_date,
                split=args.split,
                config=config,
                solvers=solvers,
            )

        path = benchmarks.save_benchmark_results(
            records,
            suite=args.suite,
            output_dir=args.output,
            config=config,
            extra=extra,
        )
    except (BenchmarkError, QuantumBackendError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.plot and records:
        figure_path = path.with_suffix(".png")
        _save_comparison_plot(extra["summary"], figure_path)
        print(f"Comparison figure: {figure_path}")

    print(f"Benchmark suite '{args.suite}' complete")
    print(f"Artifact: {path}")
    if "summary" in extra:
        for solver, stats in extra["summary"].items():
            print(
                f"- {solver}: ratio {stats['mean_approximation_ratio']:.4f} "
                f"± {stats['std_approximation_ratio']:.4f}, "
                f"optimal {stats['optimal_hit_rate']:.0%}, "
                f"feasible {stats.get('feasibility_rate', float('nan')):.0%}, "
                f"{stats['mean_elapsed_ms']:.1f} ms"
                + (
                    f", p_opt {stats['mean_p_opt']:.3f}"
                    if "mean_p_opt" in stats
                    else ""
                )
            )
    return 0


def _save_comparison_plot(summary, figure_path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from .visualization import plot_solver_comparison

    figure = plot_solver_comparison(
        [
            {
                "solver_name": solver,
                "approximation_ratio": stats["mean_approximation_ratio"],
            }
            for solver, stats in summary.items()
        ],
        metric="approximation_ratio",
    )
    figure.savefig(figure_path, bbox_inches="tight")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QAOA Portfolio Optimizer CLI")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols, e.g. AAPL,MSFT,BTC-USD",
    )
    parser.add_argument(
        "--portfolio-type",
        choices=["stock", "crypto", "mixed"],
        default="stock",
        help="Sample portfolio type when symbols/preset are not provided.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=252,
        help="Number of calendar days to load.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help=(
            "Preset portfolio name. Available: "
            f"{', '.join(list_portfolio_presets().keys())}"
        ),
    )

    subparsers = parser.add_subparsers(dest="command")
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run a Phase 5 benchmark suite and write a JSON artifact.",
    )
    benchmark.add_argument(
        "--suite",
        choices=["quality", "scaling", "layers", "market"],
        default="quality",
        help="Benchmark suite to run.",
    )
    benchmark.add_argument("--assets", type=int, default=8)
    benchmark.add_argument(
        "--target",
        type=int,
        default=None,
        help="Assets to select (default: half of --assets).",
    )
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--periods", type=int, default=252)
    benchmark.add_argument("--risk-factor", type=float, default=0.5)
    benchmark.add_argument(
        "--solvers",
        type=str,
        default="brute_force,simulated_annealing,markowitz,random,qaoa",
        help="Comma-separated solver names.",
    )
    benchmark.add_argument(
        "--asset-counts",
        type=_parse_int_list,
        default=[4, 8, 12, 16, 20],
        help="Portfolio sizes for the scaling suite (comma-separated).",
    )
    benchmark.add_argument(
        "--layers",
        type=_parse_int_list,
        default=[1, 2, 3, 5, 10],
        help="QAOA depths for the layers suite (comma-separated).",
    )
    benchmark.add_argument(
        "--qaoa-layers",
        type=int,
        default=None,
        help="QAOA circuit depth (default: benchmark preset, 1).",
    )
    benchmark.add_argument(
        "--qaoa-iterations",
        type=int,
        default=None,
        help="Max optimizer iterations per QAOA restart (default: 60).",
    )
    benchmark.add_argument(
        "--qaoa-restarts",
        type=int,
        default=None,
        help="QAOA random restarts (default: 2).",
    )
    benchmark.add_argument(
        "--qaoa-optimizer",
        type=str,
        default=None,
        help="QAOA classical optimizer (default: benchmark preset, cobyla; "
        "use adam to reproduce pre-October-2026 artifacts).",
    )
    benchmark.add_argument(
        "--qaoa-backend",
        type=str,
        default=None,
        help="PennyLane device for QAOA (default: benchmark preset, "
        "default.qubit; e.g. lightning.qubit, lightning.gpu).",
    )
    benchmark.add_argument(
        "--qaoa-feasible-decoding",
        action="store_true",
        help="Decode QAOA answers among the most probable states that select "
        "exactly --target assets.",
    )
    benchmark.add_argument(
        "--sa-schedule",
        choices=["default", "auto"],
        default="default",
        help="Simulated-annealing schedule: the Rust default (T0 = 100, "
        "10 000 moves) or an instance-scaled 'auto' schedule.",
    )
    benchmark.add_argument(
        "--sa-sweeps",
        type=int,
        default=1000,
        help="Moves per variable for --sa-schedule auto (default: 1000).",
    )
    benchmark.add_argument(
        "--sa-restarts",
        type=int,
        default=1,
        help="Independent SA runs per instance, best kept (default: 1).",
    )
    benchmark.add_argument(
        "--sa-iterations",
        type=int,
        default=None,
        help="Override the SA move count of either schedule.",
    )
    benchmark.add_argument("--symbols", dest="symbols", type=str, default=None)
    benchmark.add_argument("--start-date", type=str, default=None)
    benchmark.add_argument("--end-date", type=str, default=None)
    benchmark.add_argument("--split", type=float, default=0.7)
    benchmark.add_argument(
        "--output",
        type=str,
        default="results/benchmarks",
        help="Directory for JSON artifacts.",
    )
    benchmark.add_argument(
        "--plot",
        action="store_true",
        help="Also save a solver-comparison figure next to the artifact.",
    )
    return parser


def main() -> None:
    # Logging is configured at the application entry point, not at package
    # import time, so importing qaoa_portfolio never hijacks a host
    # application's root logger.
    config.setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "command", None) == "benchmark":
        code = _run_benchmark(args)
    else:
        code = asyncio.run(_run(args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
