"""
QAOA Portfolio Optimizer

A high-performance implementation of the Quantum Approximate Optimization
Algorithm (QAOA) for portfolio optimization problems, demonstrating
quantum-inspired solutions for real-world financial applications.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

__version__ = "0.1.0"
__author__ = "Daniel Sobral Blanco"
__email__ = "dasobral93@gmail.com"
__license__ = "CC BY-NC-ND 4.0"

# Import exceptions
from .exceptions import (
    QAOAPortfolioError,
    MarketDataError,
    DataValidationError,
    OptimizationError,
    QuantumBackendError,
    RateLimitError,
    ConfigurationError,
    VisualizationError,
    BenchmarkError,
)

# Import configuration
from .config import ConfigManager, config

# Import utilities
from .utils import (
    PerformanceTimer,
    performance_monitor,
    DataValidator,
    ensure_directory,
    safe_divide,
    format_percentage,
    validate_weights,
    normalize_weights,
)

# Import metrics
from .metrics import FinancialMetrics

# Import parameters
from .params import PortfolioParams, MarketDataParams, QAOAParams

# Import data loader
from .data_loader import (
    MarketDataLoader,
    get_free_tier_recommendations,
    setup_free_tier_environment,
)

# Import portfolio utilities
from .portfolios import (
    load_sp500_symbols,
    create_sample_portfolio,
    quick_portfolio_load,
)

# Quantum backend and visualization exports are resolved lazily (PEP 562):
# importing them eagerly pulls in PennyLane/SciPy and matplotlib/seaborn,
# adding multi-second startup cost to consumers that never touch them
# (e.g. `qaoa-portfolio --help`).
_LAZY_EXPORTS = {
    name: ".quantum_backend"
    for name in (
        "QAOAConfig",
        "QAOAResult",
        "QAOAQuantumBackend",
        "bitstring_to_solution",
        "build_cost_hamiltonian",
        "build_mixer_hamiltonian",
        "decode_solution",
        "evaluate_qubo_bitstring",
        "solve_qubo_qaoa",
    )
}
_LAZY_EXPORTS.update(
    {
        name: ".benchmarks"
        for name in (
            "BenchmarkConfig",
            "BenchmarkRecord",
            "DEFAULT_SOLVERS",
            "MAX_EXACT_ASSETS",
            "approximation_ratio",
            "generate_synthetic_prices",
            "run_solver",
            "run_quality_benchmark",
            "run_scaling_benchmark",
            "run_layer_benchmark",
            "run_market_study",
            "summarize_quality",
            "significance_test",
            "mcnemar_test",
            "SAConfig",
            "save_benchmark_results",
        )
    }
)
_LAZY_EXPORTS.update(
    {
        name: ".visualization"
        for name in (
            "VisualizationConfig",
            "normalize_qaoa_result",
            "prepare_composition_data",
            "prepare_risk_return_data",
            "prepare_probability_data",
            "prepare_top_solutions_data",
            "prepare_solver_comparison_data",
            "plot_portfolio_composition",
            "plot_risk_return_scatter",
            "plot_correlation_heatmap",
            "plot_efficient_frontier",
            "plot_qaoa_convergence",
            "plot_solution_probabilities",
            "plot_top_solutions",
            "render_qaoa_circuit_summary",
            "plot_solver_comparison",
        )
    }
)


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


# Package-level exports
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Exceptions
    "QAOAPortfolioError",
    "MarketDataError",
    "DataValidationError",
    "OptimizationError",
    "QuantumBackendError",
    "RateLimitError",
    "ConfigurationError",
    "VisualizationError",
    "BenchmarkError",
    # Configuration
    "ConfigManager",
    "config",
    # Utilities
    "PerformanceTimer",
    "performance_monitor",
    "DataValidator",
    "ensure_directory",
    "safe_divide",
    "format_percentage",
    "validate_weights",
    "normalize_weights",
    # Metrics
    "FinancialMetrics",
    # Parameters
    "PortfolioParams",
    "MarketDataParams",
    "QAOAParams",
    # Data loading
    "MarketDataLoader",
    "load_sp500_symbols",
    "create_sample_portfolio",
    "quick_portfolio_load",
    "get_free_tier_recommendations",
    "setup_free_tier_environment",
    # Quantum backend
    "QAOAConfig",
    "QAOAResult",
    "QAOAQuantumBackend",
    "bitstring_to_solution",
    "build_cost_hamiltonian",
    "build_mixer_hamiltonian",
    "decode_solution",
    "evaluate_qubo_bitstring",
    "solve_qubo_qaoa",
    # Visualization
    "VisualizationConfig",
    "normalize_qaoa_result",
    "prepare_composition_data",
    "prepare_risk_return_data",
    "prepare_probability_data",
    "prepare_top_solutions_data",
    "prepare_solver_comparison_data",
    "plot_portfolio_composition",
    "plot_risk_return_scatter",
    "plot_correlation_heatmap",
    "plot_efficient_frontier",
    "plot_qaoa_convergence",
    "plot_solution_probabilities",
    "plot_top_solutions",
    "render_qaoa_circuit_summary",
    "plot_solver_comparison",
    # Benchmarks
    "BenchmarkConfig",
    "BenchmarkRecord",
    "DEFAULT_SOLVERS",
    "MAX_EXACT_ASSETS",
    "approximation_ratio",
    "generate_synthetic_prices",
    "run_solver",
    "run_quality_benchmark",
    "run_scaling_benchmark",
    "run_layer_benchmark",
    "run_market_study",
    "summarize_quality",
    "significance_test",
    "mcnemar_test",
    "SAConfig",
    "save_benchmark_results",
]
