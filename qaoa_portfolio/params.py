"""
Parameters and constants for QAOA Portfolio Optimizer (QOPO)

This module contains parameters, constants, and default values
for portfolio optimization and market data processing.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

from typing import Dict

# ============================================================================
# Portfolio Parameters
# ============================================================================


class PortfolioParams:
    """Portfolio optimization parameters and constraints."""

    RISK_LEVELS = {
        "conservative": {"max_volatility": 0.08, "max_single_weight": 0.15},
        "moderate": {"max_volatility": 0.15, "max_single_weight": 0.25},
        "aggressive": {"max_volatility": 0.25, "max_single_weight": 0.40},
        "very_aggressive": {"max_volatility": 0.40, "max_single_weight": 0.60},
    }

    OBJECTIVES = {
        "max_sharpe": "Maximize Sharpe ratio",
        "min_volatility": "Minimize portfolio volatility",
        "max_return": "Maximize expected return",
        "efficient_frontier": "Generate efficient frontier points",
        "risk_parity": "Equal risk contribution from all assets",
    }

    REBALANCING = {
        "daily": 1,
        "weekly": 5,
        "monthly": 21,
        "quarterly": 63,
        "semi_annual": 126,
        "annual": 252,
    }

    @staticmethod
    def get_risk_params(risk_level: str) -> Dict[str, float]:
        """Get risk parameters for a given risk level."""
        return PortfolioParams.RISK_LEVELS.get(
            risk_level.lower(), PortfolioParams.RISK_LEVELS["moderate"]
        )


# ============================================================================
# Market Data Parameters
# ============================================================================


class MarketDataParams:
    """Parameters and constants for market data processing."""

    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_MONTH = 21
    TRADING_DAYS_PER_WEEK = 5

    MAX_MISSING_DATA_PCT = 20.0
    MAX_DAILY_RETURN_THRESHOLD = 0.50
    MIN_TRADING_DAYS_REQUIRED = 60


# ============================================================================
# QAOA Parameters
# ============================================================================


class QAOAParams:
    """QAOA algorithm parameters.

    Single source of truth for the optimizer/backend names accepted by
    `qaoa_portfolio.quantum_backend.QAOAConfig`.
    """

    DEFAULT_LAYERS = 3
    DEFAULT_MAX_ITERATIONS = 100
    DEFAULT_CONVERGENCE_THRESHOLD = 1e-6

    SUPPORTED_OPTIMIZERS = ("adam", "gradient_descent", "cobyla", "nelder_mead")
    SUPPORTED_BACKENDS = ("default.qubit", "lightning.qubit", "lightning.gpu")
