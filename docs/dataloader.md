# Market Data Loader API Documentation

## Overview

The QAOA Portfolio Optimizer Market Data Loader provides a comprehensive API for loading financial market data using Yahoo Finance. The implementation is optimized for free-tier usage with configuration-driven behavior, async loading, and comprehensive data validation.

## Core Classes

### MarketDataLoader

Main class for loading and processing market data with configuration integration.

#### Initialization

```python
from qaoa_portfolio import MarketDataLoader

# Initialize with default configuration
loader = MarketDataLoader()
```

The loader automatically reads configuration from the config system and sets up caching, rate limiting, and data source settings.

#### Key Methods

##### `load_portfolio_data()`

Load historical price data for multiple assets asynchronously.

```python
async def load_portfolio_data(
    symbols: List[str],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    include_volume: bool = True,
    validate_data: bool = True
) -> pd.DataFrame
```

**Parameters:**

- `symbols`: List of ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
- `start_date`: Start date ('YYYY-MM-DD' or datetime object)
- `end_date`: End date ('YYYY-MM-DD' or datetime object)
- `include_volume`: Include trading volume data
- `validate_data`: Perform data validation and cleaning

**Returns:**

- Multi-level DataFrame with columns (symbol, price_type)

The index is normalized to timezone-naive calendar dates before symbols are
combined, so mixed asset classes align correctly (crypto daily bars carry
UTC-midnight timestamps while US equities carry New York timestamps; joining
raw timestamps would never overlap). Weekend crypto rows appear as NaN for
equity columns — drop or forward-fill according to your use case.

**Example:**

```python
data = await loader.load_portfolio_data(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date='2023-01-01',
    end_date='2024-01-01'
)
```

##### `calculate_returns()`

Calculate returns from price data.

```python
def calculate_returns(
    price_data: pd.DataFrame,
    return_type: str = 'simple',
    price_column: str = 'close'
) -> pd.DataFrame
```

**Parameters:**

- `price_data`: Multi-level price DataFrame
- `return_type`: 'simple' or 'log' returns
- `price_column`: Price column to use for calculations

**Returns:**

- DataFrame with calculated returns for each symbol

##### `get_market_data_summary()`

Generate summary statistics for loaded data.

```python
def get_market_data_summary(data: pd.DataFrame) -> Dict
```

**Returns:**

- Dictionary with symbols, date range, data quality metrics

## Portfolio Utilities

### Stock Portfolios

#### `load_sp500_symbols()`

Load the S&P 500 symbol list from Wikipedia (dots replaced by dashes for Yahoo Finance; falls back to a short built-in large-cap list when the page cannot be read).

```python
from qaoa_portfolio.portfolios import load_sp500_symbols

symbols = load_sp500_symbols()
# Returns: ['AAPL', 'MSFT', 'GOOGL', ...]
```

#### `create_sample_portfolio()`

Create sample stock portfolio of specified size.

```python
from qaoa_portfolio.portfolios import create_sample_portfolio

# Default size from config (typically 5)
portfolio = create_sample_portfolio()

# Custom size
portfolio = create_sample_portfolio(size=10)
```

### Cryptocurrency Portfolios

#### `load_crypto_symbols()`

Load top cryptocurrency symbols in Yahoo Finance format.

```python
from qaoa_portfolio.portfolios import load_crypto_symbols

crypto_symbols = load_crypto_symbols()
# Returns: ['BTC-USD', 'ETH-USD', 'BNB-USD', ...]
```

#### `create_sample_crypto_portfolio()`

Create sample cryptocurrency portfolio.

```python
from qaoa_portfolio.portfolios import create_sample_crypto_portfolio

crypto_portfolio = create_sample_crypto_portfolio(size=5)
# Returns: ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
```

### Mixed Portfolios

#### `create_mixed_portfolio()`

Create portfolio combining stocks and crypto assets.

```python
from qaoa_portfolio.portfolios import create_mixed_portfolio

mixed = create_mixed_portfolio(stocks=3, crypto=2)
# Returns: ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD', 'ETH-USD']
```

### Portfolio Presets

#### `get_preset_portfolio()`

Load predefined portfolio configurations.

```python
from qaoa_portfolio.portfolios import get_preset_portfolio, list_portfolio_presets

# See available presets
presets = list_portfolio_presets()
# Returns: {'conservative_stocks': 'Large-cap defensive stocks', ...}

# Load specific preset
portfolio = get_preset_portfolio('growth_stocks')
# Returns: ['GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
```

**Available Presets:**

- `conservative_stocks`: Large-cap defensive stocks
- `growth_stocks`: High-growth technology stocks  
- `major_crypto`: Top 5 cryptocurrencies by market cap
- `defi_crypto`: DeFi and smart contract platforms
- `balanced_mixed`: Balanced mix of stocks, crypto, and index

## Quick Start Functions

### `quick_portfolio_load()`

Convenient wrapper for rapid data loading and return calculation.

```python
from qaoa_portfolio.portfolios import quick_portfolio_load

# Load default stock portfolio
price_data, returns_data = await quick_portfolio_load()

# Load crypto portfolio
price_data, returns_data = await quick_portfolio_load(
    portfolio_type='crypto',
    days_back=180
)

# Load custom symbols
price_data, returns_data = await quick_portfolio_load(
    symbols=['AAPL', 'BTC-USD', 'ETH-USD'],
    days_back=365
)

# Load a preset portfolio directly
price_data, returns_data = await quick_portfolio_load(
    preset='growth_stocks',
    days_back=180
)
```

### CLI

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run qaoa-portfolio --help
UV_PROJECT_ENVIRONMENT=qaoa-env uv run qaoa-portfolio --preset conservative_stocks --days-back 120
```

## Configuration

### Free Tier Settings

#### `get_free_tier_recommendations()`

Get current configuration and recommendations for optimal free-tier usage.

```python
from qaoa_portfolio.data_loader import get_free_tier_recommendations

recommendations = get_free_tier_recommendations()
```

**Returns configuration info:**

- Data source settings
- Cache configuration  
- Rate limiting status
- Best practices

#### `setup_free_tier_environment()`

Print a free-tier configuration summary (nothing when `logging.show_free_tier_tips` is false). It is not called automatically; `MarketDataLoader()` logs a similar summary when `logging.show_free_tier_tips` is enabled.

```python
from qaoa_portfolio.data_loader import setup_free_tier_environment

setup_free_tier_environment()
# Displays configuration summary
```

## Data Validation

The system includes dual-level validation:

### Per-Asset Validation (Fast)

- Data existence and completeness
- Required columns (OHLC)
- Non-positive price detection
- Data type validation

### Portfolio-Level Validation (Comprehensive)

- Cross-asset consistency
- Missing data analysis
- Data cleaning and forward-filling
- Portfolio quality metrics

## Error Handling

### Custom Exceptions

- `MarketDataError`: Data loading failures
- `DataValidationError`: Data quality issues
- `RateLimitError`: API rate limit exceeded
- `ConfigurationError`: Configuration problems

### Example Error Handling

```python
from qaoa_portfolio import MarketDataLoader, MarketDataError

try:
    loader = MarketDataLoader()
    data = await loader.load_portfolio_data(
        symbols=['INVALID_SYMBOL'],
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
except MarketDataError as e:
    print(f"Data loading failed: {e}")
```

## Performance Features

### Async Loading

- Concurrent data fetching for multiple symbols
- Thread pool execution for I/O operations
- Performance timing and monitoring

### Caching System

- One CSV file per symbol and window: `data/cache/<SYMBOL>_yfinance_<start>_<end>.csv`, relative to the working directory
- Configurable cache duration (default: 7 days); expired files are deleted and re-downloaded
- Unreadable cache files are deleted and re-downloaded
- `quick_portfolio_load` and the CLI derive the window from the current time, so they miss the cache on a new day; fixed `start_date`/`end_date` windows reuse it

### Rate Limiting

- Conservative rate limiting for API stability
- Configurable limits per data source
- Automatic delay management

## Usage Examples

### Basic Portfolio Analysis

```python
import asyncio
from qaoa_portfolio import MarketDataLoader
from qaoa_portfolio.portfolios import create_sample_portfolio

async def analyze_portfolio():
    # Create portfolio
    symbols = create_sample_portfolio(size=5)
    
    # Load data
    loader = MarketDataLoader()
    price_data = await loader.load_portfolio_data(
        symbols=symbols,
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    # Calculate returns
    returns = loader.calculate_returns(price_data)
    
    # Generate summary
    summary = loader.get_market_data_summary(price_data)
    
    return price_data, returns, summary

# Run analysis
price_data, returns, summary = asyncio.run(analyze_portfolio())
```

### Crypto Portfolio Analysis

```python
import asyncio

from qaoa_portfolio.portfolios import quick_portfolio_load

async def crypto_analysis():
    # Quick crypto portfolio load
    price_data, returns = await quick_portfolio_load(
        portfolio_type='crypto',
        days_back=365
    )
    
    return price_data, returns

price_data, returns = asyncio.run(crypto_analysis())
```

### Mixed Asset Portfolio

```python
import asyncio

from qaoa_portfolio.portfolios import create_mixed_portfolio, quick_portfolio_load

async def mixed_analysis():
    # Create mixed portfolio
    symbols = create_mixed_portfolio(stocks=3, crypto=2)
    
    # Load and analyze
    price_data, returns = await quick_portfolio_load(
        symbols=symbols,
        days_back=252  # ~1 trading year
    )
    
    return price_data, returns

price_data, returns = asyncio.run(mixed_analysis())
```

## Configuration Settings

Key configuration parameters (automatically loaded):

- `data_sources.cache_enabled`: Enable/disable caching
- `data_sources.cache_duration_days`: Cache retention period
- `portfolio.default_size`: Default portfolio size
- `performance.conservative_rate_limiting`: Rate limiting behavior
- `logging.show_free_tier_tips`: Display setup information

Configuration is managed through the `config` system and can be modified as needed for different use cases.

## See Also

- [Usage guide §2](usage_guide.md#2-load-market-data) — runnable loading walkthrough and troubleshooting (rate limits, cache).
- [API reference](api_reference.md#2-data-loading-qaoa_portfoliodata_loader) — signatures of the loader, presets, and validators.
- [Algorithm §1](algorithm.md#1-portfolio-selection-as-optimization) — how loaded prices become returns and covariance for the QUBO.
