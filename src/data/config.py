"""Data source configuration.

Change the *_SOURCE constants below to switch data providers globally,
or per data type.

Available sources
-----------------
DataSource.FINANCIAL_DATASETS
    financialdatasets.ai — requires FINANCIAL_DATASETS_API_KEY env var.
    Supports true point-in-time historical queries; safe for backtesting.
    Free tier: AAPL, GOOGL, MSFT, NVDA, TSLA only.

DataSource.YAHOO_FINANCE
    yfinance (unofficial Yahoo Finance wrapper) — no API key required.
    WARNING: financial statements are current snapshots, NOT point-in-time.
    Using YAHOO_FINANCE for FINANCIAL_METRICS_SOURCE or LINE_ITEMS_SOURCE
    in backtests will introduce look-ahead bias.

DataSource.ALPHA_VANTAGE
    Alpha Vantage — requires ALPHA_VANTAGE_API_KEY env var.
    Free tier: 25 requests/day.
    Best source for news with pre-computed sentiment.

DataSource.HYBRID
    Yahoo Finance first → Alpha Vantage fallback.
    Merges field-by-field: if YF returns None for a field, AV fills it.
    Chain is configured in HYBRID_CHAIN below.
    Inherits the backtesting limitations of Yahoo Finance.
"""
from enum import Enum


class DataSource(str, Enum):
    FINANCIAL_DATASETS = "financial_datasets"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    HYBRID = "hybrid"


# ── Per data-type source selection ──────────────────────────────────────────
#
# Recommended defaults for live analysis (no API key required):
PRICES_SOURCE = DataSource.HYBRID               # YF prices are free and reliable
FINANCIAL_METRICS_SOURCE = DataSource.ALPHA_VANTAGE
LINE_ITEMS_SOURCE = DataSource.ALPHA_VANTAGE
INSIDER_TRADES_SOURCE = DataSource.HYBRID       # YF free, AV fallback
NEWS_SOURCE = DataSource.HYBRID                 # YF articles + AV sentiment merge
MARKET_CAP_SOURCE = DataSource.HYBRID           # YF free, AV fallback

# ── Hybrid chain ─────────────────────────────────────────────────────────────
# Ordered list of providers tried in sequence.
# First provider with a non-None result wins; its None fields are filled
# from the next provider, and so on.
HYBRID_CHAIN: list[DataSource] = [
    DataSource.YAHOO_FINANCE,
    DataSource.ALPHA_VANTAGE,
    # DataSource.FINANCIAL_DATASETS,  # uncomment to add FD as final fallback
]

# ── API keys ─────────────────────────────────────────────────────────────────
# None = read from environment variable.
FINANCIAL_DATASETS_API_KEY: str | None = None   # env: FINANCIAL_DATASETS_API_KEY
ALPHA_VANTAGE_API_KEY: str | None = None        # env: ALPHA_VANTAGE_API_KEY
