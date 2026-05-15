"""Public data API — thin dispatcher that routes calls to the configured provider.

All agents import from this module.  The actual implementation lives in
src/data/providers/.  Switch providers by editing src/data/config.py.
"""
import logging

import pandas as pd

from src.data import config as _cfg
from src.data.config import DataSource
from src.data.models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price
from src.data.providers.alpha_vantage import AlphaVantageProvider
from src.data.providers.financial_datasets import (
    FinancialDatasetsProvider,
    _make_api_request,  # re-exported for backward compatibility with existing tests
)
from src.data.providers.hybrid import HybridProvider
from src.data.providers.yahoo_finance import YahooFinanceProvider

logger = logging.getLogger(__name__)

# ── Provider factory ──────────────────────────────────────────────────────────

def _build_provider(source: DataSource):
    if source == DataSource.FINANCIAL_DATASETS:
        return FinancialDatasetsProvider()
    if source == DataSource.YAHOO_FINANCE:
        return YahooFinanceProvider()
    if source == DataSource.ALPHA_VANTAGE:
        return AlphaVantageProvider()
    if source == DataSource.HYBRID:
        chain = [_build_provider(s) for s in _cfg.HYBRID_CHAIN]
        return HybridProvider(chain)
    raise ValueError(f"Unknown DataSource: {source}")


# Providers are instantiated once at import time.
# They are re-built if the config module is patched at runtime (e.g. tests).
def _get_provider(source: DataSource):
    return _build_provider(source)


# ── Public API (same signatures as before — agents are unchanged) ─────────────

def get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None) -> list[Price]:
    return _get_provider(_cfg.PRICES_SOURCE).get_prices(ticker, start_date, end_date, api_key)


def get_financial_metrics(ticker: str, end_date: str, period: str = "ttm", limit: int = 10, api_key: str | None = None) -> list[FinancialMetrics]:
    return _get_provider(_cfg.FINANCIAL_METRICS_SOURCE).get_financial_metrics(ticker, end_date, period, limit, api_key)


def search_line_items(ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10, api_key: str | None = None) -> list[LineItem]:
    return _get_provider(_cfg.LINE_ITEMS_SOURCE).search_line_items(ticker, line_items, end_date, period, limit, api_key)


def get_insider_trades(ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000, api_key: str | None = None) -> list[InsiderTrade]:
    return _get_provider(_cfg.INSIDER_TRADES_SOURCE).get_insider_trades(ticker, end_date, start_date, limit, api_key)


def get_company_news(ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000, api_key: str | None = None) -> list[CompanyNews]:
    return _get_provider(_cfg.NEWS_SOURCE).get_company_news(ticker, end_date, start_date, limit, api_key)


def get_market_cap(ticker: str, end_date: str, api_key: str | None = None) -> float | None:
    return _get_provider(_cfg.MARKET_CAP_SOURCE).get_market_cap(ticker, end_date, api_key)


# ── Utility functions (provider-independent) ──────────────────────────────────

def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert a list of Price objects to a DataFrame indexed by Date."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str | None = None) -> pd.DataFrame:
    """Convenience wrapper: fetch prices and return as DataFrame."""
    return prices_to_df(get_prices(ticker, start_date, end_date, api_key))
