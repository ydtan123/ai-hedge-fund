"""FinancialDatasetsProvider — wraps the financialdatasets.ai REST API.

This is the original data source. Supports true point-in-time historical
queries (report_period_lte), making it safe for backtesting.
"""
import datetime
import logging
import os
import time

import requests

from src.data.cache import get_cache
from src.data.models import (
    CompanyFacts,
    CompanyFactsResponse,
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    InsiderTrade,
    InsiderTradeResponse,
    LineItem,
    LineItemResponse,
    Price,
    PriceResponse,
)

logger = logging.getLogger(__name__)

_cache = get_cache()

_BASE = "https://api.financialdatasets.ai"


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict | None = None, max_retries: int = 3) -> requests.Response:
    """HTTP helper with linear backoff on 429s (60 s, 90 s, 120 s …)."""
    for attempt in range(max_retries + 1):
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)

        if response.status_code == 429 and attempt < max_retries:
            delay = 60 + 30 * attempt
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s …")
            time.sleep(delay)
            continue

        return response


class FinancialDatasetsProvider:
    """Fetches data from financialdatasets.ai with in-memory caching."""

    # ── prices ──────────────────────────────────────────────────────────────

    def get_prices(self, ticker: str, start_date: str, end_date: str, api_key: str | None = None) -> list[Price]:
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cached := _cache.get_prices(cache_key):
            return [Price(**p) for p in cached]

        headers = self._headers(api_key)
        url = f"{_BASE}/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            logger.warning("FD: could not fetch prices for %s (HTTP %s)", ticker, response.status_code)
            return []

        try:
            prices = PriceResponse(**response.json()).prices
        except (ValueError, KeyError) as exc:
            logger.warning("FD: failed to parse prices for %s: %s", ticker, exc)
            return []

        if prices:
            _cache.set_prices(cache_key, [p.model_dump() for p in prices])
        return prices

    # ── financial metrics ────────────────────────────────────────────────────

    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10, api_key: str | None = None) -> list[FinancialMetrics]:
        cache_key = f"{ticker}_{period}_{end_date}_{limit}"
        if cached := _cache.get_financial_metrics(cache_key):
            return [FinancialMetrics(**m) for m in cached]

        headers = self._headers(api_key)
        url = f"{_BASE}/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            logger.warning("FD: could not fetch metrics for %s (HTTP %s)", ticker, response.status_code)
            return []

        try:
            metrics = FinancialMetricsResponse(**response.json()).financial_metrics
        except (ValueError, KeyError) as exc:
            logger.warning("FD: failed to parse metrics for %s: %s", ticker, exc)
            return []

        if metrics:
            _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
        return metrics

    # ── line items ───────────────────────────────────────────────────────────

    def search_line_items(self, ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10, api_key: str | None = None) -> list[LineItem]:
        headers = self._headers(api_key)
        body = {"tickers": [ticker], "line_items": line_items, "end_date": end_date, "period": period, "limit": limit}
        response = _make_api_request(f"{_BASE}/financials/search/line-items", headers, method="POST", json_data=body)
        if response.status_code != 200:
            logger.warning("FD: could not fetch line items for %s (HTTP %s)", ticker, response.status_code)
            return []

        try:
            results = LineItemResponse(**response.json()).search_results
        except (ValueError, KeyError) as exc:
            logger.warning("FD: failed to parse line items for %s: %s", ticker, exc)
            return []

        return results[:limit]

    # ── insider trades ───────────────────────────────────────────────────────

    def get_insider_trades(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000, api_key: str | None = None) -> list[InsiderTrade]:
        cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
        if cached := _cache.get_insider_trades(cache_key):
            return [InsiderTrade(**t) for t in cached]

        headers = self._headers(api_key)
        all_trades: list[InsiderTrade] = []
        current_end = end_date

        while True:
            url = f"{_BASE}/insider-trades/?ticker={ticker}&filing_date_lte={current_end}"
            if start_date:
                url += f"&filing_date_gte={start_date}"
            url += f"&limit={limit}"

            response = _make_api_request(url, headers)
            if response.status_code != 200:
                logger.warning("FD: could not fetch insider trades for %s (HTTP %s)", ticker, response.status_code)
                break

            try:
                trades = InsiderTradeResponse(**response.json()).insider_trades
            except (ValueError, KeyError) as exc:
                logger.warning("FD: failed to parse insider trades for %s: %s", ticker, exc)
                break

            if not trades:
                break
            all_trades.extend(trades)

            if not start_date or len(trades) < limit:
                break
            current_end = min(t.filing_date for t in trades).split("T")[0]
            if current_end <= start_date:
                break

        if all_trades:
            _cache.set_insider_trades(cache_key, [t.model_dump() for t in all_trades])
        return all_trades

    # ── company news ─────────────────────────────────────────────────────────

    def get_company_news(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000, api_key: str | None = None) -> list[CompanyNews]:
        cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
        if cached := _cache.get_company_news(cache_key):
            return [CompanyNews(**n) for n in cached]

        headers = self._headers(api_key)
        all_news: list[CompanyNews] = []
        current_end = end_date

        while True:
            url = f"{_BASE}/news/?ticker={ticker}&end_date={current_end}"
            if start_date:
                url += f"&start_date={start_date}"
            url += f"&limit={limit}"

            response = _make_api_request(url, headers)
            if response.status_code != 200:
                logger.warning("FD: could not fetch news for %s (HTTP %s)", ticker, response.status_code)
                break

            try:
                news = CompanyNewsResponse(**response.json()).news
            except (ValueError, KeyError) as exc:
                logger.warning("FD: failed to parse news for %s: %s", ticker, exc)
                break

            if not news:
                break
            all_news.extend(news)

            if not start_date or len(news) < limit:
                break
            current_end = min(n.date for n in news).split("T")[0]
            if current_end <= start_date:
                break

        if all_news:
            _cache.set_company_news(cache_key, [n.model_dump() for n in all_news])
        return all_news

    # ── market cap ───────────────────────────────────────────────────────────

    def get_market_cap(self, ticker: str, end_date: str, api_key: str | None = None) -> float | None:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        if end_date == today:
            headers = self._headers(api_key)
            url = f"{_BASE}/company/facts/?ticker={ticker}"
            response = _make_api_request(url, headers)
            if response.status_code != 200:
                logger.warning("FD: could not fetch company facts for %s (HTTP %s)", ticker, response.status_code)
                return None
            try:
                return CompanyFactsResponse(**response.json()).company_facts.market_cap
            except (ValueError, KeyError):
                return None

        metrics = self.get_financial_metrics(ticker, end_date, api_key=api_key)
        if not metrics:
            return None
        return metrics[0].market_cap

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _headers(api_key: str | None) -> dict:
        key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
        return {"X-API-KEY": key} if key else {}
