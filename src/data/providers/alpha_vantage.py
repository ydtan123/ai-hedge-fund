"""AlphaVantageProvider — wraps the Alpha Vantage REST API.

Free tier: 25 requests/day.  Premium tiers allow more.
Best used as a fallback for fields yfinance cannot provide, and as the
primary news source (pre-computed sentiment via NEWS_SENTIMENT endpoint).

Note: financial statements include fiscalDateEnding, so they support basic
point-in-time filtering (filtering periods whose end date <= end_date).
However, restated historical metrics are NOT available — use FinancialDatasets
for production backtesting.
"""
import logging
import os
import threading
import time

import requests

from src.data.cache import get_cache
from src.data.models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price

logger = logging.getLogger(__name__)

_cache = get_cache()

# Raw AV statement cache: keyed by "{ticker}_{period}" → {"income": [...], "balance": [...], "cashflow": [...]}
_raw_statements: dict[str, dict] = {}
# Raw AV overview cache: keyed by ticker → dict
_raw_overview: dict[str, dict] = {}
_av_lock = threading.Lock()

_BASE = "https://www.alphavantage.co/query"

# ── Field mappings ────────────────────────────────────────────────────────────

_OVERVIEW_MAP: dict[str, str] = {
    "market_cap": "MarketCapitalization",
    "enterprise_value": "EVToRevenue",          # no direct EV — use EBITDA multiple instead
    "price_to_earnings_ratio": "PERatio",
    "price_to_book_ratio": "PriceToBookRatio",
    "price_to_sales_ratio": "PriceToSalesRatioTTM",
    "enterprise_value_to_ebitda_ratio": "EVToEBITDA",
    "enterprise_value_to_revenue_ratio": "EVToRevenue",
    "peg_ratio": "PEGRatio",
    "gross_margin": "GrossProfitTTM",           # needs revenue for ratio — handled below
    "operating_margin": "OperatingMarginTTM",
    "net_margin": "ProfitMargin",
    "return_on_equity": "ReturnOnEquityTTM",
    "return_on_assets": "ReturnOnAssetsTTM",
    "earnings_per_share": "EPS",
    "book_value_per_share": "BookValue",
    "payout_ratio": "PayoutRatio",
    "earnings_growth": "QuarterlyEarningsGrowthYOY",
    "revenue_growth": "QuarterlyRevenueGrowthYOY",
    "earnings_per_share_growth": "QuarterlyEarningsGrowthYOY",  # best approximation
    "debt_to_equity": "DebtToEquityRatioAnnual",    # may not be present in all responses
}

_INCOME_MAP: dict[str, str] = {
    "revenue": "totalRevenue",
    "gross_profit": "grossProfit",
    "operating_income": "operatingIncome",
    "net_income": "netIncome",
    "ebit": "ebit",
    "ebitda": "ebitda",
    "interest_expense": "interestExpense",
    "research_and_development": "researchAndDevelopment",
    "operating_expense": "operatingExpenses",
    "earnings_per_share": "reportedEPS",        # not always present; fallback to overview
}

_BALANCE_MAP: dict[str, str] = {
    "total_assets": "totalAssets",
    "total_liabilities": "totalLiabilities",
    "shareholders_equity": "totalShareholderEquity",
    "current_assets": "totalCurrentAssets",
    "current_liabilities": "totalCurrentLiabilities",
    "total_debt": "shortLongTermDebtTotal",
    "cash_and_equivalents": "cashAndCashEquivalentsAtCarryingValue",
    "goodwill_and_intangible_assets": "goodwill",       # approximate
    "intangible_assets": "intangibleAssets",
    "outstanding_shares": "commonStockSharesOutstanding",
}

_CASHFLOW_MAP: dict[str, str] = {
    "capital_expenditure": "capitalExpenditures",
    "free_cash_flow": "operatingCashflow",              # approximate; AV doesn't have direct FCF
    "depreciation_and_amortization": "depreciationDepletionAndAmortization",
    "dividends_and_other_cash_distributions": "dividendPayout",
    "issuance_or_purchase_of_equity_shares": "paymentsForRepurchaseOfEquity",
}

# Sentiment label mapping for news
_SENTIMENT_MAP: dict[str, str] = {
    "Bullish": "positive",
    "Somewhat-Bullish": "positive",
    "Neutral": "neutral",
    "Somewhat-Bearish": "negative",
    "Bearish": "negative",
}


def _safe_float(val) -> float | None:
    try:
        if val is None or val == "None" or val == "-":
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


class AlphaVantageProvider:
    """Fetches data from Alpha Vantage REST API."""

    def __init__(self):
        self._request_count = 0

    # ── prices ────────────────────────────────────────────────────────────────

    def get_prices(self, ticker: str, start_date: str, end_date: str, api_key: str | None = None) -> list[Price]:
        try:
            data = self._request("TIME_SERIES_DAILY", {"symbol": ticker, "outputsize": "full"}, api_key)
            if not isinstance(data, dict):
                return []
            series = data.get("Time Series (Daily)", {})
            prices: list[Price] = []
            for date_str, ohlcv in sorted(series.items(), reverse=True):
                if date_str > end_date:
                    continue
                if date_str < start_date:
                    break
                prices.append(Price(
                    open=_safe_float(ohlcv.get("1. open")),
                    close=_safe_float(ohlcv.get("4. close")),
                    high=_safe_float(ohlcv.get("2. high")),
                    low=_safe_float(ohlcv.get("3. low")),
                    volume=int(float(ohlcv.get("5. volume", 0))),
                    time=f"{date_str}T00:00:00",
                ))
            return list(reversed(prices))
        except Exception as exc:
            logger.warning("AV: get_prices failed for %s: %s", ticker, exc)
            return []

    # ── financial metrics ─────────────────────────────────────────────────────

    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10, api_key: str | None = None) -> list[FinancialMetrics]:
        try:
            with _av_lock:
                if ticker not in _raw_overview:
                    _raw_overview[ticker] = self._request("OVERVIEW", {"symbol": ticker}, api_key)
            overview = _raw_overview[ticker]
            if not isinstance(overview, dict) or not overview.get("Symbol"):
                logger.warning("AV: empty overview for %s", ticker)
                return []

            currency = overview.get("Currency", "USD")
            m: dict = {
                "ticker": ticker,
                "report_period": end_date,
                "period": period,
                "currency": currency,
            }

            for our_field, av_key in _OVERVIEW_MAP.items():
                m[our_field] = _safe_float(overview.get(av_key))

            # Market cap needs special handling (AV returns raw number)
            raw_mcap = _safe_float(overview.get("MarketCapitalization"))
            m["market_cap"] = raw_mcap

            # Gross margin = grossProfitTTM / revenueTTM
            gp = _safe_float(overview.get("GrossProfitTTM"))
            rev = _safe_float(overview.get("RevenueTTM"))
            if gp is not None and rev and rev != 0:
                m["gross_margin"] = gp / rev
            else:
                m["gross_margin"] = None

            # Fill any remaining required fields with None
            for field_name in FinancialMetrics.model_fields:
                if field_name not in m:
                    m[field_name] = None

            return [FinancialMetrics(**m)]
        except Exception as exc:
            logger.warning("AV: get_financial_metrics failed for %s: %s", ticker, exc)
            return []

    # ── line items ────────────────────────────────────────────────────────────

    def search_line_items(self, ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10, api_key: str | None = None) -> list[LineItem]:
        try:
            use_quarterly = period != "annual"
            stmt_key = f"{ticker}_{period}"
            with _av_lock:
                if stmt_key not in _raw_statements:
                    _raw_statements[stmt_key] = {
                        "income": self._request("INCOME_STATEMENT", {"symbol": ticker}, api_key),
                        "balance": self._request("BALANCE_SHEET", {"symbol": ticker}, api_key),
                        "cashflow": self._request("CASH_FLOW", {"symbol": ticker}, api_key),
                    }
            income_data = _raw_statements[stmt_key]["income"]
            balance_data = _raw_statements[stmt_key]["balance"]
            cashflow_data = _raw_statements[stmt_key]["cashflow"]

            income_reports = self._filter_reports(income_data, end_date, use_quarterly)[:limit]
            balance_reports = self._filter_reports(balance_data, end_date, use_quarterly)[:limit]
            cashflow_reports = self._filter_reports(cashflow_data, end_date, use_quarterly)[:limit]

            # Build a period → report mapping
            def index_by_date(reports: list[dict]) -> dict[str, dict]:
                return {r["fiscalDateEnding"]: r for r in reports if "fiscalDateEnding" in r}

            inc_idx = index_by_date(income_reports)
            bal_idx = index_by_date(balance_reports)
            cf_idx = index_by_date(cashflow_reports)

            all_dates = sorted(set(inc_idx) | set(bal_idx) | set(cf_idx), reverse=True)[:limit]

            results: list[LineItem] = []
            for date_str in all_dates:
                inc = inc_idx.get(date_str, {})
                bal = bal_idx.get(date_str, {})
                cf = cf_idx.get(date_str, {})
                currency = inc.get("reportedCurrency") or bal.get("reportedCurrency") or "USD"

                row: dict = {"ticker": ticker, "report_period": date_str, "period": period, "currency": currency}

                for field in line_items:
                    val = None
                    if field in _INCOME_MAP:
                        val = _safe_float(inc.get(_INCOME_MAP[field]))
                    elif field in _BALANCE_MAP:
                        val = _safe_float(bal.get(_BALANCE_MAP[field]))
                    elif field in _CASHFLOW_MAP:
                        val = _safe_float(cf.get(_CASHFLOW_MAP[field]))
                    elif field == "working_capital":
                        ca = _safe_float(bal.get("totalCurrentAssets"))
                        cl = _safe_float(bal.get("totalCurrentLiabilities"))
                        val = (ca - cl) if ca is not None and cl is not None else None
                    elif field == "gross_margin":
                        rev = _safe_float(inc.get("totalRevenue"))
                        gp = _safe_float(inc.get("grossProfit"))
                        val = (gp / rev) if rev and gp is not None else None
                    elif field == "operating_margin":
                        rev = _safe_float(inc.get("totalRevenue"))
                        oi = _safe_float(inc.get("operatingIncome"))
                        val = (oi / rev) if rev and oi is not None else None
                    elif field == "book_value_per_share":
                        equity = _safe_float(bal.get("totalShareholderEquity"))
                        shares = _safe_float(bal.get("commonStockSharesOutstanding"))
                        val = (equity / shares) if equity is not None and shares else None
                    elif field == "debt_to_equity":
                        debt = _safe_float(bal.get("shortLongTermDebtTotal"))
                        equity = _safe_float(bal.get("totalShareholderEquity"))
                        val = (debt / equity) if debt is not None and equity else None
                    elif field == "return_on_invested_capital":
                        ebit = _safe_float(inc.get("ebit"))
                        equity = _safe_float(bal.get("totalShareholderEquity"))
                        debt = _safe_float(bal.get("shortLongTermDebtTotal"))
                        if ebit is not None and equity is not None and debt is not None:
                            nopat = ebit * 0.79
                            ic = equity + debt
                            val = (nopat / ic) if ic else None
                    elif field == "free_cash_flow":
                        op_cf = _safe_float(cf.get("operatingCashflow"))
                        capex = _safe_float(cf.get("capitalExpenditures"))
                        if op_cf is not None and capex is not None:
                            val = op_cf - abs(capex)
                        elif op_cf is not None:
                            val = op_cf
                    row[field] = val

                results.append(LineItem(**row))
            return results
        except Exception as exc:
            logger.warning("AV: search_line_items failed for %s: %s", ticker, exc)
            return []

    # ── insider trades ────────────────────────────────────────────────────────

    def get_insider_trades(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000, api_key: str | None = None) -> list[InsiderTrade]:
        try:
            data = self._request("INSIDER_TRANSACTIONS", {"symbol": ticker}, api_key)
            if not isinstance(data, dict):
                return []
            raw_trades = data.get("data", [])
            trades: list[InsiderTrade] = []
            for item in raw_trades:
                filing_date = item.get("transactionDate") or item.get("filing_date") or ""
                if not filing_date:
                    continue
                if end_date and filing_date > end_date:
                    continue
                if start_date and filing_date < start_date:
                    continue
                shares_raw = item.get("transactionShares") or item.get("shares")
                trades.append(InsiderTrade(
                    ticker=ticker,
                    issuer=ticker,
                    name=item.get("executiveName") or item.get("name"),
                    title=item.get("executiveTitle") or item.get("title"),
                    is_board_director=None,
                    transaction_date=filing_date,
                    transaction_shares=_safe_float(shares_raw),
                    transaction_price_per_share=_safe_float(item.get("transactionPrice")),
                    transaction_value=None,
                    shares_owned_before_transaction=None,
                    shares_owned_after_transaction=_safe_float(item.get("sharesOwned")),
                    security_title=None,
                    filing_date=filing_date,
                ))
                if len(trades) >= limit:
                    break
            return trades
        except Exception as exc:
            logger.warning("AV: get_insider_trades failed for %s: %s", ticker, exc)
            return []

    # ── company news ──────────────────────────────────────────────────────────

    def get_company_news(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000, api_key: str | None = None) -> list[CompanyNews]:
        try:
            params: dict = {"tickers": ticker, "limit": str(min(limit, 1000))}
            if start_date:
                params["time_from"] = start_date.replace("-", "") + "T0000"
            if end_date:
                params["time_to"] = end_date.replace("-", "") + "T2359"

            data = self._request("NEWS_SENTIMENT", params, api_key)
            if not isinstance(data, dict):
                return []

            feed = data.get("feed", [])
            news: list[CompanyNews] = []
            for article in feed:
                pub_raw = article.get("time_published", "")
                # AV format: "20240315T143000"
                try:
                    dt = time.strptime(pub_raw, "%Y%m%dT%H%M%S")
                    date_str = f"{dt.tm_year:04d}-{dt.tm_mon:02d}-{dt.tm_mday:02d}T{dt.tm_hour:02d}:{dt.tm_min:02d}:{dt.tm_sec:02d}"
                except Exception:
                    date_str = end_date + "T00:00:00"

                date_only = date_str[:10]
                if end_date and date_only > end_date:
                    continue
                if start_date and date_only < start_date:
                    continue

                # Pick ticker-specific sentiment if available, else overall
                sentiment_label = article.get("overall_sentiment_label", "Neutral")
                for ts in article.get("ticker_sentiment", []):
                    if ts.get("ticker", "").upper() == ticker.upper():
                        sentiment_label = ts.get("ticker_sentiment_label", sentiment_label)
                        break

                sentiment = _SENTIMENT_MAP.get(sentiment_label, "neutral")
                authors = article.get("authors", [])
                author = authors[0] if authors else None

                news.append(CompanyNews(
                    ticker=ticker,
                    title=article.get("title", ""),
                    author=author,
                    source=article.get("source", "Alpha Vantage"),
                    date=date_str,
                    url=article.get("url", ""),
                    sentiment=sentiment,
                ))
                if len(news) >= limit:
                    break
            return news
        except Exception as exc:
            logger.warning("AV: get_company_news failed for %s: %s", ticker, exc)
            return []

    # ── market cap ────────────────────────────────────────────────────────────

    def get_market_cap(self, ticker: str, end_date: str, api_key: str | None = None) -> float | None:
        try:
            data = self._request("OVERVIEW", {"symbol": ticker}, api_key)
            if not isinstance(data, dict):
                return None
            return _safe_float(data.get("MarketCapitalization"))
        except Exception as exc:
            logger.warning("AV: get_market_cap failed for %s: %s", ticker, exc)
            return None

    # ── private helpers ───────────────────────────────────────────────────────

    def _request(self, function: str, params: dict, api_key: str | None = None) -> dict | str:
        key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not key:
            logger.warning("AV: ALPHA_VANTAGE_API_KEY not set — request will fail")

        all_params = {**params, "function": function}
        if key:
            all_params["apikey"] = key

        try:
            response = requests.get(_BASE, params=all_params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Detect rate-limit or error messages
            if "Information" in data:
                msg = data["Information"]
                if "rate limit" in msg.lower() or "api key" in msg.lower():
                    logger.warning("AV: rate limit hit — %s", msg)
                    return {}
            if "Note" in data:
                logger.warning("AV: API note — %s", data["Note"])
                return {}
            return data
        except Exception as exc:
            logger.warning("AV: HTTP request failed (%s %s): %s", function, params, exc)
            return {}

    @staticmethod
    def _filter_reports(data: dict, end_date: str, quarterly: bool) -> list[dict]:
        """Return annual or quarterly reports with fiscalDateEnding <= end_date."""
        if not isinstance(data, dict):
            return []
        key = "quarterlyReports" if quarterly else "annualReports"
        reports = data.get(key, [])
        return [r for r in reports if r.get("fiscalDateEnding", "") <= end_date]
