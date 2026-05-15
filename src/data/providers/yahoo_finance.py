"""YahooFinanceProvider — wraps yfinance for free, key-less data access.

Limitations vs FinancialDatasets:
- Financial statements are current snapshots, NOT point-in-time historical.
  Using this for backtesting introduces look-ahead bias.
- Pre-computed growth rates (revenue_growth, earnings_growth, etc.) and ROIC
  are not available directly; they are computed from 2 consecutive periods.
- Company news has no pre-computed sentiment field (sentiment=None).
"""
import logging
from datetime import datetime

import pandas as pd

from src.data.models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price

logger = logging.getLogger(__name__)

# ── Field mappings ────────────────────────────────────────────────────────────

# Maps our system field names → yfinance income_stmt row labels
_INCOME_MAP: dict[str, str] = {
    "revenue": "Total Revenue",
    "gross_profit": "Gross Profit",
    "operating_income": "Operating Income",
    "net_income": "Net Income",
    "ebit": "EBIT",
    "ebitda": "EBITDA",
    "interest_expense": "Interest Expense",
    "research_and_development": "Research And Development",
    "operating_expense": "Operating Expense",
    "earnings_per_share": "Basic EPS",
}

# Maps our system field names → yfinance balance_sheet row labels
_BALANCE_MAP: dict[str, str] = {
    "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities Net Minority Interest",
    "shareholders_equity": "Stockholders Equity",
    "current_assets": "Current Assets",
    "current_liabilities": "Current Liabilities",
    "total_debt": "Total Debt",
    "cash_and_equivalents": "Cash And Cash Equivalents",
    "goodwill_and_intangible_assets": "Goodwill And Other Intangible Assets",
    "intangible_assets": "Other Intangible Assets",
    "outstanding_shares": "Share Issued",
}

# Maps our system field names → yfinance cashflow row labels
_CASHFLOW_MAP: dict[str, str] = {
    "capital_expenditure": "Capital Expenditure",
    "free_cash_flow": "Free Cash Flow",
    "depreciation_and_amortization": "Reconciled Depreciation",
    "dividends_and_other_cash_distributions": "Cash Dividends Paid",
    "issuance_or_purchase_of_equity_shares": "Repurchase Of Capital Stock",
}

# Fields that map to yfinance.info dict keys
_INFO_METRIC_MAP: dict[str, str] = {
    "price_to_earnings_ratio": "trailingPE",
    "price_to_book_ratio": "priceToBook",
    "price_to_sales_ratio": "priceToSalesTrailing12Months",
    "peg_ratio": "pegRatio",
    "gross_margin": "grossMargins",
    "operating_margin": "operatingMargins",
    "net_margin": "profitMargins",
    "return_on_equity": "returnOnEquity",
    "return_on_assets": "returnOnAssets",
    "current_ratio": "currentRatio",
    "quick_ratio": "quickRatio",
    "debt_to_equity": "debtToEquity",
    "payout_ratio": "payoutRatio",
    "earnings_per_share": "trailingEps",
    "book_value_per_share": "bookValue",
    "market_cap": "marketCap",
    "enterprise_value": "enterpriseValue",
    "enterprise_value_to_ebitda_ratio": "enterpriseToEbitda",
    "enterprise_value_to_revenue_ratio": "enterpriseToRevenue",
    "earnings_growth": "earningsGrowth",
    "revenue_growth": "revenueGrowth",
}


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return f if pd.notna(f) else None
    except (TypeError, ValueError):
        return None


def _df_value(df: pd.DataFrame, row_label: str, col) -> float | None:
    """Safely extract a value from a yfinance statement DataFrame."""
    if df is None or df.empty:
        return None
    if row_label not in df.index:
        return None
    try:
        val = df.loc[row_label, col]
        return _safe_float(val)
    except (KeyError, Exception):
        return None


def _filter_cols_by_date(df: pd.DataFrame, end_date: str) -> list:
    """Return DataFrame columns (Timestamps) that are <= end_date, newest first."""
    if df is None or df.empty:
        return []
    end_dt = pd.Timestamp(end_date)
    cols = [c for c in df.columns if pd.Timestamp(c) <= end_dt]
    return sorted(cols, reverse=True)


class YahooFinanceProvider:
    """Fetches data from Yahoo Finance via yfinance."""

    # ── prices ───────────────────────────────────────────────────────────────

    def get_prices(self, ticker: str, start_date: str, end_date: str, api_key: str | None = None) -> list[Price]:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.history(start=start_date, end=end_date, auto_adjust=True)
            if df.empty:
                logger.warning("YF: no price data for %s (%s→%s)", ticker, start_date, end_date)
                return []
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            prices = []
            for ts, row in df.iterrows():
                prices.append(Price(
                    open=float(row["Open"]),
                    close=float(row["Close"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    volume=int(row["Volume"]),
                    time=ts.strftime("%Y-%m-%dT00:00:00"),
                ))
            return prices
        except Exception as exc:
            logger.warning("YF: get_prices failed for %s: %s", ticker, exc)
            return []

    # ── financial metrics ─────────────────────────────────────────────────────

    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10, api_key: str | None = None) -> list[FinancialMetrics]:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info or {}
            currency = info.get("currency", "USD") or "USD"
            report_period = end_date

            metrics_dict: dict = {
                "ticker": ticker,
                "report_period": report_period,
                "period": period,
                "currency": currency,
            }

            for our_field, yf_key in _INFO_METRIC_MAP.items():
                raw = info.get(yf_key)
                metrics_dict[our_field] = _safe_float(raw)

            # Compute growth rates from statements when not available in info
            if metrics_dict.get("revenue_growth") is None or metrics_dict.get("earnings_growth") is None:
                self._fill_growth_from_statements(t, end_date, period, metrics_dict)

            # ROIC is not in yfinance.info — attempt to compute from statements
            if metrics_dict.get("return_on_invested_capital") is None:
                self._fill_roic(t, end_date, period, metrics_dict)

            # Fill any remaining required fields with None (FinancialMetrics allows None for all numeric fields)
            for field_name in FinancialMetrics.model_fields:
                if field_name not in metrics_dict:
                    metrics_dict[field_name] = None

            return [FinancialMetrics(**metrics_dict)]
        except Exception as exc:
            logger.warning("YF: get_financial_metrics failed for %s: %s", ticker, exc)
            return []

    def _fill_growth_from_statements(self, t, end_date: str, period: str, metrics_dict: dict) -> None:
        try:
            income = t.income_stmt if period == "annual" else t.quarterly_income_stmt
            cols = _filter_cols_by_date(income, end_date)
            if len(cols) < 2:
                return

            rev_curr = _df_value(income, "Total Revenue", cols[0])
            rev_prev = _df_value(income, "Total Revenue", cols[1])
            if rev_curr and rev_prev and rev_prev != 0:
                metrics_dict["revenue_growth"] = (rev_curr - rev_prev) / abs(rev_prev)

            ni_curr = _df_value(income, "Net Income", cols[0])
            ni_prev = _df_value(income, "Net Income", cols[1])
            if ni_curr and ni_prev and ni_prev != 0:
                metrics_dict["earnings_growth"] = (ni_curr - ni_prev) / abs(ni_prev)

            eps_curr = _df_value(income, "Basic EPS", cols[0])
            eps_prev = _df_value(income, "Basic EPS", cols[1])
            if eps_curr and eps_prev and eps_prev != 0:
                metrics_dict["earnings_per_share_growth"] = (eps_curr - eps_prev) / abs(eps_prev)

            # Free cash flow growth from cashflow statement
            cashflow = t.cashflow if period == "annual" else t.quarterly_cashflow
            cf_cols = _filter_cols_by_date(cashflow, end_date)
            if len(cf_cols) >= 2:
                fcf_curr = _df_value(cashflow, "Free Cash Flow", cf_cols[0])
                fcf_prev = _df_value(cashflow, "Free Cash Flow", cf_cols[1])
                if fcf_curr is not None and fcf_prev and fcf_prev != 0:
                    metrics_dict["free_cash_flow_growth"] = (fcf_curr - fcf_prev) / abs(fcf_prev)
        except Exception as exc:
            logger.debug("YF: growth fill failed: %s", exc)

    def _fill_roic(self, t, end_date: str, period: str, metrics_dict: dict) -> None:
        try:
            income = t.income_stmt if period == "annual" else t.quarterly_income_stmt
            balance = t.balance_sheet if period == "annual" else t.quarterly_balance_sheet
            cashflow = t.cashflow if period == "annual" else t.quarterly_cashflow

            inc_cols = _filter_cols_by_date(income, end_date)
            bal_cols = _filter_cols_by_date(balance, end_date)
            cf_cols = _filter_cols_by_date(cashflow, end_date)

            if not inc_cols or not bal_cols:
                return

            ebit = _df_value(income, "EBIT", inc_cols[0])
            tax_rate = 0.21  # approximate US corporate tax rate
            nopat = ebit * (1 - tax_rate) if ebit is not None else None

            equity = _df_value(balance, "Stockholders Equity", bal_cols[0])
            total_debt = _df_value(balance, "Total Debt", bal_cols[0])

            if nopat is not None and equity is not None and total_debt is not None:
                invested_capital = equity + total_debt
                if invested_capital and invested_capital != 0:
                    metrics_dict["return_on_invested_capital"] = nopat / invested_capital
        except Exception as exc:
            logger.debug("YF: ROIC fill failed: %s", exc)

    # ── line items ────────────────────────────────────────────────────────────

    def search_line_items(self, ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10, api_key: str | None = None) -> list[LineItem]:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info or {}
            currency = info.get("currency", "USD") or "USD"

            use_quarterly = period != "annual"
            income = t.quarterly_income_stmt if use_quarterly else t.income_stmt
            balance = t.quarterly_balance_sheet if use_quarterly else t.balance_sheet
            cashflow = t.quarterly_cashflow if use_quarterly else t.cashflow

            # Get all available columns <= end_date, limit to requested periods
            inc_cols = _filter_cols_by_date(income, end_date)[:limit]
            bal_cols = _filter_cols_by_date(balance, end_date)[:limit]
            cf_cols = _filter_cols_by_date(cashflow, end_date)[:limit]

            # Use union of all available periods
            all_dates = sorted(set(
                [c.strftime("%Y-%m-%d") for c in inc_cols] +
                [c.strftime("%Y-%m-%d") for c in bal_cols] +
                [c.strftime("%Y-%m-%d") for c in cf_cols]
            ), reverse=True)[:limit]

            results: list[LineItem] = []
            for date_str in all_dates:
                date_ts = pd.Timestamp(date_str)
                row_data: dict = {
                    "ticker": ticker,
                    "report_period": date_str,
                    "period": period,
                    "currency": currency,
                }

                for field in line_items:
                    val = None

                    if field in _INCOME_MAP:
                        val = _df_value(income, _INCOME_MAP[field], date_ts)

                    elif field in _BALANCE_MAP:
                        val = _df_value(balance, _BALANCE_MAP[field], date_ts)

                    elif field in _CASHFLOW_MAP:
                        val = _df_value(cashflow, _CASHFLOW_MAP[field], date_ts)

                    # Computed fields
                    elif field == "working_capital":
                        ca = _df_value(balance, "Current Assets", date_ts)
                        cl = _df_value(balance, "Current Liabilities", date_ts)
                        val = (ca - cl) if ca is not None and cl is not None else None

                    elif field == "gross_margin":
                        rev = _df_value(income, "Total Revenue", date_ts)
                        gp = _df_value(income, "Gross Profit", date_ts)
                        val = (gp / rev) if rev and gp is not None else None

                    elif field == "operating_margin":
                        rev = _df_value(income, "Total Revenue", date_ts)
                        oi = _df_value(income, "Operating Income", date_ts)
                        val = (oi / rev) if rev and oi is not None else None

                    elif field == "book_value_per_share":
                        equity = _df_value(balance, "Stockholders Equity", date_ts)
                        shares = _df_value(balance, "Share Issued", date_ts) or _safe_float(info.get("sharesOutstanding"))
                        val = (equity / shares) if equity is not None and shares else None

                    elif field == "debt_to_equity":
                        debt = _df_value(balance, "Total Debt", date_ts)
                        equity = _df_value(balance, "Stockholders Equity", date_ts)
                        val = (debt / equity) if debt is not None and equity else None

                    elif field == "return_on_invested_capital":
                        # Approximate ROIC
                        ebit = _df_value(income, "EBIT", date_ts)
                        equity = _df_value(balance, "Stockholders Equity", date_ts)
                        debt = _df_value(balance, "Total Debt", date_ts)
                        if ebit is not None and equity is not None and debt is not None:
                            nopat = ebit * 0.79
                            ic = equity + debt
                            val = (nopat / ic) if ic else None

                    if val is not None:
                        row_data[field] = val

                results.append(LineItem(**row_data))

            return results
        except Exception as exc:
            logger.warning("YF: search_line_items failed for %s: %s", ticker, exc)
            return []

    # ── insider trades ────────────────────────────────────────────────────────

    def get_insider_trades(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000, api_key: str | None = None) -> list[InsiderTrade]:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.insider_transactions
            if df is None or df.empty:
                logger.warning("YF: no insider trades for %s", ticker)
                return []

            # Normalize column names (yfinance changes them between versions)
            df.columns = [c.strip() for c in df.columns]

            trades: list[InsiderTrade] = []
            for _, row in df.iterrows():
                # Extract date — yfinance may use "Start Date" or "Date"
                date_val = row.get("Start Date") or row.get("Date") or row.get("startDate")
                if date_val is None:
                    continue
                try:
                    filing_date = pd.Timestamp(date_val).strftime("%Y-%m-%d")
                except Exception:
                    continue

                if end_date and filing_date > end_date:
                    continue
                if start_date and filing_date < start_date:
                    continue

                shares_raw = row.get("Shares") or row.get("shares")
                value_raw = row.get("Value") or row.get("value")
                text_raw = str(row.get("Text") or row.get("text") or "")

                trades.append(InsiderTrade(
                    ticker=ticker,
                    issuer=ticker,
                    name=str(row.get("Insider") or row.get("insider") or ""),
                    title=str(row.get("Position") or row.get("position") or ""),
                    is_board_director=None,
                    transaction_date=filing_date,
                    transaction_shares=_safe_float(shares_raw),
                    transaction_price_per_share=None,
                    transaction_value=_safe_float(value_raw),
                    shares_owned_before_transaction=None,
                    shares_owned_after_transaction=None,
                    security_title=None,
                    filing_date=filing_date,
                ))
                if len(trades) >= limit:
                    break

            return trades
        except Exception as exc:
            logger.warning("YF: get_insider_trades failed for %s: %s", ticker, exc)
            return []

    # ── company news ──────────────────────────────────────────────────────────

    def get_company_news(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000, api_key: str | None = None) -> list[CompanyNews]:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            raw_news = t.get_news(count=min(limit, 100))
            if not raw_news:
                logger.warning("YF: no news for %s", ticker)
                return []

            news: list[CompanyNews] = []
            for article in raw_news:
                # Handle both flat and nested yfinance news structures
                if "content" in article:
                    content = article["content"]
                    title = content.get("title", "")
                    source = content.get("provider", {}).get("displayName", "")
                    pub_raw = content.get("pubDate", "")
                    url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
                    url = url_obj.get("url", "")
                    author = None
                else:
                    title = article.get("title", "")
                    source = article.get("publisher", "")
                    pub_raw = article.get("providerPublishTime", "")
                    url = article.get("link", "")
                    author = None

                # Parse date
                try:
                    if isinstance(pub_raw, (int, float)):
                        date_str = datetime.utcfromtimestamp(pub_raw).strftime("%Y-%m-%dT%H:%M:%S")
                    elif pub_raw:
                        date_str = pd.Timestamp(pub_raw).strftime("%Y-%m-%dT%H:%M:%S")
                    else:
                        date_str = end_date + "T00:00:00"
                except Exception:
                    date_str = end_date + "T00:00:00"

                date_only = date_str[:10]
                if end_date and date_only > end_date:
                    continue
                if start_date and date_only < start_date:
                    continue

                news.append(CompanyNews(
                    ticker=ticker,
                    title=title,
                    author=author,
                    source=source or "Yahoo Finance",
                    date=date_str,
                    url=url,
                    sentiment=None,  # yfinance does not provide sentiment
                ))
                if len(news) >= limit:
                    break

            return news
        except Exception as exc:
            logger.warning("YF: get_company_news failed for %s: %s", ticker, exc)
            return []

    # ── market cap ────────────────────────────────────────────────────────────

    def get_market_cap(self, ticker: str, end_date: str, api_key: str | None = None) -> float | None:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info or {}
            return _safe_float(info.get("marketCap"))
        except Exception as exc:
            logger.warning("YF: get_market_cap failed for %s: %s", ticker, exc)
            return None
