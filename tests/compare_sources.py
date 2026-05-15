"""MSFT dual-source comparison harness.

Compares data and analyst signals between:
  - FinancialDatasets provider (FINANCIAL_DATASETS_SOURCE for all types)
  - HYBRID provider (Yahoo Finance → Alpha Vantage fallback)

Usage:
    python -m tests.compare_sources

Requirements:
    - FINANCIAL_DATASETS_API_KEY env var (for FD source)
    - Optional ALPHA_VANTAGE_API_KEY for AV fallback enrichment
"""
import logging
import os
import sys
from datetime import datetime, timedelta

# Configure logging before any imports that use it
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

import src.data.config as _cfg
from src.data.config import DataSource
from src.data import config as cfg_module

TICKER = "MSFT"
END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
TOLERANCE = 0.05  # 5% numeric mismatch threshold

# ── Numeric comparison helpers ────────────────────────────────────────────────


def _pct_diff(a, b) -> float | None:
    if a is None or b is None:
        return None
    if a == 0 and b == 0:
        return 0.0
    denom = abs(a) if abs(a) > abs(b) else abs(b)
    if denom == 0:
        return None
    return abs(a - b) / denom


def _compare_field(label: str, fd_val, yf_val, *, tolerance: float = TOLERANCE):
    if fd_val is None and yf_val is None:
        logger.debug("  %s — both None", label)
        return
    if fd_val is None:
        logger.info("  [MISSING-FD]  %-45s  YF=%s", label, yf_val)
        return
    if yf_val is None:
        logger.info("  [MISSING-YF]  %-45s  FD=%s", label, fd_val)
        return

    try:
        pct = _pct_diff(float(fd_val), float(yf_val))
    except (TypeError, ValueError):
        if str(fd_val) != str(yf_val):
            logger.warning("  [MISMATCH]    %-45s  FD=%s  YF=%s", label, fd_val, yf_val)
        return

    if pct is not None and pct > tolerance:
        logger.warning("  [MISMATCH %4.0f%%]  %-40s  FD=%.4g  YF=%.4g", pct * 100, label, float(fd_val), float(yf_val))
    else:
        logger.debug("  [OK]          %-45s  FD=%.4g  YF=%.4g", label, float(fd_val), float(yf_val))


# ── Data-level comparison ─────────────────────────────────────────────────────


def _compare_prices(fd_prices, yf_prices):
    logger.info("\n── Prices ──────────────────────────────────────────────────")
    logger.info("  FD rows: %d  |  YF rows: %d", len(fd_prices), len(yf_prices))
    if not fd_prices or not yf_prices:
        logger.warning("  One source returned no prices — skipping field comparison")
        return

    fd_map = {p.time[:10]: p for p in fd_prices}
    yf_map = {p.time[:10]: p for p in yf_prices}
    common = sorted(set(fd_map) & set(yf_map))
    logger.info("  Common dates: %d", len(common))
    if common:
        last = common[-1]
        p_fd, p_yf = fd_map[last], yf_map[last]
        for field in ["open", "close", "high", "low"]:
            _compare_field(f"prices.{field} ({last})", getattr(p_fd, field), getattr(p_yf, field))


def _compare_metrics(fd_list, yf_list):
    logger.info("\n── Financial Metrics ────────────────────────────────────────")
    logger.info("  FD count: %d  |  YF count: %d", len(fd_list), len(yf_list))
    if not fd_list or not yf_list:
        logger.warning("  One source returned no metrics — skipping field comparison")
        return
    fd = fd_list[0]
    yf = yf_list[0]
    fields = [
        "return_on_equity", "return_on_assets", "return_on_invested_capital",
        "net_margin", "gross_margin", "operating_margin",
        "revenue_growth", "earnings_growth",
        "price_to_earnings_ratio", "price_to_book_ratio", "price_to_sales_ratio",
        "enterprise_value_to_ebitda", "enterprise_value_to_revenue",
        "debt_to_equity", "current_ratio",
        "earnings_per_share", "book_value_per_share",
    ]
    for f in fields:
        _compare_field(f"metrics.{f}", getattr(fd, f, None), getattr(yf, f, None))


def _compare_line_items(fd_list, yf_list):
    logger.info("\n── Line Items ───────────────────────────────────────────────")
    logger.info("  FD count: %d  |  YF count: %d", len(fd_list), len(yf_list))
    if not fd_list or not yf_list:
        logger.warning("  One source returned no line items — skipping field comparison")
        return
    fd = fd_list[0]
    yf = yf_list[0]
    for field in ["revenue", "net_income", "operating_income", "free_cash_flow", "total_assets", "total_liabilities", "shareholders_equity"]:
        _compare_field(f"line_items.{field}", getattr(fd, field, None), getattr(yf, field, None))


def _compare_news(fd_news, yf_news):
    logger.info("\n── Company News ─────────────────────────────────────────────")
    logger.info("  FD count: %d  |  YF count: %d", len(fd_news), len(yf_news))
    fd_titles = {a.title[:60].lower() for a in fd_news}
    yf_titles = {a.title[:60].lower() for a in yf_news}
    overlap = len(fd_titles & yf_titles)
    logger.info("  Title overlap (first 60 chars): %d", overlap)
    fd_sentiment = [a for a in fd_news if a.sentiment]
    yf_sentiment = [a for a in yf_news if a.sentiment]
    logger.info("  Articles with sentiment  FD: %d  |  YF: %d", len(fd_sentiment), len(yf_sentiment))


def _compare_market_cap(fd_cap, yf_cap):
    logger.info("\n── Market Cap ───────────────────────────────────────────────")
    logger.info("  FD: %s  |  YF: %s", fd_cap, yf_cap)
    _compare_field("market_cap", fd_cap, yf_cap)


# ── Agent signal comparison ───────────────────────────────────────────────────


def _run_analysis(source_label: str, source: DataSource):
    """Run full hedge fund analysis with a given source config, return signal dict."""
    logger.info("\n\n══════════════════════════════════════════")
    logger.info("  Running analysis with: %s", source_label)
    logger.info("══════════════════════════════════════════\n")

    # Patch all sources to the target source
    _cfg.PRICES_SOURCE = source
    _cfg.FINANCIAL_METRICS_SOURCE = source
    _cfg.LINE_ITEMS_SOURCE = source
    _cfg.INSIDER_TRADES_SOURCE = source
    _cfg.NEWS_SOURCE = source
    _cfg.MARKET_CAP_SOURCE = source

    try:
        from src.main import run_hedge_fund
        result = run_hedge_fund(
            ticker=TICKER,
            start_date=START_DATE,
            end_date=END_DATE,
            portfolio={"cash": 100_000, "stock": 0},
            show_reasoning=False,
        )
        return result
    except Exception as exc:
        logger.error("  Analysis failed: %s", exc)
        return None
    finally:
        # Reset to defaults so other tests aren't affected
        _cfg.PRICES_SOURCE = DataSource.HYBRID
        _cfg.FINANCIAL_METRICS_SOURCE = DataSource.FINANCIAL_DATASETS
        _cfg.LINE_ITEMS_SOURCE = DataSource.FINANCIAL_DATASETS
        _cfg.INSIDER_TRADES_SOURCE = DataSource.HYBRID
        _cfg.NEWS_SOURCE = DataSource.HYBRID
        _cfg.MARKET_CAP_SOURCE = DataSource.HYBRID


def _compare_signals(fd_result, yf_result):
    """Compare analyst signals from two runs, flag opposite conclusions."""
    logger.info("\n\n══════════════════════════════════════════")
    logger.info("  Signal Comparison")
    logger.info("══════════════════════════════════════════")

    if fd_result is None or yf_result is None:
        logger.warning("  Cannot compare signals — one or both runs failed")
        return

    # Extract decisions from result (format: {"decisions": {"MSFT": {"action": ..., "analyst_signals": {...}}}})
    fd_decisions = fd_result.get("decisions", {}).get(TICKER, {})
    yf_decisions = yf_result.get("decisions", {}).get(TICKER, {})

    fd_action = fd_decisions.get("action", "N/A")
    yf_action = yf_decisions.get("action", "N/A")

    logger.info("  FD final action: %s", fd_action)
    logger.info("  YF final action: %s", yf_action)

    opposite_actions = {("buy", "sell"), ("sell", "buy"), ("buy", "short"), ("short", "buy")}
    if (fd_action.lower(), yf_action.lower()) in opposite_actions:
        logger.warning("\n  [OPPOSITE FINAL SIGNALS] FD=%s vs YF=%s", fd_action, yf_action)

    # Compare per-analyst signals
    fd_signals = fd_decisions.get("analyst_signals", {})
    yf_signals = yf_decisions.get("analyst_signals", {})

    all_analysts = sorted(set(list(fd_signals.keys()) + list(yf_signals.keys())))
    opposites = []

    for analyst in all_analysts:
        fd_sig = fd_signals.get(analyst, {})
        yf_sig = yf_signals.get(analyst, {})
        fd_signal = fd_sig.get("signal", "N/A")
        yf_signal = yf_sig.get("signal", "N/A")
        fd_conf = fd_sig.get("confidence", "N/A")
        yf_conf = yf_sig.get("confidence", "N/A")

        same = fd_signal == yf_signal
        status = "OK" if same else "DIVERGE"

        pair = (fd_signal.lower(), yf_signal.lower())
        is_opposite = pair in {("bullish", "bearish"), ("bearish", "bullish")}

        logger.info(
            "  %-40s  FD=%-10s(%s)  YF=%-10s(%s)  [%s]%s",
            analyst, fd_signal, fd_conf, yf_signal, yf_conf, status,
            " *** OPPOSITE ***" if is_opposite else "",
        )

        if is_opposite:
            opposites.append(analyst)

    if opposites:
        logger.warning("\n  Opposite signals from: %s", ", ".join(opposites))
        logger.warning("  Likely causes:")
        logger.warning("    • Metrics/line-item values differ between providers (see data comparison above)")
        logger.warning("    • YF financial statements are current snapshots (no point-in-time); FD is point-in-time")
        logger.warning("    • Alpha Vantage pre-computed sentiment labels may shift news-sentiment agent conclusions")
        logger.warning("    • Growth rates computed differently: FD returns direct values; YF computes from consecutive statements")


# ── Data-level comparison entry point ────────────────────────────────────────


def run_data_comparison():
    """Fetch data from both providers and compare field-by-field."""
    from src.tools.api import (
        get_prices,
        get_financial_metrics,
        search_line_items,
        get_company_news,
        get_market_cap,
    )

    line_item_fields = [
        "revenue", "net_income", "operating_income", "gross_profit",
        "free_cash_flow", "capital_expenditure", "operating_cash_flow",
        "total_assets", "total_liabilities", "shareholders_equity",
        "total_debt", "cash_and_equivalents",
    ]

    logger.info("══════════════════════════════════════════")
    logger.info("  Data Comparison: FD vs YF+AV  (%s)", TICKER)
    logger.info("  Date range: %s → %s", START_DATE, END_DATE)
    logger.info("══════════════════════════════════════════")

    # --- Prices ---
    _cfg.PRICES_SOURCE = DataSource.FINANCIAL_DATASETS
    fd_prices = get_prices(TICKER, START_DATE, END_DATE)
    _cfg.PRICES_SOURCE = DataSource.YAHOO_FINANCE
    yf_prices = get_prices(TICKER, START_DATE, END_DATE)
    _cfg.PRICES_SOURCE = DataSource.HYBRID
    _compare_prices(fd_prices, yf_prices)

    # --- Metrics ---
    _cfg.FINANCIAL_METRICS_SOURCE = DataSource.FINANCIAL_DATASETS
    fd_metrics = get_financial_metrics(TICKER, END_DATE)
    _cfg.FINANCIAL_METRICS_SOURCE = DataSource.YAHOO_FINANCE
    yf_metrics = get_financial_metrics(TICKER, END_DATE)
    _cfg.FINANCIAL_METRICS_SOURCE = DataSource.FINANCIAL_DATASETS
    _compare_metrics(fd_metrics, yf_metrics)

    # --- Line Items ---
    _cfg.LINE_ITEMS_SOURCE = DataSource.FINANCIAL_DATASETS
    fd_items = search_line_items(TICKER, line_item_fields, END_DATE)
    _cfg.LINE_ITEMS_SOURCE = DataSource.YAHOO_FINANCE
    yf_items = search_line_items(TICKER, line_item_fields, END_DATE)
    _cfg.LINE_ITEMS_SOURCE = DataSource.FINANCIAL_DATASETS
    _compare_line_items(fd_items, yf_items)

    # --- News ---
    _cfg.NEWS_SOURCE = DataSource.FINANCIAL_DATASETS
    fd_news = get_company_news(TICKER, END_DATE, START_DATE)
    _cfg.NEWS_SOURCE = DataSource.HYBRID
    yf_news = get_company_news(TICKER, END_DATE, START_DATE)
    _compare_news(fd_news, yf_news)

    # --- Market Cap ---
    _cfg.MARKET_CAP_SOURCE = DataSource.FINANCIAL_DATASETS
    fd_cap = get_market_cap(TICKER, END_DATE)
    _cfg.MARKET_CAP_SOURCE = DataSource.HYBRID
    yf_cap = get_market_cap(TICKER, END_DATE)
    _compare_market_cap(fd_cap, yf_cap)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    fd_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if not fd_key:
        logger.warning("FINANCIAL_DATASETS_API_KEY not set — FD comparisons will fail or use free-tier tickers only")

    # Phase 1: data-level comparison
    run_data_comparison()

    # Phase 2: agent-level comparison (requires a working LLM key)
    llm_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not llm_key:
        logger.warning("\nNo LLM API key found — skipping agent signal comparison")
        return

    fd_result = _run_analysis("FINANCIAL_DATASETS", DataSource.FINANCIAL_DATASETS)
    yf_result = _run_analysis("HYBRID (YF+AV)", DataSource.HYBRID)
    _compare_signals(fd_result, yf_result)

    logger.info("\n\nComparison complete.")


if __name__ == "__main__":
    main()
