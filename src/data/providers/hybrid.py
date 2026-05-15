"""HybridProvider — calls providers in order, merging results field-by-field.

For list results (prices, insider trades): returns first non-empty list.
For FinancialMetrics and LineItem objects: merges field-by-field, taking the
first non-None value for each field across the provider chain.
For CompanyNews: merges articles from all providers, patching missing
sentiment fields from providers that have them.
"""
import logging

from src.data.models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price
from src.data.providers.base import DataProvider

logger = logging.getLogger(__name__)

# Fields to skip when merging (structural fields, not data)
_STRUCTURAL_FIELDS = {"ticker", "report_period", "period", "currency"}


def _merge_objects(primary, fallbacks: list, structural: set) -> object:
    """Patch None fields in `primary` from `fallbacks` in order. Returns primary."""
    if primary is None:
        return None
    primary_dict = primary.model_dump()
    for fb_obj in fallbacks:
        if fb_obj is None:
            continue
        fb_dict = fb_obj.model_dump()
        patched = False
        for key, val in fb_dict.items():
            if key in structural:
                continue
            if primary_dict.get(key) is None and val is not None:
                primary_dict[key] = val
                patched = True
                logger.debug("HYBRID: patched field '%s' from %s", key, type(fb_obj).__name__)
        if not patched:
            continue
    return primary.__class__(**primary_dict)


class HybridProvider:
    """Wraps multiple providers, merging their results field-by-field."""

    def __init__(self, chain: list[DataProvider]):
        if not chain:
            raise ValueError("HybridProvider requires at least one provider in the chain")
        self._chain = chain

    # ── prices ────────────────────────────────────────────────────────────────

    def get_prices(self, ticker: str, start_date: str, end_date: str, api_key: str | None = None) -> list[Price]:
        for provider in self._chain:
            result = provider.get_prices(ticker, start_date, end_date, api_key)
            if result:
                logger.debug("HYBRID: prices for %s from %s", ticker, type(provider).__name__)
                return result
        logger.warning("HYBRID: all providers returned empty prices for %s", ticker)
        return []

    # ── financial metrics ─────────────────────────────────────────────────────

    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10, api_key: str | None = None) -> list[FinancialMetrics]:
        results_per_provider: list[list[FinancialMetrics]] = []
        for provider in self._chain:
            r = provider.get_financial_metrics(ticker, end_date, period, limit, api_key)
            results_per_provider.append(r)

        if not any(results_per_provider):
            return []

        return _merge_financial_metrics(results_per_provider)

    # ── line items ────────────────────────────────────────────────────────────

    def search_line_items(self, ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10, api_key: str | None = None) -> list[LineItem]:
        results_per_provider: list[list[LineItem]] = []
        for provider in self._chain:
            r = provider.search_line_items(ticker, line_items, end_date, period, limit, api_key)
            results_per_provider.append(r)

        if not any(results_per_provider):
            return []

        return _merge_line_items(results_per_provider, requested_fields=line_items)

    # ── insider trades ────────────────────────────────────────────────────────

    def get_insider_trades(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000, api_key: str | None = None) -> list[InsiderTrade]:
        for provider in self._chain:
            result = provider.get_insider_trades(ticker, end_date, start_date, limit, api_key)
            if result:
                logger.debug("HYBRID: insider trades for %s from %s", ticker, type(provider).__name__)
                return result
        return []

    # ── company news ──────────────────────────────────────────────────────────

    def get_company_news(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000, api_key: str | None = None) -> list[CompanyNews]:
        results: list[list[CompanyNews]] = []
        for provider in self._chain:
            r = provider.get_company_news(ticker, end_date, start_date, limit, api_key)
            results.append(r)

        non_empty = [r for r in results if r]
        if not non_empty:
            return []

        # Merge: patch missing sentiment from later providers
        return _merge_news_sentiment(non_empty)

    # ── market cap ────────────────────────────────────────────────────────────

    def get_market_cap(self, ticker: str, end_date: str, api_key: str | None = None) -> float | None:
        for provider in self._chain:
            result = provider.get_market_cap(ticker, end_date, api_key)
            if result is not None:
                logger.debug("HYBRID: market cap for %s from %s", ticker, type(provider).__name__)
                return result
        return None


# ── Merge helpers (module-level for testability) ─────────────────────────────

def _merge_financial_metrics(results_per_provider: list[list[FinancialMetrics]]) -> list[FinancialMetrics]:
    """For each period, build one FinancialMetrics by taking the first non-None
    value for each field across providers in order."""
    # Collect the primary (first provider's) list as base
    primary_list = next((r for r in results_per_provider if r), [])
    if not primary_list:
        return []

    # If only one provider returned data, no merging needed
    fallback_lists = [r for r in results_per_provider[1:] if r]
    if not fallback_lists:
        return primary_list

    # For metrics we only have one period per provider (current snapshot) —
    # just patch the first result from each fallback onto the primary
    merged = []
    for i, primary in enumerate(primary_list):
        fallbacks = []
        for fb_list in fallback_lists:
            # Match by index or closest available
            if i < len(fb_list):
                fallbacks.append(fb_list[i])
        merged.append(_merge_objects(primary, fallbacks, _STRUCTURAL_FIELDS))
    return merged


def _merge_line_items(results_per_provider: list[list[LineItem]], requested_fields: list[str]) -> list[LineItem]:
    """For each reporting period, for each requested field, take the first non-None
    value across providers. Periods are aligned by report_period date string."""
    primary_list = next((r for r in results_per_provider if r), [])
    if not primary_list:
        return []

    fallback_lists = [r for r in results_per_provider[1:] if r]
    if not fallback_lists:
        return primary_list

    # Build lookup: period_date → LineItem for each fallback provider
    fb_lookups: list[dict[str, LineItem]] = []
    for fb_list in fallback_lists:
        fb_lookups.append({item.report_period: item for item in fb_list})

    merged = []
    for primary in primary_list:
        fallbacks = [lookup.get(primary.report_period) for lookup in fb_lookups]
        fallbacks = [f for f in fallbacks if f is not None]
        merged.append(_merge_objects(primary, fallbacks, _STRUCTURAL_FIELDS))
    return merged


def _merge_news_sentiment(non_empty_results: list[list[CompanyNews]]) -> list[CompanyNews]:
    """Merge news from multiple providers.
    - Primary: first provider's list (all articles)
    - Fallbacks: patch sentiment onto articles with matching title
    - Append any articles unique to fallback providers
    """
    primary = list(non_empty_results[0])
    fallbacks = non_empty_results[1:]

    if not fallbacks:
        return primary

    # Build title → sentiment lookup from fallback providers
    sentiment_lookup: dict[str, str] = {}
    fallback_titles: dict[str, CompanyNews] = {}
    for fb_list in fallbacks:
        for article in fb_list:
            title_key = article.title.lower().strip()[:80]
            if article.sentiment and title_key not in sentiment_lookup:
                sentiment_lookup[title_key] = article.sentiment
            fallback_titles[title_key] = article

    # Patch sentiment onto primary articles where missing
    patched_primary: list[CompanyNews] = []
    primary_title_keys: set[str] = set()
    for article in primary:
        title_key = article.title.lower().strip()[:80]
        primary_title_keys.add(title_key)
        if article.sentiment is None and title_key in sentiment_lookup:
            article = article.model_copy(update={"sentiment": sentiment_lookup[title_key]})
            logger.debug("HYBRID: patched sentiment for article '%s…'", article.title[:40])
        patched_primary.append(article)

    # Append fallback-only articles
    for title_key, article in fallback_titles.items():
        if title_key not in primary_title_keys:
            patched_primary.append(article)

    return patched_primary
