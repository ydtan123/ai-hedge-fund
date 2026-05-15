"""Unit tests for the provider abstraction layer.

Tests cover:
- HybridProvider fallback logic
- Field-by-field merge helpers
- Dispatcher routing in tools/api.py
- prices_to_df utility
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import date

from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)
from src.data.providers.hybrid import (
    HybridProvider,
    _merge_financial_metrics,
    _merge_line_items,
    _merge_news_sentiment,
    _merge_objects,
    _STRUCTURAL_FIELDS,
)
from src.tools.api import prices_to_df


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_price(**kw) -> Price:
    defaults = dict(time="2024-01-01T00:00:00Z", open=100.0, close=101.0, high=102.0, low=99.0, volume=1000)
    defaults.update(kw)
    return Price(**defaults)


def _make_metrics(**kw) -> FinancialMetrics:
    # Build defaults from the model: all float|None fields default to None,
    # required string fields get placeholder values.
    defaults: dict = {"ticker": "MSFT", "report_period": "2024-01-01", "period": "ttm", "currency": "USD"}
    for name, field in FinancialMetrics.model_fields.items():
        if name in defaults:
            continue
        ann = str(field.annotation)
        defaults[name] = None  # all numeric fields are float | None
    defaults.update(kw)
    return FinancialMetrics(**defaults)


def _make_line_item(**kw) -> LineItem:
    defaults = dict(ticker="MSFT", report_period="2024-01-01", period="ttm", currency="USD")
    defaults.update(kw)
    return LineItem(**defaults)


def _make_news(**kw) -> CompanyNews:
    defaults = dict(
        ticker="MSFT",
        title="Test article",
        author=None,
        source="Test Source",
        date="2024-01-01T00:00:00Z",
        url="https://example.com",
        summary=None,
        sentiment=None,
    )
    defaults.update(kw)
    return CompanyNews(**defaults)


def _mock_provider(prices=None, metrics=None, line_items=None, trades=None, news=None, market_cap=None):
    p = MagicMock()
    p.get_prices.return_value = prices or []
    p.get_financial_metrics.return_value = metrics or []
    p.search_line_items.return_value = line_items or []
    p.get_insider_trades.return_value = trades or []
    p.get_company_news.return_value = news or []
    p.get_market_cap.return_value = market_cap
    return p


# ── HybridProvider: fallback logic ───────────────────────────────────────────


class TestHybridFallback:
    def test_prices_returns_first_non_empty(self):
        price = _make_price()
        p1 = _mock_provider(prices=[])
        p2 = _mock_provider(prices=[price])
        h = HybridProvider([p1, p2])
        result = h.get_prices("MSFT", "2024-01-01", "2024-12-31")
        assert result == [price]
        p1.get_prices.assert_called_once()
        p2.get_prices.assert_called_once()

    def test_prices_returns_yf_when_yf_has_data(self):
        price = _make_price()
        p1 = _mock_provider(prices=[price])
        p2 = _mock_provider(prices=[_make_price(open=999.0)])
        h = HybridProvider([p1, p2])
        result = h.get_prices("MSFT", "2024-01-01", "2024-12-31")
        assert result == [price]
        p2.get_prices.assert_not_called()

    def test_market_cap_returns_first_non_none(self):
        p1 = _mock_provider(market_cap=None)
        p2 = _mock_provider(market_cap=1_000_000.0)
        h = HybridProvider([p1, p2])
        result = h.get_market_cap("MSFT", "2024-01-01")
        assert result == 1_000_000.0

    def test_market_cap_returns_p1_when_available(self):
        p1 = _mock_provider(market_cap=2_000_000.0)
        p2 = _mock_provider(market_cap=999.0)
        h = HybridProvider([p1, p2])
        result = h.get_market_cap("MSFT", "2024-01-01")
        assert result == 2_000_000.0
        p2.get_market_cap.assert_not_called()

    def test_insider_trades_returns_first_non_empty(self):
        trade = InsiderTrade(
            ticker="MSFT",
            issuer=None,
            name="CEO",
            title=None,
            is_board_director=None,
            transaction_date="2024-01-01",
            transaction_shares=100.0,
            transaction_price_per_share=300.0,
            transaction_value=30_000.0,
            transaction_acquired_disposed=None,
            shares_owned_before_transaction=None,
            shares_owned_after_transaction=None,
            security_title=None,
            filing_date="2024-01-05",
        )
        p1 = _mock_provider(trades=[])
        p2 = _mock_provider(trades=[trade])
        h = HybridProvider([p1, p2])
        result = h.get_insider_trades("MSFT", "2024-12-31")
        assert result == [trade]

    def test_empty_chain_raises(self):
        with pytest.raises(ValueError):
            HybridProvider([])


# ── Merge helpers ─────────────────────────────────────────────────────────────


class TestMergeFinancialMetrics:
    def test_patches_none_fields_from_fallback(self):
        primary = _make_metrics(net_margin=0.25, return_on_equity=None)
        fallback = _make_metrics(net_margin=0.99, return_on_equity=0.30)
        result = _merge_financial_metrics([[primary], [fallback]])
        assert len(result) == 1
        # Primary value preserved
        assert result[0].net_margin == 0.25
        # None field filled from fallback
        assert result[0].return_on_equity == 0.30

    def test_returns_primary_when_no_fallback(self):
        primary = _make_metrics(net_margin=0.25)
        result = _merge_financial_metrics([[primary]])
        assert result == [primary]

    def test_returns_empty_when_all_empty(self):
        result = _merge_financial_metrics([[], []])
        assert result == []

    def test_structural_fields_not_overwritten(self):
        primary = _make_metrics(ticker="MSFT", report_period="2024-01-01")
        fallback = _make_metrics(ticker="AAPL", report_period="2023-01-01")
        result = _merge_financial_metrics([[primary], [fallback]])
        assert result[0].ticker == "MSFT"
        assert result[0].report_period == "2024-01-01"


class TestMergeLineItems:
    def test_fills_missing_fields_from_fallback(self):
        primary = _make_line_item(revenue=1_000_000.0, net_income=None)
        fallback = _make_line_item(revenue=999.0, net_income=200_000.0)
        result = _merge_line_items([[primary], [fallback]], requested_fields=["revenue", "net_income"])
        assert len(result) == 1
        assert result[0].revenue == 1_000_000.0  # primary kept
        assert result[0].net_income == 200_000.0  # filled from fallback

    def test_aligns_by_report_period(self):
        primary_a = _make_line_item(report_period="2024-01-01", revenue=100.0, net_income=None)
        primary_b = _make_line_item(report_period="2023-01-01", revenue=90.0, net_income=None)
        fallback_a = _make_line_item(report_period="2024-01-01", revenue=0.0, net_income=10.0)
        fallback_b = _make_line_item(report_period="2023-01-01", revenue=0.0, net_income=9.0)
        result = _merge_line_items([[primary_a, primary_b], [fallback_a, fallback_b]], requested_fields=["revenue", "net_income"])
        by_period = {r.report_period: r for r in result}
        assert by_period["2024-01-01"].net_income == 10.0
        assert by_period["2023-01-01"].net_income == 9.0

    def test_returns_primary_when_no_fallback(self):
        primary = _make_line_item(revenue=1_000.0)
        result = _merge_line_items([[primary]], requested_fields=["revenue"])
        assert result == [primary]


class TestMergeNewsSentiment:
    def test_patches_missing_sentiment(self):
        primary = _make_news(title="Big News", sentiment=None)
        fallback = _make_news(title="Big News", sentiment="positive")
        result = _merge_news_sentiment([[primary], [fallback]])
        assert len(result) == 1
        assert result[0].sentiment == "positive"

    def test_does_not_overwrite_existing_sentiment(self):
        primary = _make_news(title="Big News", sentiment="negative")
        fallback = _make_news(title="Big News", sentiment="positive")
        result = _merge_news_sentiment([[primary], [fallback]])
        assert result[0].sentiment == "negative"

    def test_appends_fallback_only_articles(self):
        primary = _make_news(title="Primary Article")
        fallback_extra = _make_news(title="Exclusive AV Article", sentiment="positive")
        result = _merge_news_sentiment([[primary], [fallback_extra]])
        titles = [a.title for a in result]
        assert "Primary Article" in titles
        assert "Exclusive AV Article" in titles

    def test_returns_primary_when_no_fallbacks(self):
        primary = [_make_news(title="Only Article")]
        result = _merge_news_sentiment([primary])
        assert result == primary


# ── Dispatcher routing ────────────────────────────────────────────────────────


class TestDispatcher:
    def test_dispatcher_routes_to_financial_datasets(self):
        import src.data.config as cfg
        from src.data.config import DataSource
        from src.tools.api import _build_provider
        from src.data.providers.financial_datasets import FinancialDatasetsProvider

        provider = _build_provider(DataSource.FINANCIAL_DATASETS)
        assert isinstance(provider, FinancialDatasetsProvider)

    def test_dispatcher_routes_to_yahoo_finance(self):
        from src.data.config import DataSource
        from src.tools.api import _build_provider
        from src.data.providers.yahoo_finance import YahooFinanceProvider

        provider = _build_provider(DataSource.YAHOO_FINANCE)
        assert isinstance(provider, YahooFinanceProvider)

    def test_dispatcher_routes_to_alpha_vantage(self):
        from src.data.config import DataSource
        from src.tools.api import _build_provider
        from src.data.providers.alpha_vantage import AlphaVantageProvider

        provider = _build_provider(DataSource.ALPHA_VANTAGE)
        assert isinstance(provider, AlphaVantageProvider)

    def test_dispatcher_builds_hybrid_chain(self):
        from src.data.config import DataSource
        from src.tools.api import _build_provider
        from src.data.providers.hybrid import HybridProvider
        import src.data.config as cfg

        original_chain = cfg.HYBRID_CHAIN[:]
        cfg.HYBRID_CHAIN = [DataSource.YAHOO_FINANCE, DataSource.ALPHA_VANTAGE]
        try:
            provider = _build_provider(DataSource.HYBRID)
            assert isinstance(provider, HybridProvider)
        finally:
            cfg.HYBRID_CHAIN = original_chain

    def test_dispatcher_raises_on_unknown_source(self):
        from src.tools.api import _build_provider
        with pytest.raises(ValueError, match="Unknown DataSource"):
            _build_provider("invalid_source")


# ── prices_to_df utility ─────────────────────────────────────────────────────


class TestPricesToDf:
    def test_converts_price_list_to_dataframe(self):
        prices = [
            _make_price(time="2024-01-01T00:00:00Z", open=100.0, close=101.0),
            _make_price(time="2024-01-02T00:00:00Z", open=102.0, close=103.0),
        ]
        df = prices_to_df(prices)
        assert len(df) == 2
        assert "open" in df.columns
        assert "close" in df.columns
        assert df.index.name == "Date"

    def test_returns_empty_df_for_empty_list(self):
        df = prices_to_df([])
        assert df.empty

    def test_sorts_by_date_ascending(self):
        prices = [
            _make_price(time="2024-01-03T00:00:00Z", close=103.0),
            _make_price(time="2024-01-01T00:00:00Z", close=101.0),
            _make_price(time="2024-01-02T00:00:00Z", close=102.0),
        ]
        df = prices_to_df(prices)
        assert list(df["close"]) == [101.0, 102.0, 103.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
