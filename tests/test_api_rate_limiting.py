import os
import pytest
from unittest.mock import Mock, patch, call

from src.tools.api import _make_api_request, get_prices

# _make_api_request and its dependencies (requests, time, cache) now live in
# src.data.providers.financial_datasets — patch there, not in src.tools.api.

_MOD = "src.data.providers.financial_datasets"


class TestRateLimiting:
    """Test suite for API rate limiting functionality."""

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.requests.get")
    def test_handles_single_rate_limit(self, mock_get, mock_sleep):
        """Test that API retries once after a 429 and succeeds."""
        mock_429_response = Mock()
        mock_429_response.status_code = 429

        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"

        mock_get.side_effect = [mock_429_response, mock_200_response]

        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"

        result = _make_api_request(url, headers)

        assert result.status_code == 200
        assert result.text == "Success"
        assert mock_get.call_count == 2
        mock_get.assert_has_calls([call(url, headers=headers), call(url, headers=headers)])
        mock_sleep.assert_called_once_with(60)

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.requests.get")
    def test_handles_multiple_rate_limits(self, mock_get, mock_sleep):
        """Test that API retries multiple times after 429s."""
        mock_429_response = Mock()
        mock_429_response.status_code = 429

        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"

        mock_get.side_effect = [
            mock_429_response,
            mock_429_response,
            mock_429_response,
            mock_200_response,
        ]

        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"

        result = _make_api_request(url, headers)

        assert result.status_code == 200
        assert result.text == "Success"
        assert mock_get.call_count == 4
        assert mock_sleep.call_count == 3
        mock_sleep.assert_has_calls([call(60), call(90), call(120)])

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.requests.post")
    def test_handles_post_rate_limiting(self, mock_post, mock_sleep):
        """Test that POST requests handle rate limiting."""
        mock_429_response = Mock()
        mock_429_response.status_code = 429

        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"

        mock_post.side_effect = [mock_429_response, mock_200_response]

        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"
        json_data = {"test": "data"}

        result = _make_api_request(url, headers, method="POST", json_data=json_data)

        assert result.status_code == 200
        assert result.text == "Success"
        assert mock_post.call_count == 2
        mock_post.assert_has_calls([
            call(url, headers=headers, json=json_data),
            call(url, headers=headers, json=json_data),
        ])
        mock_sleep.assert_called_once_with(60)

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.requests.get")
    def test_ignores_other_errors(self, mock_get, mock_sleep):
        """Test that non-429 errors are returned without retrying."""
        mock_500_response = Mock()
        mock_500_response.status_code = 500
        mock_500_response.text = "Internal Server Error"

        mock_get.return_value = mock_500_response

        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"

        result = _make_api_request(url, headers)

        assert result.status_code == 500
        assert result.text == "Internal Server Error"
        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.requests.get")
    def test_normal_success_requests(self, mock_get, mock_sleep):
        """Test that successful requests return immediately without retry."""
        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"

        mock_get.return_value = mock_200_response

        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"

        result = _make_api_request(url, headers)

        assert result.status_code == 200
        assert result.text == "Success"
        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()

    @patch(f"{_MOD}._cache")
    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.requests.get")
    def test_full_integration(self, mock_get, mock_sleep, mock_cache):
        """Test that get_prices function properly handles rate limiting via the FD provider."""
        import src.data.config as cfg
        from src.data.config import DataSource

        mock_cache.get_prices.return_value = None

        mock_429_response = Mock()
        mock_429_response.status_code = 429

        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.json.return_value = {
            "ticker": "AAPL",
            "prices": [
                {
                    "time": "2024-01-01T00:00:00Z",
                    "open": 100.0,
                    "close": 101.0,
                    "high": 102.0,
                    "low": 99.0,
                    "volume": 1000,
                }
            ],
        }

        mock_get.side_effect = [mock_429_response, mock_200_response]

        original_source = cfg.PRICES_SOURCE
        cfg.PRICES_SOURCE = DataSource.FINANCIAL_DATASETS
        try:
            with patch.dict(os.environ, {"FINANCIAL_DATASETS_API_KEY": "test-key"}):
                result = get_prices("AAPL", "2024-01-01", "2024-01-02")
        finally:
            cfg.PRICES_SOURCE = original_source

        assert len(result) == 1
        assert result[0].open == 100.0
        assert result[0].close == 101.0
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(60)
        assert mock_cache.get_prices.call_count == 1
        assert mock_cache.set_prices.call_count == 1

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.requests.get")
    def test_max_retries_exceeded(self, mock_get, mock_sleep):
        """Test that function stops retrying after max_retries and returns final 429."""
        mock_429_response = Mock()
        mock_429_response.status_code = 429
        mock_429_response.text = "Too Many Requests"

        mock_get.return_value = mock_429_response

        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"

        result = _make_api_request(url, headers, max_retries=2)

        assert result.status_code == 429
        assert result.text == "Too Many Requests"
        assert mock_get.call_count == 3  # 1 initial + 2 retries
        assert mock_sleep.call_count == 2
        mock_sleep.assert_has_calls([call(60), call(90)])


if __name__ == "__main__":
    pytest.main([__file__])
