# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
poetry install          # Install all dependencies
cp .env.example .env    # Then add your API keys
```

### Run the hedge fund (CLI)
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
poetry run python src/main.py --ticker AAPL --start-date 2024-01-01 --end-date 2024-03-01
poetry run python src/main.py --ticker AAPL --ollama   # Use local LLMs
```

### Run the backtester
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

### Run the web app (both frontend + backend)
```bash
./run.sh                          # Mac/Linux - starts everything
# OR manually:
cd app/backend && poetry run uvicorn main:app --reload   # Backend at :8000
cd app/frontend && npm run dev                            # Frontend at :5173
```

### Tests
```bash
poetry run pytest tests/
poetry run pytest tests/test_api_rate_limiting.py         # Single file
poetry run pytest --cov=src --cov-report=term-missing     # With coverage
```

### Formatting
```bash
poetry run black src/ app/
poetry run isort src/ app/
```

## Architecture

### LangGraph Execution Flow

The core system is a `StateGraph` (LangGraph) where each analyst runs as a parallel node:

```
start_node → [analyst agents in parallel] → risk_management_agent → portfolio_manager → END
```

The graph state (`src/graph/state.py`) uses `AgentState` (TypedDict) with three fields:
- `messages`: accumulated LangChain messages
- `data`: shared dict including `tickers`, `portfolio`, `start_date`, `end_date`, `analyst_signals`
- `metadata`: `show_reasoning`, `model_name`, `model_provider`

### Analyst Registry

`src/utils/analysts.py` is the **single source of truth** for all analyst agents. `ANALYST_CONFIG` maps each analyst key to its display name, description, investing style, agent function, and display order. To add a new analyst: add an entry here and implement the agent function in `src/agents/`.

### Data Layer

- `src/tools/api.py` — fetches financial data from `financialdatasets.ai` API. Has built-in rate-limit handling (linear backoff on 429s). Free data available for AAPL, GOOGL, MSFT, NVDA, TSLA without an API key.
- `src/data/cache.py` — in-memory caching to avoid redundant API calls within a run
- `src/data/models.py` — Pydantic models for financial data (prices, metrics, insider trades, etc.)

### Agent Structure

Each analyst agent in `src/agents/` follows the same pattern: receives `AgentState`, calls financial data tools, constructs an LLM prompt, and writes its signal to `state["data"]["analyst_signals"][ticker]`. Agents use `src/llm/models.py` to resolve the LLM based on `model_name`/`model_provider` from state metadata.

### Web App Backend

`app/backend/` is a FastAPI app with:
- SQLite/SQLAlchemy DB (Alembic migrations in `app/backend/alembic/`)
- Routes under `app/backend/routes/` (hedge fund runs, flow storage, API key management, Ollama, language models)
- Services under `app/backend/services/` that wrap `src/` logic and handle persistence
- Runs from `app/backend/main.py` with `uvicorn`

### LLM Providers

Supported providers (configured via `.env`): OpenAI, Anthropic, Groq, DeepSeek, Google Gemini, GigaChat, xAI, Ollama (local). Model lists are in `src/llm/api_models.json` and `src/llm/ollama_models.json`.
