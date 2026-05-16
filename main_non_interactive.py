"""Non-interactive entry point for ai-hedge-fund.

Reads all configuration from ../../config.yaml instead of prompting the user.
Run from the ai-hedge-fund directory:

    cd external/ai-hedge-fund
    poetry run python main_non_interactive.py

Or pass --config to override the config file path:

    poetry run python main_non_interactive.py --config /path/to/config.yaml
"""

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime

from dateutil.relativedelta import relativedelta

import yaml
from colorama import Fore, Style, init
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.display import print_trading_output
from src.utils.llm import get_call_report
from src.utils.analysts import get_analyst_nodes
from src.utils.progress import progress

load_dotenv()
init(autoreset=True)

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _inject_api_keys(common: dict) -> None:
    mapping = {
        "DEEPSEEK_API_KEY": common.get("deepseek_api_key", ""),
        "OPENAI_API_KEY": common.get("openai_api_key", ""),
        "ANTHROPIC_API_KEY": common.get("anthropic_api_key", ""),
        "GOOGLE_API_KEY": common.get("google_api_key", ""),
        "GROQ_API_KEY": common.get("groq_api_key", ""),
    }
    for env_var, value in mapping.items():
        if value and not os.environ.get(env_var):
            os.environ[env_var] = value


def _resolve_dates(start: str, end: str) -> tuple[str, str]:
    final_end = end or datetime.now().strftime("%Y-%m-%d")
    if start:
        final_start = start
    else:
        end_dt = datetime.strptime(final_end, "%Y-%m-%d")
        final_start = (end_dt - relativedelta(months=3)).strftime("%Y-%m-%d")
    return final_start, final_end


def parse_hedge_fund_response(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None


def start(state: AgentState):
    return state


def create_workflow(selected_analysts=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    analyst_nodes = get_analyst_nodes()

    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())

    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)
    workflow.set_entry_point("start_node")
    return workflow


def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "deepseek-chat",
    model_provider: str = "DeepSeek",
):
    progress.start()
    try:
        workflow = create_workflow(selected_analysts if selected_analysts else None)
        agent = workflow.compile()

        final_state = agent.invoke(
            {
                "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            }
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        progress.stop()


def _print_llm_call_report() -> None:
    report = get_call_report()
    total = report["total"]
    by_agent = report["by_agent"]

    print(f"\n{Fore.WHITE}{Style.BRIGHT}LLM CALL REPORT{Style.RESET_ALL}")
    print("=" * 50)
    print(f"Total LLM calls: {Fore.CYAN}{total}{Style.RESET_ALL}\n")

    if by_agent:
        sorted_agents = sorted(by_agent.items(), key=lambda x: x[1], reverse=True)
        max_name_len = max(len(name) for name, _ in sorted_agents)
        for agent_name, count in sorted_agents:
            bar = "█" * count
            pct = count / total * 100 if total else 0
            print(
                f"  {Fore.CYAN}{agent_name:<{max_name_len}}{Style.RESET_ALL}"
                f"  {Fore.WHITE}{count:>3}{Style.RESET_ALL} calls"
                f"  ({pct:.0f}%)  {Fore.YELLOW}{bar}{Style.RESET_ALL}"
            )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ai-hedge-fund non-interactively from config.yaml")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers, overrides config (e.g. AAPL,MSFT)")
    parser.add_argument("--start-date", type=str, dest="start_date", help="Start date YYYY-MM-DD, overrides config")
    parser.add_argument("--end-date", type=str, dest="end_date", help="End date YYYY-MM-DD, overrides config")
    parser.add_argument(
        "--output-json", type=str, dest="output_json", default=None,
        help="Write result dict as JSON to this file path",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    _inject_api_keys(cfg.get("common", {}))

    hf = cfg.get("ai_hedge_fund", {})

    if args.tickers:
        tickers: list[str] = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = hf.get("tickers", [])
    if not tickers:
        print("Error: no tickers specified (set ai_hedge_fund.tickers in config.yaml or use --tickers)")
        sys.exit(1)

    start_date, end_date = _resolve_dates(
        args.start_date or hf.get("start_date", ""),
        args.end_date or hf.get("end_date", ""),
    )

    model_name: str = hf.get("model_name", "deepseek-chat")
    model_provider: str = hf.get("model_provider", "DeepSeek")
    initial_cash: float = float(hf.get("initial_cash", 100000.0))
    margin_requirement: float = float(hf.get("margin_requirement", 0.0))
    show_reasoning: bool = bool(hf.get("show_reasoning", False))
    selected_analysts: list[str] = hf.get("selected_analysts", [])

    print(f"{Fore.CYAN}ai-hedge-fund non-interactive run{Style.RESET_ALL}")
    print(f"  Tickers   : {', '.join(tickers)}")
    print(f"  Date range: {start_date} → {end_date}")
    print(f"  Model     : {model_provider} / {model_name}")
    print(f"  Analysts  : {len(selected_analysts) if selected_analysts else 'all'}")
    print()

    portfolio = {
        "cash": initial_cash,
        "margin_requirement": margin_requirement,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {"long": 0.0, "short": 0.0}
            for ticker in tickers
        },
    }

    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=model_provider,
    )
    print_trading_output(result)
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as _f:
            json.dump(result, _f, indent=2, default=str)
        print(f"[output] Result written to {args.output_json}")
    _print_llm_call_report()
