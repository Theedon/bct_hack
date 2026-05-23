"""
Demo script for the /recommend endpoint.

Runs four scenarios that demonstrate the key capabilities of the Task B recommendation agent:
  1. Cold Start        — new user with no review history
  2. Warm Start        — prolific user with an explicit natural-language query
  3. Multi-Turn        — same user refining their request across three conversational turns
  4. Nigerian Mode     — same warm-start user with Nigerian English contextualisation enabled

Usage:
    # Start the API server first:
    uv run uvicorn src.api:app --host 0.0.0.0 --port 8000

    # Then in another terminal:
    uv run python demos/recommend_demo.py
    uv run python demos/recommend_demo.py --base-url http://localhost:8000
"""

import argparse
import sys

import requests

# ---------------------------------------------------------------------------
# Shared user personas
# ---------------------------------------------------------------------------

# A prolific warm-start user with rich review history in the vectorstore
MARK = {
    "user_id": "-B-QEUESGWHPE_889WJaeg",
    "user_name": "Mark",
    "user_review_count": 1014,
    "average_stars": 3.41,
    "user_elite_count": 12,
    "user_fans": 200,
}

# A brand-new user — ID not present in the vectorstore → triggers cold-start path
AMARA = {
    "user_id": "demo-cold-start-001",
    "user_name": "Amara",
    "user_review_count": 0,
    "average_stars": 0.0,
    "user_elite_count": 0,
    "user_fans": 0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_server(base_url: str) -> None:
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        print(f"ERROR: Could not reach server at {base_url}/health — {exc}")
        print("Start the server with:")
        print("  uv run uvicorn src.api:app --host 0.0.0.0 --port 8000")
        sys.exit(1)


def _recommend(base_url: str, payload: dict) -> dict:
    resp = requests.post(f"{base_url}/recommend", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _print_result(result: dict, k: int = 5) -> None:
    manifesto = result["user_manifesto"].replace("\n", " ")
    print(f"  cold_start : {result['cold_start']}")
    print(f"  manifesto  : {manifesto[:160]}{'...' if len(manifesto) > 160 else ''}")
    print()
    for i, rec in enumerate(result["recommendations"][:k], 1):
        location = ", ".join(
            filter(None, [rec.get("biz_city", ""), rec.get("biz_state", "")])
        )
        print(f"  [{i}] {rec['biz_name']}  ({location})")
        print(f"       Categories : {rec['categories']}")
        print(f"       Score      : {rec['score']}")
        print(f"       Rationale  : {rec['rationale']}")
        print()


def _divider(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_cold_start(base_url: str) -> None:
    _divider("SCENARIO 1 — Cold Start (no review history)")
    print("User: Amara — brand new, zero reviews, not in the vectorstore.\n")

    result = _recommend(base_url, {**AMARA, "k": 5})
    _print_result(result)


def scenario_warm_start(base_url: str) -> None:
    _divider("SCENARIO 2 — Warm Start with Explicit Query")
    query = "quiet casual dinner, not too expensive"
    print(f"User: Mark — 1,014 reviews, 12 Elite years.")
    print(f'Query: "{query}"\n')

    result = _recommend(base_url, {**MARK, "query": query, "k": 5})
    _print_result(result)


def scenario_multi_turn(base_url: str) -> None:
    _divider("SCENARIO 3 — Multi-Turn Conversational Refinement")
    print("User: Mark — three successive turns, each refining the request.\n")

    messages: list[dict] = []

    # Turn 1 — broad general request
    print("--- Turn 1: general request (no query) ---\n")
    r1 = _recommend(base_url, {**MARK, "k": 5, "messages": messages})
    _print_result(r1)

    # Accumulate turn 1 exchange into message history
    messages.append({"role": "assistant", "content": r1["user_manifesto"]})
    messages.append(
        {"role": "user", "content": "actually I want something with seafood"}
    )

    # Turn 2 — refine toward seafood
    print("--- Turn 2: 'actually I want something with seafood' ---\n")
    r2 = _recommend(
        base_url,
        {
            **MARK,
            "query": "something with seafood",
            "k": 5,
            "messages": messages,
        },
    )
    _print_result(r2)

    # Accumulate turn 2
    messages.append({"role": "assistant", "content": r2["user_manifesto"]})
    messages.append({"role": "user", "content": "with outdoor seating if possible"})

    # Turn 3 — add outdoor seating preference
    print("--- Turn 3: 'with outdoor seating if possible' ---\n")
    r3 = _recommend(
        base_url,
        {
            **MARK,
            "query": "seafood with outdoor seating",
            "k": 5,
            "messages": messages,
        },
    )
    _print_result(r3)


def scenario_nigerian_mode(base_url: str) -> None:
    _divider("SCENARIO 4 — Nigerian Mode (nigerian_mode=true)")
    query = "quiet casual dinner, not too expensive"
    print(f"User: Mark — same query as Scenario 2, Nigerian contextualisation enabled.")
    print(f'Query: "{query}"\n')

    result = _recommend(
        base_url, {**MARK, "query": query, "k": 5, "nigerian_mode": True}
    )
    _print_result(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommendation agent demo")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running API (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    _check_server(args.base_url)
    print(f"Server healthy at {args.base_url}. Running 4 scenarios...\n")

    scenario_cold_start(args.base_url)
    scenario_warm_start(args.base_url)
    scenario_multi_turn(args.base_url)
    scenario_nigerian_mode(args.base_url)

    print("=" * 70)
    print("  All scenarios complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
