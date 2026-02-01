import asyncio
import httpx
import json
from datetime import datetime, timezone
from pathlib import Path

GAMMA_BASE = "https://gamma-api.polymarket.com"


async def fetch_all_polymarket_markets(active_only: bool = True) -> list[dict]:
    """
    Fetch Polymarket markets from Gamma, optionally only active ones.
    Handles both list and {markets: [...]} response shapes.
    """
    params = {
        "limit": 1000,   # max per Gamma docs
        "page": 1
    }
    if active_only:
        # Some Gamma deployments use `closed=false` as the filter flag
        params["closed"] = "false"

    markets: list[dict] = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            resp = await client.get(f"{GAMMA_BASE}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()

            # If data is a list, that's already the markets
            if isinstance(data, list):
                page_markets = data
                has_more = False
            else:
                # dict shape, possibly with 'markets' key
                page_markets = data.get("markets") or data.get("data") or []
                has_more = bool(data.get("hasMore") or data.get("has_more"))

            if not page_markets:
                break

            markets.extend(page_markets)

            if not has_more:
                break

            params["page"] += 1

    return markets


def normalize_polymarket_market(m: dict) -> dict:
    """
    Map a Gamma market dict into our canonical event JSON.
    """
    market_id = m.get("id")
    event_id = m.get("eventId") or m.get("event_id")
    title = m.get("question") or m.get("title") or ""
    description = m.get("description") or ""
    rules = m.get("ancillaryData") or ""

    closed = m.get("closed")
    resolved = m.get("resolved")
    if resolved:
        status = "resolved"
    elif closed:
        status = "closed"
    else:
        status = "open"

    def parse_ts(ts):
        if not ts:
            return None
        try:
            # Try ISO first
            if isinstance(ts, str):
                return (
                    datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    .astimezone(timezone.utc)
                    .isoformat()
                )
            # Fallback: unix seconds
            if isinstance(ts, (int, float)):
                return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception:
            return None

    resolution_time = parse_ts(m.get("endDate") or m.get("end_date"))
    created_time = parse_ts(m.get("createdAt") or m.get("created_at"))

    outcome_type = (m.get("outcomeType") or "").lower()
    if outcome_type in ("binary", "yes_no"):
        mtype = "binary"
    elif outcome_type in ("scalar", "numeric"):
        mtype = "scalar"
    elif outcome_type in ("categorical", "multi"):
        mtype = "categorical"
    else:
        mtype = "combo"

    outcomes_raw = m.get("outcomes") or []
    prices_raw = m.get("outcomePrices") or m.get("prices") or []

    outcomes = []
    current_prices = {}

    for i, out in enumerate(outcomes_raw):
        if isinstance(out, str):
            name = out
            token_id = None
        else:
            name = out.get("name") or out.get("outcome") or f"outcome_{i}"
            token_id = out.get("tokenId") or out.get("token_id")

        price = None
        if isinstance(prices_raw, list) and i < len(prices_raw):
            price = prices_raw[i]
        elif isinstance(prices_raw, dict):
            price = prices_raw.get(name)

        outcomes.append({"name": name, "token_id": token_id})
        if price is not None:
            current_prices[name] = price

    return {
        "platform": "polymarket",
        "market_id": market_id,
        "event_id": event_id,
        "title": title,
        "description": description,
        "rules": rules,
        "status": status,
        "resolution_time": resolution_time,
        "created_time": created_time,
        "type": mtype,
        "outcomes": outcomes,
        "current_prices": current_prices,
        "extra": {
            "raw": m
        }
    }


async def main():
    print("Fetching Polymarket markets from Gamma…")
    raw_markets = await fetch_all_polymarket_markets(active_only=True)
    print(f"Fetched {len(raw_markets)} markets")

    out_path = Path("polymarket_markets.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for m in raw_markets:
            norm = normalize_polymarket_market(m)
            f.write(json.dumps(norm) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
