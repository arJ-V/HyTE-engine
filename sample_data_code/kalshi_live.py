# kalshi_live.py
import asyncio
import json
import os
from datetime import datetime, timezone

import websockets

KALSHI_WS_URL = "wss://api.kalshi.com/trade-api/ws/v2"  # example placeholder

KALSHI_API_KEY = os.environ.get("KALSHI_API_KEY")
KALSHI_API_SECRET = os.environ.get("KALSHI_API_SECRET")

# The exact auth message + subscription format: check Kalshi WS docs
async def kalshi_ws_client(tracked_market_ids: list[str]):
    async with websockets.connect(KALSHI_WS_URL) as ws:
        auth_msg = {
            "type": "auth",
            "api_key": KALSHI_API_KEY,
            "api_secret": KALSHI_API_SECRET,
        }
        await ws.send(json.dumps(auth_msg))

        sub_msg = {
            "type": "subscribe",
            "channels": [
                {"name": "markets", "market_id": mid}
                for mid in tracked_market_ids
            ],
        }
        await ws.send(json.dumps(sub_msg))

        async for msg in ws:
            data = json.loads(msg)
            now_iso = datetime.now(timezone.utc).isoformat()

            if data.get("type") == "market_update":
                m = data["market"]
                snapshot = {
                    "source": "kalshi",
                    "market_id": m["id"],
                    "title": m.get("title") or "",
                    "description": m.get("rules") or "",
                    "category": m.get("category"),
                    "tags": m.get("tags", []),
                    "resolution_time": m.get("close_ts"),
                    "status": m.get("state", "").lower(),
                    "outcomes": [
                        {"name": "YES", "last_price": m.get("yes_price"), "volume_24h": None},
                        {"name": "NO", "last_price": m.get("no_price"), "volume_24h": None},
                    ],
                    "last_updated": now_iso,
                }
                # Here is where you would call your transformer on `snapshot`
                print("KALSHI LIVE UPDATE:", json.dumps(snapshot))

async def main():
    tracked = ["FED_2025_09_RATE", "CPI_US_2025_07"]  # example tickers or ids
    await kalshi_ws_client(tracked)

if __name__ == "__main__":
    asyncio.run(main())
