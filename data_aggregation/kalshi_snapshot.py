import json
from datetime import datetime, timezone
from pathlib import Path
import requests
import time

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

def _parse_ts(ts):
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        if isinstance(ts, str):
            return (
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                .astimezone(timezone.utc)
                .isoformat()
            )
    except Exception:
        return None
    return None

import requests
import time

def fetch_paginated(endpoint: str, params: dict, key: str, *,
                    out_jsonl_path: str | None = None,
                    resume_state_path: str | None = None,
                    max_retries: int = 12,
                    base_sleep: float = 0.25):
    """
    Paginated GET with:
      - cursor pagination
      - 429 backoff + Retry-After support
      - optional streaming to JSONL
      - optional cursor checkpointing for resume
    """
    out = [] if out_jsonl_path is None else None

    out_fp = None
    if out_jsonl_path:
        Path(out_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        out_fp = open(out_jsonl_path, "a", encoding="utf-8")

    # resume cursor if exists
    cursor = None
    if resume_state_path and Path(resume_state_path).exists():
        try:
            st = json.loads(Path(resume_state_path).read_text(encoding="utf-8"))
            cursor = st.get("cursor")
            print(f"[resume] Loaded cursor from {resume_state_path}: {cursor}")
        except Exception:
            pass

    def save_cursor(cur):
        if resume_state_path:
            Path(resume_state_path).write_text(json.dumps({"cursor": cur}), encoding="utf-8")

    retries = 0

    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor

        url = f"{BASE_URL}/{endpoint}"

        try:
            resp = requests.get(url, params=p, timeout=30)
        except requests.RequestException as e:
            # network hiccup -> backoff
            wait = min(60, base_sleep * (2 ** retries))
            print(f"[network] {e} | sleeping {wait:.1f}s")
            time.sleep(wait)
            retries += 1
            if retries > max_retries:
                raise
            continue

        # Handle rate limit
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = float(retry_after)
                except:
                    wait = min(60, base_sleep * (2 ** retries))
            else:
                wait = min(60, base_sleep * (2 ** retries))

            print(f"[429] Rate limited. Waiting {wait:.1f}s then retrying. URL={resp.url}")
            time.sleep(wait)
            retries += 1
            if retries > max_retries:
                # print body for debugging
                try:
                    print("Body JSON:", resp.json())
                except:
                    print("Body TEXT:", resp.text[:1000])
                resp.raise_for_status()
            continue

        # Other errors
        if resp.status_code >= 400:
            print("\n=== HTTP ERROR ===")
            print("URL:", resp.url)
            print("Status:", resp.status_code)
            try:
                print("Body JSON:", resp.json())
            except:
                print("Body TEXT:", resp.text[:2000])
            resp.raise_for_status()

        # Success -> reset retry counter
        retries = 0

        data = resp.json()
        page = data.get(key, [])
        cursor = data.get("cursor")

        # stream to disk or accumulate
        if out_fp:
            for item in page:
                out_fp.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            out.extend(page)

        # checkpoint cursor every page
        save_cursor(cursor)

        if not cursor:
            break

        # polite pacing (important even without 429)
        time.sleep(base_sleep)

    if out_fp:
        out_fp.close()

    return out

def fetch_markets_by_status(status):
    return fetch_paginated(
        endpoint="markets",
        params={"limit": 100, "status": status},
        key="markets",
        out_jsonl_path=f"data/raw/kalshi_markets_{status}.jsonl",
        resume_state_path=f"data/raw/kalshi_markets_{status}.cursor.json",
        base_sleep=0.35
    )

def fetch_events_to_disk():
    return fetch_paginated(
        endpoint="events",
        params={"limit": 100},        # reduce limit to reduce pressure
        key="events",
        out_jsonl_path="data/raw/kalshi_events.jsonl",
        resume_state_path="data/raw/kalshi_events.cursor.json",
        base_sleep=0.35               # slower = fewer 429s
    )


def build_event_index(events: list[dict]) -> dict:
    idx = {}
    for e in events:
        et = e.get("event_ticker") or e.get("ticker") or e.get("id")
        if et:
            idx[et] = e
    return idx

def normalize_market(m: dict, event_obj: dict | None) -> dict:
    ticker = m.get("ticker")
    event_ticker = m.get("event_ticker") or m.get("event_ticker_id")

    # event-level text (best effort)
    ev_title = (event_obj or {}).get("title") or ""
    ev_sub = (event_obj or {}).get("subtitle") or ""
    ev_desc = (event_obj or {}).get("description") or ""
    ev_rules = (event_obj or {}).get("rules") or ""

    # market-level text
    mk_title = m.get("title") or ""
    mk_desc = m.get("description") or ""
    mk_rules = m.get("rules") or ""

    # Create a richer "T"
    # (You can refine this later, but it ensures non-empty text)
    T_parts = [ev_title, ev_sub, ev_desc, ev_rules, mk_title, mk_desc, mk_rules]
    T = "\n".join([p for p in T_parts if isinstance(p, str) and p.strip()])

    status = (m.get("status") or "").lower()

    # Times (don’t fake resolution_time)
    listed_time = _parse_ts(m.get("listed_time") or m.get("listed_date") or m.get("created_time"))
    open_time = _parse_ts(m.get("open_time"))
    close_time = _parse_ts(m.get("close_time") or m.get("close_date") or m.get("expiration_time"))
    settled_time = _parse_ts(m.get("settled_time") or m.get("settlement_time") or m.get("resolved_time"))

    # Market type: keep raw + a mapped type placeholder
    raw_type = m.get("market_type") or m.get("type")
    mapped_type = raw_type or "unknown"  # refine mapping later

    return {
        "platform": "kalshi",
        "market_id": ticker,
        "event_id": event_ticker,
        "title": mk_title,             # keep market title
        "event_title": ev_title,       # add event title
        "text": T,                     # HyTE T should come from here
        "status": status,
        "times": {
            "listed_time": listed_time,
            "open_time": open_time,
            "close_time": close_time,
            "settled_time": settled_time,
        },
        "type_raw": raw_type,
        "type": mapped_type,
        "outcomes": m.get("outcomes"),
        "extra": {"raw_market": m, "raw_event": event_obj},
    }

def main():
    # 1) Fetch events once (or later restrict)
    #events = fetch_events_to_disk()
    #print("Done events")
    #ev_idx = build_event_index(events)

    # 2) Fetch markets across statuses
    all_markets = []
    for st in [ "closed", "settled"]:
        print(f"▶ Fetching markets status={st}")
        fetch_markets_by_status(st)   # writes to data/raw/kalshi_markets_{st}.jsonl
        print(f"✅ Done markets status={st}")

    print("✅ All raw downloads complete.")
    print("Next: run a separate script to merge events+markets into an event-centric dataset.")

if __name__ == "__main__":
    main()