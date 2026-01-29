import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import spacy  # pip install spacy && python -m spacy download en_core_web_sm

POLY_PATH = Path("polymarket_markets.jsonl")
KALSHI_PATH = Path("kalshi_markets.jsonl")  # or kalshi_markets_slim.jsonl
OUT_PATH = Path("events_TS_structured.jsonl")

# Optional: external domain lexicon
LEXICON_PATH = Path("data_aggregation/domain_lexicon.json")

NLP = spacy.load("en_core_web_sm")

# ---------- Load domain lexicon (optional) ----------

if LEXICON_PATH.exists():
    with LEXICON_PATH.open("r", encoding="utf-8") as f:
        DOMAIN_LEXICON: Dict[str, List[str]] = json.load(f)
else:
    DOMAIN_LEXICON = {}


# ---------- Time helpers ----------

def parse_ts(ts):
    if not ts:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
    except Exception:
        return None
    return None


def compute_tau(created, resolution, max_days: float = 365.0) -> Optional[float]:
    if not created or not resolution:
        return None
    dt_days = (resolution - created).total_seconds() / 86400.0
    tau = dt_days / max_days
    if tau < 0:
        tau = 0.0
    if tau > 1:
        tau = 1.0
    return tau


# ---------- Entity extraction ----------

def normalize_ent_text(text: str) -> str:
    return " ".join(text.split())


def extract_entities(text: str) -> List[str]:
    """
    spaCy NER + optional domain lexicon + optional ALLCAPS acronyms.
    Handles sports, politics, macro, etc. in a pluggable way.
    """
    if not text:
        return []

    ents = set()
    doc = NLP(text)

    # 1) spaCy NER entities
    for ent in doc.ents:
        # You can tweak labels here if needed
        if ent.label_ in {"PERSON", "ORG", "GPE", "NORP", "EVENT", "LOC", "PRODUCT"}:
            ents.add(normalize_ent_text(ent.text))

    lower_text = text.lower()

    # 2) Domain lexicon (macro, crypto, sports, whatever you add)
    for category, terms in DOMAIN_LEXICON.items():
        for term in terms:
            if term.lower() in lower_text:
                ents.add(normalize_ent_text(term))

    # 3) ALLCAPS acronyms (CPI, GDP, BTC, etc.)
    for token in doc:
        if token.text.isupper() and len(token.text) >= 3:
            ents.add(normalize_ent_text(token.text))

    return sorted(ents)


# ---------- psi (type), xi (threshold), pi (polarity) ----------

def infer_resolution_type(obj: Dict[str, Any]) -> str:
    mtype = (obj.get("type") or "").lower()

    if mtype in ("binary", "yes_no"):
        return "binary"
    if mtype in ("scalar", "numeric"):
        return "scalar"
    if mtype in ("categorical", "multi"):
        return "categorical"

    outcomes = obj.get("outcomes") or []
    if len(outcomes) == 2:
        return "binary"
    if len(outcomes) > 2:
        return "categorical"

    return "combo"


THRESHOLD_PATTERN = re.compile(
    r"""
    (?P<cmp>over|above|greater\ than|at\ least|under|below|less\ than|at\ most)?
    \s*
    (?P<value>[-+]?\d+(\.\d+)?)
    \s*
    (?P<unit>%|percent|percentage|bps|basis\ points|\$|dollars|points|pts|goals|runs)?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_threshold(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = THRESHOLD_PATTERN.search(text)
    if not m:
        return None

    raw = m.group(0).strip()
    val_str = m.group("value")
    unit = m.group("unit")
    try:
        value = float(val_str)
    except Exception:
        return None

    if unit:
        unit = unit.lower()

    return {"value": value, "unit": unit, "raw": raw}


POSITIVE_PHRASES = [
    "will",
    "at least",
    "or more",
    "or higher",
    "over",
    "above",
    "greater than",
    "increase",
    "rise",
    "gain",
    "win",
    "will be elected",
]

NEGATIVE_PHRASES = [
    "will not",
    "won't",
    "no",
    "under",
    "below",
    "less than",
    "at most",
    "decrease",
    "fall",
    "lose",
    "not be elected",
]


def infer_polarity(text: str) -> int:
    if not text:
        return 0
    t = text.lower()

    for phrase in NEGATIVE_PHRASES:
        if phrase in t:
            return -1

    for phrase in POSITIVE_PHRASES:
        if phrase in t:
            return +1

    return 0


# ---------- IO & TS construction ----------

def iter_markets(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def to_TS(obj: Dict[str, Any]) -> Dict[str, Any]:
    title = obj.get("title") or ""
    description = obj.get("description") or ""
    rules = obj.get("rules") or ""

    text_full = " ".join(part for part in [title, description, rules] if part)

    resolution_raw = obj.get("resolution_time")
    created_raw = obj.get("created_time")
    res_dt = parse_ts(resolution_raw)
    created_dt = parse_ts(created_raw)
    tau = compute_tau(created_dt, res_dt)

    psi = infer_resolution_type(obj)
    xi = extract_threshold(text_full)
    pi = infer_polarity(text_full)
    E = extract_entities(text_full)

    Lambda = {"pi": pi, "xi": xi, "psi": psi}

    return {
        "T": text_full,
        "S": {
            "E": E,
            "tau": tau,
            "Lambda": Lambda,
        },
    }


def main():
    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for m in iter_markets(KALSHI_PATH):
            fout.write(json.dumps(to_TS(m)) + "\n")
    print(f"Wrote", OUT_PATH)


if __name__ == "__main__":
    main()
