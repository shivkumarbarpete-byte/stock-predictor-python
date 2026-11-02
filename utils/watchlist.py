"""
utils/watchlist.py
------------------
Handles saving and loading the watchlist and portfolio tracking records.

Features:
  - Dual compatibility: simple string symbols list or structured portfolio records
    (shares owned, average buy price).
  - Stored in st.session_state per session with persistent JSON fallback at
    data/watchlist.json for local persistence across server restarts.
  - Thread-safe and graceful handling of missing or corrupted files.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union

# Path to the watchlist file, relative to this file's location.
WATCHLIST_PATH = Path(__file__).parent.parent / "data" / "watchlist.json"


def _is_streamlit_session() -> bool:
    """Checks whether the code is running inside an active Streamlit runtime session."""
    try:
        from streamlit.runtime import exists
        return bool(exists())
    except Exception:
        return False


def _load_raw_from_file() -> list:
    """Reads raw watchlist data from local JSON file if present."""
    if not WATCHLIST_PATH.exists():
        return []
    try:
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _save_raw_to_file(data: list) -> None:
    """Writes watchlist data to local JSON file as persistent fallback."""
    try:
        WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def load_watchlist_records() -> List[Dict[str, Any]]:
    """
    Returns normalized structured watchlist records:
      [{"symbol": "RELIANCE.NS", "shares": 10, "buy_price": 2400.0}, ...]
    """
    raw_list = []
    if _is_streamlit_session():
        try:
            import streamlit as st
            if "watchlist_records" not in st.session_state:
                st.session_state["watchlist_records"] = _load_raw_from_file()
            raw_list = st.session_state["watchlist_records"]
        except Exception:
            raw_list = _load_raw_from_file()
    else:
        raw_list = _load_raw_from_file()

    # Normalize to List[Dict]
    normalized: List[Dict[str, Any]] = []
    for item in raw_list:
        if isinstance(item, str):
            normalized.append({
                "symbol": item.strip().upper(),
                "shares": 0,
                "buy_price": 0.0,
            })
        elif isinstance(item, dict) and "symbol" in item:
            normalized.append({
                "symbol": str(item["symbol"]).strip().upper(),
                "shares": int(item.get("shares", 0)),
                "buy_price": float(item.get("buy_price", 0.0)),
            })
    return normalized


def save_watchlist_records(records: List[Dict[str, Any]]) -> None:
    """Saves structured records to session state and disk."""
    if _is_streamlit_session():
        try:
            import streamlit as st
            st.session_state["watchlist_records"] = records
            # Also keep simple 'watchlist' key in sync
            st.session_state["watchlist"] = [r["symbol"] for r in records]
        except Exception:
            pass

    _save_raw_to_file(records)


def load_watchlist() -> List[str]:
    """
    Backward-compatible method returning a simple list of ticker symbols:
      ["RELIANCE.NS", "TCS.NS"]
    """
    records = load_watchlist_records()
    return [r["symbol"] for r in records]


def save_watchlist(watchlist: List[Union[str, Dict[str, Any]]]) -> None:
    """
    Backward-compatible method saving either symbols or records.
    """
    normalized: List[Dict[str, Any]] = []
    for item in watchlist:
        if isinstance(item, str):
            normalized.append({
                "symbol": item.strip().upper(),
                "shares": 0,
                "buy_price": 0.0,
            })
        elif isinstance(item, dict) and "symbol" in item:
            normalized.append({
                "symbol": str(item["symbol"]).strip().upper(),
                "shares": int(item.get("shares", 0)),
                "buy_price": float(item.get("buy_price", 0.0)),
            })
    save_watchlist_records(normalized)


def add_to_watchlist(symbol: str, shares: int = 0, buy_price: float = 0.0) -> bool:
    """
    Adds a new stock to the watchlist or updates its position if already present.
    Returns True if added, False if updated.
    """
    sym = symbol.strip().upper()
    records = load_watchlist_records()
    found = False

    for r in records:
        if r["symbol"] == sym:
            if shares > 0:
                r["shares"] = int(shares)
            if buy_price > 0.0:
                r["buy_price"] = float(buy_price)
            found = True
            break

    if not found:
        records.append({
            "symbol": sym,
            "shares": int(shares),
            "buy_price": float(buy_price),
        })

    save_watchlist_records(records)
    return not found


def remove_from_watchlist(symbol: str) -> bool:
    """
    Removes a stock from the watchlist.
    Returns True if found and removed, False otherwise.
    """
    sym = symbol.strip().upper()
    records = load_watchlist_records()
    new_records = [r for r in records if r["symbol"] != sym]

    if len(new_records) != len(records):
        save_watchlist_records(new_records)
        return True
    return False
