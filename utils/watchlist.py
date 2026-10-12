"""
utils/watchlist.py
------------------
Handles saving and loading the watchlist (list of stock symbols).

The watchlist is stored in st.session_state per browser session,
with local JSON (data/watchlist.json) as a fallback for local persistence.
"""

import json
from pathlib import Path


# Path to the watchlist file, relative to this file's location.
WATCHLIST_PATH = Path(__file__).parent.parent / "data" / "watchlist.json"


def _is_streamlit_session() -> bool:
    """Checks whether the code is running inside an active Streamlit runtime session."""
    try:
        from streamlit.runtime import exists
        return bool(exists())
    except Exception:
        return False


def _load_from_file() -> list:
    """Reads watchlist from local JSON file if present."""
    if not WATCHLIST_PATH.exists():
        return []
    try:
        with open(WATCHLIST_PATH, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _save_to_file(watchlist: list) -> None:
    """Writes watchlist to local JSON file as a fallback."""
    try:
        WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WATCHLIST_PATH, "w") as f:
            json.dump(watchlist, f, indent=2)
    except Exception:
        pass


def load_watchlist() -> list:
    """
    Reads the watchlist from st.session_state if in a Streamlit session.
    Falls back to the local JSON file on initial load or outside Streamlit.
    """
    if _is_streamlit_session():
        try:
            import streamlit as st
            if "watchlist" not in st.session_state:
                st.session_state["watchlist"] = _load_from_file()
            return list(st.session_state["watchlist"])
        except Exception:
            pass

    return _load_from_file()


def save_watchlist(watchlist: list) -> None:
    """
    Saves the watchlist to st.session_state for the current browser session.
    Also writes to data/watchlist.json as fallback for local persistence.
    """
    clean_list = list(watchlist)

    if _is_streamlit_session():
        try:
            import streamlit as st
            st.session_state["watchlist"] = clean_list
        except Exception:
            pass

    _save_to_file(clean_list)
