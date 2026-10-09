"""
utils/watchlist.py
------------------
Handles saving and loading the watchlist (list of stock symbols).

The watchlist is stored as a simple JSON file: data/watchlist.json
It looks like this:  ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

Two functions:
  load_watchlist()  -> reads the file and returns a Python list
  save_watchlist()  -> takes a list and writes it back to the file

Why JSON and not a database?
  A watchlist is just a small list of strings. JSON is the simplest
  possible storage — no setup needed, human-readable, and easy to debug.
"""

import json
from pathlib import Path


# Path to the watchlist file, relative to THIS file's location.
# utils/watchlist.py lives in utils/, so we go up one level to reach
# the project root, then into data/.
WATCHLIST_PATH = Path(__file__).parent.parent / "data" / "watchlist.json"


def load_watchlist() -> list:
    """
    Reads the watchlist JSON file and returns a list of stock symbols.
    If the file doesn't exist yet, returns an empty list (safe default).
    """
    if not WATCHLIST_PATH.exists():
        # File not found — just start with an empty list
        return []

    with open(WATCHLIST_PATH, "r") as f:
        watchlist = json.load(f)

    return watchlist


def save_watchlist(watchlist: list) -> None:
    """
    Saves the given list of symbols back to the JSON file.
    Creates the data/ folder if it doesn't exist yet.

    watchlist: a Python list like ["RELIANCE.NS", "TCS.NS"]
    """
    # Make sure the data/ folder exists (create it if not)
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(WATCHLIST_PATH, "w") as f:
        # indent=2 makes the JSON file readable (not one long line)
        json.dump(watchlist, f, indent=2)
