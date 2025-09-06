# data_source.py
from __future__ import annotations
from typing import Optional, Tuple, List, Callable
from pathlib import Path
import time

import pandas as pd
import streamlit as st
from httpx import HTTPStatusError

from openf1_client import (
    sessions,            # list sessions for a meeting_key
    sessions_by_year,    # list all sessions for a season
    session_result,      # results for a given session_key
    starting_grid,       # starting grid for a Race session_key
    drivers as of1_drivers,
)

# ---------------- utils ----------------

def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _with_backoff(fn: Callable, *args, **kwargs):
    """Retry on 429 with exponential backoff."""
    delay = 1.0
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < 5:
                time.sleep(delay)
                delay *= 2
                continue
            raise

def _read_or_fetch(path: Path, fetch_fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Disk cache first; if missing, fetch and write."""
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    df = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return df

# ---------------- Streamlit cache wrappers (memory cache) ----------------

@st.cache_data(ttl=3600)
def _cached_sessions_by_year(year: int) -> pd.DataFrame:
    return sessions_by_year(year)

@st.cache_data(ttl=3600)
def _cached_sessions(meeting_key: int) -> pd.DataFrame:
    return sessions(meeting_key)

@st.cache_data(ttl=3600)
def _cached_session_result(session_key: int) -> pd.DataFrame:
    return session_result(session_key)

@st.cache_data(ttl=3600)
def _cached_starting_grid(session_key: int) -> pd.DataFrame:
    return starting_grid(session_key)

@st.cache_data(ttl=86400)
def _cached_drivers_for_meeting(meeting_key: int) -> pd.DataFrame:
    return of1_drivers(meeting_key=meeting_key)

# ---------------- Public API for app.py ----------------

def races_for_season(season: int) -> pd.DataFrame:
    """
    Return one row per Race session for a season with a clean label: 'R## – Grand Prix Name'.
    Columns: meeting_key, session_key, label, round
    """
    s = _with_backoff(_cached_sessions_by_year, season)
    if s.empty:
        return pd.DataFrame(columns=["meeting_key", "session_key", "label", "round"])

    name_col = "session_name" if "session_name" in s.columns else None
    races = s[s[name_col].astype(str).str.lower().eq("race")] if name_col else s.copy()
    if races.empty:
        return pd.DataFrame(columns=["meeting_key", "session_key", "label", "round"])

    # sort chronologically by any known timestamp column
    date_col = _first_present(races, ["date_start", "session_start_time", "date"])
    if date_col:
        races = races.sort_values(date_col)

    races = races.reset_index(drop=True)
    races["round"] = races.index + 1

    meeting_name_col = _first_present(
        races,
        ["meeting_name", "meeting", "meeting_official_name", "circuit_short_name", "meeting_name_official"]
    )

    def mk_label(row: pd.Series) -> str:
        rno = f"R{int(row['round']):02d}"
        name = str(row.get(meeting_name_col, "") or "").strip() if meeting_name_col else ""
        return f"{rno} – {name}" if name else rno

    races["label"] = races.apply(mk_label, axis=1)

    # ensure required cols exist
    for col in ["meeting_key", "session_key", "label", "round"]:
        if col not in races.columns:
            races[col] = pd.NA

    return races[["meeting_key", "session_key", "label", "round"]]

def openf1_to_sessions_results_by_meeting(season: int, meeting_key: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch Qualifying, Grid, and Race for a specific meeting and normalize to the app's schema.

    Returns:
      sessions_like: [season, round, grand_prix, driver, team, p1, p2, p3, quali]
      results_like:  [season, round, grand_prix, driver, team, grid, finish, dnf, fastest_lap]
    """
    cache_dir = Path(f"data/cache/{season}/{meeting_key}")
    # sessions metadata
    sess_meta = _read_or_fetch(cache_dir / "sessions.csv",
                               lambda: _with_backoff(_cached_sessions, meeting_key))
    if sess_meta.empty:
        return pd.DataFrame(), pd.DataFrame()

    meeting_name_col = _first_present(
        sess_meta, ["meeting_name", "meeting", "meeting_official_name", "circuit_short_name", "meeting_name_official"]
    )
    gp_vals = sess_meta[meeting_name_col].dropna().astype(str).unique().tolist() if meeting_name_col else []
    gp_name = gp_vals[0] if gp_vals else None

    # keys
    def key_for(target: str) -> Optional[int]:
        if "session_name" not in sess_meta.columns or "session_key" not in sess_meta.columns:
            return None
        hit = sess_meta[sess_meta["session_name"].astype(str).str.lower() == target.lower()]
        return int(hit.iloc[0]["session_key"]) if not hit.empty else None

    race_key = key_for("Race")
    quali_key = key_for("Qualifying")

    # fetch with disk + mem cache + backoff
    quali_raw = _read_or_fetch(cache_dir / "quali.csv",
                               lambda: _with_backoff(_cached_session_result, quali_key)) if quali_key else pd.DataFrame()
    race_raw  = _read_or_fetch(cache_dir / "race.csv",
                               lambda: _with_backoff(_cached_session_result, race_key)) if race_key else pd.DataFrame()
    grid_raw  = _read_or_fetch(cache_dir / "grid.csv",
                               lambda: _with_backoff(_cached_starting_grid, race_key)) if race_key else pd.DataFrame()
    drv_raw   = _read_or_fetch(cache_dir / "drivers.csv",
                               lambda: _with_backoff(_cached_drivers_for_meeting, meeting_key))

    # driver map
    num_to_name_team: dict[int, tuple[Optional[str], Optional[str]]] = {}
    if not drv_raw.empty:
        for _, r in drv_raw.iterrows():
            num = r.get("driver_number")
            dname = r.get("full_name") or r.get("name") or r.get("broadcast_name")
            team  = r.get("team_name")
            if pd.notna(num):
                try:
                    num_to_name_team[int(num)] = (dname, team)
                except Exception:
                    pass

    def _norm(df: pd.DataFrame, pos_col: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["driver", "team", pos_col])
        out = []
        for _, r in df.iterrows():
            num  = r.get("driver_number")
            pos  = r.get("position")
            # fallback name if mapping fails
            dflt_name = r.get("driver_name")
            if dflt_name is None and pd.notna(num):
                try:
                    dflt_name = str(int(num))
                except Exception:
                    dflt_name = str(num)
            dname, tname = num_to_name_team.get(int(num), (dflt_name, None)) if pd.notna(num) else (dflt_name, None)
            out.append({"driver": dname, "team": tname, pos_col: pos})
        return pd.DataFrame(out)

    quali_df = _norm(quali_raw, "quali")
    race_df  = _norm(race_raw,  "finish")
    grid_df  = _norm(grid_raw,  "grid")

    base = pd.merge(quali_df, grid_df, on=["driver", "team"], how="outer")
    base = pd.merge(base,      race_df,  on=["driver", "team"], how="outer")

    # compute 'round' for compatibility (from races_for_season ordering)
    try:
        races_df = races_for_season(season)
        round_val = int(races_df.loc[races_df["meeting_key"] == meeting_key, "round"].iloc[0])
    except Exception:
        round_val = None

    # sessions-like (P1/P2/P3 reserved for later)
    sessions_like = base.copy()
    sessions_like["season"] = season
    sessions_like["grand_prix"] = gp_name
    for c in ["p1", "p2", "p3"]:
        sessions_like[c] = pd.NA
    if round_val is not None:
        sessions_like["round"] = round_val
        sess_cols = ["season", "round", "grand_prix", "driver", "team", "p1", "p2", "p3", "quali"]
    else:
        sess_cols = ["season", "grand_prix", "driver", "team", "p1", "p2", "p3", "quali"]
    sessions_like = sessions_like[sess_cols]

    # results-like (DNF/fastest_lap left 0 for now; can be derived later)
    results_like = base.copy()
    results_like["season"] = season
    results_like["grand_prix"] = gp_name
    results_like["dnf"] = 0
    results_like["fastest_lap"] = 0
    if round_val is not None:
        results_like["round"] = round_val
        res_cols = ["season", "round", "grand_prix", "driver", "team", "grid", "finish", "dnf", "fastest_lap"]
    else:
        res_cols = ["season", "grand_prix", "driver", "team", "grid", "finish", "dnf", "fastest_lap"]
    results_like = results_like[res_cols]

    return sessions_like, results_like
