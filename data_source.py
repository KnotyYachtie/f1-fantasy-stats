# data_source.py
from __future__ import annotations
from typing import Optional, Tuple, List
import pandas as pd

from openf1_client import (
    sessions,            # list sessions for a meeting_key
    sessions_by_year,    # list all sessions for a season
    session_result,      # results for a given session_key
    starting_grid,       # starting grid for a Race session_key
    drivers as of1_drivers,
)

# ---------- internal utils ----------

def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name that exists in df from the candidate list."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------- public API for app.py ----------

def races_for_season(season: int) -> pd.DataFrame:
    """
    Return one row per Race session for a season with a clean human-friendly label.
    Columns returned: meeting_key, session_key, label, round
    """
    s = sessions_by_year(season)
    if s.empty:
        return pd.DataFrame(columns=["meeting_key", "session_key", "label", "round"])

    # Filter to Race sessions (case-insensitive, guard if column missing)
    name_col = "session_name" if "session_name" in s.columns else None
    races = s[s[name_col].astype(str).str.lower().eq("race")] if name_col else s.copy()
    if races.empty:
        return pd.DataFrame(columns=["meeting_key", "session_key", "label", "round"])

    # Sort chronologically by whatever timestamp column exists
    date_col = _first_present(races, ["date_start", "session_start_time", "date"])
    if date_col:
        races = races.sort_values(date_col)

    races = races.reset_index(drop=True)
    races["round"] = races.index + 1  # keep a numeric round for compatibility

    # Choose a meeting display name (fallback list is broad)
    meeting_name_col = _first_present(
        races,
        ["meeting_name", "meeting", "meeting_official_name", "circuit_short_name", "meeting_name_official"]
    )

    # Build tidy label WITHOUT date for UI
    def mk_label(row: pd.Series) -> str:
        rno = f"R{int(row['round']):02d}"
        name = str(row.get(meeting_name_col, "") or "").strip() if meeting_name_col else ""
        return f"{rno} – {name}" if name else rno

    races["label"] = races.apply(mk_label, axis=1)

    keep_cols = ["meeting_key", "session_key", "label", "round"]
    # Guard: ensure all keep_cols exist
    for col in keep_cols:
        if col not in races.columns:
            races[col] = pd.NA
    return races[keep_cols]


def openf1_to_sessions_results_by_meeting(season: int, meeting_key: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch Qualifying, Grid, and Race for a specific meeting and normalize to the app's schema.

    Returns:
      sessions_like: columns [season, round, grand_prix, driver, team, p1, p2, p3, quali]
      results_like:  columns [season, round, grand_prix, driver, team, grid, finish, dnf, fastest_lap]
    """
    # Get all sessions for this meeting
    sess_meta = sessions(meeting_key)
    if sess_meta.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Determine GP name robustly
    meeting_name_col = _first_present(
        sess_meta, ["meeting_name", "meeting", "meeting_official_name", "circuit_short_name", "meeting_name_official"]
    )
    gp_vals = sess_meta[meeting_name_col].dropna().astype(str).unique().tolist() if meeting_name_col else []
    gp_name = gp_vals[0] if gp_vals else None

    # Helper: pick a session_key for a given session name
    def key_for(target_name: str) -> Optional[int]:
        if "session_name" not in sess_meta.columns or "session_key" not in sess_meta.columns:
            return None
        hit = sess_meta[sess_meta["session_name"].astype(str).str.lower() == target_name.lower()]
        return int(hit.iloc[0]["session_key"]) if not hit.empty else None

    race_key = key_for("Race")
    quali_key = key_for("Qualifying")

    # Pull raw frames (gracefully handle missing keys)
    quali_raw = session_result(quali_key) if quali_key else pd.DataFrame()
    race_raw  = session_result(race_key)  if race_key  else pd.DataFrame()
    grid_raw  = starting_grid(race_key)   if race_key  else pd.DataFrame()

    # Build driver/team mapping for nicer display
    drv = of1_drivers(meeting_key=meeting_key)
    num_to_name_team: dict[int, tuple[Optional[str], Optional[str]]] = {}
    if not drv.empty:
        for _, r in drv.iterrows():
            num = r.get("driver_number")
            dname = r.get("full_name") or r.get("name") or r.get("broadcast_name")
            team  = r.get("team_name")
            if pd.notna(num):
                try:
                    num_to_name_team[int(num)] = (dname, team)
                except Exception:
                    pass

    def _norm(df: pd.DataFrame, pos_col: str) -> pd.DataFrame:
        """Normalize OpenF1 rows to columns [driver, team, <pos_col>]."""
        if df.empty:
            return pd.DataFrame(columns=["driver", "team", pos_col])
        out = []
        for _, r in df.iterrows():
            num  = r.get("driver_number")
            pos  = r.get("position")
            dflt = (r.get("driver_name") or (str(int(num)) if pd.notna(num) else None), None)
            dname, tname = num_to_name_team.get(int(num), dflt) if pd.notna(num) else dflt
            out.append({"driver": dname, "team": tname, pos_col: pos})
        return pd.DataFrame(out)

    quali_df = _norm(quali_raw, "quali")
    race_df  = _norm(race_raw,  "finish")
    grid_df  = _norm(grid_raw,  "grid")

    # Outer-merge all three
    base = pd.merge(quali_df, grid_df, on=["driver", "team"], how="outer")
    base = pd.merge(base,      race_df,  on=["driver", "team"], how="outer")

    # Determine round number for this meeting (optional, for compatibility)
    try:
        races_df = races_for_season(season)
        round_val = int(races_df.loc[races_df["meeting_key"] == meeting_key, "round"].iloc[0])
    except Exception:
        round_val = None

    # sessions-like (Practice slots left blank for now; fill later if needed)
    sessions_like = base.copy()
    sessions_like["season"] = season
    sessions_like["grand_prix"] = gp_name
    if round_val is not None:
        sessions_like["round"] = round_val
        cols = ["season", "round", "grand_prix", "driver", "team", "p1", "p2", "p3", "quali"]
    else:
        cols = ["season", "grand_prix", "driver", "team", "p1", "p2", "p3", "quali"]
    for c in ["p1", "p2", "p3"]:
        sessions_like[c] = pd.NA
    sessions_like = sessions_like[cols]

    # results-like (DNF/fastest_lap left as 0 for now; derive later if needed)
    results_like = base.copy()
    results_like["season"] = season
    results_like["grand_prix"] = gp_name
    if round_val is not None:
        results_like["round"] = round_val
        rcols = ["season", "round", "grand_prix", "driver", "team", "grid", "finish", "dnf", "fastest_lap"]
    else:
        rcols = ["season", "grand_prix", "driver", "team", "grid", "finish", "dnf", "fastest_lap"]
    results_like["dnf"] = 0
    results_like["fastest_lap"] = 0
    results_like = results_like[rcols]

    return sessions_like, results_like
