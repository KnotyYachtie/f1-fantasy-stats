# data_source.py
from __future__ import annotations
import pandas as pd
from typing import Tuple, Optional
from openf1_client import sessions, sessions_by_year, session_result, starting_grid, drivers as of1_drivers

def races_for_season(season: int) -> pd.DataFrame:
    """Return one row per Race session in the season with a friendly label."""
    s = sessions_by_year(season)
    if s.empty:
        return pd.DataFrame(columns=["meeting_key","session_key","meeting_name","label"])
    races = s[s["session_name"] == "Race"].copy()
    # Sort chronologically by any available timestamp
    for c in ["date_start", "session_start_time", "date"]:
        if c in races.columns:
            races = races.sort_values(c)
            break
    races = races.reset_index(drop=True)
    races["round"] = races.index + 1
    # Build label like "R03 – Italian GP (2025-09-07)"
    date_col = next((c for c in ["date_start", "session_start_time", "date"] if c in races.columns), None)
    date_str = races[date_col].astype(str) if date_col else ""
    races["label"] = races.apply(
        lambda r: f'R{int(r["round"]):02d} – {r.get("meeting_name","")} ({str(date_str[r.name])[:10]})', axis=1
    )
    return races[["meeting_key","session_key","meeting_name","label","round"]]

def openf1_to_sessions_results_by_meeting(season: int, meeting_key: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Same as before, but fetch directly by meeting_key."""
    sess_meta = sessions(meeting_key)
    if sess_meta.empty:
        return pd.DataFrame(), pd.DataFrame()
    gp_name = (sess_meta["meeting_name"].dropna().astype(str).unique().tolist() or [None])[0]

    def key_for(name: str) -> Optional[int]:
        df = sess_meta[sess_meta["session_name"] == name]
        return int(df.iloc[0]["session_key"]) if not df.empty else None

    race_key = key_for("Race")
    quali_key = key_for("Qualifying")

    quali_df_raw = session_result(quali_key) if quali_key else pd.DataFrame()
    race_df_raw  = session_result(race_key)  if race_key  else pd.DataFrame()
    grid_df_raw  = starting_grid(race_key)   if race_key  else pd.DataFrame()

    drv = of1_drivers(meeting_key=meeting_key)
    name_map = {}
    if not drv.empty:
        for _, r in drv.iterrows():
            num = r.get("driver_number")
            dname = r.get("full_name") or r.get("name") or r.get("broadcast_name")
            team  = r.get("team_name")
            if pd.notna(num):
                name_map[int(num)] = (dname, team)

    def norm(df: pd.DataFrame, pos_col: str) -> pd.DataFrame:
        if df.empty: return pd.DataFrame(columns=["driver","team",pos_col])
        out = []
        for _, r in df.iterrows():
            num = r.get("driver_number")
            pos = r.get("position")
            dname, tname = name_map.get(int(num), (None, None)) if pd.notna(num) else (None, None)
            if dname is None: dname = r.get("driver_name") or str(num)
            out.append({"driver": dname, "team": tname, pos_col: pos})
        return pd.DataFrame(out)

    quali_df = norm(quali_df_raw, "quali")
    race_df  = norm(race_df_raw,  "finish")
    grid_df  = norm(grid_df_raw,  "grid")

    base = pd.merge(quali_df, grid_df, on=["driver","team"], how="outer")
    base = pd.merge(base, race_df, on=["driver","team"], how="outer")

    sessions_like = base.copy()
    sessions_like["season"] = season
    sessions_like["grand_prix"] = gp_name
    for c in ["p1","p2","p3"]: sessions_like[c] = pd.NA
    sessions_like = sessions_like[["season","grand_prix","driver","team","p1","p2","p3","quali"]]
    # we no longer need round; race name is the key

    results_like = base.copy()
    results_like["season"] = season
    results_like["grand_prix"] = gp_name
    results_like["dnf"] = 0
    results_like["fastest_lap"] = 0
    results_like = results_like[["season","grand_prix","driver","team","grid","finish","dnf","fastest_lap"]]

    return sessions_like, results_like
