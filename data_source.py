from __future__ import annotations
import pandas as pd
from typing import Tuple, Optional
from openf1_client import meetings, sessions, results as of1_results, drivers as of1_drivers, grid as of1_grid

def pick_meeting(season: int, round_number: int) -> Optional[int]:
    m = meetings(season)
    if m.empty:
        return None
    # meetings have "round" as int (may be NaN for some)
    m = m.dropna(subset=["round"]).copy()
    m["round"] = m["round"].astype(int)
    hit = m[m["round"] == int(round_number)]
    if hit.empty:
        return None
    return int(hit.iloc[0]["meeting_key"])

def openf1_to_sessions_results(season: int, round_number: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (sessions_df, results_df) in the app's schema.
    sessions_df columns: season, round, grand_prix, driver, team, p1, p2, p3, quali
    results_df columns: season, round, grand_prix, driver, team, grid, finish, dnf, fastest_lap
    """
    meeting_key = pick_meeting(season, round_number)
    if meeting_key is None:
        return pd.DataFrame(), pd.DataFrame()
    sess_meta = sessions(meeting_key)
    gp_name = None
    if not sess_meta.empty and "meeting_name" in sess_meta.columns:
        gp_name = sess_meta["meeting_name"].dropna().astype(str).unique().tolist()
        gp_name = gp_name[0] if gp_name else None

    # Quali results
    quali = of1_results(meeting_key, "Qualifying")
    # Race results
    race = of1_results(meeting_key, "Race")
    # Grid
    grd = of1_grid(meeting_key)

    # Drivers/teams mapping (in case names differ)
    drv = of1_drivers(meeting_key=meeting_key)
    name_map = {}
    if not drv.empty:
        # Prefer "full_name" if available, else "driver_number"
        for _, r in drv.iterrows():
            key = r.get("driver_number")
            val = r.get("full_name") or r.get("name") or r.get("broadcast_name")
            team = r.get("team_name")
            if key is not None:
                name_map[int(key)] = (val, team)

    def normalize(df: pd.DataFrame, pos_col: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["driver","team",pos_col])
        out = []
        for _, r in df.iterrows():
            num = r.get("driver_number")
            finish_pos = r.get("position")
            # fastest lap flag sometimes in "points" or separate endpoints; leave 0 here
            dname, tname = name_map.get(int(num), (None, None)) if pd.notna(num) else (None, None)
            if dname is None:
                dname = r.get("driver_name") or str(num)
            out.append({"driver": dname, "team": tname, pos_col: finish_pos})
        return pd.DataFrame(out)

    quali_df = normalize(quali, "quali")
    race_df  = normalize(race,  "finish")
    grid_df  = normalize(grd,   "grid")

    # Merge frames
    base = pd.merge(quali_df, grid_df, on=["driver","team"], how="outer")
    base = pd.merge(base, race_df, on=["driver","team"], how="outer")

    # Compose sessions-like with P1/P2/P3 unavailable from OpenF1 results endpoints.
    # We leave p1..p3 as NaN for now; later we can back-fill from OpenF1 "results" with Practice sessions.
    sessions_like = base.copy()
    sessions_like["season"] = season
    sessions_like["round"] = round_number
    sessions_like["grand_prix"] = gp_name
    for c in ["p1","p2","p3"]:
        sessions_like[c] = pd.NA

    # Compose results-like
    results_like = base.copy()
    results_like["season"] = season
    results_like["round"] = round_number
    results_like["grand_prix"] = gp_name
    results_like["dnf"] = 0  # We'll refine using status if available from OpenF1
    results_like["fastest_lap"] = 0  # TODO: can be derived from results "fastest_lap" in some datasets

    # Reorder columns
    sessions_like = sessions_like[["season","round","grand_prix","driver","team","p1","p2","p3","quali"]]
    results_like  = results_like[ ["season","round","grand_prix","driver","team","grid","finish","dnf","fastest_lap"]]

    return sessions_like, results_like
