from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import json
import pandas as pd
import numpy as np

@dataclass
class PointsConfig:
    race_points_by_finish: Dict[str, int]
    pole_bonus: int = 0
    fastest_lap_bonus: int = 0
    dnf_penalty: int = 0
    position_gain_bonus: int = 0
    position_loss_penalty: int = 0
    qualifying_points_by_position: Dict[str, int] = None
    practice_points_by_position: Dict[str, int] = None

    @staticmethod
    def load(path: str) -> "PointsConfig":
        with open(path, "r") as f:
            raw = json.load(f)
        return PointsConfig(**raw)

def _to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_csvs(sessions_path: str, results_path: str):
    s = pd.read_csv(sessions_path)
    r = pd.read_csv(results_path)
    s = _to_numeric(s, ["season","round","p1","p2","p3","quali"])
    r = _to_numeric(r, ["season","round","grid","finish","dnf","fastest_lap"])
    return s, r

def join_weekend(sessions: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    keys = ["season","round","grand_prix","driver","team"]
    return pd.merge(sessions, results, on=keys, how="outer")

def correlation_report(df: pd.DataFrame) -> pd.DataFrame:
    if "finish" not in df.columns:
        return pd.DataFrame()
    rows = []
    for col in [c for c in ["p1","p2","p3","quali","grid"] if c in df.columns]:
        sub = df[[col, "finish"]].dropna()
        n = len(sub)
        if n < 3:
            rows.append({"metric": col, "pearson": None, "spearman": None, "n": n})
            continue
        pear = sub[[col, "finish"]].corr(method="pearson").iloc[0,1]
        spear = sub[[col, "finish"]].corr(method="spearman").iloc[0,1]
        rows.append({"metric": col, "pearson": float(pear), "spearman": float(spear), "n": n})
    out = pd.DataFrame(rows)
    return out.sort_values("spearman", na_position="last")

def rolling_consistency(results: pd.DataFrame, window:int=5) -> pd.DataFrame:
    rr = results.sort_values(["driver","season","round"]).copy()
    rr["finish_std"] = rr.groupby("driver")["finish"].rolling(window, min_periods=3).std().reset_index(level=0, drop=True)
    latest = rr.dropna(subset=["finish_std"]).groupby("driver").tail(1)
    return latest[["driver","finish_std"]].sort_values("finish_std")

def compute_points_row(row: pd.Series, cfg: PointsConfig) -> int:
    pts = 0
    if not pd.isna(row.get("finish", np.nan)):
        pts += int(cfg.race_points_by_finish.get(str(int(row["finish"])), 0))
    if not pd.isna(row.get("quali", np.nan)) and int(row["quali"]) == 1:
        pts += cfg.pole_bonus
    if int(row.get("fastest_lap", 0)) == 1:
        pts += cfg.fastest_lap_bonus
    if not pd.isna(row.get("grid", np.nan)) and not pd.isna(row.get("finish", np.nan)):
        delta = int(row["grid"]) - int(row["finish"])
        if delta > 0:
            pts += delta * cfg.position_gain_bonus
        elif delta < 0:
            pts += abs(delta) * (-cfg.position_loss_penalty)
    if int(row.get("dnf", 0)) == 1:
        pts -= cfg.dnf_penalty
    if cfg.qualifying_points_by_position:
        qpts = cfg.qualifying_points_by_position.get(str(int(row.get("quali", 0))), 0)
        pts += int(qpts)
    if cfg.practice_points_by_position:
        for col in ["p1","p2","p3"]:
            if col in row and not pd.isna(row[col]):
                pts += int(cfg.practice_points_by_position.get(str(int(row[col])), 0))
    return int(pts)

def simulate_points(df: pd.DataFrame, cfg: PointsConfig) -> pd.DataFrame:
    out = df.copy()
    out["sim_points"] = out.apply(lambda r: compute_points_row(r, cfg), axis=1)
    return out

def aggregate_points(sim_df: pd.DataFrame) -> pd.DataFrame:
    agg = sim_df.groupby(["season","round","grand_prix","driver","team"], as_index=False)["sim_points"].sum()
    return agg.sort_values(["season","round","sim_points"], ascending=[True, True, False])

def teammate_h2h(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute head-to-head per race vs teammate based on finish.
    Rules:
      - Lower finish wins.
      - If one DNF and the other finished, finisher wins.
      - If both DNF or missing finishes, no result.
      - If a team has != 2 drivers for an event, skip that team/event.
    Returns:
      per_race: rows with [season, round, grand_prix, team, driver, teammate, finish, tm_finish, dnf, tm_dnf, beat_teammate (0/1)]
      summary: rows with [driver, wins, losses, total, win_pct]
    """
    needed = {"season","round","grand_prix","driver","team","finish","dnf"}
    if not needed.issubset(df.columns):
        return pd.DataFrame(), pd.DataFrame()
    # pick only needed columns
    cols = list(needed)
    sub = df[cols].copy()
    # group by event+team
    grp = sub.groupby(["season","round","grand_prix","team"], dropna=False)
    rows = []
    for (season, rnd, gp, team), g in grp:
        g = g.dropna(subset=["driver"])
        if len(g) != 2:
            continue
        a, b = g.iloc[0].copy(), g.iloc[1].copy()
        # define comparable finishes
        def outcome(x):
            f = x["finish"]; d = x.get("dnf", 0)
            if pd.isna(f):
                return None, int(d) if not pd.isna(d) else 0
            return int(f), int(d) if not pd.isna(d) else 0
        fa, da = outcome(a); fb, db = outcome(b)
        # decide winners
        def decide(f1, d1, f2, d2):
            # returns 1 if 1 beats 2, 0 if loses, None if no result
            if f1 is None and f2 is None:
                return None
            if d1==1 and d2==1:
                return None
            if d1==1 and d2==0:
                return 0
            if d1==0 and d2==1:
                return 1
            if f1 is None or f2 is None:
                return None
            if f1 < f2:
                return 1
            if f1 > f2:
                return 0
            return None  # exact tie / identical
        a_win = decide(fa, da, fb, db)
        b_win = decide(fb, db, fa, da) if a_win is not None else None
        rows.append({
            "season": int(season) if not pd.isna(season) else season,
            "round": int(rnd) if not pd.isna(rnd) else rnd,
            "grand_prix": gp, "team": team,
            "driver": a["driver"], "teammate": b["driver"],
            "finish": fa, "tm_finish": fb, "dnf": da, "tm_dnf": db,
            "beat_teammate": a_win
        })
        rows.append({
            "season": int(season) if not pd.isna(season) else season,
            "round": int(rnd) if not pd.isna(rnd) else rnd,
            "grand_prix": gp, "team": team,
            "driver": b["driver"], "teammate": a["driver"],
            "finish": fb, "tm_finish": fa, "dnf": db, "tm_dnf": da,
            "beat_teammate": b_win
        })
    per_race = pd.DataFrame(rows)
    # summary
    if per_race.empty:
        return per_race, pd.DataFrame()
    valid = per_race.dropna(subset=["beat_teammate"]).copy()
    valid["beat_teammate"] = valid["beat_teammate"].astype(int)
    summ = valid.groupby("driver", as_index=False).agg(
        wins=("beat_teammate","sum"),
        total=("beat_teammate","count")
    )
    summ["losses"] = summ["total"] - summ["wins"]
    summ["win_pct"] = (summ["wins"] / summ["total"]).round(3)
    summ = summ[["driver","wins","losses","total","win_pct"]].sort_values(["win_pct","total","wins"], ascending=[False, False, False])
    return per_race.sort_values(["season","round","team","driver"]), summ
