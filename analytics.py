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
