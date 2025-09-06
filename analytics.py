# analytics.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import pandas as pd

# -------------------------
# Config model (simple, JSON-friendly)
# -------------------------
@dataclass
class PointsConfig:
    race_points_by_finish: Dict[str, int] = field(default_factory=dict)
    qualifying_points_by_position: Dict[str, int] = field(default_factory=dict)

    pole_bonus: int = 0
    fastest_lap_bonus: int = 0
    dnf_penalty: int = 0

    # If you later want to use these, wire them in compute_points_row
    position_gain_bonus: int = 0
    position_loss_penalty: int = 0

    practice_points_by_position: Dict[str, int] = field(default_factory=dict)


# -------------------------
# Helpers
# -------------------------
def _pos_key(val: Any) -> str | None:
    """
    Convert a qualifying/finish/grid value into a normalized string key: "1".."20".
    Returns None if missing/invalid/outside 1..20.
    """
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        v = int(float(val))
        if 1 <= v <= 20:
            return str(v)
        return None
    except Exception:
        return None


# -------------------------
# Row scorer (robust to NaN/strings)
# -------------------------
def compute_points_row(row: pd.Series, cfg: PointsConfig) -> int:
    # Qualifying points
    q_key = _pos_key(row.get("quali"))
    qpts = cfg.qualifying_points_by_position.get(q_key, 0) if q_key else 0

    # Race points
    f_key = _pos_key(row.get("finish"))
    rpts = cfg.race_points_by_finish.get(f_key, 0) if f_key else 0

    # Optional practice points (kept zero unless you’ve populated the table)
    # p1_key = _pos_key(row.get("p1")); p1pts = cfg.practice_points_by_position.get(p1_key, 0) if p1_key else 0
    # p2_key = _pos_key(row.get("p2")); p2pts = cfg.practice_points_by_position.get(p2_key, 0) if p2_key else 0
    # p3_key = _pos_key(row.get("p3")); p3pts = cfg.practice_points_by_position.get(p3_key, 0) if p3_key else 0

    # Bonuses / penalties
    pole_bonus = cfg.pole_bonus if q_key == "1" else 0
    fl_flag = int(row.get("fastest_lap", 0) or 0)
    fl_bonus = cfg.fastest_lap_bonus if fl_flag == 1 else 0
    dnf_flag = int(row.get("dnf", 0) or 0)
    dnf_pen  = cfg.dnf_penalty if dnf_flag == 1 else 0

    # Position gain/loss adjustment (disabled by default; enable if you set values)
    pos_adj = 0
    g_key = _pos_key(row.get("grid"))
    if g_key and f_key and (cfg.position_gain_bonus or cfg.position_loss_penalty):
        # NOTE: lower number is better in positions (P1 best).
        # gain_neg means positions gained; gain_pos means positions lost.
        gain = int(f_key) - int(g_key)
        if gain < 0 and cfg.position_gain_bonus:
            pos_adj += abs(gain) * cfg.position_gain_bonus
        elif gain > 0 and cfg.position_loss_penalty:
            pos_adj -= abs(gain) * cfg.position_loss_penalty

    total = qpts + rpts + pole_bonus + fl_bonus - dnf_pen + pos_adj
    # If you enable practice points, add: + p1pts + p2pts + p3pts
    return int(total)


# -------------------------
# Simulation API
# -------------------------
def simulate_points(df: pd.DataFrame, cfg: PointsConfig) -> pd.DataFrame:
    """
    Returns a copy of df with a new column 'sim_points' computed per row.
    Expects columns like: ['driver','team','quali','grid','finish','dnf','fastest_lap', 'p1','p2','p3' (optional)]
    """
    out = df.copy()

    # Sanitize numeric columns to avoid crashes on strings/None
    for col in ["p1", "p2", "p3", "quali", "grid", "finish", "dnf", "fastest_lap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["sim_points"] = out.apply(lambda r: compute_points_row(r, cfg), axis=1)
    return out


def aggregate_points(sim: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate simulated points per driver/team (and per event if available).
    Returns a tidy leaderboard sorted by points desc.
    """
    cols_present = [c for c in ["season", "round", "grand_prix"] if c in sim.columns]
    group_cols = cols_present + [c for c in ["driver", "team"] if c in sim.columns]

    if not group_cols:
        # fallback to a simple sum if no identifiers are present
        return pd.DataFrame({"sim_points": [sim["sim_points"].sum()]})

    agg = (
        sim.groupby(group_cols, dropna=False, as_index=False)["sim_points"]
        .sum()
        .sort_values("sim_points", ascending=False)
        .reset_index(drop=True)
    )
    return agg
