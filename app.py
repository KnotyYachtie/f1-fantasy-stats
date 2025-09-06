import streamlit as st
import pandas as pd
import json
from analytics import (
    PointsConfig, load_csvs, join_weekend, correlation_report,
    rolling_consistency, simulate_points, aggregate_points
)

st.set_page_config(page_title="F1 Fantasy Analytics (GridRival)", layout="wide")

st.title("F1 Fantasy Analytics — GridRival Companion")
st.caption("Lightweight build for Streamlit Cloud: cached loads, no heavy deps, on-demand compute.")

@st.cache_data
def _load_data(sess_path, res_path):
    sessions, results = load_csvs(sess_path, res_path)
    df = join_weekend(sessions, results)
    return sessions, results, df

@st.cache_data
def _corr(df):
    return correlation_report(df)

@st.cache_data
def _cons(results, window):
    return rolling_consistency(results, window=window)

@st.cache_data
def _sim(df, cfg_dict):
    cfg = PointsConfig(**cfg_dict)
    sim = simulate_points(df, cfg)
    agg = aggregate_points(sim)
    return agg

with st.sidebar:
    st.header("Data")
    use_sample = st.toggle("Use sample Monza data", value=True)
    if use_sample:
        sess_path = "data/sessions.csv"
        res_path = "data/results.csv"
        st.success("Using bundled sample data.")
        sessions_file = None
        results_file = None
    else:
        sessions_file = st.file_uploader("Upload sessions.csv (practice & quali)", type=["csv"])
        results_file  = st.file_uploader("Upload results.csv (grid & finish)", type=["csv"])
        sess_path = sessions_file
        res_path = results_file

    st.divider()
    st.header("Scoring")
    cfg_raw = json.load(open("points_config.json","r"))
    col1, col2 = st.columns(2)
    with col1:
        cfg_raw["pole_bonus"] = st.number_input("Pole bonus", value=int(cfg_raw["pole_bonus"]), step=1)
        cfg_raw["fastest_lap_bonus"] = st.number_input("Fastest lap bonus", value=int(cfg_raw["fastest_lap_bonus"]), step=1)
        cfg_raw["dnf_penalty"] = st.number_input("DNF penalty", value=int(cfg_raw["dnf_penalty"]), step=1)
    with col2:
        cfg_raw["position_gain_bonus"] = st.number_input("Per-place gain bonus", value=int(cfg_raw["position_gain_bonus"]), step=1)
        cfg_raw["position_loss_penalty"] = st.number_input("Per-place loss penalty", value=int(cfg_raw["position_loss_penalty"]), step=1)

    with st.expander("Race points by finish (1–20)"):
        cols = st.columns(5)
        for i in range(1, 21):
            with cols[(i-1)%5]:
                cfg_raw["race_points_by_finish"][str(i)] = st.number_input(
                    f"{i}", value=int(cfg_raw["race_points_by_finish"].get(str(i), 0)), step=1
                )

    save_clicked = st.button("Save scoring to points_config.json")
    if save_clicked:
        with open("points_config.json","w") as f:
            json.dump(cfg_raw, f, indent=2)
        st.toast("Scoring saved.", icon="✅")

if sess_path is None or res_path is None:
    st.info("Upload both CSVs or toggle sample data in the sidebar.")
    st.stop()

# Lazy data load
sessions, results, df = _load_data(sess_path, res_path)

tab1, tab2, tab3, tab4 = st.tabs(["Correlations", "Consistency", "Practice Trends", "Points Simulation"])

with tab1:
    st.subheader("Practice/Quali/Grid ↔ Race Finish")
    if st.button("Compute correlations"):
        corr = _corr(df)
        st.dataframe(corr, use_container_width=True)
        st.caption("Spearman is rank-based and often more robust week-to-week; Pearson captures linearity. Lower positions are better.")

with tab2:
    st.subheader("Race-to-race Consistency")
    window = st.slider("Rolling window (races)", min_value=3, max_value=10, value=5, step=1)
    if st.button("Compute consistency"):
        cons = _cons(results, window)
        st.dataframe(cons, use_container_width=True)

with tab3:
    st.subheader("Practice Trends")
    pr_cols = [c for c in ["p1","p2","p3"] if c in df.columns]
    if not pr_cols:
        st.info("No practice columns present.")
    else:
        if st.button("Compute practice trends"):
            tmp = df.melt(
                id_vars=["driver","season","round","grand_prix"],
                value_vars=pr_cols,
                var_name="session",
                value_name="pos"
            ).dropna()
            st.dataframe(
                tmp.groupby("driver")["pos"].mean().reset_index().rename(columns={"pos":"avg_practice_pos"}).sort_values("avg_practice_pos"),
                use_container_width=True
            )

with tab4:
    st.subheader("Simulated Points (per scoring config)")
    if st.button("Run simulation"):
        agg = _sim(df, cfg_raw)
        st.dataframe(agg, use_container_width=True)
        st.download_button("Download simulated points (CSV)", data=agg.to_csv(index=False), file_name="sim_points.csv")
