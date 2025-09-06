from pathlib import Path
from data_source import openf1_to_sessions_results
import streamlit as st
import pandas as pd
import json
from analytics import (
    PointsConfig, load_csvs, join_weekend, correlation_report,
    rolling_consistency, simulate_points, aggregate_points, teammate_h2h
)

st.set_page_config(page_title="F1 Fantasy Analytics (GridRival)", layout="wide")

# Ensure cache folder exists for OpenF1 pulls
Path("data").mkdir(exist_ok=True)
if "openf1_loaded" not in st.session_state:
    st.session_state["openf1_loaded"] = False

st.title("F1 Fantasy Analytics — GridRival Companion")
st.caption("Fast-start build with Teammate H2H — cached, no heavy deps, on-demand compute.")

# ---------- Cache wrappers ----------
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
def _sim(df, cfg_json):
    cfg = PointsConfig(**json.loads(cfg_json))
    sim = simulate_points(df, cfg)
    agg = aggregate_points(sim)
    return agg

@st.cache_data
def _h2h(df):
    return teammate_h2h(df)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Data")
    source = st.radio("Data source", ["Sample", "Upload CSVs", "OpenF1"], index=0, horizontal=True)

    if source == "Sample":
        sess_path = "data/sessions.csv"
        res_path  = "data/results.csv"
        st.success("Using bundled sample data.")
        st.session_state["openf1_loaded"] = False

    elif source == "Upload CSVs":
        sessions_file = st.file_uploader("Upload sessions.csv (practice & quali)", type=["csv"])
        results_file  = st.file_uploader("Upload results.csv (grid & finish)", type=["csv"])
        if not sessions_file or not results_file:
            st.stop()
        sess_path = sessions_file
        res_path  = results_file
        st.session_state["openf1_loaded"] = False

    else:  # OpenF1
        season = st.number_input("Season (year)", value=2025, step=1)

        # Build race list for the season
        from data_source import races_for_season, openf1_to_sessions_results_by_meeting
        races_df = races_for_season(int(season))
        if races_df.empty:
            st.error("No Race sessions found for that season.")
            st.stop()

        # Dropdown of race labels
        race_label = st.selectbox(
            "Race",
            races_df["label"].tolist(),
            index=0
        )
        selected = races_df[races_df["label"] == race_label].iloc[0]
        meeting_key = int(selected["meeting_key"])

        if st.button("Fetch from OpenF1"):
            with st.spinner("Fetching OpenF1 data..."):
                s_like, r_like = openf1_to_sessions_results_by_meeting(int(season), meeting_key)
            if s_like.empty or r_like.empty:
                st.error("No data found for that race.")
                st.stop()
            s_like.to_csv("data/_openf1_sessions_cache.csv", index=False)
            r_like.to_csv("data/_openf1_results_cache.csv", index=False)
            st.session_state["openf1_loaded"] = True
            st.success("OpenF1 data loaded.")

        if st.session_state.get("openf1_loaded"):
            sess_path = "data/_openf1_sessions_cache.csv"
            res_path  = "data/_openf1_results_cache.csv"
        else:
            st.stop()

            # Cache to files so downstream code can read like CSVs
            s_like.to_csv("data/_openf1_sessions_cache.csv", index=False)
            r_like.to_csv("data/_openf1_results_cache.csv", index=False)
            st.session_state["openf1_loaded"] = True
            st.success("OpenF1 data loaded.")

        if st.session_state.get("openf1_loaded"):
            sess_path = "data/_openf1_sessions_cache.csv"
            res_path  = "data/_openf1_results_cache.csv"
        else:
            st.info("Choose season/round and click 'Fetch from OpenF1'.")
            st.stop()

    st.divider()
    st.header("Scoring")
    try:
        cfg_raw = json.load(open("points_config.json","r"))
    except Exception as e:
        st.error(f"Couldn't read points_config.json: {e}. Recreating defaults.")
        cfg_raw = {
            "race_points_by_finish": {str(i): p for i,p in enumerate([25,18,15,12,10,8,6,4,2,1] + [0]*20, start=1)},
            "pole_bonus": 0, "fastest_lap_bonus": 0, "dnf_penalty": 0,
            "position_gain_bonus": 0, "position_loss_penalty": 0,
            "qualifying_points_by_position": {}, "practice_points_by_position": {}
        }

    col1, col2 = st.columns(2)
    with col1:
        cfg_raw["pole_bonus"] = st.number_input("Pole bonus", value=int(cfg_raw.get("pole_bonus",0)), step=1, key="pole_bonus")
        cfg_raw["fastest_lap_bonus"] = st.number_input("Fastest lap bonus", value=int(cfg_raw.get("fastest_lap_bonus",0)), step=1, key="fl_bonus")
        cfg_raw["dnf_penalty"] = st.number_input("DNF penalty", value=int(cfg_raw.get("dnf_penalty",0)), step=1, key="dnf_pen")
    with col2:
        cfg_raw["position_gain_bonus"] = st.number_input("Per-place gain bonus", value=int(cfg_raw.get("position_gain_bonus",0)), step=1, key="gain_bonus")
        cfg_raw["position_loss_penalty"] = st.number_input("Per-place loss penalty", value=int(cfg_raw.get("position_loss_penalty",0)), step=1, key="loss_pen")

    with st.expander("Race points by finish (1–20)"):
        cols = st.columns(5)
        for i in range(1, 21):
            with cols[(i-1)%5]:
                cfg_raw["race_points_by_finish"][str(i)] = st.number_input(
                    f"{i}", value=int(cfg_raw["race_points_by_finish"].get(str(i), 0)), step=1, key=f"finish_pts_{i}"
                )

    if st.button("Save scoring to points_config.json", key="save_scoring"):
        with open("points_config.json","w") as f:
            json.dump(cfg_raw, f, indent=2)
        st.toast("Scoring saved.", icon="✅")

# ---------- Guard ----------
if sess_path is None or res_path is None:
    st.info("Upload both CSVs or toggle sample data in the sidebar.")
    st.stop()

# ---------- Data ----------
sessions, results, df = _load_data(sess_path, res_path)

# For caching dicts reliably, serialize to JSON
cfg_json = json.dumps(cfg_raw, sort_keys=True)

# ---------- UI Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Correlations", "Consistency", "Practice Trends", "Points Simulation", "Teammate H2H"])

with tab1:
    st.subheader("Practice/Quali/Grid ↔ Race Finish")
    if st.button("Compute correlations", key="btn_corr"):
        corr = _corr(df)
        st.dataframe(corr, use_container_width=True)
        st.caption("Spearman is rank-based and often more robust week-to-week; Pearson captures linearity. Lower positions are better.")

with tab2:
    st.subheader("Race-to-race Consistency")
    window = st.slider("Rolling window (races)", min_value=3, max_value=10, value=5, step=1, key="win_slider")
    if st.button("Compute consistency", key="btn_cons"):
        cons = _cons(results, window)
        st.dataframe(cons, use_container_width=True)

with tab3:
    st.subheader("Practice Trends")
    pr_cols = [c for c in ["p1","p2","p3"] if c in df.columns]
    if not pr_cols:
        st.info("No practice columns present.")
    else:
        if st.button("Compute practice trends", key="btn_practice"):
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
    if st.button("Run simulation", key="btn_sim"):
        agg = _sim(df, cfg_json)
        st.dataframe(agg, use_container_width=True)
        st.download_button("Download simulated points (CSV)", data=agg.to_csv(index=False), file_name="sim_points.csv")

with tab5:
    st.subheader("Teammate Head-to-Head (Race Result)")
    st.caption("Rules: lower finish wins; finisher beats a DNF; double DNF or missing finish = no result. Teams with ≠2 drivers for an event are skipped.")
    if st.button("Compute H2H", key="btn_h2h"):
        per_race, summary = _h2h(df)
        if per_race.empty:
            st.info("No valid H2H rows found. Make sure results.csv has finish & dnf, and both drivers per team are present.")
        else:
            st.markdown("**Per race H2H**")
            st.dataframe(per_race, use_container_width=True)
            st.download_button("Download per-race H2H (CSV)", data=per_race.to_csv(index=False), file_name="h2h_per_race.csv")

            st.markdown("**Season summary**")
            st.dataframe(summary, use_container_width=True)
            st.download_button("Download H2H summary (CSV)", data=summary.to_csv(index=False), file_name="h2h_summary.csv")
