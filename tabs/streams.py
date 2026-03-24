"""
tabs/streams.py — Raw Streams Explorer tab render function.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import _dist_fmt
from data_loader import streams_to_df


def render(data: dict, settings: dict) -> None:
    use_miles = settings["use_miles"]
    show_streams_tab = settings["show_streams_tab"]

    df_range = data["df_range"]
    streams_by_id = data["streams_by_id"]

    if not show_streams_tab:
        st.info("Enable **\U0001f527 Raw streams explorer** in the sidebar to use this tab.")
        return

    st.subheader("Strava Streams Explorer")
    st.caption("Inspect per-second/per-sample streams (pace, HR, cadence, grade, etc.) for individual activities.")

    if not isinstance(streams_by_id, dict) or len(streams_by_id) == 0:
        st.warning("No streams found in streams file.")
        return

    # Build a nice label: date -- name -- distance
    df_act = df_range.copy()
    df_act["date_str"] = df_act["start_dt_local"].dt.strftime("%Y-%m-%d")
    df_act["label"] = (df_act["date_str"].astype(str) + " \u2014 " +
                       df_act["name"].fillna("").astype(str).str.slice(0, 50) + " \u2014 " +
                       df_act["distance_km"].fillna(0).map(lambda x: _dist_fmt(x, use_miles)))

    # Keep only activities that have streams available
    df_act["has_streams"] = df_act["id"].astype(int).isin(streams_by_id.keys())
    df_streamable = df_act[df_act["has_streams"]].sort_values("start_dt_local", ascending=False)

    if len(df_streamable) == 0:
        st.info("No activities with streams available in the selected date range. Try widening the date range.")
        return

    # Select activity by label but map to id
    label_to_id = dict(zip(df_streamable["label"], df_streamable["id"].astype(int)))
    selected_label = st.selectbox("Activity", list(label_to_id.keys()))
    activity_id = label_to_id[selected_label]

    streams = streams_by_id.get(int(activity_id), {})
    df = streams_to_df(streams)

    # Choose x-axis
    x_options = []
    for col in ["time", "distance", "distance_km"]:
        if col in df.columns:
            x_options.append(col)

    if not x_options:
        st.error("No 'time' or 'distance' stream found for this activity.")
        return

    default_x = "distance_km" if "distance_km" in x_options else x_options[0]
    x_axis = st.radio("X axis", x_options, horizontal=True, index=x_options.index(default_x))

    # y-axis candidates: numeric columns excluding x and lat/lng
    exclude = set(["lat", "lng"]) | set(x_options)
    y_candidates = [c for c in df.columns if c not in exclude]

    numeric_cols = []
    for c in y_candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            numeric_cols.append(c)

    if not numeric_cols:
        st.error("No numeric metrics available to plot for this activity.")
        return

    default_metric = "pace_min_per_km" if "pace_min_per_km" in numeric_cols else ("heartrate" if "heartrate" in numeric_cols else numeric_cols[0])
    metric = st.selectbox("Metric (Y axis)", numeric_cols, index=numeric_cols.index(default_metric))

    smooth_window = st.slider("Rolling mean window", 1, 200, 1, help="Window size for rolling mean calculation.")
    plot_df = df.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df[x_axis] = pd.to_numeric(plot_df[x_axis], errors="coerce")

    if smooth_window > 1:
        plot_df[metric] = plot_df[metric].rolling(smooth_window, min_periods=1).mean()

    # For pace, strip pre-run standing noise (values > 15 min/km = not yet running)
    if metric == "pace_min_per_km":
        plot_df = plot_df[plot_df[metric] < 15.0]

    plot_df = plot_df.dropna(subset=[x_axis, metric])

    _act_row = df_streamable[df_streamable["id"].astype(int) == activity_id].iloc[0]
    _act_title = f"{pd.to_datetime(_act_row['start_dt_local']).strftime('%Y-%m-%d')} \u2014 {str(_act_row.get('name',''))[:50]}"
    fig = px.line(
        plot_df,
        x=x_axis,
        y=metric,
        title=f"{_act_title} \u00b7 {metric}",
    )
    fig.update_layout(height=600)

    # Invert pace axis (lower is better)
    if metric == "pace_min_per_km":
        fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)
