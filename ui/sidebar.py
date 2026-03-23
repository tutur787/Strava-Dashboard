"""
ui/sidebar.py — render_sidebar() renders all sidebar widgets and returns the settings dict.
"""
import os
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st

from analytics import make_hr_zones
from config import (
    KM_TO_MILES,
    LONG_RUN_DEFAULTS,
    RACE_EFFORT_DEFAULTS,
    RACE_PRESETS_KM,
)
from database import (
    _SUPABASE_ENABLED,
    sb_load_athlete,
    sb_save_preferences,
)


def render_sidebar(
    _OAUTH_ENABLED: bool,
    _STRAVA_CLIENT_ID: str,
    _STRAVA_CLIENT_SECRET: str,
    _cookies: Any,
    _COOKIES_ENABLED: bool,
    STRAVA_CACHE_PATH: str,
    _get_supabase,
) -> dict:
    """
    Render the first sidebar block (controls, settings, filters, unit toggle).
    Returns the settings dict.
    The date range block is rendered in app.py after activities are loaded.
    """
    with st.sidebar:
        st.header("Controls")

        # Load saved preferences (populated after first auth)
        _prefs = st.session_state.get("_prefs", {})

        max_hr = st.number_input(
            "Max HR (bpm)",
            min_value=120,
            max_value=230,
            value=int(_prefs.get("max_hr", 180)),
            step=1,
            help="Used to compute intensity = avgHR / maxHR.",
        )

        with st.expander("HR Zone Boundaries (% of max HR)"):
            st.caption("Drag to adjust where each zone begins and ends. These boundaries drive both run classification and the zone breakdown chart.")
            hr_z1 = st.slider("Z1/Z2 boundary", 50, 75, int(_prefs.get("hr_z1", 60)), step=1,
                              help="Below this = Z1 Recovery") / 100.0
            hr_z2 = st.slider("Z2/Z3 boundary (Easy threshold)", 65, 88, int(_prefs.get("hr_z2", 80)), step=1,
                              help="Runs with median HR below this are classified as Easy") / 100.0
            hr_z3 = st.slider("Z3/Z4 boundary (Tempo threshold)", 78, 95, int(_prefs.get("hr_z3", 87)), step=1,
                              help="Runs with median HR above this are classified as Tempo") / 100.0
            hr_z4 = st.slider("Z4/Z5 boundary", 85, 100, int(_prefs.get("hr_z4", 93)), step=1,
                              help="Above this = Z5 VO\u2082max") / 100.0
            # Clamp to prevent inversions
            hr_z1 = min(hr_z1, hr_z2 - 0.01)
            hr_z3 = max(hr_z3, hr_z2 + 0.01)
            hr_z4 = max(hr_z4, hr_z3 + 0.01)
            _hr_zones = make_hr_zones(hr_z1, hr_z2, hr_z3, hr_z4)

        _race_keys = list(RACE_PRESETS_KM.keys())
        _saved_race = _prefs.get("race_choice", "Half Marathon")
        _race_idx = _race_keys.index(_saved_race) if _saved_race in _race_keys else 2
        race_choice = st.selectbox("Target race distance", _race_keys, index=_race_idx)
        if race_choice == "Custom (km)":
            race_km = st.number_input("Custom race distance (km)", min_value=1.0, max_value=200.0,
                                      value=float(_prefs.get("race_km", 21.0975)), step=0.5)
        else:
            race_km = float(RACE_PRESETS_KM[race_choice])

        lo_def, hi_def = RACE_EFFORT_DEFAULTS.get(race_choice, (0.84, 0.92))
        st.subheader("Race-effort HR band")
        effort_band = st.slider(
            "HR intensity range (% of max HR)",
            min_value=0.50,
            max_value=1.00,
            value=(float(_prefs.get("lo_hr", lo_def)), float(_prefs.get("hi_hr", hi_def))),
            step=0.01,
            help="Used in Tab 2 to track efficiency at race-relevant intensity.",
        )

        lr_def = float(LONG_RUN_DEFAULTS.get(race_choice, 0.60))
        st.subheader("Long-run threshold")
        long_run_ratio_thresh = st.slider(
            "Minimum long-run distance (% of race distance)",
            min_value=0.30,
            max_value=1.00,
            value=float(_prefs.get("long_run_ratio_thresh", lr_def)),
            step=0.05,
            help="Tab 3 considers runs longer than this threshold as 'long runs' for fatigue modeling.",
        )

        st.subheader("Readiness windows")
        readiness_window_days = st.slider(
            "Readiness lookback (days)",
            min_value=14,
            max_value=90,
            value=int(_prefs.get("readiness_window_days", 42)),
            step=7,
            help="Tab 4 uses this window to summarize recent risk/readiness.",
        )

        st.subheader("Prediction window")
        prediction_lookback_days = st.slider(
            "Race prediction lookback (days)",
            min_value=30,
            max_value=365,
            value=int(_prefs.get("prediction_lookback_days", 180)),
            step=15,
            help="Tab 5 uses this window to predict race time.",
        )

        st.divider()
        st.subheader("Filters")
        only_runs = st.checkbox("Only running activities", value=bool(_prefs.get("only_runs", True)))
        exclude_manual = st.checkbox("Exclude manual activities", value=bool(_prefs.get("exclude_manual", True)))
        exclude_trainer = st.checkbox("Exclude trainer/treadmill", value=bool(_prefs.get("exclude_trainer", False)))

        st.divider()
        use_miles = st.toggle("Show distances in miles \U0001f1fa\U0001f1f8", value=bool(_prefs.get("use_miles", False)))
        show_streams_tab = st.checkbox("\U0001f527 Raw streams explorer", value=False,
                                       help="Adds a Raw Streams tab to inspect per-second GPS, HR, and cadence data for individual activities.")
        # Date range is added below after activities are loaded

        # Save Settings button (only shown when authenticated)
        if _OAUTH_ENABLED and "strava_tokens" in st.session_state:
            st.divider()
            if st.button("\U0001f4be Save Settings", use_container_width=True,
                         help="Save current sidebar values so they reload next time"):
                _prefs_to_save = {
                    "max_hr": int(max_hr),
                    "hr_z1": int(hr_z1 * 100), "hr_z2": int(hr_z2 * 100),
                    "hr_z3": int(hr_z3 * 100), "hr_z4": int(hr_z4 * 100),
                    "race_choice": race_choice,
                    "race_km": float(race_km),
                    "lo_hr": float(effort_band[0]), "hi_hr": float(effort_band[1]),
                    "long_run_ratio_thresh": float(long_run_ratio_thresh),
                    "readiness_window_days": int(readiness_window_days),
                    "prediction_lookback_days": int(prediction_lookback_days),
                    "only_runs": bool(only_runs),
                    "exclude_manual": bool(exclude_manual),
                    "exclude_trainer": bool(exclude_trainer),
                    "use_miles": bool(use_miles),
                }
                _save_aid = st.session_state.get("strava_athlete_id")
                if _SUPABASE_ENABLED and _save_aid:
                    _pref_err = sb_save_preferences(_save_aid, _prefs_to_save)
                    if _pref_err:
                        st.error(f"Save failed: {_pref_err}")
                    else:
                        st.session_state["_prefs"] = _prefs_to_save
                        st.success("Settings saved \u2713")
                else:
                    st.session_state["_prefs"] = _prefs_to_save
                    st.success("Settings saved locally \u2713")

        # Strava connection controls (shown when authenticated)
        if _OAUTH_ENABLED and "strava_tokens" in st.session_state:
            st.divider()
            _athlete_id = st.session_state.get("strava_athlete_id")
            _athlete_display = st.session_state.get("strava_athlete_name", "")
            if _SUPABASE_ENABLED and _athlete_id:
                _ath_row = sb_load_athlete(_athlete_id)
                if _ath_row:
                    _athlete_display = _ath_row.get("display_name", _athlete_display)
            if _athlete_display:
                st.caption(f"\U0001f517 Connected as **{_athlete_display}**")

            # Last synced timestamp
            _fetched_at = st.session_state.get("strava_fetched_at", "")
            if _fetched_at == "supabase":
                # Use the athletes table fetched_at for the real sync time
                _real_ts = (_ath_row or {}).get("fetched_at", "") if _SUPABASE_ENABLED and _athlete_id else ""
                if _real_ts:
                    try:
                        _age = datetime.utcnow() - datetime.fromisoformat(_real_ts.replace("Z", "+00:00").split("+")[0])
                        _age_str = (f"{int(_age.total_seconds() // 3600)}h ago"
                                    if _age.total_seconds() >= 3600
                                    else f"{int(_age.total_seconds() // 60)}m ago")
                        st.caption(f"Last synced: {_age_str}")
                    except Exception:
                        st.caption("Last synced: from cloud")
                else:
                    st.caption("Last synced: from cloud")
            elif _fetched_at:
                try:
                    _age = datetime.utcnow() - datetime.fromisoformat(_fetched_at)
                    _age_str = (f"{int(_age.total_seconds() // 3600)}h ago"
                                if _age.total_seconds() >= 3600
                                else f"{int(_age.total_seconds() // 60)}m ago")
                    st.caption(f"Last synced: {_age_str}")
                except Exception:
                    pass

            _col1, _col2 = st.columns(2)
            if _col1.button("Refresh data", help="Re-fetch all activities from Strava"):
                _refresh_athlete_id = st.session_state.get("strava_athlete_id")
                for _k in ["strava_activities", "strava_streams", "gear_details", "strava_fetched_at"]:
                    st.session_state.pop(_k, None)
                if os.path.exists(STRAVA_CACHE_PATH):
                    os.remove(STRAVA_CACHE_PATH)
                # Clear Supabase activities so they re-fetch from API
                if _SUPABASE_ENABLED and _refresh_athlete_id:
                    try:
                        _get_supabase().table("activities").delete().eq("athlete_id", _refresh_athlete_id).execute()
                        _get_supabase().table("streams").delete().eq("athlete_id", _refresh_athlete_id).execute()
                    except Exception:
                        pass
                # Signal app.py to bypass the disk-stream cache and force API fetch
                st.session_state["_stream_refresh_needed"] = True
                st.rerun()
            if _col2.button("Disconnect"):
                for _k in ["strava_tokens", "strava_activities", "strava_streams", "gear_details",
                            "strava_fetched_at", "_auth_persisted", "_prefs"]:
                    st.session_state.pop(_k, None)
                if _COOKIES_ENABLED and _cookies is not None:
                    try:
                        _cookies.remove("strava_refresh_token")
                        _cookies.remove("strava_athlete_id")
                    except Exception:
                        pass
                st.rerun()

    # Build derived unit helpers
    _pace_factor = (1.0 / KM_TO_MILES) if use_miles else 1.0
    _pace_lbl = f"min/{'mi' if use_miles else 'km'}"
    _dist_unit = "mi" if use_miles else "km"

    # Build race_km_label
    race_km_label = f"{race_km:.2f} km"

    settings = {
        "use_miles": use_miles,
        "max_hr": int(max_hr),
        "race_choice": race_choice,
        "race_km": float(race_km),
        "race_km_label": race_km_label,
        "effort_band": effort_band,
        "long_run_ratio_thresh": float(long_run_ratio_thresh),
        "readiness_window_days": int(readiness_window_days),
        "prediction_lookback_days": int(prediction_lookback_days),
        "hr_z1": float(hr_z1),
        "hr_z2": float(hr_z2),
        "hr_z3": float(hr_z3),
        "hr_z4": float(hr_z4),
        "only_runs": bool(only_runs),
        "exclude_manual": bool(exclude_manual),
        "exclude_trainer": bool(exclude_trainer),
        "show_streams_tab": bool(show_streams_tab),
        # derived unit helpers
        "KM_TO_MILES": KM_TO_MILES,
        "_pace_factor": _pace_factor,
        "_pace_lbl": _pace_lbl,
        "_dist_unit": _dist_unit,
        # internal zone list (not in spec but needed by tabs)
        "_hr_zones": _hr_zones,
    }
    return settings
