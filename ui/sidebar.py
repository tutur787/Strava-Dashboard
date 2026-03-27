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

        # ── Athlete settings ──────────────────────────────────────────────
        max_hr = st.number_input(
            "Max HR (bpm)",
            min_value=120,
            max_value=230,
            value=int(_prefs.get("max_hr", 180)),
            step=1,
            help="Used to compute intensity = avgHR / maxHR.",
        )
        st.caption("Not sure? A common starting estimate is **220 − your age**. Measure properly with a max-effort test for best accuracy.")

        rest_hr = st.number_input(
            "Resting HR (bpm)",
            min_value=30,
            max_value=100,
            value=int(_prefs.get("rest_hr", 50)),
            step=1,
            help="Used in the full Banister TRIMP formula for training load. Measure lying down first thing in the morning.",
        )

        _gender_options = ["Men", "Women", "Non-binary", "Prefer not to say"]
        # Priority: saved prefs → Strava athlete sex (first login) → Men
        _strava_sex_default = st.session_state.get("strava_athlete_sex", "Men")
        _saved_gender = _prefs.get("gender", _strava_sex_default)
        _gender_idx = _gender_options.index(_saved_gender) if _saved_gender in _gender_options else 0
        gender = st.selectbox(
            "Sex / gender (for TRIMP & VO\u2082max norms)",
            _gender_options,
            index=_gender_idx,
            help=(
                "Banister's TRIMP uses sex-specific exponential coefficients "
                "(men: 0.64\u20091.92, women: 0.86\u20091.67). "
                "Non-binary and 'Prefer not to say' use averaged coefficients (0.75\u20091.795) "
                "and pooled ACSM VO\u2082max norms. "
                "Pre-filled from your Strava profile on first login."
            ),
        )

        age = st.number_input(
            "Age (years)",
            min_value=15,
            max_value=85,
            value=int(_prefs.get("age", 35)),
            step=1,
            help="Used to select the correct ACSM VO\u2082max norms bracket for your age/sex and to estimate max HR if needed (220 \u2212 age).",
        )

        # ── HR Zone Boundaries ────────────────────────────────────────────
        with st.expander("HR Zone Boundaries (% of max HR)"):
            st.caption("Drag to adjust where each zone begins and ends. These boundaries drive both run classification and the zone breakdown chart.")
            _hz1_pct = st.slider("Z1/Z2 boundary", 50, 75, int(_prefs.get("hr_z1", 60)), step=1,
                                 help="Below this = Z1 Recovery")
            _hz2_pct = st.slider("Z2/Z3 boundary (Easy threshold)", 65, 88, int(_prefs.get("hr_z2", 80)), step=1,
                                 help="Runs with median HR below this are classified as Easy")
            _hz3_pct = st.slider("Z3/Z4 boundary (Tempo threshold)", 78, 95, int(_prefs.get("hr_z3", 87)), step=1,
                                 help="Runs with median HR above this are classified as Tempo")
            _hz4_pct = st.slider("Z4/Z5 boundary", 85, 100, int(_prefs.get("hr_z4", 93)), step=1,
                                 help="Above this = Z5 VO\u2082max")
            hr_z1 = _hz1_pct / 100.0
            hr_z2 = _hz2_pct / 100.0
            hr_z3 = _hz3_pct / 100.0
            hr_z4 = _hz4_pct / 100.0
            # Clamp to prevent inversions
            hr_z1 = min(hr_z1, hr_z2 - 0.01)
            hr_z3 = max(hr_z3, hr_z2 + 0.01)
            hr_z4 = max(hr_z4, hr_z3 + 0.01)
            # Show actual BPM equivalents
            _b1 = int(hr_z1 * max_hr)
            _b2 = int(hr_z2 * max_hr)
            _b3 = int(hr_z3 * max_hr)
            _b4 = int(hr_z4 * max_hr)
            st.caption(
                f"**Z1** < {_b1} bpm  ·  **Z2** {_b1}–{_b2} bpm  ·  "
                f"**Z3** {_b2}–{_b3} bpm  ·  **Z4** {_b3}–{_b4} bpm  ·  **Z5** > {_b4} bpm"
            )
            _hr_zones = make_hr_zones(hr_z1, hr_z2, hr_z3, hr_z4)

        # ── Target race ───────────────────────────────────────────────────
        st.divider()
        st.subheader("Target race")
        _race_keys = list(RACE_PRESETS_KM.keys())
        _saved_race = _prefs.get("race_choice", "Half Marathon")
        _race_idx = _race_keys.index(_saved_race) if _saved_race in _race_keys else 2
        race_choice = st.selectbox("Target race distance", _race_keys, index=_race_idx)
        if race_choice == "Custom (km)":
            race_km = st.number_input("Custom race distance (km)", min_value=1.0, max_value=200.0,
                                      value=float(_prefs.get("race_km", 21.0975)), step=0.5)
        else:
            race_km = float(RACE_PRESETS_KM[race_choice])

        # ── Long-run threshold ────────────────────────────────────────────
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
        _lr_min_km = long_run_ratio_thresh * race_km
        st.caption(f"= **{_lr_min_km:.1f} km** / **{_lr_min_km * KM_TO_MILES:.1f} mi** minimum long run")

        # ── Race-effort HR band ───────────────────────────────────────────
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
        st.caption(f"**{int(effort_band[0] * max_hr)}–{int(effort_band[1] * max_hr)} bpm** at your current max HR of {max_hr} bpm")

        # ── Prediction window ─────────────────────────────────────────────
        st.subheader("Prediction window")
        prediction_lookback_days = st.slider(
            "Race prediction lookback (days)",
            min_value=30,
            max_value=365,
            value=int(_prefs.get("prediction_lookback_days", 180)),
            step=15,
            help="Tab 5 uses this window to predict race time.",
        )

        # ── Display ───────────────────────────────────────────────────────
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
                # Collect PMC planner values from session state (set by training_load tab widgets)
                _pmc_race_date = st.session_state.get("pmc_race_date")
                _pmc_taper_weeks = st.session_state.get("pmc_taper_weeks")
                _pmc_build_choice = st.session_state.get("pmc_build_choice")
                _prefs_to_save = {
                    "max_hr": int(max_hr),
                    "rest_hr": int(rest_hr),
                    "gender": str(gender),
                    "age": int(age),
                    "hr_z1": int(hr_z1 * 100), "hr_z2": int(hr_z2 * 100),
                    "hr_z3": int(hr_z3 * 100), "hr_z4": int(hr_z4 * 100),
                    "race_choice": race_choice,
                    "race_km": float(race_km),
                    "lo_hr": float(effort_band[0]), "hi_hr": float(effort_band[1]),
                    "long_run_ratio_thresh": float(long_run_ratio_thresh),
                    "prediction_lookback_days": int(prediction_lookback_days),
                    "use_miles": bool(use_miles),
                    # PMC race day planner
                    **({"pmc_race_date": str(_pmc_race_date)} if _pmc_race_date else {}),
                    **({"pmc_taper_weeks": int(_pmc_taper_weeks)} if _pmc_taper_weeks is not None else {}),
                    **({"pmc_build_choice": str(_pmc_build_choice)} if _pmc_build_choice else {}),
                }
                _save_aid = st.session_state.get("strava_athlete_id")
                if _SUPABASE_ENABLED and _save_aid:
                    _pref_err = sb_save_preferences(_save_aid, _prefs_to_save)
                    if _pref_err:
                        st.error(f"Save failed: {_pref_err}")
                    else:
                        st.session_state["_prefs"] = _prefs_to_save
                        st.rerun()  # m6: recompute TRIMP/zones immediately with new HR values
                else:
                    st.session_state["_prefs"] = _prefs_to_save
                    st.rerun()  # m6: recompute TRIMP/zones immediately with new HR values

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
                _cache_path = (f"data/strava_cache_{_refresh_athlete_id}.json"
                               if _refresh_athlete_id else STRAVA_CACHE_PATH)
                if os.path.exists(_cache_path):
                    os.remove(_cache_path)
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
        "rest_hr": int(rest_hr),
        "gender": str(gender),
        "age": int(age),
        "race_choice": race_choice,
        "race_km": float(race_km),
        "race_km_label": race_km_label,
        "effort_band": effort_band,
        "long_run_ratio_thresh": float(long_run_ratio_thresh),
        "prediction_lookback_days": int(prediction_lookback_days),
        "hr_z1": float(hr_z1),
        "hr_z2": float(hr_z2),
        "hr_z3": float(hr_z3),
        "hr_z4": float(hr_z4),
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
