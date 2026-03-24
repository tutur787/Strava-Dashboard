"""
app.py — Thin orchestrator for the Allure Strava dashboard.

Responsibility:
  1. Page config + CSS
  2. Cookie/OAuth bootstrap
  3. Preference loading (before sidebar so saved values apply)
  4. Sidebar → settings
  5. Landing page (unauthenticated)
  6. Token refresh / persistence
  7. Activity loading (Supabase → disk cache → API)
  8. Date range filter (sidebar block)
  9. Stream loading + on-demand fetching
  10. Run classification, weather, analytics pre-computation
  11. Build shared `data` dict
  12. Render 9 tabs by calling each tab module's render()
"""

import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Allure.run",
    page_icon="🏃",
    layout="wide",
)

# ── Cookie controller ─────────────────────────────────────────────────
# Must come before any other st calls.
# Skip during OAuth callback — CookieController's JS causes a RerunException
# that interrupts the token exchange before tokens are saved.
_in_oauth_callback = "code" in st.query_params
try:
    from streamlit_cookies_controller import CookieController as _CookieController
    _cookies = None if _in_oauth_callback else _CookieController()
    _COOKIES_ENABLED = not _in_oauth_callback
except Exception:
    _cookies = None
    _COOKIES_ENABLED = False

# ── Project imports (dependency order) ───────────────────────────────
from config import (
    ACTIVITIES_PATH,
    STREAMS_PATH,
    STRAVA_CACHE_PATH,
    KM_TO_MILES,
)
from analytics import (
    classify_all_runs,
    build_daily_weekly,
    compute_personal_bests,
    estimate_vo2max,
    compute_consistency,
    compute_cadence_stats,
    compute_activity_gap,
    generate_insight,
    make_hr_zones,
)
from database import (
    _SUPABASE_ENABLED,
    _get_supabase,
    sb_save_athlete,
    sb_load_athlete,
    sb_save_activities,
    sb_load_activities,
    sb_save_streams,
    sb_load_streams,
    sb_load_preferences,
)
from auth import (
    get_strava_auth_url,
    exchange_strava_code,
    refresh_strava_token,
    get_valid_token,
    fetch_all_activities_api,
    fetch_activity_streams_api,
    fetch_gear_api,
)
from data_loader import (
    parse_activities_raw,
    load_activities,
    load_streams,
    load_strava_disk_cache,
    save_strava_disk_cache,
    fetch_weather_for_activities,
)
from ui.styles import inject_css
from ui.sidebar import render_sidebar
import tabs.overview as tab_overview
import tabs.training_load as tab_training_load
import tabs.pace as tab_pace
import tabs.long_runs as tab_long_runs
import tabs.recovery as tab_recovery
import tabs.race_predictor as tab_race_predictor
import tabs.gear as tab_gear
import tabs.guide as tab_guide
import tabs.streams as tab_streams

# ── CSS ───────────────────────────────────────────────────────────────
inject_css()

# ── Strava OAuth credentials ──────────────────────────────────────────
_strava_secrets    = st.secrets.get("strava", {})
_STRAVA_CLIENT_ID  = _strava_secrets.get("client_id", "")
_STRAVA_CLIENT_SECRET = _strava_secrets.get("client_secret", "")
_REDIRECT_URI      = _strava_secrets.get("redirect_uri", "")
_OAUTH_ENABLED     = bool(_STRAVA_CLIENT_ID and _STRAVA_CLIENT_SECRET)

# ── OAuth callback handling ───────────────────────────────────────────
# The _exchanged_code guard prevents a second token exchange on the rerun
# that Streamlit triggers immediately after st.query_params.clear().
if _in_oauth_callback:
    _auth_code = st.query_params.get("code", "")
    if _auth_code and st.session_state.get("_exchanged_code") != _auth_code:
        st.session_state["_exchanged_code"] = _auth_code
        with st.spinner("Connecting to Strava…"):
            _tok = exchange_strava_code(
                _STRAVA_CLIENT_ID,
                _STRAVA_CLIENT_SECRET,
                _auth_code,
                _REDIRECT_URI,
            )
        if _tok and "access_token" in _tok:
            st.session_state["strava_tokens"] = _tok
            _ath = _tok.get("athlete", {})
            if isinstance(_ath, dict):
                _athlete_id_cb   = _ath.get("id")
                _athlete_name_cb = (
                    f"{_ath.get('firstname', '')} {_ath.get('lastname', '')}".strip()
                )
                if _athlete_id_cb:
                    st.session_state["strava_athlete_id"]   = _athlete_id_cb
                    st.session_state["strava_athlete_name"] = _athlete_name_cb
        st.query_params.clear()
        st.rerun()

# ── Silent token refresh from cookies ────────────────────────────────
if "strava_tokens" not in st.session_state and _COOKIES_ENABLED and _cookies is not None:
    try:
        _cookie_rt  = _cookies.get("strava_refresh_token")
        _cookie_aid = _cookies.get("strava_athlete_id")
        if _cookie_rt and _OAUTH_ENABLED:
            _refreshed = refresh_strava_token(
                _STRAVA_CLIENT_ID, _STRAVA_CLIENT_SECRET, _cookie_rt
            )
            if _refreshed and "access_token" in _refreshed:
                st.session_state["strava_tokens"] = _refreshed
                if _cookie_aid:
                    try:
                        _aid_int = int(_cookie_aid)
                        st.session_state["strava_athlete_id"] = _aid_int
                        # Restore display name from Supabase — the token refresh API
                        # does not return the athlete object, so we must look it up.
                        if _SUPABASE_ENABLED:
                            try:
                                _ath_row_cookie = sb_load_athlete(_aid_int)
                                if _ath_row_cookie and _ath_row_cookie.get("display_name"):
                                    st.session_state["strava_athlete_name"] = _ath_row_cookie["display_name"]
                            except Exception:
                                pass
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass

# ── Load saved preferences before sidebar (so widgets use saved values) ──
if "strava_tokens" in st.session_state and "_prefs" not in st.session_state:
    _aid_prefs = st.session_state.get("strava_athlete_id")
    if _SUPABASE_ENABLED and _aid_prefs:
        try:
            _saved_prefs = sb_load_preferences(int(_aid_prefs))
            if _saved_prefs:
                st.session_state["_prefs"] = _saved_prefs
                # Rerun so sidebar widgets initialise with saved default values
                st.rerun()
        except Exception:
            pass

# ── Page title ────────────────────────────────────────────────────────
st.title("Allure.run")

# ── Sidebar → settings ───────────────────────────────────────────────
settings = render_sidebar(
    _OAUTH_ENABLED,
    _STRAVA_CLIENT_ID,
    _STRAVA_CLIENT_SECRET,
    _cookies,
    _COOKIES_ENABLED,
    STRAVA_CACHE_PATH,
    _get_supabase,
)

# Unpack frequently used settings
use_miles        = settings["use_miles"]
max_hr           = settings["max_hr"]
race_km          = settings["race_km"]
effort_band      = settings["effort_band"]
only_runs        = settings["only_runs"]
exclude_manual   = settings["exclude_manual"]
exclude_trainer  = settings["exclude_trainer"]
_hr_zones        = settings["_hr_zones"]

# ── Landing page (unauthenticated) ────────────────────────────────────
if "strava_tokens" not in st.session_state:
    if _OAUTH_ENABLED:
        col_land, _ = st.columns([2, 1])
        with col_land:
            st.markdown("### Connect your Strava account to get started")
            st.markdown(
                "Allure analyses your complete running history — training load, "
                "pace trends, HR zones, race predictions, recovery risk and more. "
                "Your data is only used to power your personal dashboard."
            )
            _auth_url = get_strava_auth_url(_STRAVA_CLIENT_ID, _REDIRECT_URI)
            st.markdown(
                f"""<a href="{_auth_url}" target="_self" style="
                    display:inline-block;
                    background-color:#FC4C02;
                    color:#ffffff;
                    text-decoration:none;
                    font-weight:600;
                    font-size:1rem;
                    padding:0.55rem 1.4rem;
                    border-radius:6px;
                    margin-top:0.5rem;
                ">🔗 Connect with Strava</a>""",
                unsafe_allow_html=True,
            )
    else:
        st.info(
            "Strava credentials are not configured. "
            "Add `strava.client_id` and `strava.client_secret` to your secrets."
        )
    st.stop()

# ── Get a valid access token ──────────────────────────────────────────
_access_token = get_valid_token(_STRAVA_CLIENT_ID, _STRAVA_CLIENT_SECRET)

# ── Persist auth to cookies + Supabase (once per session) ────────────
if not st.session_state.get("_auth_persisted") and _access_token:
    _tok_state = st.session_state.get("strava_tokens", {})
    _rt        = _tok_state.get("refresh_token", "")
    _aid_save  = st.session_state.get("strava_athlete_id")
    _aname     = st.session_state.get("strava_athlete_name", "")

    # Cookies — deferred below the insight banner so the iframes they inject
    # do not land between the title and banner (where they would consume flex-gap).
    # Store what needs to be written; the actual .set() calls happen after the banner.
    if _COOKIES_ENABLED and _cookies is not None and _rt:
        st.session_state["_pending_rt"]  = _rt
        st.session_state["_pending_aid"] = str(_aid_save) if _aid_save else None

    # Supabase — only write display_name when we actually have it;
    # an empty upsert would overwrite the stored name with "".
    if _SUPABASE_ENABLED and _aid_save and _rt:
        try:
            if _aname:
                sb_save_athlete(int(_aid_save), _aname, _rt)
            else:
                # Token-only update: preserve the existing display_name
                _get_supabase().table("athletes").update({
                    "refresh_token": _rt,
                    "fetched_at": datetime.utcnow().isoformat(),
                }).eq("athlete_id", int(_aid_save)).execute()
        except Exception:
            pass

    st.session_state["_auth_persisted"] = True

# ── Load activities ───────────────────────────────────────────────────
activities = None
_athlete_id = st.session_state.get("strava_athlete_id")

# 1. Try Supabase
if _SUPABASE_ENABLED and _athlete_id:
    try:
        _raw_sb = sb_load_activities(int(_athlete_id))
        if _raw_sb:
            activities = parse_activities_raw(_raw_sb)
            st.session_state["strava_fetched_at"] = "supabase"
    except Exception:
        pass

# 2. Try disk cache
if activities is None or len(activities) == 0:
    try:
        _disk = load_strava_disk_cache()
        if _disk and "activities" in _disk and _disk["activities"]:
            activities = parse_activities_raw(_disk["activities"])
    except Exception:
        pass

# 3. Fetch live from Strava API
if (activities is None or len(activities) == 0) and _access_token:
    with st.spinner("Fetching activities from Strava…"):
        _raw_acts = fetch_all_activities_api(_access_token)
    if _raw_acts:
        activities = parse_activities_raw(_raw_acts)
        # Persist to Supabase
        if _SUPABASE_ENABLED and _athlete_id:
            try:
                sb_save_activities(int(_athlete_id), _raw_acts)
            except Exception:
                pass
        # Persist to disk cache (streams come later)
        try:
            save_strava_disk_cache(_raw_acts, {})
        except Exception:
            pass
        st.session_state["strava_fetched_at"] = datetime.utcnow().isoformat()

if activities is None:
    activities = pd.DataFrame()

# ── Normalise start_dt_local ──────────────────────────────────────────
if len(activities) > 0 and "start_dt_local" in activities.columns:
    activities["start_dt_local"] = pd.to_datetime(
        activities["start_dt_local"], errors="coerce", utc=False
    )
    if activities["start_dt_local"].dt.tz is not None:
        activities["start_dt_local"] = activities["start_dt_local"].dt.tz_localize(None)

# ── Apply type filters to activities (affects ALL analytics) ──────────
# Done here so build_daily_weekly, compute_personal_bests, etc. only
# ever see the activity types the user has selected.
if len(activities) > 0:
    if only_runs:
        activities = activities[
            activities["type"].astype(str).str.lower() == "run"
        ].copy()
    if exclude_manual and "manual" in activities.columns:
        activities = activities[~activities["manual"].fillna(False)].copy()
    if exclude_trainer and "trainer" in activities.columns:
        activities = activities[~activities["trainer"].fillna(False)].copy()

# ── Load streams from cache (Supabase → disk) ────────────────────────
# Done before the sidebar block so we can decide whether to show
# "Fetch Streams" or not based on what's already cached.
streams_by_id: dict = {}

# Consume flags set by sidebar buttons
_stream_refresh_needed  = st.session_state.pop("_stream_refresh_needed", False)
_fetch_streams_requested = st.session_state.pop("_fetch_streams_requested", False)

# 1. Supabase
if not _stream_refresh_needed and _SUPABASE_ENABLED and _athlete_id:
    try:
        streams_by_id = sb_load_streams(int(_athlete_id))
    except Exception:
        pass

# 2. Disk fallback (skipped on forced refresh to avoid stale cache)
if not _stream_refresh_needed and not streams_by_id:
    try:
        _disk_streams = load_streams(STREAMS_PATH)
        if _disk_streams:
            streams_by_id = _disk_streams
    except Exception:
        pass

# ── Date range filter + stream button (sidebar block) ─────────────────
if len(activities) > 0 and "start_dt_local" in activities.columns:
    _act_min = activities["start_dt_local"].min().date()
    _act_max = activities["start_dt_local"].max().date()

    _default_start = max(
        _act_min,
        (pd.Timestamp(_act_max) - pd.Timedelta(days=90)).date(),
    )

    with st.sidebar:
        st.divider()
        st.subheader("Date Range")
        _date_range = st.date_input(
            "Filter activities",
            value=(_default_start, _act_max),
            min_value=_act_min,
            max_value=_act_max,
        )
        st.caption(
            "ℹ️ Per-second stream data (GPS, HR, cadence) is fetched from Strava "
            "only for activities within the selected date range and cached for "
            "future sessions."
        )
        # Show "Fetch Streams" when no streams are cached yet
        if not streams_by_id and _access_token:
            if st.button(
                "⬇️ Fetch Streams",
                use_container_width=True,
                help="Download per-second HR, pace and cadence data for the selected date range.",
            ):
                st.session_state["_fetch_streams_requested"] = True
                st.rerun()

    if isinstance(_date_range, (list, tuple)) and len(_date_range) == 2:
        _start_date, _end_date = _date_range
    else:
        _start_date, _end_date = _act_min, _act_max

    _start_ts = pd.Timestamp(_start_date)
    _end_ts   = pd.Timestamp(_end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df_range = activities[
        (activities["start_dt_local"] >= _start_ts) &
        (activities["start_dt_local"] <= _end_ts)
    ].copy()
else:
    df_range = pd.DataFrame()

# ── On-demand stream fetch (button or forced refresh) ─────────────────
if _access_token and len(df_range) > 0 and (_fetch_streams_requested or _stream_refresh_needed):
    _ids_in_range = set(df_range["id"].dropna().astype(int).tolist())
    _ids_have     = set(streams_by_id.keys())
    _MAX_STREAM_FETCH = 50
    _ids_to_fetch = sorted(_ids_in_range - _ids_have, reverse=True)[:_MAX_STREAM_FETCH]
    if _ids_to_fetch:
        _new_streams: dict = {}
        _fetch_total = len(_ids_to_fetch)
        _fetch_bar   = st.progress(0, text="Fetching activity streams…")
        for _fi, _fid in enumerate(_ids_to_fetch):
            try:
                _s = fetch_activity_streams_api(int(_fid), _access_token)
                if _s:
                    _new_streams[int(_fid)]  = _s
                    streams_by_id[int(_fid)] = _s
            except Exception:
                pass
            _fetch_bar.progress(
                (_fi + 1) / _fetch_total,
                text=f"Fetching streams… {_fi + 1}/{_fetch_total}",
            )
        _fetch_bar.empty()
        if _new_streams and _SUPABASE_ENABLED and _athlete_id:
            try:
                sb_save_streams(int(_athlete_id), _new_streams)
            except Exception:
                pass

# ── Run classification ────────────────────────────────────────────────
if len(activities) > 0:
    _long_run_km  = settings.get("long_run_ratio_thresh", 0.60) * race_km
    _easy_thresh  = settings.get("hr_z2", 0.80)
    _tempo_thresh = settings.get("hr_z3", 0.87)

    activities["run_type"] = classify_all_runs(
        activities, streams_by_id,
        long_run_km=_long_run_km, max_hr=max_hr,
        easy_thresh=_easy_thresh, tempo_thresh=_tempo_thresh,
    )
    if len(df_range) > 0:
        df_range["run_type"] = classify_all_runs(
            df_range, streams_by_id,
            long_run_km=_long_run_km, max_hr=max_hr,
            easy_thresh=_easy_thresh, tempo_thresh=_tempo_thresh,
        )

# ── Weather enrichment ────────────────────────────────────────────────
_weather_df = pd.DataFrame()
if len(df_range) > 0:
    try:
        _weather_cols  = ["id", "date", "start_lat", "start_lng", "start_dt_local"]
        _weather_input = df_range[
            [c for c in _weather_cols if c in df_range.columns]
        ].to_json(orient="records")
        _weather_df = fetch_weather_for_activities(_weather_input)
    except Exception:
        _weather_df = pd.DataFrame()

# Merge temp_c into df_range so tabs can access it directly via df_range["temp_c"]
if len(_weather_df) > 0 and "id" in _weather_df.columns and len(df_range) > 0:
    df_range = df_range.merge(_weather_df[["id", "temp_c"]], on="id", how="left")

# ── Daily/weekly aggregates for full history ──────────────────────────
daily_all  = pd.DataFrame()
weekly_all = pd.DataFrame()
if len(activities) > 0 and "start_dt_local" in activities.columns:
    try:
        _hist_start = pd.Timestamp(activities["start_dt_local"].min())
        _hist_end   = pd.Timestamp(activities["start_dt_local"].max())
        daily_all, weekly_all = build_daily_weekly(
            activities, max_hr=max_hr, date_range=(_hist_start, _hist_end),
            rest_hr=settings.get("rest_hr", 50),
            gender=settings.get("gender", "Men"),
        )
    except Exception:
        pass

# ── Shared analytics pre-computation ─────────────────────────────────
bests       = compute_personal_bests(activities) if len(activities) > 0 else {}
vo2max_est  = estimate_vo2max(bests)
consistency = compute_consistency(activities) if len(activities) > 0 else {}

cadence_df = pd.DataFrame()
if len(df_range) > 0 and streams_by_id:
    try:
        cadence_df = compute_cadence_stats(df_range, streams_by_id)
    except Exception:
        pass

# ── Per-activity GAP pace (grade-adjusted) ────────────────────────────
if len(df_range) > 0 and streams_by_id:
    try:
        _gap_map = {}
        for _aid, _s in streams_by_id.items():
            _g = compute_activity_gap(_s)
            if _g is not None:
                _gap_map[int(_aid)] = _g
        if _gap_map:
            df_range = df_range.copy()
            df_range["gap_pace_min_per_km"] = df_range["id"].astype(int).map(_gap_map)
    except Exception:
        pass

# ── Gear details ──────────────────────────────────────────────────────
if "gear_details" not in st.session_state:
    _gear_details: dict = {}
    if _access_token and len(activities) > 0 and "gear_id" in activities.columns:
        _gear_ids = activities["gear_id"].dropna().unique().tolist()
        _gear_ids = [g for g in _gear_ids if g and str(g) != "nan"]
        for _gid in _gear_ids:
            try:
                _gdata = fetch_gear_api(str(_gid), _access_token)
                if _gdata:
                    _gear_details[str(_gid)] = _gdata.get("name", str(_gid))
            except Exception:
                pass
    st.session_state["gear_details"] = _gear_details

gear_details = st.session_state.get("gear_details", {})

# ── Insight banner ────────────────────────────────────────────────────
if len(daily_all) > 0 and len(activities) > 0:
    try:
        _insight_msg, _insight_level = generate_insight(daily_all, activities)
        if _insight_msg:
            _banner_styles = {
                "success": "background:rgba(0,200,83,0.13);border-left:4px solid #00c853;color:rgba(255,255,255,0.92)",
                "warning": "background:rgba(255,171,0,0.13);border-left:4px solid #ffab00;color:rgba(255,255,255,0.92)",
                "info":    "background:rgba(0,148,255,0.13);border-left:4px solid #0094ff;color:rgba(255,255,255,0.92)",
            }
            _bstyle = _banner_styles.get(_insight_level, _banner_styles["info"])
            st.markdown(
                f'<div style="{_bstyle};padding:0.7rem 1.5rem;border-radius:6px;margin-bottom:1.5rem;">'
                f'{_insight_msg}</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

# ── Deferred cookie writes (after banner so iframes don't gap the title) ─
_pending_rt  = st.session_state.pop("_pending_rt",  None)
_pending_aid = st.session_state.pop("_pending_aid", None)
if _pending_rt and _COOKIES_ENABLED and _cookies is not None:
    try:
        _max_age = 60 * 60 * 24 * 60  # 60 days
        _cookies.set("strava_refresh_token", _pending_rt, max_age=_max_age)
        if _pending_aid:
            _cookies.set("strava_athlete_id", _pending_aid, max_age=_max_age)
    except Exception:
        pass

# ── Build shared data dict ────────────────────────────────────────────
data: dict = {
    "activities":    activities,
    "df_range":      df_range,
    "streams_by_id": streams_by_id,
    "daily_all":     daily_all,
    "weekly_all":    weekly_all,
    "bests":         bests,
    "vo2max_est":    vo2max_est,
    "consistency":   consistency,
    "cadence_df":    cadence_df,
    "_weather_df":   _weather_df,
    "_hr_zones":     _hr_zones,
    "gear_details":  gear_details,
}

# ── Tab layout ────────────────────────────────────────────────────────
_tab_labels = [
    "Overview",
    "Training Load",
    "Pace & Efficiency",
    "Long Runs",
    "Recovery & Risk",
    "Race Predictor",
    "Gear",
    "Guide",
    "Raw Streams",
]

_tabs = st.tabs(_tab_labels)

with _tabs[0]:
    tab_overview.render(data, settings)

with _tabs[1]:
    tab_training_load.render(data, settings)

with _tabs[2]:
    tab_pace.render(data, settings)

with _tabs[3]:
    tab_long_runs.render(data, settings)

with _tabs[4]:
    tab_recovery.render(data, settings)

with _tabs[5]:
    tab_race_predictor.render(data, settings)

with _tabs[6]:
    tab_gear.render(data, settings)

with _tabs[7]:
    tab_guide.render(data, settings)

with _tabs[8]:
    tab_streams.render(data, settings)
