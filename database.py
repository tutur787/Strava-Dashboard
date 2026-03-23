"""
database.py — Supabase helpers.
"""
import streamlit as st
from datetime import datetime
from typing import Dict, Optional


# ── Supabase setup ────────────────────────────────────────────────────
# Run in Supabase SQL editor:
# CREATE TABLE athletes (athlete_id BIGINT PRIMARY KEY, display_name TEXT, refresh_token TEXT, fetched_at TIMESTAMPTZ DEFAULT NOW(), preferences JSONB DEFAULT '{}');
# If you already created the table: ALTER TABLE athletes ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}';
# CREATE TABLE activities (athlete_id BIGINT, activity_id BIGINT, data JSONB, PRIMARY KEY (athlete_id, activity_id));
# CREATE TABLE streams (athlete_id BIGINT, activity_id BIGINT, data JSONB, PRIMARY KEY (athlete_id, activity_id));
# CREATE INDEX idx_activities_athlete ON activities(athlete_id);
# CREATE INDEX idx_streams_athlete ON streams(athlete_id);
_SUPABASE_ENABLED = False
_SUPABASE_ERROR = ""
try:
    from supabase import create_client as _sb_create_client
    _sb_url = st.secrets.get("supabase", {}).get("url", "")
    _sb_key = st.secrets.get("supabase", {}).get("key", "")
    _SUPABASE_ENABLED = bool(_sb_url and _sb_key)
    if not _sb_url:
        _SUPABASE_ERROR = "Missing supabase.url in secrets"
    elif not _sb_key:
        _SUPABASE_ERROR = "Missing supabase.key in secrets"
except ImportError:
    _SUPABASE_ERROR = "supabase package not installed"
except Exception as _e:
    _SUPABASE_ERROR = str(_e)


@st.cache_resource
def _get_supabase():
    from supabase import create_client
    return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])


def sb_save_athlete(athlete_id: int, display_name: str, refresh_token: str) -> Optional[str]:
    """Returns error string on failure, None on success."""
    try:
        _get_supabase().table("athletes").upsert({
            "athlete_id": athlete_id,
            "display_name": display_name,
            "refresh_token": refresh_token,
            "fetched_at": datetime.utcnow().isoformat(),
        }).execute()
        return None
    except Exception as e:
        return str(e)


def sb_load_athlete(athlete_id: int) -> Optional[dict]:
    try:
        resp = _get_supabase().table("athletes").select("*").eq("athlete_id", athlete_id).maybe_single().execute()
        return resp.data
    except Exception:
        return None


def sb_save_activities(athlete_id: int, activities_raw: list) -> Optional[str]:
    """Returns error string on failure, None on success."""
    try:
        rows = [{"athlete_id": athlete_id, "activity_id": int(a["id"]), "data": a} for a in activities_raw]
        for i in range(0, len(rows), 500):
            _get_supabase().table("activities").upsert(rows[i:i+500]).execute()
        return None
    except Exception as e:
        return str(e)


def sb_load_activities(athlete_id: int) -> Optional[list]:
    try:
        resp = _get_supabase().table("activities").select("data").eq("athlete_id", athlete_id).execute()
        if resp.data:
            return [row["data"] for row in resp.data]
    except Exception:
        pass
    return None


def sb_save_preferences(athlete_id: int, prefs: dict) -> Optional[str]:
    """Save sidebar preferences to athletes table. Returns error string or None."""
    try:
        _get_supabase().table("athletes").update({"preferences": prefs}).eq("athlete_id", athlete_id).execute()
        return None
    except Exception as e:
        return str(e)


def sb_load_preferences(athlete_id: int) -> Optional[dict]:
    """Load saved sidebar preferences. Returns dict or None."""
    try:
        resp = (_get_supabase().table("athletes")
                .select("preferences").eq("athlete_id", athlete_id)
                .maybe_single().execute())
        if resp.data and resp.data.get("preferences"):
            return resp.data["preferences"]
    except Exception:
        pass
    return None


def sb_save_streams(athlete_id: int, streams: Dict[int, dict]) -> None:
    try:
        rows = [{"athlete_id": athlete_id, "activity_id": int(aid), "data": data}
                for aid, data in streams.items()]
        for i in range(0, len(rows), 200):
            _get_supabase().table("streams").upsert(rows[i:i+200]).execute()
    except Exception:
        pass


def sb_load_streams(athlete_id: int) -> Dict[int, dict]:
    try:
        resp = _get_supabase().table("streams").select("activity_id,data").eq("athlete_id", athlete_id).execute()
        return {int(row["activity_id"]): row["data"] for row in (resp.data or [])}
    except Exception:
        return {}
