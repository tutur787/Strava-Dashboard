"""
auth.py — Strava OAuth helpers and token management.
"""
import time
from typing import Optional

import requests
import streamlit as st

from config import STRAVA_API_BASE, STRAVA_AUTH_URL, STRAVA_TOKEN_URL


def get_strava_auth_url(client_id: str, redirect_uri: str) -> str:
    from urllib.parse import urlencode
    # Scope is appended separately — urlencode would encode commas/colons as %2C/%3A
    # which Strava's auth server rejects. Scope must be a plain comma-separated string.
    params = urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "approval_prompt": "auto",
    })
    return f"{STRAVA_AUTH_URL}?{params}&scope=read,activity:read_all"


def exchange_strava_code(client_id, client_secret, code, redirect_uri: str = "") -> dict:
    resp = requests.post(STRAVA_TOKEN_URL, data={
        "client_id": client_id, "client_secret": client_secret,
        "code": code, "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    })
    return resp.json()


def refresh_strava_token(client_id, client_secret, refresh_token) -> dict:
    resp = requests.post(STRAVA_TOKEN_URL, data={
        "client_id": client_id, "client_secret": client_secret,
        "refresh_token": refresh_token, "grant_type": "refresh_token",
    })
    return resp.json()


def get_valid_token(client_id, client_secret) -> Optional[str]:
    """Return a valid access token, refreshing if needed. Returns None if not authenticated."""
    tokens = st.session_state.get("strava_tokens")
    if not tokens or "access_token" not in tokens:
        return None
    expires_at = tokens.get("expires_at", 0)
    # If expires_at is 0 or missing, trust the token we just received (don't refresh)
    if expires_at and time.time() > expires_at - 300:
        new_tokens = refresh_strava_token(client_id, client_secret, tokens["refresh_token"])
        if "access_token" in new_tokens:
            st.session_state["strava_tokens"] = new_tokens
            return new_tokens["access_token"]
        # Refresh failed but we still have access_token -- try it anyway
        return tokens.get("access_token")
    return tokens["access_token"]


def _strava_get(url: str, access_token: str, params: dict = None, max_retries: int = 4) -> requests.Response:
    """GET with exponential back-off on 429 (rate limit) and transient 5xx errors."""
    headers = {"Authorization": f"Bearer {access_token}"}
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
            wait = min(wait, 60)
            time.sleep(wait)
            continue
        if resp.status_code >= 500 and attempt < max_retries - 1:
            time.sleep(2 ** attempt)
            continue
        return resp
    return resp  # return last response even if still failing


def fetch_all_activities_api(access_token: str) -> list:
    """Fetch all activities from Strava API (paginated, rate-limit safe)."""
    all_acts = []
    page = 1
    while True:
        resp = _strava_get(
            f"{STRAVA_API_BASE}/athlete/activities", access_token,
            params={"per_page": 200, "page": page},
        )
        if resp.status_code != 200:
            break
        batch = resp.json()
        if not isinstance(batch, list) or len(batch) == 0:
            break
        all_acts.extend(batch)
        if len(batch) < 200:
            break
        page += 1
    return all_acts


def fetch_activity_streams_api(activity_id: int, access_token: str) -> dict:
    """Fetch streams for a single activity (rate-limit safe)."""
    resp = _strava_get(
        f"{STRAVA_API_BASE}/activities/{activity_id}/streams", access_token,
        params={"keys": "time,heartrate,velocity_smooth,cadence,altitude,latlng,grade_smooth", "key_by_type": "true"},
    )
    return resp.json() if resp.status_code == 200 else {}


def fetch_gear_api(gear_id: str, access_token: str) -> dict:
    """Fetch gear (shoe) details."""
    resp = requests.get(
        f"{STRAVA_API_BASE}/gear/{gear_id}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    return resp.json() if resp.status_code == 200 else {}
