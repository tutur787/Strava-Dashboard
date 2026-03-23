"""
tabs/guide.py — Metrics Guide tab render function.
"""
import streamlit as st


def render(data: dict, settings: dict) -> None:
    st.subheader("\U0001f4da Metrics Guide (How to read this dashboard)")
    st.caption(
        "These are explainable, HR-based training analytics. Use this tab to understand what each metric means, how it's computed, and how to interpret it."
    )

    def _bullets(items):
        return "\n".join([f"- {x}" for x in items])

    # ---------- Card: Raw Streams ----------
    with st.expander("Raw Streams \u2014 per-second sensor data", expanded=False):
        st.markdown(
            """
**What this tab is for**
This is the "raw truth" layer: per-sample signals (often 1 Hz) from Strava streams. Use it to validate anything you see in later tabs.

**Key streams you'll see**
- **time (s)**: elapsed seconds since start.
- **distance / distance_km**: cumulative distance.
- **velocity_smooth (m/s)**: GPS-smoothed speed.
- **pace_min_per_km**: derived from speed; lower is faster.
- **heartrate (bpm)**: sensor readings when available.
- **cadence / grade_smooth / altitude / lat/lng**: present depending on device + Strava export.

**How derived fields are calculated**
- `distance_km = distance / 1000`
- `pace_min_per_km = (1000 / velocity_smooth) / 60`

**How to interpret**
- Look for **trends**, not noise: pacing drift, HR drift, surges, stop/start artifacts.
- **Stable pace + rising HR** \u2192 cardiovascular drift (heat, dehydration, fatigue).
- **Rising pace (slower) + rising HR** \u2192 fatigue or poor fueling/pacing.
- Pace is inverted on plots (lower is better).
"""
        )

    # ---------- Card: Training Load ----------
    with st.expander("Training Load \u2014 volume, intensity & fitness trend", expanded=False):
        st.markdown(
            """
**Goal**
Quantify "how much stress" you're absorbing and how quickly training is changing.

**Core metric: HR intensity**
- **What:** Relative effort.
- **Calc:** `hr_intensity = avg_hr / max_hr`
- **Interpretation:** Higher = harder. Accuracy depends on realistic max HR.

**Daily HR load**
- **What:** Stress for the day (duration \u00d7 intensity).
- **Calc:** `daily_load = duration_min \u00d7 hr_intensity`
- **Interpretation:** 60 min @ 0.75 \u2248 45 min @ 1.00 in load terms.

**Acute load (7d EWMA)**
- **What:** Short-term fatigue proxy.
- **Calc:** EWMA of daily load over ~7 days.
- **Interpretation:** Rising fast = accumulating fatigue; falling = recovery.

**Chronic load (28d EWMA)**
- **What:** Longer-term fitness proxy.
- **Calc:** EWMA of daily load over ~28 days.
- **Interpretation:** Gradual rise = sustainable build.

**ACWR (Acute:Chronic Workload Ratio)**
- **What:** Change management proxy.
- **Calc:** `acwr = acute_load / chronic_load`
- **Interpretation:**
  - ~0.8-1.3: generally "balanced"
  - greater than 1.5: higher risk of overreaching (heuristic, not medical advice)

**Race-normalized volume**
- **Weekly distance / race distance**: `weekly_km / race_km`
- **Long run % of race**: `longest_run_km / race_km`
**Interpretation:** Helps compare readiness across race distances.

**Monotony & strain**
- **Monotony:** mean daily load / std(daily load) within the week
- **Strain:** weekly load \u00d7 monotony
**Interpretation:** High monotony means low variation; high strain = big week + little variation \u2192 watch recovery.
"""
        )

    # ---------- Card: Pace & Efficiency ----------
    with st.expander("Pace & Efficiency \u2014 getting faster at the same effort", expanded=False):
        st.markdown(
            """
**Goal**
Track fitness changes while controlling for effort.

**Pace vs HR scatter**
- **What:** Output (pace) vs physiological effort (HR).
- **Interpretation:** Over time, "better" usually shifts toward faster paces at similar HR.
- **Pitfalls:** Hills, wind, heat, and GPS error increase scatter.

**Race-effort HR band**
- **What:** Filter for comparable intensity (e.g., HM effort).
- **Calc:** `%maxHR` range picked in sidebar.
- **Interpretation:** Compare runs inside the band to avoid "easy day vs workout" bias.

**Efficiency index (speed / HR)**
- **What:** Output per cardiovascular cost.
- **Calc:** `speed_per_hr = avg_speed_mps / avg_hr`
- **Interpretation:** Higher = better economy/fitness at similar conditions.
- **Pitfalls:** Sensitive to heat, fatigue, dehydration, terrain.

**Rolling medians/trends**
- **What:** Smoothed view of noisy field data.
- **Interpretation:** Rising efficiency trend = improving; falling trend can signal fatigue/overreach.
"""
        )

    # ---------- Card: Long Runs ----------
    with st.expander("Long Runs \u2014 durability & fatigue within each run", expanded=False):
        st.markdown(
            """
**Goal**
Measure durability: how well you hold pace/efficiency as the run progresses.

**Pace fade**
- **What:** How much slower you get later in the run.
- **Calc:** `(pace_second_half - pace_first_half) / pace_first_half`
- **Interpretation:**
  - Small fade (<~3\u20135%) = strong pacing/durability
  - Larger fade can indicate fatigue, pacing error, fueling issues

**HR drift**
- **What:** HR increase later in the run at similar pace.
- **Calc:** `(hr_second_half - hr_first_half) / hr_first_half`
- **Interpretation:** Elevated drift often correlates with heat, dehydration, or low aerobic base.

**Decoupling (efficiency loss)**
- **What:** Drop in (speed/HR) from first to second half.
- **Calc:** `( (speed/hr)_second / (speed/hr)_first ) - 1`
- **Interpretation:** Negative = efficiency decline; closer to 0 = better aerobic durability.

**Run inspector plot**
- Pace & HR vs distance helps you visually confirm whether fatigue was gradual, abrupt, or due to stops/terrain.
"""
        )

    # ---------- Card: Recovery & Risk ----------
    with st.expander("Recovery & Risk \u2014 overreach and injury-risk proxies", expanded=False):
        st.markdown(
            """
**Goal**
Provide decision-support indicators for recovery and overuse risk.

**Rest days (last 7)**
- **What:** How much true downtime you've had.
- **Calc:** Count of days with ~0 HR load in last 7 days.
- **Interpretation:** Consistently low rest \u2192 watch cumulative fatigue.

**Daily risk score (0\u2013100)**
- **What:** Composite proxy based on workload + recovery signals.
- **Inputs:** ACWR, acute load relative to baseline, low rest.
- **Interpretation:**
  - Lower = likely fresher
  - Higher = more caution (not medical advice)

**Flags**
- ACWR high / very high
- Low rest
- "Big day" (outlier daily load vs recent distribution)
**Interpretation:** Multiple flags close together = consider backing off.

**Weekly patterns**
- **Load change vs 4-week avg:** highlights spikes (often where injuries happen).
- **Load vs monotony (bubble size = strain):** big bubbles at high monotony = stressful weeks with little variation.
"""
        )

    # ---------- Card: Race Predictor ----------
    with st.expander("Race Predictor \u2014 how the prediction is built", expanded=False):
        st.markdown(
            """
**Goal**
Estimate race-day time using recent efforts, distance scaling, efficiency trend, and a small readiness penalty.

**Minimum source distance & extrapolation cap**
- **What:** Two guards against unreliable extrapolation.
- **Min distance:** Excludes short runs (intervals, warm-ups) that are too noisy.
- **Max ratio cap (4×):** Source run must be ≥ target distance ÷ 4 — prevents predicting a marathon from a 5 km jog.
- **Pace sanity gate:** Discards degenerate paces (<2 min/km or >15 min/km) that indicate GPS or stop/start artifacts.

**Consensus from top efforts**
- Rather than using the single best-equivalent run (which is sensitive to one exceptional day), the predictor takes the **median of the top 5 best-equivalent performances**, giving a more stable estimate.

**Distance scaling (Riegel-style power law)**
- **Idea:** Predict time at distance D2 from effort at D1:
  `T2 = T1 \u00d7 (D2 / D1)^k`
- **Interpretation:** Good for endurance extrapolation when using representative efforts.
- **Pitfalls:** Less reliable if the input run is not steady-state (intervals, lots of stops).

**Efficiency adjustment**
- Uses the trend in speed/HR to nudge prediction based on recent fitness direction.

**Risk penalty**
- Small conservative penalty if Tab 4 indicates higher load/risk (reduces overconfidence).

**How to interpret the result**
- Treat it as a conditional estimate: "If conditions match recent training and I'm not overreached, this is plausible."
"""
        )
