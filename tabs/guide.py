"""
tabs/guide.py — Metrics Guide tab render function.
"""
import streamlit as st


def render(data: dict, settings: dict) -> None:
    st.subheader("\U0001f4da Metrics Guide \u2014 how to read this dashboard")
    st.caption(
        "Every metric here is explainable and based on published sports science. "
        "Use this tab to understand what each number means, how it is computed, and how to interpret it."
    )

    # ---------- Overview ----------
    with st.expander("Overview \u2014 all-time KPIs, VDOT & training paces", expanded=False):
        st.markdown(
            """
**VO\u2082max estimate (VDOT)**
- **What:** A performance-derived fitness score, not a lab measurement.
- **Calc:** Jack Daniels & Gilbert (1979) quadratic:
  - `v = distance_m / time_min` (m/min at your best recorded effort)
  - `VO\u2082 = \u22124.60 + 0.182258v + 0.000104v\u00b2`
  - `%VO\u2082max = 0.8 + 0.1894393\u00b7e^(\u22120.012778t) + 0.2989558\u00b7e^(\u22120.1932605t)`
  - `VDOT = VO\u2082 / %VO\u2082max`
- **Priority:** Marathon \u2192 HM \u2192 10K \u2192 5K (longer efforts give more reliable estimates).
- **Interpretation:** Each +1 VDOT \u2248 1\u20132% faster race times. Reflects race-pace fitness, not maximal oxygen uptake in a lab.

**Jack Daniels training paces**
Derived from VDOT by back-calculating the velocity at each training intensity:

| Zone | % VO\u2082max | Purpose |
|------|-----------|---------|
| Easy | 70% | Base building, recovery. Most of your running. |
| Marathon | 80% | Aerobic threshold. Goal race pace. |
| Threshold | 86% | Comfortably hard. Raises lactate threshold. |
| Interval | 98% | VO\u2082max pace. 3\u20135 min reps. |
| Repetition | 105% | Speed & economy. 200\u2013400 m reps, full recovery. |

Easy pace is intentionally slow \u2014 most runners train their easy runs 30\u201360 s/km too fast.

**Consistency metrics**
- **Week streak:** consecutive weeks with \u22651 run.
- **Consistent weeks (last 12):** percentage of the last 12 weeks with \u22653 runs.
- **Personal bests:** fastest recorded pace within a distance window (e.g. 4\u20137 km for 5K).
"""
        )

    # ---------- Training Load ----------
    with st.expander("Training Load \u2014 TRIMP, fitness & fatigue", expanded=False):
        st.markdown(
            """
**Training load (Banister TRIMP)**
- **What:** Quantifies the physiological stress of each run.
- **Calc (Men):** `TRIMP = duration\u2009\u00d7\u2009HRr\u2009\u00d7\u20090.64\u2009\u00d7\u2009e^(1.92\u2009\u00d7\u2009HRr)`
- **Calc (Women):** `TRIMP = duration\u2009\u00d7\u2009HRr\u2009\u00d7\u20090.86\u2009\u00d7\u2009e^(1.67\u2009\u00d7\u2009HRr)`
- **Where:** `HRr = (avg_hr \u2212 rest_hr) / (max_hr \u2212 rest_hr)` (heart-rate reserve fraction)
- **Why the exponential?** Metabolic cost increases non-linearly with intensity. A tempo run generates roughly 3\u00d7 the load of an easy run of the same duration \u2014 the linear model underestimates this.
- **Set max HR and resting HR in the sidebar.** Resting HR should be measured lying down first thing in the morning.
- **Source:** Banister EW (1991). *Physiological Testing of Elite Athletes.*

**Acute load / ATL ("Fatigue")**
- **Calc:** 7-day exponentially-weighted moving average (EWMA) of daily TRIMP.
- **Interpretation:** Rises quickly with training, falls quickly with rest. Tracks short-term fatigue.

**Chronic load / CTL ("Fitness")**
- **Calc:** 28-day EWMA of daily TRIMP.
- **Interpretation:** Builds slowly over months. A rising CTL means your fitness base is growing.

**Form / TSB (Training Stress Balance)**
- **Calc:** `TSB = CTL \u2212 ATL`
- **Interpretation (Banister TRIMP units):**

| TSB | State |
|-----|-------|
| > +50 | Fresh \u2014 good window to race or do a breakthrough session |
| +10 to +50 | Good form \u2014 training will be well absorbed |
| \u221225 to +10 | Productive zone \u2014 slightly fatigued but adapting |
| \u221275 to \u221225 | Carrying fatigue \u2014 monitor recovery closely |
| < \u221275 | Heavy fatigue \u2014 consider a recovery block |

Note: TSB is in full TRIMP units (not TrainingPeaks TSS units), so absolute values are larger than you may expect from other platforms.

**ACWR (Acute:Chronic Workload Ratio)**
- **Calc:** `ACWR = ATL / CTL`
- **Interpretation:**
  - 0.8\u20131.3: balanced "sweet spot"
  - \u22721.5: elevated overreach risk
  - \u22721.8: high risk \u2014 back off
- ACWR is a ratio so it is scale-invariant; these thresholds hold regardless of load units.
- **Source:** Gabbett TJ (2016). *British Journal of Sports Medicine*, 50(5), 273\u2013280.

**Training monotony & strain**
- **Monotony:** `mean(daily_load) / std(daily_load)` within the week. > 2.0 = little day-to-day variation \u2014 watch for overtraining.
- **Strain:** `weekly_load \u00d7 monotony`. Big week + low variation = highest risk combination.
- **Source:** Foster C (1998). *Medicine & Science in Sports & Exercise*, 30(7), 1164\u20131168.

**80/20 polarization**
- Approximately 80% easy/long volume and 20% quality (tempo + workout) is the evidence-backed distribution for endurance athletes.
- **Source:** Seiler S & T\u00f8nnessen E (2009). *Sportscience*, 13, 32\u201353.

**HR zones**
Zones are set as % of max HR in the sidebar. Defaults:

| Zone | % Max HR | Purpose |
|------|----------|---------|
| Z1 Recovery | 0\u201360% | Active recovery |
| Z2 Aerobic | 60\u201375% | Base building, fat adaptation |
| Z3 Tempo | 75\u201387% | Marathon to HM effort |
| Z4 Threshold | 87\u201393% | 10K race effort, lactate threshold |
| Z5 VO\u2082max | 93%+ | Short intervals, 5K effort |
"""
        )

    # ---------- Pace & Efficiency ----------
    with st.expander("Pace & Efficiency \u2014 getting faster at the same effort", expanded=False):
        st.markdown(
            """
**Pace vs HR scatter**
- **What:** Output (pace) vs physiological effort (HR).
- **Interpretation:** Over time, the trend should shift toward faster paces at similar HR \u2014 meaning better fitness.
- **Pitfalls:** Hills, wind, heat, and GPS error increase scatter. Use the race-effort HR band to filter for comparable efforts.

**Race-effort HR band**
- A HR intensity range (as % of max HR) set per race distance, e.g. 84\u201392% for a half marathon.
- Filters runs to comparable intensity so you can track efficiency without mixing easy days and hard sessions.

**Efficiency index (speed / HR)**
- **Calc:** `speed_per_hr = avg_speed_mps / avg_hr`
- **Interpretation:** Higher = better economy at the same cardiac cost. Equivalent to Garmin's "Aerobic Efficiency" or Friel's "Efficiency Factor."
- **Pitfalls:** Sensitive to heat, fatigue, dehydration, terrain.

**Grade-adjusted pace (GAP)**
- **What:** What your pace would have been on flat ground, removing the elevation effect.
- **Calc:** Uses the Minetti et al. metabolic cost polynomial:
  `C(g) = 280.5g\u2075 \u2212 58.7g\u2074 \u2212 76.8g\u00b3 + 51.9g\u00b2 + 19.6g + 2.5`
  where g = gradient (fraction). GAP = actual pace \u00d7 (C_flat / C_grade).
- **Requires** the `grade_smooth` stream from Strava (GPS + elevation data). Re-fetch streams if GAP is not showing.
- **Source:** Minetti AE et al. (2002). *Journal of Applied Physiology*, 93(3), 1039\u20131046.
"""
        )

    # ---------- Long Runs ----------
    with st.expander("Long Runs \u2014 durability & aerobic decoupling", expanded=False):
        st.markdown(
            """
**Goal**
Measure durability: how well you hold pace and efficiency as a run progresses.

**Pace fade**
- **Calc:** `(pace_second_half \u2212 pace_first_half) / pace_first_half`
- **Interpretation:** < 5% = strong pacing. > 10% = significant fade (fatigue, pacing error, fueling).

**HR drift**
- **Calc:** `(hr_second_half \u2212 hr_first_half) / hr_first_half`
- **Interpretation:** Rising HR at similar pace = cardiovascular drift (heat, dehydration, low aerobic base).

**Aerobic decoupling (Pa:HR)**
- **What:** The single most informative long-run fitness metric. Measures how much your speed\u00f7HR efficiency drops from first to second half.
- **Calc:** `decoupling = ( (speed/hr)_second / (speed/hr)_first ) \u2212 1`
- **Interpretation:**
  - < 5%: aerobic system held up well \u2014 good aerobic base for this duration.
  - > 5%: system becoming stressed \u2014 normal early in training, should decrease as base develops.
- **Trend over time:** A falling decoupling trend across long runs is one of the clearest signs of improving aerobic fitness.
- **Source:** Friel J. *The Triathlete\u2019s Training Bible.* (Pa:HR decoupling concept.)

**Grade-adjusted pace in run inspector**
- Actual pace vs GAP plotted against distance to show whether pace variation was driven by terrain or true fatigue.
"""
        )

    # ---------- Recovery & Risk ----------
    with st.expander("Recovery & Risk \u2014 overreach and injury-risk proxies", expanded=False):
        st.markdown(
            """
**Composite risk score (0\u2013100)**
A weighted combination of three normalised components:

| Component | Weight | Definition |
|-----------|--------|-----------|
| ACWR | 50% | 0 at ACWR \u22641.0; 1.0 at ACWR \u22652.0 |
| Acute load z-score | 30% | 0 at or below historical mean ATL; 1.0 at +2 SD above it |
| Low rest days | 20% | 0 with \u22653 rest days in last 7; 1.0 with 0 rest days |

ACWR carries the most weight because it has the strongest evidence base for injury risk.

**Alert flags (table)**
Days where one or more of the following conditions are true:

| Flag | Condition |
|------|-----------|
| ACWR >1.5 | Elevated overreach risk zone |
| ACWR >1.8 | High overreach risk zone |
| Low rest | Fewer than 2 rest days in the past 7 |
| Outlier load | Daily TRIMP > 28-day mean + 2 SD |

**Weekly patterns**
- **Load change vs 4-week average:** highlights spikes. Load increases > 25% above a 4-week average are flagged.
- **Load vs monotony (bubble = strain):** large bubbles at high monotony = stressful weeks with little variation \u2014 the most injury-prone pattern.

**Low-efficiency run detector**
- Flags runs where speed/HR fell below the rolling 20th percentile of the last 12 HR-bearing runs.
- A run below this baseline suggests fatigue, illness, or heat stress.

*These are proxies, not medical advice.*
"""
        )

    # ---------- Race Predictor ----------
    with st.expander("Race Predictor \u2014 how the prediction is built", expanded=False):
        st.markdown(
            """
**Riegel power-law model**
- **Calc:** `T2 = T1 \u00d7 (D2 / D1)^k`
- **Default exponent k = 1.06** (Riegel\u2019s original value from world record data).
  - Optimistic: 1.03 (less drop-off at longer distances)
  - Conservative: 1.09 (more drop-off)
- **Source:** Riegel PS (1981). *American Scientist*, 69(3), 285\u2013290.

**Prediction guards**
- **Min distance gate (5 km default):** Excludes short runs (intervals, warm-ups).
- **4\u00d7 extrapolation cap:** Source run must be \u2265 target distance \u00f7 4. Prevents predicting a marathon from a short tempo.
- **Pace sanity gate:** Discards paces outside 2\u201315 min/km to remove GPS artifacts.
- **Median of top 5:** Rather than the single best run (sensitive to one exceptional day), uses the median of the 5 best equivalent performances.

**Efficiency adjustment**
- Compares recent median speed/HR vs historical median speed/HR at race-effort intensity.
- Factor = `clip(ratio^0.6, 0.90, 1.10)`. Dampened and bounded to remain conservative.
- Requires \u22656 HR-bearing runs in the effort band; shows "Not enough data" otherwise.

**Load adjustment**
- A small conservative penalty (up to +10% time) when the composite risk score exceeds 55, indicating high recent load.

**Pacing strategy**
- Even split: goal pace throughout.
- Negative split: first half at pace \u00d7 1.025, second half at pace \u00d7 0.975 (\u00b12.5%).
- **Source:** Abbiss CR & Laursen PB (2008). *Sports Medicine*, 38(3), 239\u2013252.
"""
        )

    # ---------- Raw Streams ----------
    with st.expander("Raw Streams \u2014 per-second sensor data", expanded=False):
        st.markdown(
            """
**What this tab is for**
The raw layer: per-sample signals (typically 1 Hz) directly from Strava. Use it to validate anything in the other tabs or inspect individual run quality.

**Key streams**
- **time (s):** elapsed seconds since start.
- **distance / distance_km:** cumulative distance.
- **velocity_smooth (m/s):** GPS-smoothed speed.
- **pace_min_per_km:** `(1000 / velocity_smooth) / 60`. Lower = faster.
- **heartrate (bpm):** sensor readings when a HR monitor is worn.
- **grade_smooth (%):** terrain gradient, used for grade-adjusted pace (GAP).
- **cadence / altitude / lat\u2013lng:** present depending on device and Strava export.

**How to interpret**
- Look for **trends**, not noise.
- **Stable pace + rising HR** \u2192 cardiovascular drift (heat, dehydration, fatigue).
- **Rising pace (slower) + rising HR** \u2192 fatigue or poor fueling/pacing.
- Pace is inverted on plots \u2014 lower on the y-axis = faster.
"""
        )
