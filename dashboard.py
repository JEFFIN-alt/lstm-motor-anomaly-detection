"""
dashboard.py  —  Professional Dark-Theme Monitoring Dashboard
Run with:  streamlit run dashboard.py
"""
import streamlit as st
import serial
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from collections import deque
from tensorflow.keras.models import load_model
import joblib
import time

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be very first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Motor Health Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
#  GLOBAL CSS  — dark theme + custom components
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0d14;
    color: #e2e8f0;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 100%; }

/* ── Header banner ── */
.dash-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 28px;
    background: linear-gradient(135deg, #0f1623 0%, #141c2e 50%, #0f1a2e 100%);
    border: 1px solid #1e2d45;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.dash-header .icon { font-size: 2.4rem; }
.dash-header h1 {
    margin: 0; font-size: 1.5rem; font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.3px;
}
.dash-header .sub {
    margin: 0; font-size: 0.8rem; color: #64748b; font-weight: 400;
}

/* ── Status badge ── */
.status-normal {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 24px;
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 1px solid #065f46;
    border-radius: 14px;
    animation: none;
}
.status-anomaly {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 24px;
    background: linear-gradient(135deg, #1c0505, #2d0a0a);
    border: 1px solid #7f1d1d;
    border-radius: 14px;
    animation: pulseBorder 1.2s ease-in-out infinite;
}
.status-blink {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 24px;
    background: linear-gradient(135deg, #1c0f00, #2d1800);
    border: 1px solid #92400e;
    border-radius: 14px;
    animation: pulseBorder 0.6s ease-in-out infinite;
}
@keyframes pulseBorder {
    0%  { box-shadow: 0 0 0 0 rgba(220,38,38,0.5); }
    50% { box-shadow: 0 0 0 8px rgba(220,38,38,0); }
    100%{ box-shadow: 0 0 0 0 rgba(220,38,38,0); }
}
.status-dot-normal  { width:16px; height:16px; border-radius:50%; background:#22c55e;
    box-shadow:0 0 10px #22c55e; flex-shrink:0; }
.status-dot-anomaly { width:16px; height:16px; border-radius:50%; background:#ef4444;
    box-shadow:0 0 14px #ef4444; flex-shrink:0;
    animation: blinkDot 1.2s step-start infinite; }
.status-dot-blink   { width:16px; height:16px; border-radius:50%; background:#f97316;
    box-shadow:0 0 14px #f97316; flex-shrink:0;
    animation: blinkDot 0.5s step-start infinite; }
@keyframes blinkDot {
    0%,100%{ opacity:1; } 50%{ opacity:0.15; }
}
.status-text-normal  { font-size:1.15rem; font-weight:700; color:#4ade80; }
.status-text-anomaly { font-size:1.15rem; font-weight:700; color:#f87171; }
.status-text-blink   { font-size:1.15rem; font-weight:700; color:#fb923c; }
.status-sub { font-size:0.75rem; color:#94a3b8; margin-top:2px; }

/* ── Metric card ── */
.metric-card {
    background: #0f1623;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    height: 100%;
}
.metric-card .label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #64748b; margin-bottom: 8px;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem; font-weight: 500; line-height: 1;
}
.metric-card .unit { font-size: 0.75rem; color: #64748b; margin-top: 4px; }

/* ── Section header ── */
.section-title {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: #475569;
    border-left: 3px solid #3b82f6;
    padding-left: 10px; margin: 18px 0 10px 0;
}

/* ── Score bar ── */
.score-bar-wrap {
    background: #141c2e; border-radius: 8px;
    padding: 14px 18px; border: 1px solid #1e2d45;
}
.bar-label { font-size:0.72rem; color:#94a3b8; margin-bottom:6px; display:flex; justify-content:space-between; }
.bar-track  { height:8px; background:#1e293b; border-radius:4px; overflow:hidden; }
.bar-fill   { height:100%; border-radius:4px; transition: width 0.3s ease; }

/* ── Log entry ── */
.log-box {
    background: #080b12; border: 1px solid #1e2d45; border-radius: 10px;
    padding: 12px 16px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #94a3b8; max-height: 160px;
    overflow-y: auto; line-height: 1.7;
}
.log-anomaly { color: #f87171; }
.log-blink   { color: #fb923c; }
.log-normal  { color: #4ade80; }

/* ── Table styling ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────
COM_PORT            = "COM7"
BAUD_RATE           = 9600
SEQUENCE_LEN        = 30
FEATURE_COLS        = ["ax","ay","az","gx","gy","gz","temp","flame"]
SENSOR_IDX          = 7
NUM_FEATURES        = len(FEATURE_COLS)

LSTM_THRESHOLD      = 0.0628
VIBRATION_THRESHOLD = 60000
VIBRATION_WINDOW    = 10
VIBRATION_STD_THRESH= 20000
VIB_WEIGHT          = 0.4
LSTM_WEIGHT         = 0.6

MAX_HIST            = 120   # samples kept in chart history

# ─────────────────────────────────────────────────────────────────
#  LOAD MODEL & SCALER  (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading LSTM model…")
def load_resources():
    m = load_model("motor_anomaly_lstm.keras")
    s = joblib.load("scaler.save")
    m.predict(np.zeros((1, SEQUENCE_LEN, NUM_FEATURES)), verbose=0)  # warm-up
    return m, s

model, scaler = load_resources()

# ─────────────────────────────────────────────────────────────────
#  SERIAL  (cached — opens only once)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Arduino…")
def open_serial():
    s = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.5, write_timeout=1)
    time.sleep(0.5)
    s.reset_input_buffer()
    s.write(b'0')   # start green
    return s

ser = open_serial()

# ─────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────
def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("sequence",      deque(maxlen=SEQUENCE_LEN))
_init("mse_hist",      deque(maxlen=MAX_HIST))
_init("vib_hist",      deque(maxlen=MAX_HIST))
_init("score_hist",    deque(maxlen=MAX_HIST))
_init("status",        "STARTING")    # NORMAL / ANOMALY / CRITICAL
_init("led_state",     'G')
_init("event_log",     [])            # list of (time_str, msg, kind)
_init("last_mse",      0.0)
_init("last_vib",      0.0)
_init("last_combined", 0.0)
_init("last_temp",     0.0)
_init("last_sensor",   1)
_init("accel_hist",    deque(maxlen=VIBRATION_WINDOW))

def add_event(msg, kind="normal"):
    ts = time.strftime("%H:%M:%S")
    st.session_state.event_log.insert(0, (ts, msg, kind))
    if len(st.session_state.event_log) > 30:
        st.session_state.event_log.pop()

def set_led(state: str):
    if state == st.session_state.led_state:
        return
    st.session_state.led_state = state
    try:
        if   state == 'B': ser.write(b'2')
        elif state == 'R': ser.write(b'1')
        else:              ser.write(b'0')
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────
#  READ + PROCESS ONE SAMPLE
# ─────────────────────────────────────────────────────────────────
def process_sample():
    try:
        raw = ser.readline()
    except Exception:
        return

    if not raw:
        return

    try:
        line = raw.decode("utf-8", errors="ignore").strip()
    except Exception:
        return

    parts = line.split(",")
    if len(parts) != NUM_FEATURES:
        return

    try:
        values = [float(v) for v in parts]
    except ValueError:
        return

    ax, ay, az   = values[0], values[1], values[2]
    sensor_val   = int(values[SENSOR_IDX])
    temp_val     = values[6]

    # ── Vibration ──
    accel_mag = (ax**2 + ay**2 + az**2) ** 0.5
    st.session_state.accel_hist.append(accel_mag)

    vib_spike_score = min(accel_mag / VIBRATION_THRESHOLD, 1.0)
    vib_std_score   = 0.0
    if len(st.session_state.accel_hist) == VIBRATION_WINDOW:
        vib_std = float(np.std(st.session_state.accel_hist))
        vib_std_score = min(vib_std / VIBRATION_STD_THRESH, 1.0)
    vib_score = max(vib_spike_score, vib_std_score)

    st.session_state.vib_hist.append(accel_mag)
    st.session_state.last_vib    = accel_mag
    st.session_state.last_temp   = temp_val
    st.session_state.last_sensor = sensor_val

    # ── Sensor = 0 (critical threshold breach) ──
    if sensor_val == 0:
        set_led('B')
        st.session_state.status = "CRITICAL"
        add_event("Sensor threshold breached — anomaly detected", "blink")
        # still update buffer
        raw_df = pd.DataFrame([values], columns=FEATURE_COLS)
        scaled = scaler.transform(raw_df)[0]
        st.session_state.sequence.append(scaled)
        st.session_state.score_hist.append(1.0)
        st.session_state.last_combined = 1.0
        return

    # ── LSTM ──
    raw_df = pd.DataFrame([values], columns=FEATURE_COLS)
    scaled = scaler.transform(raw_df)[0]
    st.session_state.sequence.append(scaled)

    if len(st.session_state.sequence) < SEQUENCE_LEN:
        return

    X     = np.array(st.session_state.sequence, dtype=np.float32).reshape(1, SEQUENCE_LEN, NUM_FEATURES)
    recon = model.predict(X, verbose=0)
    mse   = float(np.mean(np.square(X - recon)))

    lstm_score    = min(mse / LSTM_THRESHOLD, 1.0)
    combined      = (LSTM_WEIGHT * lstm_score) + (VIB_WEIGHT * vib_score)
    anomaly_this  = combined >= 1.0 or vib_spike_score >= 1.0 or vib_std_score >= 1.0

    st.session_state.mse_hist.append(mse)
    st.session_state.score_hist.append(combined)
    st.session_state.last_mse      = mse
    st.session_state.last_combined = combined

    if anomaly_this:
        set_led('R')
        if st.session_state.status != "ANOMALY":
            st.session_state.status = "ANOMALY"
            add_event(f"Anomaly detected — score={combined:.3f}  MSE={mse:.5f}", "anomaly")
    else:
        if st.session_state.led_state != 'B':
            set_led('G')
        if st.session_state.status not in ("NORMAL", "STARTING"):
            add_event(f"System returned to normal — combined={combined:.3f}", "normal")
        if st.session_state.led_state == 'G':
            st.session_state.status = "NORMAL"

# ─────────────────────────────────────────────────────────────────
#  RENDER UI
# ─────────────────────────────────────────────────────────────────
def render():
    status       = st.session_state.status
    mse          = st.session_state.last_mse
    vib          = st.session_state.last_vib
    combined     = st.session_state.last_combined
    temp         = st.session_state.last_temp
    sensor_val   = st.session_state.last_sensor

    # ── Header ──
    st.markdown("""
    <div class="dash-header">
      <span class="icon">⚙️</span>
      <div>
        <h1>Motor Health Monitoring</h1>
        <p class="sub">LSTM Anomaly Detection  •  Real-time sensor fusion</p>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Status + Metrics row ──
    col_status, col_mse, col_vib, col_score, col_temp = st.columns([2.2, 1, 1, 1, 1])

    with col_status:
        if status == "CRITICAL":
            cls, dot_cls, txt_cls, label = "status-blink","status-dot-blink","status-text-blink","⚠ ANOMALY DETECTED"
            sub = "Sensor threshold breached — red LED blinking"
        elif status == "ANOMALY":
            cls, dot_cls, txt_cls, label = "status-anomaly","status-dot-anomaly","status-text-anomaly","⚠ ANOMALY DETECTED"
            sub = "Combined score exceeded threshold"
        else:
            cls, dot_cls, txt_cls, label = "status-normal","status-dot-normal","status-text-normal","✓ SYSTEM NORMAL"
            sub = "All parameters within normal range"

        st.markdown(f"""
        <div class="{cls}">
          <div class="{dot_cls}"></div>
          <div>
            <div class="{txt_cls}">{label}</div>
            <div class="status-sub">{sub}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    with col_mse:
        mse_color = "#ef4444" if mse > LSTM_THRESHOLD else "#60a5fa"
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Reconstruction Error</div>
          <div class="value" style="color:{mse_color}">{mse:.5f}</div>
          <div class="unit">threshold {LSTM_THRESHOLD}</div>
        </div>""", unsafe_allow_html=True)

    with col_vib:
        vib_pct   = min(vib / VIBRATION_THRESHOLD * 100, 100)
        vib_color = "#ef4444" if vib_pct > 80 else ("#f59e0b" if vib_pct > 50 else "#60a5fa")
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Vibration Magnitude</div>
          <div class="value" style="color:{vib_color}">{int(vib):,}</div>
          <div class="unit">raw accel units</div>
        </div>""", unsafe_allow_html=True)

    with col_score:
        sc_color = "#ef4444" if combined >= 1.0 else ("#f59e0b" if combined > 0.6 else "#4ade80")
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Combined Score</div>
          <div class="value" style="color:{sc_color}">{combined:.3f}</div>
          <div class="unit">threshold 1.000</div>
        </div>""", unsafe_allow_html=True)

    with col_temp:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Temperature</div>
          <div class="value" style="color:#38bdf8">{temp:.1f}</div>
          <div class="unit">°C</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Score bars ──
    mse_pct  = min(mse / LSTM_THRESHOLD * 100, 100)
    sc_pct   = min(combined * 100, 100)

    mse_bar_color  = "#ef4444" if mse_pct  >= 100 else ("#f59e0b" if mse_pct  > 60 else "#3b82f6")
    sc_bar_color   = "#ef4444" if sc_pct   >= 100 else ("#f59e0b" if sc_pct   > 60 else "#22c55e")
    vib_bar_color  = "#ef4444" if vib_pct  >= 100 else ("#f59e0b" if vib_pct  > 50 else "#8b5cf6")

    col_bars1, col_bars2, col_bars3 = st.columns(3)
    with col_bars1:
        st.markdown(f"""
        <div class="score-bar-wrap">
          <div class="bar-label"><span>LSTM Error</span><span>{mse_pct:.0f}%</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:{mse_pct}%;background:{mse_bar_color};"></div></div>
        </div>""", unsafe_allow_html=True)
    with col_bars2:
        st.markdown(f"""
        <div class="score-bar-wrap">
          <div class="bar-label"><span>Combined Score</span><span>{sc_pct:.0f}%</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:{sc_pct}%;background:{sc_bar_color};"></div></div>
        </div>""", unsafe_allow_html=True)
    with col_bars3:
        st.markdown(f"""
        <div class="score-bar-wrap">
          <div class="bar-label"><span>Vibration Load</span><span>{vib_pct:.0f}%</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:{vib_pct}%;background:{vib_bar_color};"></div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphs ──
    col_g1, col_g2 = st.columns([3, 2])

    DARK_BG   = "rgba(9,12,22,0)"
    PLOT_BG   = "rgba(15,22,35,1)"
    GRID_CLR  = "#1e2d45"
    TICK_CLR  = "#475569"
    layout_base = dict(
        paper_bgcolor=DARK_BG, plot_bgcolor=PLOT_BG,
        margin=dict(l=10, r=10, t=36, b=10),
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=TICK_CLR)),
        yaxis=dict(gridcolor=GRID_CLR, zeroline=False, tickfont=dict(color=TICK_CLR)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    )

    with col_g1:
        st.markdown('<div class="section-title">RECONSTRUCTION ERROR  vs  THRESHOLD</div>', unsafe_allow_html=True)
        mse_vals   = list(st.session_state.mse_hist)
        sc_vals    = list(st.session_state.score_hist)
        x_idx      = list(range(len(mse_vals)))

        fig1 = go.Figure()
        # Fill under MSE line
        fig1.add_trace(go.Scatter(
            x=x_idx, y=mse_vals, mode="lines", name="MSE",
            line=dict(color="#38bdf8", width=2),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
        ))
        # Combined score
        fig1.add_trace(go.Scatter(
            x=x_idx, y=sc_vals, mode="lines", name="Combined Score",
            line=dict(color="#a78bfa", width=1.5, dash="dot"),
        ))
        # Threshold line
        fig1.add_hline(
            y=LSTM_THRESHOLD, line_dash="dash", line_color="#ef4444", line_width=1.5,
            annotation_text=f"Threshold ({LSTM_THRESHOLD})",
            annotation_font_color="#ef4444", annotation_font_size=11,
        )
        fig1.update_layout(height=280, title=None, **layout_base)
        fig1.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    with col_g2:
        st.markdown('<div class="section-title">VIBRATION MAGNITUDE</div>', unsafe_allow_html=True)
        vib_vals = list(st.session_state.vib_hist)
        v_x      = list(range(len(vib_vals)))

        # Color each bar based on level
        bar_colors = []
        for v in vib_vals:
            pct = v / VIBRATION_THRESHOLD
            bar_colors.append("#ef4444" if pct >= 1.0 else ("#f59e0b" if pct > 0.6 else "#8b5cf6"))

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=v_x, y=vib_vals, name="Accel Magnitude",
            marker_color=bar_colors, marker_line_width=0,
        ))
        fig2.add_hline(
            y=VIBRATION_THRESHOLD, line_dash="dash", line_color="#ef4444", line_width=1.2,
            annotation_text="Spike threshold",
            annotation_font_color="#ef4444", annotation_font_size=10,
        )
        fig2.update_layout(height=280, title=None, bargap=0.1, **layout_base)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Bottom: sensor table + event log ──
    col_t, col_log = st.columns([1, 1])

    with col_t:
        st.markdown('<div class="section-title">LIVE SENSOR READINGS</div>', unsafe_allow_html=True)
        if st.session_state.mse_hist:
            ax = ay = az = gx = gy = gz = 0.0  # placeholder if no raw saved
        sensor_icon = "🔴" if sensor_val == 0 else "🟢"
        raw_df_show = pd.DataFrame({
            "Parameter": ["Accel X", "Accel Y", "Accel Z",
                          "Gyro X", "Gyro Y", "Gyro Z",
                          "Temperature", "Sensor"],
            "Value":     [
                f"{vib:.0f} (mag)",
                "—", "—", "—", "—", "—",
                f"{temp:.1f} °C",
                f"{sensor_icon}  {'Triggered' if sensor_val == 0 else 'Normal'}",
            ],
        })
        st.dataframe(raw_df_show, use_container_width=True, hide_index=True)

    with col_log:
        st.markdown('<div class="section-title">EVENT LOG</div>', unsafe_allow_html=True)
        log_html = ""
        for ts, msg, kind in st.session_state.event_log[:10]:
            css = f"log-{kind}"
            log_html += f'<div class="{css}">[{ts}] {msg}</div>\n'
        if not log_html:
            log_html = '<div style="color:#475569">No events yet — waiting for data…</div>'
        st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────
SAMPLES_PER_RENDER = 3   # process this many samples before re-rendering

for _ in range(SAMPLES_PER_RENDER):
    process_sample()

render()

# Re-run immediately to keep the dashboard live
time.sleep(0.05)
st.rerun()