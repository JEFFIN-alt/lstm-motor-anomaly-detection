"""
ui_detector.py  —  Dark-themed real-time motor anomaly monitor
Run with: python ui_detector.py
Same conda env and folder as real_time_detector.py
"""

import threading
import time
import tkinter as tk
from collections import deque

import joblib
import numpy as np
import pandas as pd
import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tensorflow.keras.models import load_model

# ──────────────────────────────────────────────
#  CONFIG  (same as real_time_detector.py)
# ──────────────────────────────────────────────
COM_PORT             = "COM7"
BAUD_RATE            = 9600
SEQUENCE_LEN         = 30
FEATURE_COLS         = ["ax","ay","az","gx","gy","gz","temp","flame"]
SENSOR_IDX           = 7

LSTM_THRESHOLD       = 0.0628
VIBRATION_THRESHOLD  = 60000
VIBRATION_STD_THRESH = 20000
VIBRATION_WINDOW     = 10
VIB_WEIGHT           = 0.4
LSTM_WEIGHT          = 0.6
HIST                 = 80   # points on chart

# ──────────────────────────────────────────────
#  LOAD MODEL & SCALER
# ──────────────────────────────────────────────
print("Loading model and scaler…")
model  = load_model("motor_anomaly_lstm.keras")
scaler = joblib.load("scaler.save")
model.predict(np.zeros((1, SEQUENCE_LEN, len(FEATURE_COLS))), verbose=0)
print("Model ready.")

# ──────────────────────────────────────────────
#  SERIAL  (same retry logic as real_time_detector.py)
# ──────────────────────────────────────────────
print(f"Opening {COM_PORT} …")
ser = None
for attempt in range(1, 4):
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1, write_timeout=1)
        try: ser.reset_input_buffer()
        except Exception: pass
        time.sleep(0.5)
        print(f"Connected on attempt {attempt}.")
        break
    except serial.SerialException as e:
        print(f"Attempt {attempt}/3 failed: {e}")
        if attempt < 3: time.sleep(2)
        else: raise SystemExit("Cannot open COM port. Close Arduino Serial Monitor and retry.")

# Let Arduino finish resetting, then flush garbled startup bytes
time.sleep(2)
ser.reset_input_buffer()
ser.write(b'0')   # start GREEN
print("Ready.\n")

# ──────────────────────────────────────────────
#  SHARED STATE
# ──────────────────────────────────────────────
_lock   = threading.Lock()
_state  = {
    "status"   : "BUFFERING",
    "mse"      : 0.0,
    "vib"      : 0.0,
    "combined" : 0.0,
    "led"      : "G",
    "buf"      : 0,
    "values"   : [0.0] * len(FEATURE_COLS),
    "log"      : [],          # list of (timestamp, msg, kind)
}
_mse_hist   = deque([0.0]*HIST, maxlen=HIST)
_score_hist = deque([0.0]*HIST, maxlen=HIST)

# ──────────────────────────────────────────────
#  DETECTOR THREAD  (exact logic from real_time_detector.py)
# ──────────────────────────────────────────────
_seq        = deque(maxlen=SEQUENCE_LEN)
_accel_hist = deque(maxlen=VIBRATION_WINDOW)

def _set_led(new):
    if _state["led"] == new: return
    _state["led"] = new
    try: ser.write({"B":b"2","R":b"1","G":b"0"}[new])
    except Exception: pass

def _log(msg, kind):
    ts = time.strftime("%H:%M:%S")
    _state["log"].insert(0, (ts, msg, kind))
    _state["log"] = _state["log"][:30]

def detector():
    while True:
        try:  raw = ser.readline()
        except serial.SerialException: time.sleep(0.5); continue
        if not raw: continue
        try:   line = raw.decode("utf-8", errors="ignore").strip()
        except Exception: continue
        parts = line.split(",")
        if len(parts) != len(FEATURE_COLS): continue
        try:   values = [float(v) for v in parts]
        except ValueError: continue

        ax, ay, az = values[0], values[1], values[2]
        sensor_val = int(values[SENSOR_IDX])
        accel_mag  = (ax**2 + ay**2 + az**2)**0.5
        _accel_hist.append(accel_mag)

        vib_spike = min(accel_mag / VIBRATION_THRESHOLD, 1.0)
        vib_std_s = 0.0
        if len(_accel_hist) == VIBRATION_WINDOW:
            vib_std_s = min(np.std(_accel_hist) / VIBRATION_STD_THRESH, 1.0)
        vib_score = max(vib_spike, vib_std_s)

        # scale & buffer
        raw_df = pd.DataFrame([values], columns=FEATURE_COLS)
        scaled = scaler.transform(raw_df)[0]
        _seq.append(scaled)

        with _lock:
            _state["values"] = values
            _state["vib"]    = accel_mag

        # instant sensor breach
        if sensor_val == 0:
            with _lock:
                if _state["status"] != "CRITICAL":
                    _log("Sensor threshold breach — ANOMALY", "critical")
                _set_led("B")
                _state["status"]   = "CRITICAL"
                _state["combined"] = 1.0
                _score_hist.append(1.0)
            continue

        buf_now = len(_seq)
        if buf_now < SEQUENCE_LEN:
            with _lock:
                _state["buf"]    = buf_now
                _state["status"] = "BUFFERING"
            continue

        # LSTM inference
        X     = np.array(_seq, dtype=np.float32).reshape(1, SEQUENCE_LEN, len(FEATURE_COLS))
        recon = model.predict(X, verbose=0)
        mse   = float(np.mean(np.square(X - recon)))

        lstm_s   = min(mse / LSTM_THRESHOLD, 1.0)
        combined = LSTM_WEIGHT * lstm_s + VIB_WEIGHT * vib_score
        anomaly  = combined >= 1.0 or vib_spike >= 1.0 or vib_std_s >= 1.0

        with _lock:
            _mse_hist.append(mse)
            _score_hist.append(combined)
            _state["mse"]      = mse
            _state["combined"] = combined
            prev = _state["status"]
            if anomaly:
                _set_led("R")
                _state["status"] = "ANOMALY"
                if prev != "ANOMALY":
                    _log(f"Anomaly  score={combined:.3f}  MSE={mse:.5f}", "anomaly")
            else:
                if _state["led"] != "B":
                    _set_led("G")
                    if prev not in ("NORMAL","BUFFERING","STARTING"):
                        _log(f"Normal  combined={combined:.3f}", "normal")
                    _state["status"] = "NORMAL"

threading.Thread(target=detector, daemon=True).start()


# ──────────────────────────────────────────────
#  COLOUR PALETTE
# ──────────────────────────────────────────────
BG     = "#0d1117"
BG2    = "#161b22"
BG3    = "#21262d"
BORDER = "#30363d"
CYAN   = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
ORANGE = "#e3b341"
TXT    = "#c9d1d9"
TXT2   = "#8b949e"


# ──────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────
class MonitorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Motor Anomaly Monitor")
        self.configure(bg=BG)
        self.geometry("900x600")
        self.resizable(True, True)
        self._build()
        self.after(300, self._refresh)

    # ── layout ───────────────────────────────
    def _build(self):
        # ── top bar ──
        bar = tk.Frame(self, bg=BG2, height=44)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="⬡  Motor Anomaly Monitor", bg=BG2, fg=CYAN,
                 font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT, padx=16, pady=10)
        self._clock = tk.Label(bar, text="", bg=BG2, fg=TXT2, font=("Consolas", 10))
        self._clock.pack(side=tk.RIGHT, padx=16)
        tk.Label(bar, text=f"COM7 · {BAUD_RATE}", bg=BG2, fg=TXT2,
                 font=("Consolas", 10)).pack(side=tk.RIGHT, padx=8)

        tk.Frame(self, bg=BORDER, height=1).pack(fill=tk.X)

        # ── main body ──
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)

        # left panel (250 px)
        left = tk.Frame(body, bg=BG2, width=250)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        tk.Frame(body, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)

        # right panel
        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_left(left)
        self._build_right(right)

    # ── left panel ───────────────────────────
    def _build_left(self, p):
        pad = dict(padx=14, pady=5)

        # status
        self._lbl(p, "STATUS").pack(anchor="w", **pad)
        self._status_frame = tk.Frame(p, bg=BG3, bd=0, padx=12, pady=10)
        self._status_frame.pack(fill=tk.X, padx=10, pady=(0,4))

        dot_row = tk.Frame(self._status_frame, bg=BG3)
        dot_row.pack(anchor="w")
        self._dot = tk.Label(dot_row, text="●", bg=BG3, fg=GREEN,
                              font=("Segoe UI", 20))
        self._dot.pack(side=tk.LEFT)
        self._status_txt = tk.Label(dot_row, text="Connecting…", bg=BG3, fg=GREEN,
                                     font=("Segoe UI", 12, "bold"))
        self._status_txt.pack(side=tk.LEFT, padx=6)
        self._status_sub = tk.Label(self._status_frame, text="", bg=BG3, fg=TXT2,
                                     font=("Consolas", 8), wraplength=200, justify="left")
        self._status_sub.pack(anchor="w")

        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, padx=10, pady=4)

        # MSE
        self._lbl(p, "RECONSTRUCTION ERROR (MSE)").pack(anchor="w", **pad)
        self._mse_val = tk.Label(p, text="—", bg=BG2, fg=CYAN,
                                  font=("Consolas", 22, "bold"))
        self._mse_val.pack(anchor="w", padx=14)
        tk.Label(p, text=f"threshold  {LSTM_THRESHOLD}", bg=BG2, fg=TXT2,
                 font=("Consolas", 8)).pack(anchor="w", padx=14)

        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, padx=10, pady=6)

        # combined + vib
        self._lbl(p, "COMBINED SCORE").pack(anchor="w", **pad)
        self._sc_val = tk.Label(p, text="—", bg=BG2, fg=GREEN,
                                 font=("Consolas", 16, "bold"))
        self._sc_val.pack(anchor="w", padx=14)

        self._lbl(p, "VIBRATION MAG").pack(anchor="w", padx=14, pady=(8,2))
        self._vib_val = tk.Label(p, text="—", bg=BG2, fg=TXT2,
                                  font=("Consolas", 11))
        self._vib_val.pack(anchor="w", padx=14)

        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, padx=10, pady=6)

        # LED state
        self._lbl(p, "ARDUINO LED").pack(anchor="w", **pad)
        self._led_val = tk.Label(p, text="—", bg=BG2, fg=GREEN,
                                  font=("Consolas", 10))
        self._led_val.pack(anchor="w", padx=14)

        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, padx=10, pady=6)

        # event log
        self._lbl(p, "EVENT LOG").pack(anchor="w", **pad)
        log_f = tk.Frame(p, bg=BG3)
        log_f.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        self._log_txt = tk.Text(log_f, bg=BG3, fg=TXT2, font=("Consolas", 8),
                                 relief="flat", bd=0, state="disabled",
                                 highlightthickness=0, wrap="word")
        self._log_txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._log_txt.tag_config("anomaly",  foreground=RED)
        self._log_txt.tag_config("critical", foreground=ORANGE)
        self._log_txt.tag_config("normal",   foreground=GREEN)
        self._log_txt.tag_config("ts",       foreground=TXT2)

    # ── right panel ──────────────────────────
    def _build_right(self, p):
        # chart
        fig = Figure(figsize=(1,1), facecolor=BG)
        self._ax = fig.add_subplot(111)
        self._ax.set_facecolor(BG2)
        self._ax.tick_params(colors=TXT2, labelsize=8)
        for sp in self._ax.spines.values():
            sp.set_color(BORDER); sp.set_linewidth(0.6)
        self._ax.axhline(LSTM_THRESHOLD, color=RED, lw=1, ls="--", alpha=0.7)
        self._ax.set_ylabel("MSE / Score", color=TXT2, fontsize=9)
        self._ax.set_xticks([])

        xs = list(range(HIST))
        self._ln_mse,   = self._ax.plot(xs, [0]*HIST, color=CYAN,   lw=1.8, label="MSE")
        self._ln_score, = self._ax.plot(xs, [0]*HIST, color=ORANGE, lw=1.2,
                                         ls="--", label="Combined")
        self._ax.legend(loc="upper left", fontsize=8, framealpha=0,
                        labelcolor=[CYAN, ORANGE])
        fig.tight_layout(pad=1)

        canvas = FigureCanvasTkAgg(fig, master=p)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._canvas = canvas
        self._fig    = fig

        # separator
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X)

        # sensor chips row
        chip_area = tk.Frame(p, bg=BG2, height=70)
        chip_area.pack(fill=tk.X)
        chip_area.pack_propagate(False)
        self._chips = {}
        for col in FEATURE_COLS:
            f = tk.Frame(chip_area, bg=BG3, padx=6, pady=4)
            f.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=3, pady=8)
            tk.Label(f, text=col.upper(), bg=BG3, fg=TXT2,
                     font=("Segoe UI", 7, "bold")).pack()
            v = tk.StringVar(value="—")
            lbl = tk.Label(f, textvariable=v, bg=BG3, fg=CYAN, font=("Consolas", 9))
            lbl.pack()
            self._chips[col] = (v, lbl)

    def _lbl(self, parent, txt):
        return tk.Label(parent, text=txt, bg=BG2, fg=TXT2,
                        font=("Segoe UI", 7, "bold"))

    # ── refresh (every 300 ms) ────────────────
    def _refresh(self):
        self._clock.config(text=time.strftime("%H:%M:%S"))

        with _lock:
            s   = dict(_state)
            mh  = list(_mse_hist)
            sh  = list(_score_hist)
            log = list(_state["log"])

        status = s["status"]

        # status card
        if status == "BUFFERING":
            c = CYAN
            self._status_txt.config(text=f"Buffering  {s['buf']}/{SEQUENCE_LEN}", fg=c)
            self._status_sub.config(text="Filling sequence window…")
            self._dot.config(fg=c)
        elif status == "CRITICAL":
            c = ORANGE
            self._status_txt.config(text="⚠ CRITICAL", fg=c)
            self._status_sub.config(text="Sensor threshold breach")
            self._dot.config(fg=c)
        elif status == "ANOMALY":
            c = RED
            self._status_txt.config(text="⚠ ANOMALY", fg=c)
            self._status_sub.config(text=f"score={s['combined']:.3f}   MSE={s['mse']:.5f}")
            self._dot.config(fg=c)
        else:
            c = GREEN
            self._status_txt.config(text="✓ Normal", fg=c)
            self._status_sub.config(text=f"combined={s['combined']:.3f}")
            self._dot.config(fg=c)

        self._status_frame.config(bg=BG3)
        self._status_txt.config(bg=BG3)
        self._dot.config(bg=BG3)
        self._status_sub.config(bg=BG3)

        # MSE
        mse_color = RED if s["mse"] > LSTM_THRESHOLD else CYAN
        self._mse_val.config(text=f"{s['mse']:.6f}", fg=mse_color)

        # combined score
        sc = s["combined"]
        sc_color = RED if sc >= 1 else (ORANGE if sc > 0.6 else GREEN)
        self._sc_val.config(text=f"{sc:.4f}", fg=sc_color)

        # vibration
        self._vib_val.config(text=f"{int(s['vib']):,}")

        # LED
        led_map = {"G": (f"● GREEN  (normal)",  GREEN),
                   "R": (f"● RED    (anomaly)", RED),
                   "B": (f"● BLINK  (critical)",ORANGE)}
        lt, lc = led_map.get(s["led"], ("—", TXT2))
        self._led_val.config(text=lt, fg=lc)

        # chips
        for i, col in enumerate(FEATURE_COLS):
            v_var, lbl = self._chips[col]
            val = s["values"][i]
            if col == "flame":
                v_var.set("⚡ 0" if val == 0 else "✓ 1")
                lbl.config(fg=ORANGE if val == 0 else CYAN)
            else:
                v_var.set(f"{val:.0f}")
                lbl.config(fg=CYAN)

        # event log
        self._log_txt.config(state="normal")
        self._log_txt.delete("1.0", tk.END)
        for ts, msg, kind in log:
            self._log_txt.insert(tk.END, ts + "  ", "ts")
            self._log_txt.insert(tk.END, msg + "\n", kind)
        self._log_txt.config(state="disabled")

        # chart
        self._ln_mse.set_ydata(mh)
        self._ln_score.set_ydata(sh)
        top = max(max(mh or [0]), max(sh or [0]), LSTM_THRESHOLD) * 1.2
        self._ax.set_ylim(0, max(top, LSTM_THRESHOLD * 2))
        self._canvas.draw_idle()

        self.after(300, self._refresh)


# ──────────────────────────────────────────────
#  LAUNCH
# ──────────────────────────────────────────────
if __name__ == "__main__":
    MonitorApp().mainloop()
