"""
ui_detector.py  —  Dark-themed real-time motor anomaly monitor
Run: python ui_detector.py   (sensor_ai conda env, same folder as real_time_detector.py)

UI uses pure tkinter Canvas for the live chart — no matplotlib blocking.
"""

import threading
import time
import tkinter as tk
from collections import deque

import joblib
import numpy as np
import pandas as pd
import serial
from tensorflow.keras.models import load_model

# ──────────────────────────────────────────────────────
#  CONFIG  (identical to real_time_detector.py)
# ──────────────────────────────────────────────────────
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
HIST                 = 60    # history points on chart (keep low for fast redraw)

# ──────────────────────────────────────────────────────
#  LOAD MODEL & SCALER
# ──────────────────────────────────────────────────────
print("Loading model…")
model  = load_model("motor_anomaly_lstm.keras")
scaler = joblib.load("scaler.save")
model.predict(np.zeros((1, SEQUENCE_LEN, len(FEATURE_COLS))), verbose=0)
print("Model ready.")

# ──────────────────────────────────────────────────────
#  SERIAL  (same retry logic as real_time_detector.py)
# ──────────────────────────────────────────────────────
print(f"Opening {COM_PORT}…")
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

# Wait for Arduino to finish its DTR reset, then flush garbage bytes
time.sleep(2)
ser.reset_input_buffer()
ser.write(b'0')
print("Ready.\n")

# ──────────────────────────────────────────────────────
#  SHARED STATE  (written by detector thread, read by UI)
# ──────────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "status"  : "BUFFERING",
    "mse"     : 0.0,
    "vib"     : 0.0,
    "combined": 0.0,
    "led"     : "G",
    "buf"     : 0,
    "values"  : [0.0] * len(FEATURE_COLS),
    "log"     : [],
}
_mse_hist   = deque([0.0] * HIST, maxlen=HIST)
_score_hist = deque([0.0] * HIST, maxlen=HIST)

# ──────────────────────────────────────────────────────
#  DETECTOR THREAD  (same logic as real_time_detector.py)
# ──────────────────────────────────────────────────────
_seq        = deque(maxlen=SEQUENCE_LEN)
_accel_hist = deque(maxlen=VIBRATION_WINDOW)

def _set_led(new):
    if _state["led"] == new: return
    _state["led"] = new
    try: ser.write({"B": b"2", "R": b"1", "G": b"0"}[new])
    except Exception: pass

def _log(msg, kind):
    ts = time.strftime("%H:%M:%S")
    _state["log"].insert(0, (ts, msg, kind))
    _state["log"] = _state["log"][:25]

def _reconnect_serial():
    """Try to reopen ser if it dropped. Blocks until reconnected."""
    global ser
    print("Serial disconnected — attempting reconnect…")
    _log("Serial disconnected — reconnecting…", "critical")
    while True:
        try:
            if ser:
                try: ser.close()
                except Exception: pass
            ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1, write_timeout=1)
            time.sleep(2)
            ser.reset_input_buffer()
            ser.write(b'0')
            print("Reconnected.")
            _log("Serial reconnected", "normal")
            return
        except serial.SerialException:
            time.sleep(3)

def _detector():
    while True:
        try:
            # ── read one line ──────────────────────────────────────
            try:
                raw = ser.readline()
            except serial.SerialException:
                _reconnect_serial()
                continue

            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            parts = line.split(",")
            if len(parts) != len(FEATURE_COLS):
                continue
            try:
                values = [float(v) for v in parts]
            except ValueError:
                continue

            ax, ay, az = values[0], values[1], values[2]
            sensor_val = int(values[SENSOR_IDX])
            accel_mag  = (ax**2 + ay**2 + az**2) ** 0.5
            _accel_hist.append(accel_mag)

            vib_spike = min(accel_mag / VIBRATION_THRESHOLD, 1.0)
            vib_std_s = 0.0
            if len(_accel_hist) == VIBRATION_WINDOW:
                vib_std_s = min(float(np.std(_accel_hist)) / VIBRATION_STD_THRESH, 1.0)
            vib_score = max(vib_spike, vib_std_s)

            df     = pd.DataFrame([values], columns=FEATURE_COLS)
            scaled = scaler.transform(df)[0]
            _seq.append(scaled)

            with _lock:
                _state["values"] = values
                _state["vib"]    = accel_mag

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

            # ── LSTM inference ────────────────────────────────────
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
                        if prev not in ("NORMAL", "BUFFERING", "STARTING"):
                            _log(f"Normal  combined={combined:.3f}", "normal")
                        _state["status"] = "NORMAL"

        except Exception as e:
            # Catch-all: log the error and keep running — never let the thread die
            print(f"[detector] Unexpected error: {e}")
            _log(f"Error (recovering): {e}", "critical")
            time.sleep(1)

threading.Thread(target=_detector, daemon=True, name="detector").start()


# ──────────────────────────────────────────────────────
#  COLOUR PALETTE
# ──────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────
#  PURE-TKINTER LINE CHART  (no matplotlib — never blocks)
# ──────────────────────────────────────────────────────
class LineChart(tk.Canvas):
    """Simple, fast rolling line chart drawn with tkinter Canvas primitives."""

    def __init__(self, parent, series: list, **kw):
        """
        series: list of dicts  {data: deque, color: str, label: str}
        """
        super().__init__(parent, bg=BG2, highlightthickness=0, bd=0, **kw)
        self._series  = series
        self._thresh  = LSTM_THRESHOLD
        self._pad     = (36, 8, 8, 20)   # left, top, right, bottom

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return

        pl, pt, pr, pb = self._pad
        cw = w - pl - pr   # chart width
        ch = h - pt - pb   # chart height

        # gather all data
        all_vals = []
        snaps = [list(s["data"]) for s in self._series]
        for snap in snaps:
            all_vals.extend(snap)

        y_max = max(max(all_vals) if all_vals else 0, self._thresh) * 1.2
        y_max = max(y_max, self._thresh * 2)
        y_min = 0.0

        def px(i, n):   return pl + (i / max(n - 1, 1)) * cw
        def py(v):      return pt + ch - ((v - y_min) / (y_max - y_min)) * ch

        # grid lines (3 horizontal)
        for k in range(4):
            yv = y_min + (y_max - y_min) * k / 3
            yp = py(yv)
            self.create_line(pl, yp, pl + cw, yp, fill=BORDER, width=1)
            self.create_text(pl - 4, yp, text=f"{yv:.4f}", anchor="e",
                             fill=TXT2, font=("Consolas", 7))

        # threshold line
        th_y = py(self._thresh)
        self.create_line(pl, th_y, pl + cw, th_y, fill=RED, width=1, dash=(6, 4))
        self.create_text(pl + 4, th_y - 6, text=f"thresh {self._thresh}",
                         anchor="w", fill=RED, font=("Consolas", 7))

        # series lines
        for s, snap in zip(self._series, snaps):
            n = len(snap)
            if n < 2: continue
            pts = []
            for i, v in enumerate(snap):
                pts.append(px(i, n))
                pts.append(py(v))
            self.create_line(*pts, fill=s["color"], width=2, smooth=False)

        # legend
        lx = pl + 4
        for s in self._series:
            self.create_rectangle(lx, pt + 2, lx + 14, pt + 8, fill=s["color"], outline="")
            self.create_text(lx + 18, pt + 3, text=s["label"], anchor="nw",
                             fill=TXT2, font=("Consolas", 7))
            lx += 80


# ──────────────────────────────────────────────────────
#  MAIN WINDOW
# ──────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Motor Anomaly Monitor")
        self.configure(bg=BG)
        self.geometry("960x580")
        self._build()
        self._log_cache = []          # avoids rebuilding Text widget every tick
        self.after(200,  self._refresh_fast)   # labels  — lightweight, 200 ms
        self.after(1000, self._refresh_slow)   # chart + log — heavier, 1 s

    # ── layout ──────────────────────────────────────────
    def _build(self):
        # top bar
        bar = tk.Frame(self, bg=BG2, height=42)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="⬡  Motor Anomaly Monitor", bg=BG2, fg=CYAN,
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=16, pady=10)
        self._clk = tk.Label(bar, text="", bg=BG2, fg=TXT2, font=("Consolas", 10))
        self._clk.pack(side=tk.RIGHT, padx=16)
        tk.Label(bar, text=f"COM7 · {BAUD_RATE} baud", bg=BG2, fg=TXT2,
                 font=("Consolas", 10)).pack(side=tk.RIGHT, padx=12)
        tk.Frame(self, bg=BORDER, height=1).pack(fill=tk.X)

        # body
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)

        # left sidebar (240 px)
        left = tk.Frame(body, bg=BG2, width=240)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        tk.Frame(body, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)

        # right area
        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_left(left)
        self._build_right(right)

    # ── left sidebar ────────────────────────────────────
    def _build_left(self, p):
        def sec(text):
            tk.Label(p, text=text, bg=BG2, fg=TXT2,
                     font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=12, pady=(10,2))

        def divider():
            tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, padx=8, pady=4)

        # Status
        sec("STATUS")
        sf = tk.Frame(p, bg=BG3, padx=10, pady=8)
        sf.pack(fill=tk.X, padx=10)
        row = tk.Frame(sf, bg=BG3)
        row.pack(anchor="w")
        self._dot = tk.Label(row, text="●", bg=BG3, fg=GREEN, font=("Segoe UI", 18))
        self._dot.pack(side=tk.LEFT)
        self._stxt = tk.Label(row, text="Buffering…", bg=BG3, fg=GREEN,
                               font=("Segoe UI", 11, "bold"))
        self._stxt.pack(side=tk.LEFT, padx=6)
        self._ssub = tk.Label(sf, text="", bg=BG3, fg=TXT2,
                               font=("Consolas", 8), wraplength=200, justify="left")
        self._ssub.pack(anchor="w")

        divider()

        # MSE
        sec("MSE  (Reconstruction Error)")
        self._mse_lbl = tk.Label(p, text="—", bg=BG2, fg=CYAN,
                                  font=("Consolas", 20, "bold"))
        self._mse_lbl.pack(anchor="w", padx=14)
        tk.Label(p, text=f"threshold {LSTM_THRESHOLD}", bg=BG2, fg=TXT2,
                 font=("Consolas", 8)).pack(anchor="w", padx=14)

        divider()

        # Scores
        sec("COMBINED SCORE")
        self._sc_lbl = tk.Label(p, text="—", bg=BG2, fg=GREEN,
                                 font=("Consolas", 15, "bold"))
        self._sc_lbl.pack(anchor="w", padx=14)

        sec("VIBRATION MAG")
        self._vib_lbl = tk.Label(p, text="—", bg=BG2, fg=TXT2, font=("Consolas", 11))
        self._vib_lbl.pack(anchor="w", padx=14)

        divider()

        # LED
        sec("ARDUINO LED")
        self._led_lbl = tk.Label(p, text="—", bg=BG2, fg=GREEN, font=("Consolas", 10))
        self._led_lbl.pack(anchor="w", padx=14)

        divider()

        # Event log
        sec("EVENT LOG")
        lf = tk.Frame(p, bg=BG3)
        lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self._log = tk.Text(lf, bg=BG3, fg=TXT2, font=("Consolas", 8),
                             relief="flat", bd=0, state="disabled",
                             highlightthickness=0, wrap="word")
        self._log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._log.tag_config("anomaly",  foreground=RED)
        self._log.tag_config("critical", foreground=ORANGE)
        self._log.tag_config("normal",   foreground=GREEN)
        self._log.tag_config("ts",       foreground=TXT2)

    # ── right panel ─────────────────────────────────────
    def _build_right(self, p):
        # chart (takes most of the space)
        self._chart = LineChart(
            p,
            series=[
                {"data": _mse_hist,   "color": CYAN,   "label": "MSE"},
                {"data": _score_hist, "color": ORANGE, "label": "Combined"},
            ]
        )
        self._chart.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X)

        # sensor chips
        chips_row = tk.Frame(p, bg=BG2, height=62)
        chips_row.pack(fill=tk.X)
        chips_row.pack_propagate(False)
        self._chips = {}
        for col in FEATURE_COLS:
            f = tk.Frame(chips_row, bg=BG3, padx=4, pady=4)
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3, pady=8)
            tk.Label(f, text=col.upper(), bg=BG3, fg=TXT2,
                     font=("Segoe UI", 6, "bold")).pack()
            v = tk.StringVar(value="—")
            lbl = tk.Label(f, textvariable=v, bg=BG3, fg=CYAN, font=("Consolas", 9))
            lbl.pack()
            self._chips[col] = (v, lbl)

    # ── FAST refresh  (200 ms) — labels only, never touches canvas ──────────
    def _refresh_fast(self):
        self._clk.config(text=time.strftime("%H:%M:%S"))

        with _lock:
            s = dict(_state)

        st = s["status"]
        if st == "BUFFERING":
            c, txt, sub = CYAN,   f"Buffering  {s['buf']}/{SEQUENCE_LEN}", "Filling window…"
        elif st == "CRITICAL":
            c, txt, sub = ORANGE, "⚠ CRITICAL", "Sensor threshold breach!"
        elif st == "ANOMALY":
            c, txt, sub = RED,    "⚠ ANOMALY",  f"score={s['combined']:.3f}  MSE={s['mse']:.5f}"
        else:
            c, txt, sub = GREEN,  "✓ Normal",   f"combined={s['combined']:.3f}"

        self._dot.config(fg=c)
        self._stxt.config(text=txt, fg=c)
        self._ssub.config(text=sub)

        mc = RED if s["mse"] > LSTM_THRESHOLD else CYAN
        self._mse_lbl.config(text=f"{s['mse']:.6f}", fg=mc)

        sc = s["combined"]
        sc_c = RED if sc >= 1 else (ORANGE if sc > 0.6 else GREEN)
        self._sc_lbl.config(text=f"{sc:.4f}", fg=sc_c)
        self._vib_lbl.config(text=f"{int(s['vib']):,}")

        led_map = {"G": ("● GREEN  (normal)", GREEN),
                   "R": ("● RED    (anomaly)", RED),
                   "B": ("● BLINK  (critical)", ORANGE)}
        lt, lc = led_map.get(s["led"], ("—", TXT2))
        self._led_lbl.config(text=lt, fg=lc)

        for i, col in enumerate(FEATURE_COLS):
            v_var, lbl = self._chips[col]
            val = s["values"][i]
            if col == "flame":
                v_var.set("⚡ 0" if val == 0 else "✓ 1")
                lbl.config(fg=ORANGE if val == 0 else CYAN)
            else:
                v_var.set(f"{val:.0f}")
                lbl.config(fg=CYAN)

        self.after(200, self._refresh_fast)

    # ── SLOW refresh  (1000 ms) — chart + log; heavier but infrequent ────────
    def _refresh_slow(self):
        with _lock:
            log = list(_state["log"])

        # Only rebuild Text widget when log actually changed
        if log != self._log_cache:
            self._log_cache = log
            self._log.config(state="normal")
            self._log.delete("1.0", tk.END)
            for ts, msg, kind in log:
                self._log.insert(tk.END, ts + "  ", "ts")
                self._log.insert(tk.END, msg + "\n", kind)
            self._log.config(state="disabled")

        # Redraw chart — pure canvas, no blocking
        self._chart.redraw()

        self.after(1000, self._refresh_slow)


# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()
