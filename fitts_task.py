"""指向校準測驗：ISO 9241-411 多方向點選（multi-directional tapping）。

螢幕上一圈 9 個圓，亮起的那個是目標。把游標停在目標上 dwell 秒就選中，
接著目標跳到圓的對面。共 6 組條件（2 種距離 × 3 種目標大小），約 3 分鐘。

執行：
    python fitts_task.py --participant P01            # 手部追蹤
    python fitts_task.py --participant P01 --mouse    # 用滑鼠測試

輸出到 sessions/fitts_<participant>_<時間>/：
    trials.csv  每次選取一列（Fitts' law 用）
    frames.csv  每一幀一列（晃動範圍、路過時間用）
    meta.json   設定與每組條件的目標位置

操作鍵（英文輸入法）：空白鍵 開始下一組；q 或 Esc 中途結束（已完成的資料會保留）
"""
import argparse
import csv
import json
import math
import platform
import random
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from filters import PointFilter
from tracking import HandInput, MouseInput

W, H = 1280, 760               # 畫面大小；字母板用同樣的大小，像素尺度才一致
CENTER = (W // 2, H // 2 + 20)
N_TARGETS = 9
AMPLITUDES = (400, 600)        # 圓圈直徑（距離 A）
WIDTHS = (45, 80, 130)         # 目標直徑（大小 W）
# 相鄰兩個圓的中心距離 = A·sin(π/9) ≈ 0.342·A，必須大於 W 才不會重疊：
assert min(AMPLITUDES) * math.sin(math.pi / N_TARGETS) > max(WIDTHS), "目標會重疊"
WINDOW = "SteadyDwell - Pointing Calibration"

BG = (28, 28, 32)
IDLE = (90, 90, 98)
ACTIVE = (60, 200, 255)        # OpenCV 是 BGR：這是黃橘色
PROGRESS = (80, 200, 80)
TEXT = (235, 235, 235)
DIM = (150, 150, 150)


# ---------------------------------------------------------------- 幾何

def target_positions(amplitude, n=N_TARGETS, center=CENTER):
    """n 個目標平均排在直徑 = amplitude 的圓上，第 0 個在正上方，順時針。"""
    cx, cy = center
    r = amplitude / 2
    return [(cx + r * math.sin(2 * math.pi * i / n),
             cy - r * math.cos(2 * math.pi * i / n)) for i in range(n)]


def selection_order(n=N_TARGETS):
    """ISO 順序：每次跳到圓的對面。n=9 → 0,5,1,6,2,7,3,8,4,0（第一個是起點，不計時）"""
    step = (n + 1) // 2
    return [(i * step) % n for i in range(n + 1)]


def make_conditions(seed=None):
    """6 組 (A, W) 條件，隨機順序（避免每個人都從簡單的開始，造成練習效應）。"""
    conds = [(a, w) for a in AMPLITUDES for w in WIDTHS]
    random.Random(seed).shuffle(conds)
    return conds


# ---------------------------------------------------------------- 測驗邏輯（不碰畫面，可以單獨測試）

class FittsTask:
    """測驗的狀態機。每一幀呼叫 update()。

    phase:
      "rest" 兩組之間休息，等使用者按空白鍵
      "run"  進行中
      "done" 全部完成
    """

    def __init__(self, conditions, dwell=1.0, timeout=10.0, n=N_TARGETS):
        self.conditions = conditions
        self.dwell = dwell
        self.timeout = timeout
        self.n = n
        self.order = selection_order(n)
        self.seq = -1                  # 目前第幾組（0 起算）
        self.phase = "rest"
        self.trials = []               # 每次計時的選取一筆 dict

    # --- 組與組之間

    def start_next_sequence(self, now):
        self.seq += 1
        if self.seq >= len(self.conditions):
            self.phase = "done"
            return
        self.amplitude, self.width = self.conditions[self.seq]
        self.positions = target_positions(self.amplitude, self.n)
        self.step = 0                  # 走到 order 的第幾個；step 0 是起點
        self.phase = "run"
        self._reset_trial(now)

    def _reset_trial(self, now):
        self.t_start = now             # 這次移動的開始時間（= 上一次選中的時間）
        self.inside = False
        self.elapsed = 0.0             # 目前在目標上已停留幾秒
        self.entries = 0               # 這次移動進入目標幾次（> 1 代表進去又跑出來）
        self.t_first_entry = None
        self.t_last_entry = None

    @property
    def active(self):
        return self.order[self.step]

    @property
    def progress(self):
        return min(self.elapsed / self.dwell, 1.0)

    def hover_index(self, cursor):
        """游標在哪個圓裡面（任何一個，不只目標）；都不在則回傳 -1。"""
        if cursor is None:
            return -1
        i = min(range(self.n), key=lambda k: math.dist(cursor, self.positions[k]))
        return i if math.dist(cursor, self.positions[i]) <= self.width / 2 else -1

    # --- 每一幀

    def update(self, cursor, now, dt):
        """回傳這一幀的事件字串："" / "select" / "timeout" / "start"（選中起點）"""
        if self.phase != "run":
            return ""
        hover = self.hover_index(cursor)
        in_target = hover == self.active

        if in_target and not self.inside:        # 剛進入目標
            self.entries += 1
            self.t_last_entry = now
            if self.t_first_entry is None:
                self.t_first_entry = now
        if in_target:
            self.elapsed += dt
        else:
            self.elapsed = 0.0                   # 離開目標，停留計時歸零
        self.inside = in_target

        if in_target and self.elapsed >= self.dwell - 1e-6:
            return self._finish(cursor, now, timeout=False)
        if self.step > 0 and now - self.t_start > self.timeout:
            return self._finish(cursor, now, timeout=True)
        return ""

    def _finish(self, cursor, now, timeout):
        if self.step == 0:
            event = "start"                      # 起點：只是讓手就位，不記錄
        else:
            event = "timeout" if timeout else "select"
            prev = self.order[self.step - 1]
            p0, p1 = self.positions[prev], self.positions[self.active]
            self.trials.append({
                "seq": self.seq, "trial": self.step,
                "A": self.amplitude, "W": self.width,
                "from_idx": prev, "to_idx": self.active,
                "distance": round(math.dist(p0, p1), 2),   # 實際距離（9 是奇數，不是正對面）
                "target_x": round(p1[0], 1), "target_y": round(p1[1], 1),
                "t_start": round(self.t_start, 4), "t_end": round(now, 4),
                "mt_total": round(now - self.t_start, 4),
                # 移動時間：到「最後一次進入目標」為止，不含停留時間
                "mt_move": round(self.t_last_entry - self.t_start, 4) if self.t_last_entry and not timeout else "",
                "t_first_entry": round(self.t_first_entry - self.t_start, 4) if self.t_first_entry else "",
                "entries": self.entries,
                "timeout": int(timeout),
                "end_x": round(cursor[0], 1) if cursor else "",
                "end_y": round(cursor[1], 1) if cursor else "",
            })
        self.step += 1
        if self.step >= len(self.order):          # 這一組做完
            self.phase = "rest"
        else:
            self._reset_trial(now)
        return event


# ---------------------------------------------------------------- 畫面

def put_text(img, text, org, scale=0.7, color=TEXT, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_center(img, text, y, scale=0.8, color=TEXT, thickness=2):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    put_text(img, text, ((W - tw) // 2, y), scale, color, thickness)


def render(task, cursor, raw, frame, smoothing):
    img = np.full((H, W, 3), BG, np.uint8)
    n_seq = len(task.conditions)

    if task.phase == "run":
        r = int(task.width / 2)
        for i, (x, y) in enumerate(task.positions):
            c = (int(x), int(y))
            if i == task.active:
                cv2.circle(img, c, r, ACTIVE, -1, cv2.LINE_AA)
                if task.progress > 0:               # 停留進度：外圈綠色弧線
                    cv2.ellipse(img, c, (r + 8, r + 8), -90, 0, 360 * task.progress,
                                PROGRESS, 4, cv2.LINE_AA)
            else:
                cv2.circle(img, c, r, IDLE, 2, cv2.LINE_AA)
        label = "start: dwell on the target" if task.step == 0 else f"target {task.step}/{len(task.order) - 1}"
        put_text(img, f"set {task.seq + 1}/{n_seq}   {label}", (20, 32))
    elif task.phase == "rest":
        if task.seq < 0:
            put_center(img, "Pointing calibration", 240, 1.1)
            put_center(img, "Move the cursor with your index finger.", 300, 0.7, DIM, 1)
            put_center(img, "Keep it on the yellow circle until it is selected.", 335, 0.7, DIM, 1)
            put_center(img, "Move at a comfortable pace - accuracy matters more than speed.", 370, 0.7, DIM, 1)
        else:
            put_center(img, f"Set {task.seq + 1}/{n_seq} done", 300, 1.1)
            put_center(img, "Rest your arm if you need to.", 350, 0.7, DIM, 1)
        put_center(img, "Press SPACE to start", 440, 0.8, ACTIVE)
    else:
        put_center(img, "All done - thank you!", 340, 1.1)
        put_center(img, "Press q to save and exit", 400, 0.7, DIM, 1)

    if raw is not None and smoothing:
        cv2.circle(img, tuple(map(int, raw)), 3, DIM, -1)
    if cursor is not None:
        cv2.circle(img, tuple(map(int, cursor)), 8, TEXT, 2, cv2.LINE_AA)
    elif task.phase == "run":
        put_center(img, "NO HAND DETECTED", 70, 0.7, (80, 80, 230))

    if frame is not None:
        img[10:136, W - 234:W - 10] = cv2.resize(frame, (224, 126))
    return img


def play_click():
    sound = "/System/Library/Sounds/Tink.aiff"   # macOS 內建音效
    if shutil.which("afplay") and Path(sound).exists():
        subprocess.Popen(["afplay", sound])


# ---------------------------------------------------------------- 記錄

FRAME_FIELDS = ["t", "seq", "A", "W", "trial", "active_idx", "detected",
                "raw_x", "raw_y", "x", "y", "hover_idx", "dwell_progress", "event"]
TRIAL_FIELDS = ["seq", "trial", "A", "W", "from_idx", "to_idx", "distance",
                "target_x", "target_y", "t_start", "t_end", "mt_total", "mt_move",
                "t_first_entry", "entries", "timeout", "end_x", "end_y"]


def fmt(v):
    return "" if v is None else round(v, 1)


def frame_row(task, t, raw, cursor, event):
    return {
        "t": round(t, 4), "seq": task.seq, "A": task.amplitude, "W": task.width,
        "trial": task.step if event == "" else task.step - 1, "active_idx": task.active if event == "" else "",
        "detected": int(raw is not None),
        "raw_x": fmt(raw[0]) if raw else "", "raw_y": fmt(raw[1]) if raw else "",
        "x": fmt(cursor[0]) if cursor else "", "y": fmt(cursor[1]) if cursor else "",
        "hover_idx": task.hover_index(cursor), "dwell_progress": round(task.progress, 3),
        "event": event,
    }


def save_session(out_dir, meta, task):
    meta["end"] = datetime.now().isoformat()
    meta["completed_sets"] = len({t["seq"] for t in task.trials if t["trial"] == task.n})
    meta["sets"] = [{"seq": i, "A": a, "W": w,
                     "positions": [[round(x, 1), round(y, 1)] for x, y in target_positions(a)]}
                    for i, (a, w) in enumerate(task.conditions)]
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    with open(out_dir / "trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        writer.writerows(task.trials)


# ---------------------------------------------------------------- 主程式

def main():
    ap = argparse.ArgumentParser(description="Pointing calibration task (ISO 9241-411)")
    ap.add_argument("--participant", default="P00")
    ap.add_argument("--dwell", type=float, default=1.0, help="選取所需停留秒數")
    ap.add_argument("--timeout", type=float, default=10.0, help="單次選取的時間上限（秒）")
    ap.add_argument("--seed", type=int, default=None, help="條件順序的亂數種子")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--min-cutoff", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="hand_landmarker.task")
    ap.add_argument("--mouse", action="store_true", help="用滑鼠代替手部追蹤")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--out", default="sessions")
    args = ap.parse_args()

    conditions = make_conditions(args.seed)
    task = FittsTask(conditions, args.dwell, args.timeout)
    smoothing = not args.no_smooth
    pfilter = PointFilter(min_cutoff=args.min_cutoff, beta=args.beta)

    cv2.namedWindow(WINDOW)
    source = (MouseInput(WINDOW) if args.mouse
              else HandInput((W, H), args.model, args.camera, args.margin))

    out_dir = Path(args.out) / f"fitts_{args.participant}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"task": "fitts", "participant": args.participant, "start": datetime.now().isoformat(),
            "input": "mouse" if args.mouse else "hand", "screen": [W, H], "center": CENTER,
            "n_targets": N_TARGETS, "conditions": conditions, "args": vars(args),
            "platform": platform.platform()}
    frames_file = open(out_dir / "frames.csv", "w", newline="")
    frames = csv.DictWriter(frames_file, fieldnames=FRAME_FIELDS)
    frames.writeheader()

    t0 = t_prev = time.monotonic()
    try:
        while True:
            raw = source.read()
            now = time.monotonic()
            dt, t_prev = now - t_prev, now
            if raw is None:
                pfilter.reset()
                cursor = None
            else:
                cursor = pfilter(raw[0], raw[1], now) if smoothing else raw

            if task.phase == "run":
                event = task.update(cursor, now - t0, dt)
                frames.writerow(frame_row(task, now - t0, raw, cursor, event))
                if event and not args.no_sound:
                    play_click()

            cv2.imshow(WINDOW, render(task, cursor, raw, source.frame, smoothing))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and task.phase == "rest":
                task.start_next_sequence(time.monotonic() - t0)
                t_prev = time.monotonic()
    finally:
        frames_file.close()
        save_session(out_dir, meta, task)
        source.close()
        cv2.destroyAllWindows()
        n_ok = sum(1 for t in task.trials if not t["timeout"])
        print(f"完成 {n_ok}/{len(task.trials)} 次選取（其餘逾時）。資料已存到 {out_dir}/")
        print(f"下一步：python analyze_fitts.py {out_dir}")


if __name__ == "__main__":
    main()
