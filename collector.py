"""字母板打字：用 webcam 追蹤食指指尖，停留選字，並逐幀記錄互動資料。

用來驗證校準的效果：同一個人分別用「預設設定」和「建議設定」打同樣的句子。
    python collector.py --participant P01 --condition default
    python collector.py --participant P01 --condition recommended --key-size 120 --dwell 1.2
    python collector.py --mouse          # 不開攝影機，用滑鼠測試

輸出到 sessions/typing_<participant>_<condition>_<時間>/（frames.csv、meta.json）

操作鍵（英文輸入法下）：
    n           跳過這一句
    q 或 Esc    結束並存檔
加上 --demo 時另外可以用（實驗時不要用，避免同一份資料混到不同設定）：
    1 / 2 / 3   停留時間 0.5 / 1.0 / 1.5 秒
    s           平滑化（One Euro Filter）開／關
    h           遲滯開／關
"""
import argparse
import csv
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from board import Board, DwellSelector, apply_key, next_target
from filters import PointFilter
from tracking import HandInput, MouseInput

W, H = 1280, 760                 # 畫面大小；和指向測驗相同，像素尺度才一致
WINDOW = "SteadyDwell - Letterboard"

# 顏色（OpenCV 用 BGR）
BG = (28, 28, 32)
KEY = (70, 70, 78)
KEY_HOVER = (120, 110, 60)
KEY_FLASH = (80, 170, 80)
TEXT = (235, 235, 235)
DIM = (150, 150, 150)
ACCENT = (60, 200, 255)


# ---------------------------------------------------------------- 畫面

def put_text(img, text, org, scale=1.0, color=TEXT, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render(state):
    img = np.full((H, W, 3), BG, np.uint8)
    board, sel = state["board"], state["selector"]

    # 目標句子與已輸入文字
    put_text(img, "TARGET", (60, 50), 0.6, DIM, 1)
    put_text(img, state["phrase"], (160, 52), 1.1, DIM)
    put_text(img, "TYPED", (60, 110), 0.6, DIM, 1)
    put_text(img, state["typed"] + "_", (160, 112), 1.1, TEXT)

    # 按鍵
    flash_key = state["flash_key"] if time.monotonic() < state["flash_until"] else None
    for k in board.keys:
        color = KEY_FLASH if k is flash_key else KEY_HOVER if k is sel.current else KEY
        cv2.rectangle(img, (int(k.x), int(k.y)), (int(k.x + k.w), int(k.y + k.h)), color, -1)
        scale = (1.2 if len(k.label) == 1 else 0.6) * board.key_size / 90
        (tw, th), _ = cv2.getTextSize(k.label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        if k is sel.current and not sel.armed:     # 剛選過、還沒離開：外框提示
            cv2.rectangle(img, (int(k.x), int(k.y)), (int(k.x + k.w), int(k.y + k.h)), DIM, 2)
        cx, cy = k.center
        put_text(img, k.label, (int(cx - tw / 2), int(cy + th / 2)), scale)
        # 停留進度：按鍵底部的進度條
        if k is sel.current and sel.progress > 0:
            x1 = int(k.x + k.w * sel.progress)
            bar = max(6, int(board.key_size / 9))
            cv2.rectangle(img, (int(k.x), int(k.y + k.h - bar)), (x1, int(k.y + k.h)), ACCENT, -1)

    # 游標：灰點是平滑前的原始位置，白圈是實際游標
    if state["raw"] is not None and state["smoothing"]:
        cv2.circle(img, tuple(map(int, state["raw"])), 4, DIM, -1)
    if state["cursor"] is not None:
        c = tuple(map(int, state["cursor"]))
        cv2.circle(img, c, 10, ACCENT, 2, cv2.LINE_AA)
    else:
        put_text(img, "NO HAND DETECTED", (W // 2 - 150, 200), 0.8, (80, 80, 230))

    # 攝影機縮圖
    if state["frame"] is not None:
        thumb = cv2.resize(state["frame"], (224, 126))
        img[10:136, W - 234:W - 10] = thumb

    # 狀態列
    hud = (f"{state['condition']}   key {board.key_size}px   dwell {sel.dwell_time:.1f}s   "
           f"smoothing {'ON' if state['smoothing'] else 'OFF'}   hysteresis {sel.hysteresis_px:.0f}px   "
           f"phrase {state['phrase_idx'] + 1}/{state['n_phrases']} [n]   quit [q]")
    put_text(img, hud, (60, H - 12), 0.5, DIM, 1)
    return img


# ---------------------------------------------------------------- 主程式

FIELDS = ["t", "frame", "input", "condition", "key_size", "detected", "raw_x", "raw_y", "x", "y",
          "hover_key", "dwell_progress", "dwell_time", "smoothing", "hysteresis_px",
          "phrase_idx", "target_phrase", "typed", "target_key", "events"]


def speak(text, rate=None):
    if text.strip() and shutil.which("say"):  # macOS 內建的語音合成
        subprocess.Popen(["say"] + (["-r", str(rate)] if rate else []) + [text])


def feedback(label):
    """選中時的回饋：唸出字母（像 HoloBoard 一樣）。"""
    spoken = {"SPACE": "space", "DEL": "delete"}.get(label, label)
    speak(spoken, rate=260)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="P00")
    ap.add_argument("--condition", default="default", help="標籤，例如 default 或 recommended")
    ap.add_argument("--key-size", type=int, default=90, help="按鍵邊長（像素），40–136")
    ap.add_argument("--dwell", type=float, default=1.0)
    ap.add_argument("--n-phrases", type=int, default=5, help="要打幾句")
    ap.add_argument("--hysteresis", type=float, default=20.0, help="遲滯範圍（像素）")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--min-cutoff", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--margin", type=float, default=0.15, help="攝影機畫面邊緣不使用的比例")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="hand_landmarker.task")
    ap.add_argument("--phrases", default="phrases.txt")
    ap.add_argument("--mouse", action="store_true", help="用滑鼠代替手部追蹤")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--demo", action="store_true", help="允許用 1/2/3/s/h 即時切換設定")
    ap.add_argument("--out", default="sessions")
    args = ap.parse_args()
    if not 40 <= args.key_size <= 136:
        raise SystemExit("--key-size 要在 40–136 之間，字母板才放得進畫面。")

    phrases = [p.strip().upper() for p in Path(args.phrases).read_text().splitlines() if p.strip()]
    phrases = phrases[:args.n_phrases]
    board = Board(key_size=args.key_size, screen_w=W)
    selector = DwellSelector(board, args.dwell, args.hysteresis)
    pfilter = PointFilter(min_cutoff=args.min_cutoff, beta=args.beta)

    cv2.namedWindow(WINDOW)
    source = (MouseInput(WINDOW) if args.mouse
              else HandInput((W, H), args.model, args.camera, args.margin))

    session_dir = Path(args.out) / f"typing_{args.participant}_{args.condition}_{datetime.now():%Y%m%d_%H%M%S}"
    session_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "task": "typing", "participant": args.participant, "condition": args.condition,
        "key_size": args.key_size, "start": datetime.now().isoformat(),
        "input": "mouse" if args.mouse else "hand", "screen": [W, H],
        "args": vars(args), "phrases": phrases, "layout": board.to_dict(),
        "platform": platform.platform(),
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    csv_file = open(session_dir / "frames.csv", "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    writer.writeheader()

    state = {"board": board, "selector": selector, "typed": "", "phrase_idx": 0,
             "condition": args.condition,
             "phrase": phrases[0], "n_phrases": len(phrases), "smoothing": not args.no_smooth,
             "raw": None, "cursor": None, "frame": None, "flash_key": None, "flash_until": 0.0}
    hyst_default = args.hysteresis
    t_start = t_prev = time.monotonic()
    frame_no = 0

    try:
        while True:
            raw = source.read()
            now = time.monotonic()
            dt, t_prev = now - t_prev, now

            if raw is None:
                pfilter.reset()
                cursor = None
            elif state["smoothing"]:
                cursor = pfilter(raw[0], raw[1], now)
            else:
                cursor = raw

            target_key = next_target(state["typed"], state["phrase"])
            events = selector.update(*(cursor or (None, None)), dt)
            log_events = []
            for ev in events:
                kind, key = ev[0], ev[1]
                if kind == "leave":
                    log_events.append(f"leave:{key.label}:{ev[2]}")
                else:
                    log_events.append(f"{kind}:{key.label}")
                if kind == "select":
                    if key.label == "SPEAK":
                        speak(state["typed"])
                    elif not args.no_sound:
                        feedback(key.label)
                    state["typed"] = apply_key(state["typed"], key.label)
                    state["flash_key"], state["flash_until"] = key, now + 0.25

            writer.writerow({
                "t": round(now - t_start, 4), "frame": frame_no,
                "input": "mouse" if args.mouse else "hand", "condition": args.condition,
                "key_size": args.key_size, "detected": int(raw is not None),
                "raw_x": round(raw[0], 1) if raw else "", "raw_y": round(raw[1], 1) if raw else "",
                "x": round(cursor[0], 1) if cursor else "", "y": round(cursor[1], 1) if cursor else "",
                "hover_key": selector.current.label if selector.current else "",
                "dwell_progress": round(selector.progress, 3), "dwell_time": selector.dwell_time,
                "smoothing": int(state["smoothing"]), "hysteresis_px": selector.hysteresis_px,
                "phrase_idx": state["phrase_idx"], "target_phrase": state["phrase"],
                "typed": state["typed"], "target_key": target_key, "events": "|".join(log_events),
            })
            frame_no += 1

            # 完成一句 → 自動換下一句
            if state["typed"] == state["phrase"]:
                writer.writerow({**{f: "" for f in FIELDS}, "t": round(now - t_start, 4),
                                 "frame": frame_no, "phrase_idx": state["phrase_idx"],
                                 "events": "phrase_done"})
                if state["phrase_idx"] + 1 >= len(phrases):
                    print("所有句子已完成。")
                    break
                state["phrase_idx"] += 1
                state["phrase"], state["typed"] = phrases[state["phrase_idx"]], ""

            state.update(raw=raw, cursor=cursor, frame=source.frame)
            cv2.imshow(WINDOW, render(state))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif args.demo and key in (ord("1"), ord("2"), ord("3")):
                selector.dwell_time = {ord("1"): 0.5, ord("2"): 1.0, ord("3"): 1.5}[key]
            elif args.demo and key == ord("s"):
                state["smoothing"] = not state["smoothing"]
                pfilter.reset()
            elif args.demo and key == ord("h"):
                selector.hysteresis_px = 0.0 if selector.hysteresis_px else hyst_default
            elif key == ord("n") and state["phrase_idx"] + 1 < len(phrases):
                state["phrase_idx"] += 1
                state["phrase"], state["typed"] = phrases[state["phrase_idx"]], ""
    finally:
        csv_file.close()
        source.close()
        cv2.destroyAllWindows()
        print(f"資料已存到 {session_dir}/")
        print("比較兩種設定：python analyze_typing.py sessions/typing_*")


if __name__ == "__main__":
    main()
