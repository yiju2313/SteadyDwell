"""產生「模擬使用者」的指向測驗資料，用來在沒有攝影機時測試分析程式。

⚠️ 這是合成資料，不代表任何真實使用者。輸出資料夾名稱以 SIM_ 開頭以便區分。

    python simulate_fitts.py --tremor 6     # 手比較穩
    python simulate_fitts.py --tremor 18    # 手比較晃

模型：移動時間依 Fitts' law（MT = a + b·ID）加上隨機變化；軌跡為 minimum-jerk 曲線；
落點有偏差；停留時游標以 tremor 標準差隨機晃動；再經過與真實程式相同的 One Euro Filter。
"""
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

import fitts_task as ft
from filters import PointFilter


def simulate(tremor=10.0, a=0.35, b=0.30, fps=15.0, dwell=1.0, seed=0, out="sessions",
             participant="SIM"):
    rng = np.random.default_rng(seed)
    task = ft.FittsTask(ft.make_conditions(seed), dwell=dwell, timeout=10.0)
    pfilter = PointFilter(min_cutoff=1.0, beta=0.01)

    out_dir = Path(out) / f"fitts_{participant}_t{tremor:g}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"task": "fitts", "participant": participant, "synthetic": True,
            "sim_params": {"tremor": tremor, "a": a, "b": b, "fps": fps, "seed": seed},
            "start": datetime.now().isoformat(), "input": "simulated", "screen": [ft.W, ft.H],
            "center": ft.CENTER, "n_targets": ft.N_TARGETS, "conditions": task.conditions,
            "args": {"dwell": dwell, "timeout": 10.0}}
    f = open(out_dir / "frames.csv", "w", newline="")
    frames = csv.DictWriter(f, fieldnames=ft.FRAME_FIELDS)
    frames.writeheader()

    t = 0.0
    pos = np.array(ft.CENTER, float)
    drift = np.zeros(2)
    while task.phase != "done":
        task.start_next_sequence(t)
        if task.phase == "done":
            break
        while task.phase == "run":
            target = np.array(task.positions[task.active])
            dist = np.linalg.norm(target - pos)
            ide = math.log2(dist / task.width + 1)
            mt = max(0.15, a + b * ide + rng.normal(0, 0.08))
            aim = target + rng.normal(0, task.width * 0.12, 2)        # 落點偏差
            start, t_move = pos.copy(), 0.0
            while task.phase == "run":
                dt = 1 / fps * rng.uniform(0.85, 1.15)
                t += dt
                t_move += dt
                if t_move < mt:                                        # minimum-jerk 移動
                    s = t_move / mt
                    s = 10 * s**3 - 15 * s**4 + 6 * s**5
                    pos = start + (aim - start) * s
                else:
                    pos = aim
                drift = 0.7 * drift + rng.normal(0, tremor * 0.7, 2)  # 有相關性的晃動
                raw = (float(np.clip(pos[0] + drift[0], 0, ft.W)),
                       float(np.clip(pos[1] + drift[1], 0, ft.H)))
                cursor = pfilter(raw[0], raw[1], t)
                event = task.update(cursor, t, dt)
                frames.writerow(ft.frame_row(task, t, raw, cursor, event))
                if event:
                    break
        t += 3.0                                                       # 組間休息
    f.close()
    ft.save_session(out_dir, meta, task)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tremor", type=float, default=10.0, help="晃動程度（像素）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--participant", default="SIM")
    args = ap.parse_args()
    d = simulate(args.tremor, seed=args.seed, participant=args.participant)
    trials = list(csv.DictReader(open(d / "trials.csv")))
    print(f"{d}: {len(trials)} trials, {sum(int(r['timeout']) for r in trials)} timeouts")
    print(json.loads((d / "meta.json").read_text())["completed_sets"], "sets completed")
