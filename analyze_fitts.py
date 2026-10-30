"""分析指向校準資料，建議個人化的按鍵大小和停留時間。

    python analyze_fitts.py sessions/fitts_P01_20261027_190000

輸出（存在同一個資料夾）：
    fitts_report.png      三張圖
    recommendation.json   建議設定與所有指標

三個量測：
1. Fitts' law（描述整體指向能力）
   移動時間 MT = a + b·ID，ID = log2(D/W + 1)（Shannon 公式，D 用實際距離）。
   MT 只算到「最後一次進入目標」，不含停留時間。用每組條件的平均值做線性迴歸。
   throughput = ID/MT 的平均（bits/s），數字越大代表指向越有效率。

2. 一次到位率 → 按鍵大小
   「一次到位」= 沒有逾時，而且只進入目標一次（沒有進去又晃出來）。
   計算每種目標大小 W 的一次到位率，找出達到 90% 的最小大小（在測過的大小之間線性內插）。

3. 移動途中的滯留時間 → 停留時間
   字母板的按鍵是一格一格緊鄰的，手移過去的路上會經過其他按鍵。
   在畫面上鋪一層「按鍵大小」的虛擬格子，計算移動途中游標在每一格停留多久。
   停留時間必須比 95% 的滯留時間更長，才不會誤選路過的按鍵。
   滯留時間會隨格子大小改變，所以在建議的按鍵大小下計算。

另外報告停留時游標的晃動（平滑前 vs 平滑後），用來說明 One Euro Filter 的效果。
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import plotstyle as ps
from plotstyle import plt

KEY_MIN, KEY_MAX = 40, 136      # 字母板（1280×760、8 欄 × 4 列）放得下的按鍵大小範圍
DWELL_MIN, DWELL_MAX = 0.4, 2.0


def load(session):
    session = Path(session)
    meta = json.loads((session / "meta.json").read_text())
    trials = pd.read_csv(session / "trials.csv")
    frames = pd.read_csv(session / "frames.csv")
    return session, meta, trials, frames


# ---------------------------------------------------------------- 1. Fitts' law

def fitts(trials):
    ok = trials[trials.timeout == 0]
    by = (trials.groupby(["A", "W"])
          .agg(n=("trial", "size"), timeouts=("timeout", "sum"), entries=("entries", "mean"))
          .join(ok.groupby(["A", "W"]).agg(D=("distance", "mean"), mt=("mt_move", "mean"),
                                           mt_sd=("mt_move", "std")))
          .reset_index())
    by["ID"] = np.log2(by.D / by.W + 1)
    rows = by.dropna(subset=["mt"])
    b, a = np.polyfit(rows.ID, rows.mt, 1)
    pred = a + b * rows.ID
    ss_res = ((rows.mt - pred) ** 2).sum()
    ss_tot = ((rows.mt - rows.mt.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    tp = (rows.ID / rows.mt).mean()
    return by, {"a": a, "b": b, "r2": r2, "throughput": tp}


# ---------------------------------------------------------------- 2. 一次到位率 → 按鍵大小

def clean_rate(trials):
    t = trials.assign(clean=(trials.timeout == 0) & (trials.entries == 1))
    return t.groupby("W").clean.agg(["mean", "size"]).rename(columns={"mean": "rate", "size": "n"})


def recommend_key(rates, target_rate):
    ws, rs = rates.index.to_numpy(float), rates.rate.to_numpy(float)
    if rs[0] >= target_rate:
        key, note = ws[0], "smallest tested size already meets the target rate"
    elif rs.max() < target_rate:
        key, note = KEY_MAX, "no tested size met the target rate; using the largest key that fits"
    else:
        i = int(np.argmax(rs >= target_rate))           # 第一個達標的大小
        w0, w1, r0, r1 = ws[i - 1], ws[i], rs[i - 1], rs[i]
        key = w0 + (target_rate - r0) * (w1 - w0) / (r1 - r0) if r1 > r0 else w1
        note = "interpolated between tested sizes"
    key = int(np.clip(math.ceil(key / 5) * 5, KEY_MIN, KEY_MAX))
    return key, note


# ---------------------------------------------------------------- 3. 移動途中的滯留時間 → 停留時間

def linger_durations(trials, frames, cell):
    """在 cell×cell 的虛擬格子上，計算每次移動途中游標在各格停留的時間。

    排除每次移動的第一段（剛選完、還在上一個目標）和最後一段（已到達目標）。
    """
    f = frames.dropna(subset=["x", "y"])
    dt = frames.t.diff().median()
    out = []
    for _, tr in trials.iterrows():
        t_end = tr.t_start + tr.mt_move if tr.timeout == 0 else tr.t_end
        seg = f[(f.seq == tr.seq) & (f.t > tr.t_start) & (f.t <= t_end)]
        if len(seg) < 3:
            continue
        cx, cy = (seg.x // cell).to_numpy(), (seg.y // cell).to_numpy()
        change = np.r_[True, (cx[1:] != cx[:-1]) | (cy[1:] != cy[:-1])]
        ids = np.cumsum(change)
        ts = seg.t.to_numpy()
        for k in range(2, ids[-1]):                      # 去掉第一段和最後一段
            m = ids == k
            out.append(ts[m][-1] - ts[m][0] + dt)
    return np.array(out)


def recommend_dwell(trials, frames, key):
    d = linger_durations(trials, frames, key)
    if len(d) < 5:
        return 1.0, float("nan"), d
    p95 = float(np.percentile(d, 95))
    dwell = float(np.clip(math.ceil((p95 + 0.1) * 10) / 10, DWELL_MIN, DWELL_MAX))
    return dwell, p95, d


# ---------------------------------------------------------------- 晃動（說明用）

def hold_jitter(trials, frames, dwell):
    """選中前最後半個 dwell 時間內，游標相對「自己的平均位置」的晃動（95 百分位，像素）。"""
    res = {"raw": [], "smoothed": []}
    f = frames.dropna(subset=["x", "y", "raw_x", "raw_y"])
    for _, tr in trials[trials.timeout == 0].iterrows():
        seg = f[(f.seq == tr.seq) & (f.t > tr.t_end - dwell / 2) & (f.t <= tr.t_end)]
        if len(seg) < 3:
            continue
        for name, xs, ys in [("raw", seg.raw_x, seg.raw_y), ("smoothed", seg.x, seg.y)]:
            res[name].extend(np.hypot(xs - xs.mean(), ys - ys.mean()))
    return {k: round(float(np.percentile(v, 95)), 1) if v else None for k, v in res.items()}


# ---------------------------------------------------------------- 圖

def plot(by, fit, rates, target_rate, curve, rec, title, path):
    ps.apply()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=ps.INK)

    # A. Fitts' law
    ax = axes[0]
    for amp, color in zip(sorted(by.A.unique()), [ps.BLUE, ps.ORANGE]):
        d = by[by.A == amp].dropna(subset=["mt"])
        ax.errorbar(d.ID, d.mt, yerr=d.mt_sd, fmt="o", ms=8, color=color, ecolor=color,
                    elinewidth=1, capsize=3, label=f"A = {amp}px", zorder=3)
    xs = np.linspace(by.ID.min() - 0.2, by.ID.max() + 0.2, 50)
    ax.plot(xs, fit["a"] + fit["b"] * xs, color=ps.MUTED, lw=1.5, ls="--", zorder=2)
    ax.text(0.03, 0.95, f"MT = {fit['a']:.2f} + {fit['b']:.2f}·ID\n"
            f"R² = {fit['r2']:.2f}   TP = {fit['throughput']:.2f} bits/s",
            transform=ax.transAxes, va="top", fontsize=9, color=ps.INK_2)
    ax.set(title="Movement time vs difficulty", xlabel="Index of difficulty, ID (bits)",
           ylabel="Movement time (s, excl. dwell)")
    ax.legend(loc="lower right")

    # B. 一次到位率
    ax = axes[1]
    ax.plot(rates.index, rates.rate * 100, "-o", color=ps.BLUE, ms=8, zorder=3)
    for w, r in rates.rate.items():
        ax.annotate(f"{r * 100:.0f}%", (w, r * 100), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color=ps.INK_2)
    ax.axhline(target_rate * 100, color=ps.MUTED, ls=":", lw=1.2)
    ax.axvline(rec["key_size_px"], color=ps.ORANGE, ls="--", lw=1.5)
    right = rec["key_size_px"] > 100                       # 靠右時把文字放在線的左邊
    ax.text(rec["key_size_px"], 4, f"recommended \nkey {rec['key_size_px']}px " if right
            else f" recommended\n key {rec['key_size_px']}px",
            color=ps.INK_2, fontsize=9, va="bottom", ha="right" if right else "left")
    ax.set(title="Clean acquisitions by target size", xlabel="Target size W (px)",
           ylabel="Selected on first entry (%)", ylim=(0, 112),
           xlim=(min(KEY_MIN, rates.index.min()) - 5, KEY_MAX + 5))

    # C. 滯留時間
    ax = axes[2]
    ok = curve.dropna()
    ax.plot(ok.cell, ok.p95, "-", color=ps.BLUE, lw=2, label="95th pct linger time")
    ax.axvline(rec["key_size_px"], color=ps.ORANGE, ls="--", lw=1.5)
    ax.axhline(rec["dwell_s"], color=ps.ORANGE, ls="--", lw=1.5)
    ax.text(KEY_MIN, rec["dwell_s"], f"recommended dwell {rec['dwell_s']:.1f}s",
            va="bottom", fontsize=9, color=ps.INK_2)
    ax.set(title="Unintended lingering on key-sized cells", xlabel="Key size (px)",
           ylabel="Linger time while moving (s)", xlim=(KEY_MIN - 5, KEY_MAX + 5),
           ylim=(0, max(ok.p95.max(), rec["dwell_s"]) * 1.25 if len(ok) else 2))
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 主程式

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="fitts_task.py 產生的資料夾")
    ap.add_argument("--target-rate", type=float, default=0.9, help="一次到位率的目標")
    args = ap.parse_args()

    session, meta, trials, frames = load(args.session)
    dwell_cal = meta["args"]["dwell"]
    if (trials.timeout == 0).sum() < 6:
        raise SystemExit("成功的選取太少，無法分析。")

    by, fit = fitts(trials)
    rates = clean_rate(trials)
    key, key_note = recommend_key(rates, args.target_rate)
    dwell, p95, _ = recommend_dwell(trials, frames, key)
    cells = list(range(KEY_MIN, KEY_MAX + 1, 8))
    curve = pd.DataFrame({"cell": cells,
                          "p95": [np.percentile(d, 95) if len(d := linger_durations(trials, frames, c)) >= 5
                                  else np.nan for c in cells]})
    jitter = hold_jitter(trials, frames, dwell_cal)

    rec = {"key_size_px": key, "key_note": key_note, "target_rate": args.target_rate,
           "clean_rate_by_W": {int(w): round(float(r), 3) for w, r in rates.rate.items()},
           "dwell_s": dwell, "linger_p95_at_key_s": round(p95, 3)}
    who = meta["participant"] + (" (SYNTHETIC)" if meta.get("synthetic") else "")
    plot(by, fit, rates, args.target_rate, curve, rec,
         f"Pointing calibration — {who}, input: {meta['input']}", session / "fitts_report.png")

    result = {
        "participant": meta["participant"], "synthetic": bool(meta.get("synthetic")),
        "input": meta["input"], "calibration_dwell_s": dwell_cal,
        "trials": int(len(trials)), "timeouts": int(trials.timeout.sum()),
        "fitts": {k: round(float(v), 3) for k, v in fit.items()},
        "hold_jitter_p95_px": jitter,
        "conditions": by.round(3).to_dict(orient="records"),
        "recommendation": rec,
    }
    (session / "recommendation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n=== {who} ===")
    print(by[["A", "W", "ID", "n", "timeouts", "mt", "entries"]].round(2).to_string(index=False))
    print(f"\nFitts' law: MT = {fit['a']:.3f} + {fit['b']:.3f}·ID   R² = {fit['r2']:.3f}"
          f"   throughput = {fit['throughput']:.2f} bits/s")
    print("一次到位率：" + "  ".join(f"W={int(w)}: {r:.0%}" for w, r in rates.rate.items()))
    print(f"→ 建議按鍵大小 {key} px（{key_note}）")
    print(f"移動途中滯留時間 P95 = {p95:.2f} s → 建議停留時間 {dwell} s")
    print(f"停留時晃動（95 百分位）：平滑前 {jitter['raw']} px → 平滑後 {jitter['smoothed']} px")
    print(f"\n圖：{session / 'fitts_report.png'}")
    print("用建議設定打字：")
    print(f"  python collector.py --participant {meta['participant']} --condition recommended "
          f"--key-size {key} --dwell {dwell}")


if __name__ == "__main__":
    main()
