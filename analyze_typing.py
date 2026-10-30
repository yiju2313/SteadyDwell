"""比較不同設定下的打字表現（例如預設 vs 建議設定）。

    python analyze_typing.py sessions/typing_*

輸出（存在 sessions/ 底下）：
    typing_summary.csv    每個 session 一列
    typing_report.png     各指標的比較圖

指標（只算完整打完的句子）：
    WPM            (|T| - 1) / 秒數 × 60 / 5，從第一次選取算到打完
    錯誤選取率      選到的按鍵 ≠ 當下該選的按鍵，佔所有選取的比例
    KSPC           每個字元平均選取幾次（最理想是 1.0；選錯要刪除、重打，就會變大）
    猶豫次數        每句話裡「停留進度超過 50% 才離開」的次數
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import plotstyle as ps
from plotstyle import plt


def load_events(csv_path):
    frames = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
    rows = []
    for _, r in frames[frames.events != ""].iterrows():
        for ev in r.events.split("|"):
            parts = ev.split(":")
            rows.append({
                "t": float(r.t), "phrase_idx": int(r.phrase_idx), "target_key": r.target_key,
                "kind": parts[0], "key": parts[1] if len(parts) > 1 else "",
                "progress": float(parts[2]) if len(parts) > 2 else np.nan,
            })
    return pd.DataFrame(rows, columns=["t", "phrase_idx", "target_key", "kind", "key", "progress"])


def phrase_metrics(events, phrases):
    out = []
    for p, g in events.groupby("phrase_idx"):
        sel = g[g.kind == "select"]
        done = g[g.kind == "phrase_done"]
        text = phrases[p]
        row = {"phrase_idx": p, "chars": len(text), "selections": len(sel),
               "errors": int((sel.key != sel.target_key).sum()),
               "hesitations": int(((g.kind == "leave") & (g.progress >= 0.5) & (g.progress < 1)).sum()),
               "completed": len(done) > 0}
        if row["completed"] and len(sel) > 1:
            seconds = done.t.iloc[0] - sel.t.iloc[0]
            row["wpm"] = (len(text) - 1) / seconds * 60 / 5
        out.append(row)
    return pd.DataFrame(out)


def summarize(session):
    session = Path(session)
    meta = json.loads((session / "meta.json").read_text())
    pm = phrase_metrics(load_events(session / "frames.csv"), meta["phrases"])
    c = pm[pm.completed]
    return {
        "session": session.name, "participant": meta["participant"],
        "condition": meta.get("condition", "unlabeled"),
        "key_size": meta.get("key_size", meta.get("layout", {}).get("cell", [None, None])[1]),
        "dwell": meta["args"]["dwell"], "input": meta["input"],
        "phrases_completed": int(len(c)), "phrases_attempted": int(len(pm)),
        "wpm": c.wpm.mean() if len(c) else np.nan,
        "error_rate": c.errors.sum() / c.selections.sum() if len(c) else np.nan,
        "kspc": c.selections.sum() / c.chars.sum() if len(c) else np.nan,
        "hesitations_per_phrase": c.hesitations.mean() if len(c) else np.nan,
    }


def plot(df, path):
    ps.apply()
    metrics = [("wpm", "Typing speed (WPM)", "higher is better"),
               ("error_rate", "Error selection rate", "lower is better"),
               ("kspc", "Selections per character (KSPC)", "1.0 is ideal"),
               ("hesitations_per_phrase", "Hesitations per phrase", "lower is better")]
    people = sorted(df.participant.unique())
    conds = sorted(df.condition.unique(), key=lambda c: (c != "default", c))
    colors = [ps.BLUE, ps.ORANGE, ps.MUTED]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    fig.suptitle("Letterboard typing by condition", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=ps.INK)
    width = 0.8 / max(len(conds), 1)
    for ax, (col, title, hint) in zip(axes, metrics):
        for j, cond in enumerate(conds):
            vals = [df[(df.participant == p) & (df.condition == cond)][col].mean() for p in people]
            xs = np.arange(len(people)) + (j - (len(conds) - 1) / 2) * width
            bars = ax.bar(xs, vals, width * 0.92, color=colors[j % 3], label=cond, zorder=3)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                                textcoords="offset points", ha="center", fontsize=8, color=ps.INK_2)
        ax.set_xticks(range(len(people)), people)
        ax.set_title(title, pad=20)
        ax.text(0, 1.02, hint, transform=ax.transAxes, fontsize=8, color=ps.MUTED, va="bottom")
        ax.grid(axis="x", visible=False)
        if col == "kspc":
            ax.axhline(1.0, color=ps.MUTED, ls=":", lw=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(conds), bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="collector.py 產生的資料夾")
    ap.add_argument("--out", default="sessions")
    args = ap.parse_args()

    dirs = [d for d in args.sessions
            if (Path(d) / "frames.csv").exists() and (Path(d) / "meta.json").exists()]
    if not dirs:
        raise SystemExit("找不到含有 frames.csv 和 meta.json 的資料夾。")
    df = pd.DataFrame([summarize(d) for d in dirs])
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df.to_csv(out / "typing_summary.csv", index=False)
    cols = ["participant", "condition", "key_size", "dwell", "phrases_completed",
            "wpm", "error_rate", "kspc", "hesitations_per_phrase"]
    print(df[cols].round(3).to_string(index=False))
    plot(df, out / "typing_report.png")
    print(f"\n圖：{out / 'typing_report.png'}\n表：{out / 'typing_summary.csv'}")


if __name__ == "__main__":
    main()
