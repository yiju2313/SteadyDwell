"""不需要攝影機的邏輯測試：python test_logic.py"""
import random
import statistics

from board import Board, DwellSelector, apply_key, next_target
from filters import PointFilter

board = Board()
A = next(k for k in board.keys if k.label == "A")
B = next(k for k in board.keys if k.label == "B")
ax, ay = A.center
bx, by = B.center

# 1. 命中判定
assert board.key_at(ax, ay) is A
assert board.key_at(0, 0) is None

# 2. 停留選取：1 秒內不選，滿 1 秒選中
sel = DwellSelector(board, dwell_time=1.0, hysteresis_px=20)
assert sel.update(ax, ay, 0.0) == [("enter", A)]
for _ in range(9):
    assert sel.update(ax, ay, 0.1) == []
assert sel.update(ax, ay, 0.1) == [("select", A)]
# 選中後繼續停留，不會再選一次（要先離開）
for _ in range(30):
    assert sel.update(ax, ay, 0.1) == []
assert sel.progress == 0.0

# 離開再回來，可以再選一次（打出 LL）
sel.update(bx, by, 0.1)
sel.update(ax, ay, 0.1)
for _ in range(9):
    sel.update(ax, ay, 0.1)
assert sel.update(ax, ay, 0.1) == [("select", A)]

# 3. 遲滯：剛好超出 A 的右邊界 5px（在間隙中），仍算在 A 上
sel = DwellSelector(board, dwell_time=1.0, hysteresis_px=20)
sel.update(ax, ay, 0.0)
edge_x = A.x + A.w + 5
assert board.key_at(edge_x, ay) is None
assert sel.update(edge_x, ay, 0.1) == []
assert sel.current is A
# 沒有遲滯時，同樣位置會離開 A
sel0 = DwellSelector(board, 1.0, hysteresis_px=0)
sel0.update(ax, ay, 0.0)
assert sel0.update(edge_x, ay, 0.1)[0][0] == "leave"

# 4. 中斷的停留：停 0.5 秒後移到 B
sel = DwellSelector(board, 1.0, 20)
sel.update(ax, ay, 0.0)
for _ in range(5):
    sel.update(ax, ay, 0.1)
ev = sel.update(bx, by, 0.1)
assert ev[0] == ("leave", A, 0.5) and ev[1] == ("enter", B), ev

# 5. 手消失 → 離開
assert sel.update(None, None, 0.1)[0][0] == "leave"
assert sel.current is None

# 6. 不同按鍵大小都放得進 1280×760 的畫面
for size in (40, 90, 136):
    b = Board(key_size=size)
    assert min(k.x for k in b.keys) >= 0 and max(k.x + k.w for k in b.keys) <= 1280
    assert max(k.y + k.h for k in b.keys) <= 740, size
    assert len(b.keys) == 29

# 7. 文字與目標
assert apply_key("HI", "SPACE") == "HI " and apply_key("HI", "DEL") == "H"
assert apply_key("HI", "SPEAK") == "HI" and apply_key("HI", "X") == "HIX"
assert next_target("", "HI YOU") == "H"
assert next_target("HI", "HI YOU") == "SPACE"
assert next_target("HX", "HI YOU") == "DEL"
assert next_target("HI YOU", "HI YOU") == ""

# 8. One Euro Filter 降低靜止時的抖動
random.seed(0)
f = PointFilter(min_cutoff=1.0, beta=0.01)
raw, smooth = [], []
for i in range(300):
    x = 500 + random.gauss(0, 8)
    raw.append(x)
    smooth.append(f(x, 300, i / 30)[0])
r, s = statistics.stdev(raw[30:]), statistics.stdev(smooth[30:])
print(f"抖動標準差：原始 {r:.2f}px → 平滑後 {s:.2f}px")
assert s < r * 0.5

print("全部測試通過")

# ---------------------------------------------------------------- 指向測驗
import math
import fitts_task as ft

assert ft.selection_order() == [0, 5, 1, 6, 2, 7, 3, 8, 4, 0]
conds = ft.make_conditions(seed=1)
assert sorted(conds) == sorted((a, w) for a in ft.AMPLITUDES for w in ft.WIDTHS)
for a in ft.AMPLITUDES:                       # 目標不重疊、不超出畫面
    p = ft.target_positions(a)
    assert math.dist(p[0], p[1]) > max(ft.WIDTHS)
    assert all(0 <= x - 65 and x + 65 <= ft.W and 0 <= y - 65 and y + 65 <= ft.H for x, y in p)

task = ft.FittsTask([(600, 80)], dwell=1.0, timeout=10.0)
task.start_next_sequence(0.0)
start = task.positions[task.active]
for i in range(11):                          # 起點：停 1 秒 → "start"，不記錄
    ev = task.update(start, 0.1 * i, 0.1 if i else 0.0)
assert ev == "start" and task.trials == []
tgt = task.positions[task.active]            # 第 1 個目標：先在外面晃，再進去停 1 秒
task.update((tgt[0] + 100, tgt[1]), 1.5, 0.1)
task.update(tgt, 1.6, 0.1)
task.update((tgt[0] + 100, tgt[1]), 1.7, 0.1)  # 跑出去 → 重來
for i in range(11):
    ev = task.update(tgt, 1.8 + 0.1 * i, 0.1 if i else 0.0)
assert ev == "select", ev
tr = task.trials[0]
assert tr["entries"] == 2 and tr["timeout"] == 0 and abs(tr["mt_move"] - 0.8) < 1e-6
ev = task.update(None, 1.8 + 1.0 + 10.5, 0.1)  # 第 2 個目標：10 秒沒到 → 逾時
assert ev == "timeout" and task.trials[1]["timeout"] == 1
print("指向測驗測試通過")
