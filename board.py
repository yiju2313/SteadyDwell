"""字母板配置、命中判定（含遲滯）與停留選取邏輯。

這個檔案不依賴攝影機或 MediaPipe，所以可以單獨測試。
"""
from dataclasses import dataclass, asdict

# 8 欄 × 4 列，正方形按鍵；數字代表寬度（以一格為單位），預設 1
DEFAULT_ROWS = [
    ["A", "B", "C", "D", "E", "F", "G", "H"],
    ["I", "J", "K", "L", "M", "N", "O", "P"],
    ["Q", "R", "S", "T", "U", "V", "W", "X"],
    ["Y", "Z", ("SPACE", 3), "DEL", "SPEAK"],
]


@dataclass
class Key:
    label: str
    x: float  # 左上角（像素）
    y: float
    w: float
    h: float

    def contains(self, px, py, margin=0.0):
        """margin > 0 表示把按鍵範圍往外擴張（用於遲滯）。"""
        return (self.x - margin <= px <= self.x + self.w + margin and
                self.y - margin <= py <= self.y + self.h + margin)

    @property
    def center(self):
        return self.x + self.w / 2, self.y + self.h / 2


class Board:
    """key_size：正方形按鍵的邊長（像素）。字母板水平置中，從 top 開始往下排。"""

    def __init__(self, key_size=90, gap=10, top=150, screen_w=1280, rows=None):
        rows = rows or DEFAULT_ROWS
        n_cols = max(sum(it[1] if isinstance(it, tuple) else 1 for it in row) for row in rows)
        board_w = n_cols * key_size + (n_cols - 1) * gap
        origin = ((screen_w - board_w) / 2, top)
        cell = (key_size, key_size)
        self.key_size = key_size
        self.origin, self.cell, self.gap = origin, cell, gap
        self.keys = []
        ox, oy = origin
        cw, ch = cell
        for r, row in enumerate(rows):
            col = 0
            for item in row:
                label, span = item if isinstance(item, tuple) else (item, 1)
                x = ox + col * (cw + gap)
                y = oy + r * (ch + gap)
                w = span * cw + (span - 1) * gap
                self.keys.append(Key(label, x, y, w, ch))
                col += span

    def key_at(self, px, py):
        for k in self.keys:
            if k.contains(px, py):
                return k
        return None

    def to_dict(self):
        """存成 JSON，分析工具用它把軌跡畫回鍵盤上。"""
        return {"key_size": self.key_size, "origin": self.origin, "cell": self.cell, "gap": self.gap,
                "keys": [asdict(k) for k in self.keys]}


def apply_key(typed, label):
    """把選中的按鍵套用到已輸入文字上。"""
    if label == "SPACE":
        return typed + " "
    if label == "DEL":
        return typed[:-1]
    if len(label) == 1:
        return typed + label
    return typed  # SPEAK 等功能鍵不改變文字


def next_target(typed, phrase):
    """依照目標句子，回傳「此刻正確的下一個按鍵」。

    已輸入的是目標的前綴 → 下一個字母（空白為 SPACE）
    已經打錯             → DEL（要先刪掉錯字）
    已經完成             → ""
    """
    if not phrase.startswith(typed):
        return "DEL"
    if len(typed) == len(phrase):
        return ""
    ch = phrase[len(typed)]
    return "SPACE" if ch == " " else ch


class DwellSelector:
    """停留選取的狀態機。

    每一幀呼叫 update()，回傳這一幀發生的事件清單：
      ("enter", key)            游標進入按鍵
      ("leave", key, progress)  游標離開按鍵；progress > 0 且 < 1 代表「中斷的停留」
      ("select", key)           停留滿 dwell_time，選中

    選中之後，游標必須先離開這個按鍵，才能再選它一次（避免停著不動就連續輸入）。
    所以要打 HELLO 的 LL，選完第一個 L 要移出去再移回來。
    """

    def __init__(self, board, dwell_time=1.0, hysteresis_px=20.0):
        self.board = board
        self.dwell_time = dwell_time
        self.hysteresis_px = hysteresis_px
        self.current = None    # 目前停留中的 Key
        self.elapsed = 0.0     # 目前已停留秒數
        self.armed = True      # False = 剛選過，要先離開才能再選

    @property
    def progress(self):
        return min(self.elapsed / self.dwell_time, 1.0) if self.current and self.armed else 0.0

    def _resolve_key(self, px, py):
        # 遲滯：已經在某個按鍵上時，要離開「擴張後」的範圍才算離開
        if self.current and self.current.contains(px, py, self.hysteresis_px):
            return self.current
        return self.board.key_at(px, py)

    def update(self, px, py, dt):
        """px, py 為 None 代表這一幀沒偵測到手。"""
        events = []
        key = None if px is None else self._resolve_key(px, py)

        if key is not self.current:
            if self.current is not None:
                events.append(("leave", self.current, round(self.progress, 3)))
            if key is not None:
                events.append(("enter", key))
            self.current, self.elapsed, self.armed = key, 0.0, True
            return events

        if key is not None and self.armed:
            self.elapsed += dt
            if self.elapsed >= self.dwell_time - 1e-6:  # 容許浮點誤差
                events.append(("select", key))
                self.elapsed, self.armed = 0.0, False
        return events
