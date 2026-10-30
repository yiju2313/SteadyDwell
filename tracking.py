"""輸入來源：webcam 手部追蹤（食指指尖）或滑鼠。

指向測驗（fitts_task.py）和字母板（collector.py）共用這個模組。
兩種輸入都提供相同的介面：
    read()  → 回傳 (x, y) 螢幕像素座標；沒偵測到則回傳 None
    frame   → 最近一張攝影機畫面（滑鼠模式為 None）
    close() → 釋放資源
"""
import time

import cv2
import numpy as np

INDEX_FINGER_TIP = 8  # MediaPipe 手部 21 個關鍵點中，食指指尖的編號


class HandInput:
    """用 MediaPipe Hand Landmarker 取得食指指尖位置。"""

    def __init__(self, screen_size, model_path="hand_landmarker.task", camera=0, margin=0.15):
        import mediapipe as mp
        self.mp = mp
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            raise RuntimeError("打不開攝影機。請確認終端機有相機權限，或換 --camera 1 試試。")
        self.screen_w, self.screen_h = screen_size
        self.margin = margin
        self.t0 = time.monotonic()
        self.last_ts = -1
        self.frame = None

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)  # 鏡像：手往右，游標也往右
        self.frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - self.t0) * 1000)
        ts = max(ts, self.last_ts + 1)  # VIDEO 模式要求時間戳嚴格遞增
        self.last_ts = ts
        result = self.landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return None
        tip = result.hand_landmarks[0][INDEX_FINGER_TIP]
        return self._to_screen(tip.x, tip.y)

    def _to_screen(self, nx, ny):
        # 只用攝影機畫面中央的區域對應到整個螢幕，手不用伸到畫面邊緣
        m = self.margin
        sx = np.clip((nx - m) / (1 - 2 * m), 0, 1) * self.screen_w
        sy = np.clip((ny - m) / (1 - 2 * m), 0, 1) * self.screen_h
        return float(sx), float(sy)

    def close(self):
        self.cap.release()
        self.landmarker.close()


class MouseInput:
    """不開攝影機，用滑鼠當游標（除錯用）。必須在 cv2.namedWindow 之後建立。"""

    def __init__(self, window):
        self.pos = None
        self.frame = None
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        self.pos = (float(x), float(y))

    def read(self):
        return self.pos

    def close(self):
        pass
