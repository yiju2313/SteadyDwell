"""One Euro Filter (Casiez, Roussel & Vogel, CHI 2012).

慢速移動時強力平滑（去抖動），快速移動時減少延遲。
兩個主要參數：
  min_cutoff：越小越平滑，但慢速時延遲越大
  beta：越大，快速移動時越跟手
"""
import math


class LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        self.y = x if self.y is None else alpha * x + (1 - alpha) * self.y
        return self.y


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPass()
        self.dx_filter = LowPass()
        self.t_prev = None
        self.x_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.__init__(self.min_cutoff, self.beta, self.d_cutoff)

    def __call__(self, x, t):
        """x：目前數值，t：時間（秒）。回傳平滑後的數值。"""
        if self.t_prev is None:
            self.t_prev, self.x_prev = t, x
            self.x_filter(x, 1.0)
            self.dx_filter(0.0, 1.0)
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        edx = self.dx_filter(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        result = self.x_filter(x, self._alpha(cutoff, dt))
        self.t_prev, self.x_prev = t, x
        return result


class PointFilter:
    """對 (x, y) 各用一個 One Euro Filter。"""

    def __init__(self, **kw):
        self.fx = OneEuroFilter(**kw)
        self.fy = OneEuroFilter(**kw)

    def reset(self):
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x, y, t):
        return self.fx(x, t), self.fy(y, t)
