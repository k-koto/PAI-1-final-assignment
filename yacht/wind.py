"""風(傾ける力)の強弱とリズムのデータを生成・読み込みする。

風の強さ w(t) は 0〜1 に正規化した値で、1 のとき船を傾けるトルクが最大になる。
"""

from pathlib import Path

import numpy as np

DT = 0.05  # データの時間刻み[s] (制御周期と同じ)
DURATION = 60.0
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "wind"


def _gust_shape(t, t0, rise, hold, decay):
    """t0から立ち上がり、一定時間続いて消える1回の突風の形 (0〜1)。"""
    up = np.clip((t - t0) / rise, 0, 1)
    down = np.clip(1 - (t - (t0 + rise + hold)) / decay, 0, 1)
    return np.minimum(up, down)


def generate(rng, *, base, period, jitter, amp, rise, hold, decay, lull_prob=0.0, noise=0.03, cap=1.0):
    """周期的な突風(ブロー)と急な弱まり(ラル)を含む風の時系列を作る。

    パラメータは (min, max) のタプルで渡すと突風ごとにばらつく。
    """
    def pick(v):
        return rng.uniform(*v) if isinstance(v, tuple) else v

    t = np.arange(0, DURATION, DT)
    w = np.full_like(t, pick(base))
    t0 = rng.uniform(1.0, 3.0)
    while t0 < DURATION:
        g = _gust_shape(t, t0, pick(rise), pick(hold), pick(decay))
        if rng.random() < lull_prob:
            w = w * (1 - 0.9 * g)  # 急に風が抜ける
        else:
            w = w + pick(amp) * g
        t0 += pick(period) * (1 + rng.uniform(-jitter, jitter))
    # 細かな揺らぎ(オルンシュタイン・ウーレンベック過程)
    n = np.zeros_like(t)
    for i in range(1, len(t)):
        n[i] = n[i - 1] * 0.95 + rng.normal(0, noise)
    return t, np.clip(w + n, 0, cap)


TRAIN_PARAMS = dict(base=(0.15, 0.45), period=(3.0, 8.0), jitter=0.2, amp=(0.15, 0.45),
                    rise=(0.3, 1.5), hold=(0.5, 2.5), decay=(0.4, 2.0), lull_prob=0.15, cap=0.9)

# 学習に使わない未知のパターン
TEST_PATTERNS = {
    "storm": dict(base=0.45, period=(2.0, 2.5), jitter=0.1, amp=(0.4, 0.55), rise=(0.15, 0.3),
                  hold=(0.4, 0.8), decay=(0.3, 0.6), noise=0.04),
    "irregular": dict(base=(0.2, 0.5), period=(1.0, 10.0), jitter=0.5, amp=(0.1, 0.6), rise=(0.1, 2.0),
                      hold=(0.2, 3.0), decay=(0.1, 2.5), lull_prob=0.3),
    "lull": dict(base=0.8, period=(4.0, 6.0), jitter=0.3, amp=(0.0, 0.0), rise=(0.1, 0.2),
                 hold=(0.8, 1.5), decay=(0.1, 0.3), lull_prob=1.0),
}


def save(path, t, w):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([t, w]), delimiter=",", header="t,wind", comments="", fmt="%.3f")


def load(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)[:, 1]


def make_dataset(n_train=200, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n_train):
        save(DATA_DIR / "train" / f"train_{i:03d}.csv", *generate(rng, **TRAIN_PARAMS))
    rng = np.random.default_rng(seed + 1000)
    for name, params in TEST_PATTERNS.items():
        save(DATA_DIR / "test" / f"{name}.csv", *generate(rng, **params))
