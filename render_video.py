"""提出用の動画を作る。

    uv run python render_video.py --early ppo_25000_steps --mid ppo_100000_steps --final ppo_final

構成: タイトル → 制御なし → 学習の初期/中期/適合後 → 未知の風でのテスト → まとめ
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from yacht import wind
from yacht.env import GOAL, YachtBalanceEnv
from yacht.model import ROOT

W, H = 1280, 720
FPS = 20
FONT = ROOT / "assets" / "fonts" / "NotoSansJP.ttf"
HIST_SEC = 10.0


def font(size, bold=False):
    f = ImageFont.truetype(str(FONT), size)
    f.set_variation_by_axes([700 if bold else 400])
    return f


WIND_WEAK = np.array([0.2, 0.55, 1.0])  # 弱い風: 青
WIND_STRONG = np.array([1.0, 0.15, 0.1])  # 強い風: 赤


def wind_color(w):
    """風の強さ0〜1を青→赤のグラデーションに変換する。"""
    w = float(np.clip(w, 0, 1))
    return np.array([*(WIND_WEAK + (WIND_STRONG - WIND_WEAK) * w), 0.85], np.float32)


class Recorder:
    def __init__(self, env, writer):
        self.env = env
        self.writer = writer  # メモリに溜めず、1フレームずつ動画へ書き出す
        self.main = mujoco.Renderer(env.model, H, W)
        self.inset = mujoco.Renderer(env.model, 216, 384)

    def _wind_arrows(self, scene, w):
        """風上から船へ向かう風の矢印。長さと色の濃さが風の強さを表す。"""
        if w < 0.02:
            return
        for z in (0.25, 0.5, 0.75):
            for x in (-0.2, 0.15):
                if scene.ngeom >= scene.maxgeom:
                    return
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                                    wind_color(w))
                start = np.array([x, 0.55 + 0.45 * w, z])
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012 + 0.012 * w, start,
                                     start - np.array([0, 0.45 * w, 0]))
                scene.ngeom += 1

    def render3d(self, w):
        self.main.update_scene(self.env.data, camera="quarter")
        self._wind_arrows(self.main.scene, w)
        img = Image.fromarray(self.main.render())
        self.inset.update_scene(self.env.data, camera="stern")
        self._wind_arrows(self.inset.scene, w)
        inset = Image.fromarray(self.inset.render())
        img.paste(inset, (W - 384 - 16, 16))
        d = ImageDraw.Draw(img)
        d.rectangle([W - 384 - 16, 16, W - 16, 16 + 216], outline=(255, 255, 255), width=2)
        d.text((W - 384 - 8, 20), "後方から", font=font(18), fill="white")
        return img

    def hud(self, img, title, subtitle, hist, info, speedup, banner=None):
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, 0, 760, 92], fill=(0, 0, 0, 140))
        d.text((20, 10), title, font=font(36, True), fill="white")
        d.text((22, 56), subtitle, font=font(20), fill=(220, 230, 240))
        if speedup != 1:
            d.rounded_rectangle([W - 400 - 16 - 110, 20, W - 400 - 16 - 10, 60], 8, fill=(230, 60, 60, 220))
            d.text((W - 400 - 16 - 96, 24), f"{speedup}倍速", font=font(22, True), fill="white")

        # 下部パネル
        top = H - 170
        d.rectangle([0, top, W, H], fill=(0, 0, 0, 150))
        heel = np.rad2deg(info["heel"])
        col = (90, 220, 120) if abs(heel) < 10 else (250, 200, 60) if abs(heel) < 25 else (255, 80, 80)
        d.text((20, top + 10), "傾き", font=font(20), fill="white")
        d.text((20, top + 34), f"{heel:+5.1f}°", font=font(44, True), fill=col)
        self._bar(d, 20, top + 100, 200, "風の強さ", info["wind"], tuple(int(c * 255) for c in wind_color(info["wind"])[:3]))
        self._bar(d, 20, top + 132, 200, "ハイク量", info["hike"], (255, 140, 40))

        # 傾きと風の履歴グラフ
        gx, gy, gw, gh = 260, top + 14, 560, 142
        d.rectangle([gx, gy, gx + gw, gy + gh], fill=(255, 255, 255, 25))
        mid = gy + gh / 2
        for a in (-30, 0, 30):
            y = mid - a / 45 * gh / 2
            d.line([gx, y, gx + gw, y], fill=(255, 255, 255, 60 if a else 120))
            d.text((gx + 4, y - 14), f"{a:+d}°", font=font(12), fill=(255, 255, 255, 160))
        n = int(HIST_SEC * FPS)
        if len(hist) > 1:
            h = np.array(hist[-n:])
            xs = gx + gw - (len(h) - 1 - np.arange(len(h))) / n * gw
            wy = gy + gh - h[:, 1] * gh
            d.line(list(zip(xs, wy)), fill=(80, 200, 255, 200), width=2)
            hy = mid - np.clip(np.rad2deg(h[:, 0]), -45, 45) / 45 * gh / 2
            d.line(list(zip(xs, hy)), fill=(255, 255, 255), width=3)
        d.text((gx + gw - 170, gy + 2), "— 傾き  — 風", font=font(14), fill=(255, 255, 255, 200))

        # 速度とゴールまでの進み具合
        px = 850
        d.text((px, top + 10), "速度", font=font(20), fill="white")
        d.text((px, top + 34), f"{info['speed'] * 1.944:4.1f} kt", font=font(40, True), fill="white")
        d.text((px + 200, top + 10), "経過時間", font=font(20), fill="white")
        d.text((px + 200, top + 34), f"{info['time']:5.1f} s", font=font(40, True), fill="white")
        prog = min(info["distance"] / GOAL, 1.0)
        d.text((px, top + 96), f"ゴールまで {max(GOAL - info['distance'], 0):5.1f} m", font=font(18), fill="white")
        d.rounded_rectangle([px, top + 126, W - 20, top + 150], 6, fill=(255, 255, 255, 50))
        if prog > 0:
            d.rounded_rectangle([px, top + 126, px + (W - 20 - px) * prog, top + 150], 6, fill=(90, 220, 120))

        if banner:
            text, color = banner
            f = font(80, True)
            tw = d.textlength(text, font=f)
            d.rounded_rectangle([W / 2 - tw / 2 - 40, 250, W / 2 + tw / 2 + 40, 380], 20, fill=color)
            d.text((W / 2 - tw / 2, 262), text, font=f, fill="white")
        return img

    @staticmethod
    def _bar(d, x, y, w, label, v, color):
        d.text((x, y - 2), label, font=font(16), fill="white")
        d.rounded_rectangle([x + 80, y + 2, x + w, y + 20], 4, fill=(255, 255, 255, 50))
        d.rounded_rectangle([x + 80, y + 2, x + 80 + (w - 80) * float(np.clip(v, 0, 1)), y + 20], 4, fill=color)

    def episode(self, policy, wind_index, title, subtitle, speedup=1, max_time=60.0, hold=1.5):
        """1本走らせて動画に追加し、結果(転覆したか, ゴール時刻)を返す。"""
        env = self.env
        obs, info = env.reset(options={"wind_index": wind_index})
        hist, goal_time, end_k = [], None, None
        k = 0
        while True:
            action = np.array([-1.0]) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, info = env.step(action)
            hist.append((info["heel"], info["wind"]))
            if info["capsized"]:
                # 転覆した瞬間で画面を止める(続けると重心の低い船体が起き上がってしまう)
                img = self.render3d(info["wind"])
                frame = np.asarray(self.hud(img, title, subtitle, hist, info, speedup,
                                            ("転覆!", (220, 50, 50, 230))))
                for _ in range(int(hold * FPS)):
                    self.writer.append_data(frame)
                print(f"  {title}: 転覆 {info['time']:.1f}s", flush=True)
                return True, None
            if info["goal"] and goal_time is None:
                goal_time = info["time"]
            if end_k is None and (goal_time is not None or info["time"] >= max_time):
                end_k = k + int(hold * FPS)
            banner = (f"ゴール! {goal_time:.1f}秒", (40, 170, 90, 230)) if goal_time is not None else None
            if k % speedup == 0 or k == end_k:
                img = self.render3d(info["wind"])
                self.writer.append_data(np.asarray(self.hud(img, title, subtitle, hist, info, speedup, banner)))
            if truncated or (end_k is not None and k >= end_k):
                print(f"  {title}: ゴール {goal_time}", flush=True)
                return False, goal_time
            k += 1

    def card(self, lines, seconds=3.0):
        img = Image.new("RGB", (W, H), (12, 40, 70))
        d = ImageDraw.Draw(img)
        y = H / 2 - sum(s for _, s, _ in lines) * 0.7
        for text, size, bold in lines:
            f = font(size, bold)
            d.text((W / 2 - d.textlength(text, font=f) / 2, y), text, font=f, fill="white")
            y += size * 1.4
        frame = np.asarray(img)
        for _ in range(int(seconds * FPS)):
            self.writer.append_data(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--early", required=True)
    parser.add_argument("--mid", required=True)
    parser.add_argument("--final", default="ppo_final")
    parser.add_argument("--train-wind", type=int, default=43)
    parser.add_argument("--out", default="video/yacht_balance.mp4")
    args = parser.parse_args()

    stages = [("学習初期", args.early), ("学習中期", args.mid), ("適合後", args.final)]
    policies = {k: PPO.load(Path("checkpoints") / v, device="cpu") for k, v in stages}

    def steps_label(name):
        return "学習完了" if name == "ppo_final" else f"{int(name.split('_')[1]):,}ステップ学習"

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    writer = imageio.get_writer(out, fps=FPS, quality=8, macro_block_size=8)
    train = Recorder(YachtBalanceEnv(), writer)
    test_files = sorted((wind.DATA_DIR / "test").glob("*.csv"))
    test = Recorder(YachtBalanceEnv(test_files), writer)

    train.card([("ヨットのバランスを取る Physical AI", 52, True),
                ("470級ヨット × SO-101アーム × 強化学習(PPO)", 30, False)], 3.5)
    train.card([("風が帆を押して船を傾ける", 34, True),
                ("AIはアームで重りを風上へ乗り出し、傾きを抑える(ハイクアウト)", 26, False),
                ("傾くほど船は遅くなる ── 平らに保って30m先のゴールを目指す", 26, False)], 5.0)

    capsized, _ = train.episode(None, args.train_wind, "制御なし", "アームは畳んだまま", hold=2.0)
    results = [("制御なし", "転覆" if capsized else "-")]

    for label, ckpt in stages:
        train.card([(label, 56, True), (steps_label(ckpt), 30, False)], 1.5)
        capsized, goal = train.episode(policies[label], args.train_wind, f"AI {label}",
                                       f"{steps_label(ckpt)} / 学習に使った風", max_time=20.0)
        results.append((label, "転覆" if capsized else f"{goal:.1f}秒" if goal else "未到達"))

    train.card([("未知の風でテスト", 56, True), ("学習に使っていない強さ・リズムの風", 30, False)], 2.0)
    titles = {"storm": "強風・短い周期の突風", "irregular": "不規則なリズム", "lull": "強風からの急な風の抜け"}
    for i, f in enumerate(test_files):
        name = titles.get(f.stem, f.stem)
        capsized, goal = test.episode(policies["適合後"], i, f"未知の風: {name}", "適合後のAI", max_time=20.0)
        results.append((f"未知: {name}", "転覆" if capsized else f"{goal:.1f}秒" if goal else "未到達"))

    train.card([("結果(ゴールまでの時間)", 44, True)] + [(f"{a}: {b}", 30, False) for a, b in results], 5.0)

    writer.close()
    print(f"{out}: 書き出し完了")
    for a, b in results:
        print(f"  {a}: {b}")


if __name__ == "__main__":
    main()
