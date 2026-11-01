"""学習の途中経過(チェックポイント)を、学習用の風と未知の風で評価する。

    uv run python evaluate.py
"""

import re
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from yacht import wind
from yacht.env import YachtBalanceEnv

CKPT_DIR = Path("checkpoints")


def run_episode(env, policy, wind_index):
    """1エピソード走らせ、傾きの記録とゴール時刻を返す。policy=Noneなら制御なし。"""
    obs, info = env.reset(options={"wind_index": wind_index})
    heels, goal_time = [], None
    while True:
        action = np.array([-1.0]) if policy is None else policy.predict(obs, deterministic=True)[0]
        obs, _, terminated, truncated, info = env.step(action)
        heels.append(info["heel"])
        if info["goal"] and goal_time is None:
            goal_time = info["time"]
        if terminated or truncated:
            return np.rad2deg(np.abs(heels)), info["capsized"], goal_time


def summarize(env, policy, indices):
    rows = [run_episode(env, policy, i) for i in indices]
    mean_heel = np.mean([r[0].mean() for r in rows])
    capsized = sum(r[1] for r in rows)
    goals = [r[2] for r in rows if r[2] is not None]
    goal = f"{np.mean(goals):5.1f}s ({len(goals)}/{len(rows)})" if goals else "   -  "
    return f"平均傾き {mean_heel:5.1f}°  転覆 {capsized}/{len(rows)}  ゴール {goal}"


def checkpoints():
    def steps(p):
        m = re.search(r"_(\d+)_steps", p.name)
        return int(m.group(1)) if m else 10**12
    return sorted(CKPT_DIR.glob("ppo_*.zip"), key=steps)


def main():
    train_env = YachtBalanceEnv()
    test_files = sorted((wind.DATA_DIR / "test").glob("*.csv"))
    test_env = YachtBalanceEnv(test_files)
    train_idx = range(0, len(train_env.winds), 20)
    print(f"{'モデル':24s} | 学習用の風({len(train_idx)}本) | 未知の風({len(test_files)}本)")
    print(f"{'制御なし':24s} | {summarize(train_env, None, train_idx)} | {summarize(test_env, None, range(len(test_files)))}")
    for ckpt in checkpoints():
        policy = PPO.load(ckpt, device="cpu")
        print(f"{ckpt.stem:24s} | {summarize(train_env, policy, train_idx)} | "
              f"{summarize(test_env, policy, range(len(test_files)))}", flush=True)


if __name__ == "__main__":
    main()
