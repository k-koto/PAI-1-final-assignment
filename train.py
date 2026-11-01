"""PPOでバランス制御を学習する。学習の途中経過は checkpoints/ に保存される。

    uv run python train.py --steps 500000
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from yacht import wind
from yacht.env import YachtBalanceEnv

CKPT_DIR = Path("checkpoints")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5_000, help="途中経過を保存する間隔[ステップ]")
    args = parser.parse_args()

    if not (wind.DATA_DIR / "train").exists():
        wind.make_dataset()

    env = make_vec_env(YachtBalanceEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=256, gamma=0.98, learning_rate=3e-4,
                policy_kwargs=dict(net_arch=[64, 64]), seed=args.seed, verbose=1,
                tensorboard_log=None)
    CKPT_DIR.mkdir(exist_ok=True)
    model.save(CKPT_DIR / "ppo_0_steps")  # 学習前(ランダムな初期状態)
    model.learn(total_timesteps=args.steps,
                callback=CheckpointCallback(save_freq=max(args.save_every // args.envs, 1), save_path=str(CKPT_DIR),
                                            name_prefix="ppo"))
    model.save(CKPT_DIR / "ppo_final")


if __name__ == "__main__":
    main()
