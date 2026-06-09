import argparse
import glob
import os
import random

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent_policy import AgentPolicy
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default


def make_env(configs, rank, seed=0):
    def _init():
        env = LuxEnvironment(
            configs=configs,
            learning_agent=AgentPolicy(mode="train"),
            opponent_agent=Agent()
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def get_command_line_arguments():
    parser = argparse.ArgumentParser(description='Phase 1: PPO fine-tuning with shaped rewards')
    parser.add_argument('--id', help='Run identifier', type=str, default=str(random.randint(0, 10000)))
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.999)
    parser.add_argument('--gae_lambda', help='GAE Lambda', type=float, default=0.95)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2048 * 8)
    parser.add_argument('--step_count', help='Total training steps', type=int, default=3000000)
    parser.add_argument('--n_steps', help='Steps per update', type=int, default=2048 * 8)
    parser.add_argument('--n_envs', help='Number of parallel environments', type=int, default=1)
    parser.add_argument('--bc_weights', help='Path to behavior cloning weights (.pth)', type=str, default=None)
    parser.add_argument('--path', help='Path to a PPO checkpoint to resume training', type=str, default=None)
    return parser.parse_args()


def load_bc_weights(model, bc_weights_path):
    """Load behavior cloning weights into the PPO actor network."""
    import torch
    bc_state_dict = torch.load(bc_weights_path, map_location=model.device)

    policy_state_dict = model.policy.state_dict()
    loaded_keys = []
    for key in bc_state_dict:
        policy_key = key
        if policy_key in policy_state_dict:
            if policy_state_dict[policy_key].shape == bc_state_dict[key].shape:
                policy_state_dict[policy_key] = bc_state_dict[key]
                loaded_keys.append(policy_key)

    model.policy.load_state_dict(policy_state_dict)
    print(f"Loaded {len(loaded_keys)} parameters from behavior cloning weights: {bc_weights_path}")
    print(f"Loaded keys: {loaded_keys}")


def train(args):
    print(args)
    configs = LuxMatchConfigs_Default
    opponent = Agent()

    # Create environment(s)
    env_eval = None
    if args.n_envs == 1:
        env = LuxEnvironment(
            configs=configs,
            learning_agent=AgentPolicy(mode="train"),
            opponent_agent=opponent
        )
    else:
        env = SubprocVecEnv([make_env(configs, i) for i in range(args.n_envs)])

    run_id = args.id
    print(f"Run id: {run_id}")

    # Create or load model
    if args.path:
        model = PPO.load(args.path, env=env)
        model.learning_rate = args.learning_rate
        print(f"Resumed training from checkpoint: {args.path}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./phase1_tensorboard/",
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            ent_coef=0.01,
        )
        print("Created new PPO model")

        # Load behavior cloning weights if provided
        if args.bc_weights:
            load_bc_weights(model, args.bc_weights)

    # Callbacks
    callbacks = []

    player_replay = AgentPolicy(mode="inference", model=model)
    callbacks.append(
        SaveReplayAndModelCallback(
            save_freq=100000,
            save_path='./phase1_models/',
            name_prefix=f'phase1_{run_id}',
            replay_env=LuxEnvironment(
                configs=configs,
                learning_agent=player_replay,
                opponent_agent=Agent()
            ),
            replay_num_episodes=5
        )
    )

    if args.n_envs > 1:
        env_eval = SubprocVecEnv([make_env(configs, i) for i in range(4)])
        callbacks.append(
            EvalCallback(
                env_eval,
                best_model_save_path=f'./phase1_logs_{run_id}/',
                log_path=f'./phase1_logs_{run_id}/',
                eval_freq=args.n_steps * 2,
                n_eval_episodes=30
            )
        )

    # Train
    print(f"Phase 1: Training with shaped rewards for {args.step_count} steps...")
    model.learn(total_timesteps=args.step_count, callback=callbacks)

    # Save final model
    final_path = f'phase1_models/phase1_{run_id}_{args.step_count}_steps'
    model.save(path=final_path)
    print(f"Final model saved to {final_path}.zip")

    # Quick inference test
    print("Running inference test...")
    if args.n_envs == 1:
        test_env = env
    else:
        test_env = LuxEnvironment(
            configs=configs,
            learning_agent=AgentPolicy(mode="inference", model=model),
            opponent_agent=Agent()
        )
    obs, info = test_env.reset()
    for i in range(600):
        action_code, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = test_env.step(action_code)
        if i % 50 == 0:
            print(f"Turn {i}")
        if terminated or truncated:
            print("Episode done, resetting.")
            obs, info = test_env.reset()
    print("Done")


if __name__ == "__main__":
    local_args = get_command_line_arguments()
    train(local_args)