"""
Evaluate a trained model against the default (do-nothing) opponent.
Usage:
    python evaluate.py --model model.zip --games 10
    python evaluate.py --model ../project/model.zip --games 20
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_policy import AgentPolicy
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.env.agent import Agent
from luxai2021.game.constants import LuxMatchConfigs_Default
from stable_baselines3 import PPO


def evaluate(model_path, num_games, seed_start=0):
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Must use mode="train" for LuxEnvironment to work correctly.
    # LuxEnvironment.step() needs the agent to yield control via the generator,
    # which only happens for LEARNING type agents (mode="train").
    # The model still uses model.predict() for action selection via env.step().
    player = AgentPolicy(mode="train", model=model)
    opponent = Agent()
    env = LuxEnvironment(
        configs=LuxMatchConfigs_Default,
        learning_agent=player,
        opponent_agent=opponent
    )

    wins = 0
    total_city_tiles = 0
    total_opponent_city_tiles = 0

    for i in range(num_games):
        obs, info = env.reset(seed=seed_start + i)
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        my_city_tiles = 0
        opp_city_tiles = 0
        for city in env.game.cities.values():
            if city.team == player.team:
                my_city_tiles += len(city.city_cells)
            else:
                opp_city_tiles += len(city.city_cells)

        won = env.game.get_winning_team() == player.team
        if won:
            wins += 1

        total_city_tiles += my_city_tiles
        total_opponent_city_tiles += opp_city_tiles

        print(f"Game {i+1}/{num_games}: "
              f"{'WIN' if won else 'LOSS'}, "
              f"my_cities={my_city_tiles}, opp_cities={opp_city_tiles}, "
              f"turns={env.game.state['turn']}")

    print(f"\n{'='*50}")
    print(f"Results over {num_games} games:")
    print(f"  Win rate:     {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"  Avg my cities:    {total_city_tiles/num_games:.1f}")
    print(f"  Avg opp cities:   {total_opponent_city_tiles/num_games:.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Lux AI model")
    parser.add_argument("--model", type=str, default="model.zip", help="Path to model.zip")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--seed", type=int, default=0, help="Random seed offset")
    args = parser.parse_args()
    evaluate(args.model, args.games, args.seed)