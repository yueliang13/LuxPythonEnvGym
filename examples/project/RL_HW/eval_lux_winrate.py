#!/usr/bin/env python3
# Run repeated LuxAI 2021 matches and report win rates for the two trained agents.
#
# Usage:
#   cd /home/liuxiaoyang/data/RL_HW
#   ./eval_lux_winrate.py [options]
#
# Default agents:
#   G4  = LuxAI-G4/main.py
#   PPO = LuxPythonEnvGym/kaggle_submissions/main_lux-ai-2021.py
#
# Options:
#   --games N          Number of seeds to evaluate. Default: 10.
#                      With --both-sides, total matches are 2 * N.
#   --seed-start S     First seed to use. Seeds are S, S+1, ... Default: 42.
#   --both-sides       For each seed, run both player orders: G4 vs PPO and PPO vs G4.
#                      This is the recommended setting for fairer evaluation.
#   --fixed-order      Without --both-sides, always keep G4 as player 0 and PPO as player 1.
#                      If omitted, player order alternates every game.
#   --store-replay     Save replay JSON files. Default: off.
#   --store-logs       Save agent stderr/error logs. Default: off.
#   --maxtime MS       Per-turn time limit passed to lux-ai-2021. Default: 10000.
#   --loglevel LEVEL   lux-ai-2021 log level. Default: 0 to suppress warning spam.
#   --workers N        Number of matches to run in parallel. Default: 1.
#                      Start with 2 or 4; each worker loads both agents/models.
#   --threads-per-match N
#                      BLAS/PyTorch CPU threads available to each match. Default: 16.
#                      Increase this if one match is too slow; decrease it if workers compete heavily.
#
# Examples:
#   ./eval_lux_winrate.py --games 20
#   ./eval_lux_winrate.py --games 20 --both-sides
#   ./eval_lux_winrate.py --games 50 --seed-start 100 --both-sides
#   ./eval_lux_winrate.py --games 50 --both-sides --workers 4 --threads-per-match 16
#   ./eval_lux_winrate.py --games 5 --store-replay --store-logs --loglevel 2
#
# Output:
#   Each match prints seed, player order, winner, and parsed ranks.
#   The final summary prints G4/PPO wins, ties, failures, win_rate, and
#   score_rate where a tie counts as 0.5.
#
# The script fixes the Python interpreter to the conda lux environment and
# calls the official lux-ai-2021 CLI by absolute path.
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(ROOT, "..", "..", "..", "..", ".venv", "bin", "python3")
_venv_python = os.path.normpath(_venv_python)
PYTHON = _venv_python if os.path.isfile(_venv_python) else sys.executable
LUX_CLI = "lux-ai-2021"
G4 = "LuxAI-G4/main.py"
PPO = "LuxPythonEnvGym/kaggle_submissions/main_lux-ai-2021.py"
RANK_RE = re.compile(r"rank:\s*(\d+),\s*agentID:\s*(\d+),\s*name:\s*'([^']+)'")


def bool_arg(value):
    return "true" if value else "false"


def run_match_task(task):
    index, total, seed, swapped, args = task
    first, second = (PPO, G4) if swapped else (G4, PPO)
    lux_cmd = [
        LUX_CLI,
        f"--seed={seed}",
        f"--maxtime={args.maxtime}",
        f"--python={PYTHON}",
        f"--storeReplay={bool_arg(args.store_replay)}",
        f"--storeLogs={bool_arg(args.store_logs)}",
        f"--loglevel={args.loglevel}",
        first,
        second,
    ]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(args.threads_per_match),
        "MKL_NUM_THREADS": str(args.threads_per_match),
        "OPENBLAS_NUM_THREADS": str(args.threads_per_match),
        "NUMEXPR_NUM_THREADS": str(args.threads_per_match),
        "VECLIB_MAXIMUM_THREADS": str(args.threads_per_match),
    })
    proc = subprocess.run(lux_cmd, cwd=ROOT, env=env, start_new_session=True, text=True, capture_output=True)
    output = proc.stdout + proc.stderr
    ranks = []
    for rank, agent_id, name in RANK_RE.findall(output):
        ranks.append({"rank": int(rank), "agent_id": int(agent_id), "name": name})
    if proc.returncode != 0 or len(ranks) < 2:
        return {
            "ok": False,
            "index": index,
            "total": total,
            "seed": seed,
            "swapped": swapped,
            "returncode": proc.returncode,
            "output": output[-3000:],
        }
    min_rank = min(item["rank"] for item in ranks)
    winners = [item for item in ranks if item["rank"] == min_rank]
    if len(winners) != 1:
        winner = "tie"
    elif winners[0]["name"] == G4:
        winner = "g4"
    elif winners[0]["name"] == PPO:
        winner = "ppo"
    else:
        winner = "unknown"
    return {
        "ok": True,
        "index": index,
        "total": total,
        "seed": seed,
        "swapped": swapped,
        "winner": winner,
        "ranks": ranks,
    }


def main():
    parser = argparse.ArgumentParser(description="Run repeated LuxAI 2021 matches and report win rate.")
    parser.add_argument("--games", type=int, default=10, help="number of seeds to evaluate; with --both-sides this runs 2x matches")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--both-sides", action="store_true", help="run normal and swapped player order for every seed")
    parser.add_argument("--fixed-order", action="store_true", help="do not alternate first/second player when --both-sides is off")
    parser.add_argument("--store-replay", action="store_true")
    parser.add_argument("--store-logs", action="store_true")
    parser.add_argument("--maxtime", type=int, default=10000)
    parser.add_argument("--loglevel", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="number of matches to run in parallel")
    parser.add_argument("--threads-per-match", type=int, default=16, help="BLAS/PyTorch CPU threads available to each match")
    args = parser.parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.threads_per_match <= 0:
        raise SystemExit("--threads-per-match must be positive")

    tasks = []
    for i in range(args.games):
        seed = args.seed_start + i
        if args.both_sides:
            tasks.append((seed, False))
            tasks.append((seed, True))
        else:
            swapped = False if args.fixed_order else (i % 2 == 1)
            tasks.append((seed, swapped))

    counts = {"g4": 0, "ppo": 0, "tie": 0, "failed": 0}

    def handle_result(result):
        side = "swap" if result["swapped"] else "normal"
        prefix = f"[{result['index']:03d}/{result['total']:03d} seed={result['seed']} {side}]"
        if not result["ok"]:
            counts["failed"] += 1
            print(f"{prefix} FAILED returncode={result['returncode']}", flush=True)
            print(result["output"], file=sys.stderr)
            return
        winner = result["winner"]
        counts[winner] = counts.get(winner, 0) + 1
        rank_text = ", ".join(f"{item['name']}=rank{item['rank']}" for item in result["ranks"])
        print(f"{prefix} winner={winner} | {rank_text}", flush=True)

    task_args = [(index, len(tasks), seed, swapped, args) for index, (seed, swapped) in enumerate(tasks, 1)]
    if args.workers == 1:
        for task in task_args:
            handle_result(run_match_task(task))
    else:
        max_workers = min(args.workers, len(task_args))
        print(f"Running {len(task_args)} matches with workers={max_workers}, threads_per_match={args.threads_per_match}", flush=True)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_match_task, task) for task in task_args]
            for future in as_completed(futures):
                handle_result(future.result())

    scored = counts["g4"] + counts["ppo"] + counts["tie"]
    print("\nSummary")
    print(f"matches={len(tasks)} scored={scored} failed={counts['failed']}")
    print(f"G4 wins={counts['g4']} PPO wins={counts['ppo']} ties={counts['tie']}")
    if scored:
        g4_win_rate = counts["g4"] / scored
        ppo_win_rate = counts["ppo"] / scored
        g4_score_rate = (counts["g4"] + 0.5 * counts["tie"]) / scored
        print(f"G4 win_rate={g4_win_rate:.2%} score_rate(tie=0.5)={g4_score_rate:.2%}")
        print(f"PPO win_rate={ppo_win_rate:.2%}")


if __name__ == "__main__":
    main()
