#!/usr/bin/env bash
# Run the official LuxAI 2021 tournament mode for the two trained agents.
#
# Usage:
#   cd /home/liuxiaoyang/data/RL_HW
#   ./run_lux_tournament.sh [options] [lux-ai-2021 tournament options...]
#
# Default agents:
#   LuxAI-G4/main.py
#   LuxPythonEnvGym/kaggle_submissions/main_lux-ai-2021.py
#
# What this script does:
#   Calls the official lux-ai-2021 CLI with --tournament.
#   This is different from eval_lux_winrate.py:
#     - tournament mode is the official built-in ranking mode.
#     - it is good for quick official-style ranking checks.
#     - it does not expose the same clear seed range, fixed number of games,
#       or both-sides control that eval_lux_winrate.py provides.
#
# Script options:
#   --rank-system NAME       Ranking system passed to --rankSystem.
#                            Allowed by official CLI: trueskill, elo, wins.
#                            Default: wins.
#   --max-concurrent N       Max concurrent tournament matches.
#                            Passed to --maxConcurrentMatches. Default: 1.
#   --store-replay true|false
#                            Save tournament replay JSON files. Default: false.
#   --store-logs true|false  Save agent stderr/error logs. Default: true.
#   --loglevel LEVEL         Tournament log level. Default: 2.
#   --maxtime MS             Per-turn time limit. Default: 10000.
#   --memory MB              Bot memory limit. Default: 4000.
#
# Extra arguments are passed directly to lux-ai-2021 after the defaults above.
# Useful official tournament options include:
#   --rankSystem trueskill|elo|wins
#   --maxConcurrentMatches N
#   --storeReplay true|false
#   --storeLogs true|false
#   --loglevel 0|1|2|3|4
#
# Examples:
#   ./run_lux_tournament.sh
#   ./run_lux_tournament.sh --rank-system wins --max-concurrent 4
#   ./run_lux_tournament.sh --rank-system trueskill --store-replay true --store-logs true
#   ./run_lux_tournament.sh --loglevel 0 --store-replay false --store-logs false
#
# Notes:
#   The official help says --seed and --out are ignored in --tournament mode.
#   For controlled seed sweeps and win-rate tables, use eval_lux_winrate.py.
#
# The script fixes the Python interpreter to the conda lux environment and
# loads nvm so the official lux-ai-2021 CLI is available.
set -euo pipefail

ROOT=/home/liuxiaoyang/data/RL_HW
PYTHON=/data/user/liuxiaoyang/.conda/envs/lux/bin/python
RANK_SYSTEM=wins
MAX_CONCURRENT=1
STORE_REPLAY=false
STORE_LOGS=true
LOGLEVEL=2
MAXTIME=10000
MEMORY=4000
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rank-system)
      RANK_SYSTEM="$2"
      shift 2
      ;;
    --max-concurrent)
      MAX_CONCURRENT="$2"
      shift 2
      ;;
    --store-replay)
      STORE_REPLAY="$2"
      shift 2
      ;;
    --store-logs)
      STORE_LOGS="$2"
      shift 2
      ;;
    --loglevel)
      LOGLEVEL="$2"
      shift 2
      ;;
    --maxtime)
      MAXTIME="$2"
      shift 2
      ;;
    --memory)
      MEMORY="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

source /home/liuxiaoyang/.nvm/nvm.sh >/dev/null 2>&1
cd "$ROOT"

lux-ai-2021 \
  --tournament \
  --rankSystem="$RANK_SYSTEM" \
  --maxConcurrentMatches="$MAX_CONCURRENT" \
  --storeReplay="$STORE_REPLAY" \
  --storeLogs="$STORE_LOGS" \
  --loglevel="$LOGLEVEL" \
  --maxtime="$MAXTIME" \
  --memory="$MEMORY" \
  --python="$PYTHON" \
  "${EXTRA_ARGS[@]}" \
  LuxAI-G4/main.py \
  LuxPythonEnvGym/kaggle_submissions/main_lux-ai-2021.py
