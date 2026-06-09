#!/usr/bin/env bash
# Run one LuxAI 2021 match between the two trained agents.
#
# Usage:
#   cd /home/liuxiaoyang/data/RL_HW
#   ./run_lux_match.sh [--seed SEED] [--swap] [lux-ai-2021 options...]
#
# Default agents/order:
#   Player 0: LuxAI-G4/main.py
#   Player 1: LuxPythonEnvGym/kaggle_submissions/main_lux-ai-2021.py
#
# Script options:
#   --seed SEED   Set the LuxAI random seed. Default: 42.
#   --swap        Swap player order, so PPO is player 0 and G4 is player 1.
#
# Extra arguments are passed directly to lux-ai-2021. Common examples:
#   --storeReplay=false   Do not save replay JSON files.
#   --storeReplay=true    Save replay JSON files for visualization.
#   --storeLogs=false     Do not save stderr/error logs.
#   --storeLogs=true      Save agent stderr/error logs under errorlogs/.
#   --loglevel=0          Suppress engine logs; useful for quick checks.
#   --loglevel=2          Show warnings such as collisions/invalid moves.
#   --out PATH            Choose replay output path when replay saving is on.
#
# Examples:
#   ./run_lux_match.sh --seed 42 --storeReplay=false --storeLogs=true
#   ./run_lux_match.sh --seed 42 --swap --storeReplay=true --storeLogs=true
#
# The script fixes the Python interpreter to the conda lux environment and
# loads nvm so the official lux-ai-2021 CLI is available.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""  # Disable GPU usage for both agents

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR"
_venv_python="$ROOT/../../../../.venv/bin/python3"
if [ -f "$_venv_python" ]; then
    PYTHON="$(cd "$(dirname "$_venv_python")" && pwd)/$(basename "$_venv_python")"
else
    PYTHON="$(which python3)"
fi
SEED=42
SWAP=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --swap)
      SWAP=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

source /home/liuxiaoyang/.nvm/nvm.sh >/dev/null 2>&1
cd "$ROOT"

G4=LuxAI-G4/main.py
PPO=LuxPythonEnvGym/kaggle_submissions/main_lux-ai-2021.py
if [[ "$SWAP" == 1 ]]; then
  A="$PPO"
  B="$G4"
else
  A="$G4"
  B="$PPO"
fi

lux-ai-2021   --seed="$SEED"   --maxtime=10000   --python="$PYTHON"   "${EXTRA_ARGS[@]}"   "$A" "$B"
