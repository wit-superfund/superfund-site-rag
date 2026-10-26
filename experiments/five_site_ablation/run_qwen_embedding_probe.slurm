#!/bin/bash
#SBATCH --job-name=qwen_vec_probe
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40|vram48|vram80|vram102|vram143
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/pi_larsonj_wit_edu/superfund/project/superfund/logs/%x-%j.out
#SBATCH --error=/work/pi_larsonj_wit_edu/superfund/project/superfund/logs/%x-%j.err

set -euo pipefail

ROOT="${SUPERFUND_ROOT:-/work/pi_larsonj_wit_edu/superfund/project/superfund}"
STAGE="${STAGE_ROOT:-$ROOT/staging/rag_matrix_v3_20261025}"

module load python/3.12.3 2>/dev/null || true
source "$HOME/.venv_rag/bin/activate"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$STAGE"
python qwen_embedding_probe.py "$@"
