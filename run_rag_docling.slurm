#!/bin/bash
#SBATCH --job-name=rag_desrochesl
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|l40s"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rag_desrochesl_%j.out
#SBATCH --error=logs/rag_desrochesl_%j.err

set -e
mkdir -p logs

WORK_DIR="/work/pi_larsonj_wit_edu/superfund/project/superfund/run_rag_07302026"
cd "$WORK_DIR"

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "$WORK_DIR/.venv/bin/activate"

MODEL="gemma3-27b"
DATE=$(date +%Y%m%d_%H%M)
OUTPUT="output/rag_docling_${MODEL}_${DATE}.csv"
SCRIPT="/work/pi_larsonj_wit_edu/superfund/project/superfund/run_rag_07302026/rag_run_july_30.py"

mkdir -p output

# uv sync -q

echo "Starting ($SCRIPT)"

uv run $SCRIPT \
    --pdfs_filter Wisconsin_Tomah Wisconsin_Unity "Wisconsin_Waste" Wisconsin_Wausau Wyoming \
    --llm_models qwen3-4b \
    --embed_model qwen3-embed-8b \
    --questions_file questions.txt \
    --out_dir /work/pi_larsonj_wit_edu/superfund/project/superfund/run_rag_07302026/output \
    --use_docling \


echo "Results saved to: $OUTPUT"
echo "Job finished: $(date)"
