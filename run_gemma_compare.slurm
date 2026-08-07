#!/bin/bash
#SBATCH --job-name=rag_gemma_compare
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|l40s"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/comparison/rag_gemma_compare_%j.out
#SBATCH --error=logs/comparison/rag_gemma_compare_%j.err

set -e
mkdir -p logs

WORK_DIR="/work/pi_larsonj_wit_edu/superfund/project/superfund/run_rag_07302026"
cd "$WORK_DIR"

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "$WORK_DIR/.venv/bin/activate"

GEM3="gemma3-27b-it"
GEM4="gemma4-31B-it"

DATE=$(date +%Y%m%d_%H%M)
DOCLING_ON="output/DOCLING_ON"
DOCLING_OFF="output/DOCLING_OFF"
SCRIPT="/work/pi_larsonj_wit_edu/superfund/project/superfund/run_rag_07302026/rag_run_july_30.py"

mkdir -p output

# --------------------------------------------------------------------------------------------------------------------------------

echo "---------------------------"
echo "--- Testing: Docling On ---"
echo "---------------------------"

uv run $SCRIPT \
    --pdfs_filter Wisconsin_Tomah \
    --llm_models $GEM3 $GEM4 \
    --embed_model qwen3-embed-8b \
    --questions_file questions.txt \
    --out_dir "${WORK_DIR}/${DATE}/${DOCLING_ON}/" \
    --use_docling \

# --------------------------------------------------------------------------------------------------------------------------------

echo "----------------------------"
echo "--- Testing: Docling Off ---"
echo "----------------------------"

uv run $SCRIPT \
    --pdfs_filter Wisconsin_Tomah \
    --llm_models $GEM3 $GEM4 \
    --embed_model qwen3-embed-8b \
    --questions_file questions.txt \
    --out_dir "${WORK_DIR}/${DATE}/${DOCLING_OFF}/" \

# --------------------------------------------------------------------------------------------------------------------------------

echo "Results saved to: ${WORK_DIR}/output"

echo "Job finished: $(date)"
