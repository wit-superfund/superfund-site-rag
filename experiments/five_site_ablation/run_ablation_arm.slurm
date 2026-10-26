#!/bin/bash
#SBATCH --job-name=rag_matrix
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram102|vram143
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-4
#SBATCH --output=/work/pi_larsonj_wit_edu/superfund/project/superfund/logs/%x-%A_%a.out
#SBATCH --error=/work/pi_larsonj_wit_edu/superfund/project/superfund/logs/%x-%A_%a.err

# Run one frozen arm over the five mining sites. Required: ARM=<matrix ID>.
# SMOKE=1 selects one question, three pages per document, and one source.
# SMOKE_SITE_INDEX defaults to Silver Bow (zero-based array row 1) when SMOKE=1.
set -euo pipefail

ROOT="${SUPERFUND_ROOT:-/work/pi_larsonj_wit_edu/superfund/project/superfund}"
STAGE="${STAGE_ROOT:-$ROOT/staging/rag_matrix_v3_20261025}"
MATRIX="$STAGE/ablation_matrix.tsv"
FIVE="$ROOT/five_sites"
SITE_FILE="$ROOT/five_site_benchmark.tsv"
QUESTIONS="${QUESTIONS_FILE:-$STAGE/questions_mining.tsv}"
CONDENSED="$ROOT/npl-state-scraper/condensing mining 10-15-26/multisite"
CACHE_ROOT="${CACHE_ROOT:-/scratch3/workspace/${USER}-superfund/rag_matrix_v2_cache/v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/output/rag_matrix_v3_pdf_fallback}"
ARM="${ARM:?Set ARM to an ID in ablation_matrix.tsv}"
CONDENSER="${CONDENSER:-pymupdf}"
LLM_MODEL="${LLM_MODEL:-gemma4-27b}"
CHUNK_SIZE="${CHUNK_SIZE:-2400}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-300}"
BM25_K1="${BM25_K1:-1.2}"
BM25_B="${BM25_B:-0.75}"
BM25_TOKENIZER="${BM25_TOKENIZER:-unicode-alnum-2-64-v1}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-$ROOT/models/qwen3-embedding-8b-1d8ad4ca-complete}"

ROW=$(awk -F '\t' -v arm="$ARM" 'NR > 1 && $1 == arm {print; exit}' "$MATRIX")
if [ -z "$ROW" ]; then
    echo "Unknown ARM=$ARM" >&2
    exit 2
fi
IFS=$'\t' read -r ARM_ID SERIES STATUS MAX_PAGES EMBED_MODEL QUERY_MODE RETRIEVAL_MODE VERIFIER_MODE CANDIDATE_K TOP_K RRF_K REUSE_FROM DESCRIPTION <<< "$ROW"
if [ "$STATUS" = "blocked" ]; then
    echo "ARM=$ARM is blocked: $DESCRIPTION" >&2
    exit 2
fi
if [ "$STATUS" = "reuse" ]; then
    echo "ARM=$ARM reuses $REUSE_FROM and must not be recomputed: $DESCRIPTION" >&2
    exit 2
fi
EMBED_MODEL_PATH=""
if [ "$EMBED_MODEL" = "qwen3-embed-8b" ]; then
    EMBED_MODEL_PATH="$QWEN_MODEL_PATH"
fi
QUERY_INSTRUCTION_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
if [ "$QUERY_MODE" = "superfund" ]; then
    QUERY_INSTRUCTION_SHA256=fc6e7e118938eafb71009851781003cbab03129eee13cf942616ec1265d3a340
fi

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SOURCES=(pdf condensed)
if [ "${SMOKE:-0}" = 1 ]; then
    EXPECTED_INDEX="${SMOKE_SITE_INDEX:-1}"
    if [ "$TASK_INDEX" != "$EXPECTED_INDEX" ]; then
        echo "SMOKE=1 runs only array index $EXPECTED_INDEX; task $TASK_INDEX exits cleanly"
        exit 0
    fi
    MAX_PAGES=3
    SOURCES=("${SMOKE_SOURCE:-condensed}")
    QUESTIONS="${SMOKE_QUESTIONS_FILE:-$STAGE/questions_smoke_one.txt}"
fi

IFS=$'\t' read -r SITE FOLDER NAME < <(sed -n "$((TASK_INDEX + 1))p" "$SITE_FILE")
if [ -z "${SITE:-}" ] || [ ! -d "$FIVE/$FOLDER" ]; then
    echo "Invalid site row or missing PDF directory for task $TASK_INDEX" >&2
    exit 2
fi

RUN_TAG="$ARM"
if [ "${SMOKE:-0}" = 1 ]; then
    RUN_TAG="smoke_${ARM}"
fi
OUT="$OUTPUT_ROOT/$RUN_TAG"
mkdir -p "$ROOT/logs" "$CACHE_ROOT" "$OUT/manifests" "$OUT/resources"

module load python/3.12.3 2>/dev/null || true
source "$HOME/.venv_rag/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MANIFEST="$OUT/manifests/${SITE}.tsv"
CODE_SHA=$(sha256sum "$STAGE/compare_rag.py" | awk '{print $1}')
CACHE_CODE_SHA=$(sha256sum "$STAGE/rag_cache.py" | awk '{print $1}')
PIPELINE_BUNDLE_SHA=$(sha256sum "$STAGE/compare_rag.py" "$STAGE/rag_cache.py" "$STAGE/embedding_modes.py" "$STAGE/bm25_rrf.py" "$STAGE/retrieval_modes.py" "$STAGE/citation_eval.py" "$STAGE/question_io.py" "$STAGE/pdf_extract.py" | sha256sum | awk '{print $1}')
MATRIX_SHA=$(sha256sum "$MATRIX" | awk '{print $1}')
QUESTION_SHA=$(sha256sum "$QUESTIONS" | awk '{print $1}')
SITE_FILE_SHA=$(sha256sum "$SITE_FILE" | awk '{print $1}')
ARM_CONFIG_SHA=$(printf '%s\n' "$ROW" "$MAX_PAGES" "${SOURCES[*]}" "$LLM_MODEL" "$CONDENSER" "$CHUNK_SIZE" "$CHUNK_OVERLAP" "$BM25_K1" "$BM25_B" "$EMBED_MODEL_PATH" "$PIPELINE_BUNDLE_SHA" "$MATRIX_SHA" "$QUESTION_SHA" "$SITE_FILE_SHA" | sha256sum | awk '{print $1}')
if [ -e "$MANIFEST" ]; then
    EXISTING_CONFIG_SHA=$(awk -F '\t' '$1 == "arm_config_sha" {print $2; exit}' "$MANIFEST")
    if [ "$EXISTING_CONFIG_SHA" != "$ARM_CONFIG_SHA" ]; then
        echo "Refusing to mix configurations in RUN_TAG=$RUN_TAG for site $SITE" >&2
        exit 2
    fi
fi

MANIFEST_TMP="${MANIFEST}.tmp-${SLURM_JOB_ID:-$$}"
{
    printf 'field\tvalue\n'
    printf 'arm_config_sha\t%s\n' "$ARM_CONFIG_SHA"
    printf 'arm\t%s\n' "$ARM"
    printf 'series\t%s\n' "$SERIES"
    printf 'description\t%s\n' "$DESCRIPTION"
    printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'slurm_job_id\t%s\n' "${SLURM_JOB_ID:-}"
    printf 'site_id\t%s\n' "$SITE"
    printf 'site_key\t%s\n' "$FOLDER"
    printf 'site_name\t%s\n' "$NAME"
    printf 'sources\t%s\n' "${SOURCES[*]}"
    printf 'max_pages\t%s\n' "$MAX_PAGES"
    printf 'embed_model\t%s\n' "$EMBED_MODEL"
    printf 'embed_model_path_override\t%s\n' "$EMBED_MODEL_PATH"
    printf 'query_instruction_mode\t%s\n' "$QUERY_MODE"
    printf 'query_instruction_sha256\t%s\n' "$QUERY_INSTRUCTION_SHA256"
    printf 'retrieval_mode\t%s\n' "$RETRIEVAL_MODE"
    printf 'verifier_mode\t%s\n' "$VERIFIER_MODE"
    printf 'candidate_k\t%s\n' "$CANDIDATE_K"
    printf 'top_k\t%s\n' "$TOP_K"
    printf 'rrf_k\t%s\n' "$RRF_K"
    printf 'llm_model\t%s\n' "$LLM_MODEL"
    printf 'condenser\t%s\n' "$CONDENSER"
    printf 'chunk_size\t%s\n' "$CHUNK_SIZE"
    printf 'chunk_overlap\t%s\n' "$CHUNK_OVERLAP"
    printf 'bm25_k1\t%s\n' "$BM25_K1"
    printf 'bm25_b\t%s\n' "$BM25_B"
    printf 'bm25_tokenizer\t%s\n' "$BM25_TOKENIZER"
    printf 'cache_root\t%s\n' "$CACHE_ROOT"
    printf 'compare_rag_sha256\t%s\n' "$CODE_SHA"
    printf 'rag_cache_sha256\t%s\n' "$CACHE_CODE_SHA"
    printf 'pipeline_bundle_sha256\t%s\n' "$PIPELINE_BUNDLE_SHA"
    printf 'matrix_sha256\t%s\n' "$MATRIX_SHA"
    printf 'questions_sha256\t%s\n' "$QUESTION_SHA"
    printf 'site_file_sha256\t%s\n' "$SITE_FILE_SHA"
} > "$MANIFEST_TMP"
mv "$MANIFEST_TMP" "$MANIFEST"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-} task=$TASK_INDEX arm=$ARM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
RESOURCE_FILE="$OUT/resources/${SITE}.time.txt"
EMBED_PATH_ARGS=()
if [ -n "$EMBED_MODEL_PATH" ]; then
    EMBED_PATH_ARGS=(--embed_model_path "$EMBED_MODEL_PATH")
fi
/usr/bin/time -v -o "$RESOURCE_FILE" python "$STAGE/compare_rag.py" \
    --site "$SITE" \
    --site_name "$NAME" \
    --sources "${SOURCES[@]}" \
    --pdf_dir "$FIVE/$FOLDER" \
    --condensed_root "$CONDENSED" \
    --condenser "$CONDENSER" \
    --questions_file "$QUESTIONS" \
    --embed_model "$EMBED_MODEL" \
    "${EMBED_PATH_ARGS[@]}" \
    --query_instruction_mode "$QUERY_MODE" \
    --llm_model "$LLM_MODEL" \
    --max_pages "$MAX_PAGES" \
    --chunk_size "$CHUNK_SIZE" \
    --chunk_overlap "$CHUNK_OVERLAP" \
    --retrieval_mode "$RETRIEVAL_MODE" \
    --candidate_k "$CANDIDATE_K" \
    --top_k "$TOP_K" \
    --bm25_k1 "$BM25_K1" \
    --bm25_b "$BM25_B" \
    --bm25_tokenizer "$BM25_TOKENIZER" \
    --rrf_k "$RRF_K" \
    --verifier_mode "$VERIFIER_MODE" \
    --cache_dir "$CACHE_ROOT" \
    --output_dir "$OUT"

printf 'finished_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
echo "finished $(date)"
