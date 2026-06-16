#!/usr/bin/env bash
set -euo pipefail

BENCH_BIN="${BENCH_BIN:-./build/bin/llama-bench}"
OUT_DIR="${OUT_DIR:-bench-results}"

NGL="${NGL:-999}"
P="${P:-512}"
N="${N:-128}"
R="${R:-5}"
MMP="${MMP:-1}"

# Default model list for a 16 GB VRAM motivation experiment.
# To override, edit this array directly.
MODELS=(
  "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
  "bartowski/Qwen2.5-14B-Instruct-1M-GGUF:Q4_K_M"
  "bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q4_K_M"
  "bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q5_K_M"
  "Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M"
)

mkdir -p "$OUT_DIR"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "ERROR: llama-bench not found or not executable: $BENCH_BIN" >&2
  exit 1
fi

sanitize_name() {
  local s="$1"
  s="${s//\//__}"
  s="${s//:/-}"
  s="${s// /_}"
  echo "$s"
}

run_case() {
  local model="$1"
  local model_dir="$2"
  local variant="$3"
  shift 3

  local jsonl="$model_dir/${variant}.jsonl"
  local log="$model_dir/${variant}.log"
  local status_file="$model_dir/status.tsv"

  local common_args=(
    -hf "$model"
    -ngl "$NGL"
    -p "$P"
    -n "$N"
    -r "$R"
    -mmp "$MMP"
    --no-warmup
    -o jsonl
  )

  echo
  echo "============================================================"
  echo "Model:   $model"
  echo "Variant: $variant"
  echo "Output:  $jsonl"
  echo "Log:     $log"
  echo "============================================================"

  set +e

  env \
    -u GGML_CUDA_ENABLE_UNIFIED_MEMORY \
    -u GGML_CUDA_MANAGED_PREFETCH \
    -u GGML_CUDA_MANAGED_ADVISE \
    "$@" \
    "$BENCH_BIN" "${common_args[@]}" \
    > "$jsonl" \
    2> "$log"

  local exit_code=$?

  set -e

  if [[ "$exit_code" -eq 0 ]]; then
    echo -e "${variant}\tOK\t${jsonl}\t${log}" >> "$status_file"
    echo "Done: $variant"
  else
    echo -e "${variant}\tFAILED_EXIT_${exit_code}\t${jsonl}\t${log}" >> "$status_file"
    echo "FAILED: $variant with exit code $exit_code"
    echo "Check log: $log"
  fi
}

run_model() {
  local model="$1"

  if [[ -z "$model" ]]; then
    echo "ERROR: empty model string" >&2
    exit 1
  fi

  local model_name
  model_name="$(sanitize_name "$model")"

  local model_dir="$OUT_DIR/$model_name"
  mkdir -p "$model_dir"

  local status_file="$model_dir/status.tsv"

  {
    echo "model=$model"
    echo "bench_bin=$BENCH_BIN"
    echo "ngl=$NGL"
    echo "p=$P"
    echo "n=$N"
    echo "r=$R"
    echo "mmp=$MMP"
  } > "$model_dir/config.txt"

  echo -e "variant\tstatus\tjsonl\tlog" > "$status_file"

  run_case "$model" "$model_dir" "default"

  run_case "$model" "$model_dir" "managed" \
    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1


  run_case "$model" "$model_dir" "prefetch" \
    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    GGML_CUDA_MANAGED_PREFETCH=1

  run_case "$model" "$model_dir" "all" \
    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    GGML_CUDA_MANAGED_ADVISE=1 \
    GGML_CUDA_MANAGED_PREFETCH=1
}

echo
echo "Benchmark configuration"
echo "BENCH_BIN: $BENCH_BIN"
echo "OUT_DIR:   $OUT_DIR"
echo "NGL:       $NGL"
echo "P:         $P"
echo "N:         $N"
echo "R:         $R"
echo "MMP:       $MMP"
echo
echo "Models:"
for model in "${MODELS[@]}"; do
  echo "  - $model"
done

for model in "${MODELS[@]}"; do
  run_model "$model"
done

echo
echo "All benchmark attempts finished."
echo "Results are in: $OUT_DIR"
echo
echo "Status files:"
find "$OUT_DIR" -name status.tsv -print