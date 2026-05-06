#!/usr/bin/env bash
# Run the 6 SOTA model evaluations on benchmark_v2_full.jsonl with controlled parallelism.
#
# Usage:
#   ./scripts/run_milestone_eval.sh                  # 3-way parallel (default)
#   PARALLELISM=2 ./scripts/run_milestone_eval.sh    # 2-way
#   PARALLELISM=6 ./scripts/run_milestone_eval.sh    # all-at-once (Docker-heavy)
#
# Each model writes its own log file under logs/milestone_eval_<timestamp>/.
# Tail any log with:  tail -f logs/milestone_eval_*/<model>.log

set -uo pipefail
cd "$(dirname "$0")/.."

# ── Config ────────────────────────────────────────────────────────────────────
MODELS=(
    "gpt-5.4"
    "claude-sonnet"
    "claude-opus"
    "gemini-3.1-pro"
    "deepseek-v4-pro"
    "qwen-3.6-plus"
)

BENCHMARK="data/benchmark/benchmark_v2_full.jsonl"
N_SAMPLES=5
TEMPERATURE=0.8
PARALLELISM="${PARALLELISM:-3}"          # max concurrent pipelines
PYTHON="${PYTHON:-/Users/ender_yang/opt/anaconda3/bin/python3}"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
[[ -f "$BENCHMARK" ]] || { echo "ERROR: benchmark not found at $BENCHMARK" >&2; exit 1; }
command -v "$PYTHON" >/dev/null || { echo "ERROR: python not found at $PYTHON" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: docker is not running" >&2; exit 1; }

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/milestone_eval_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# ── Banner ────────────────────────────────────────────────────────────────────
cat <<EOF
================================================================
  Milestone evaluation
  Benchmark : $BENCHMARK ($(wc -l < "$BENCHMARK" | tr -d ' ') items)
  Models    : ${#MODELS[@]} (${MODELS[*]})
  Samples   : n=$N_SAMPLES, T=$TEMPERATURE
  Parallel  : $PARALLELISM
  Logs      : $LOG_DIR/
  Started   : $(date)
================================================================
EOF

# ── Cleanup on Ctrl-C ─────────────────────────────────────────────────────────
# Each background job runs in its own process group (set -m enables job control,
# which makes & put each job in a new pgid). On Ctrl-C we send SIGTERM to the
# entire process group of every background job — this propagates to the python
# subprocesses (run_full_pipeline.py → run_baseline_eval.py → etc.) so no orphans.
set -m

cleanup() {
    echo
    echo "Interrupted — killing background job groups..."
    for pid in $(jobs -p); do
        # Negative PID = process group. Try SIGTERM first, then SIGKILL after 2s.
        kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in $(jobs -p); do
        kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    done
    # Defensive sweep in case anything escaped
    pkill -9 -f "run_full_pipeline.py"     2>/dev/null || true
    pkill -9 -f "run_baseline_eval.py"     2>/dev/null || true
    pkill -9 -f "run_perturbed_eval.py"    2>/dev/null || true
    pkill -9 -f "run_classification.py"    2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

# ── Run with controlled parallelism ───────────────────────────────────────────
run_one() {
    local model="$1"
    local safe="${model//\//_}"
    local log="$LOG_DIR/${safe}.log"

    if "$PYTHON" scripts/run_full_pipeline.py \
        --model "$model" \
        --benchmark "$BENCHMARK" \
        --n-samples "$N_SAMPLES" \
        --temperature "$TEMPERATURE" \
        > "$log" 2>&1
    then
        echo "[$(date +%H:%M:%S)] DONE   $model  ✓"
    else
        echo "[$(date +%H:%M:%S)] FAIL   $model  ✗  → $log"
    fi
}

for model in "${MODELS[@]}"; do
    # Block until we have a free slot
    while (( $(jobs -rp | wc -l) >= PARALLELISM )); do
        sleep 5
    done

    echo "[$(date +%H:%M:%S)] START  $model"
    run_one "$model" &
done

# Wait for the last batch
wait

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "  All evaluations finished at $(date)"
echo "================================================================"

DONE=0; FAIL=0
for model in "${MODELS[@]}"; do
    safe="${model//\//_}"
    log="$LOG_DIR/${safe}.log"
    if grep -q "Pipeline complete in" "$log" 2>/dev/null; then
        DONE=$((DONE + 1))
        elapsed=$(grep "Pipeline complete in" "$log" | tail -1 | sed 's/.*complete in //')
        echo "  ✓ $model  ($elapsed)"
    else
        FAIL=$((FAIL + 1))
        echo "  ✗ $model  → $log"
    fi
done
echo
echo "  ${DONE}/${#MODELS[@]} succeeded, ${FAIL}/${#MODELS[@]} failed"
echo "  Result files in data/results/  (baseline_/perturbed_/classified_<model>_${TIMESTAMP%_*}*.jsonl)"
echo "================================================================"