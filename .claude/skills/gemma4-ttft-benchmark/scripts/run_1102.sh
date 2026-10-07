#!/bin/bash
# Gemma-4 26B-A4B prefill TTFT, spyre-inference#1102: hf-adapters (#620) vs
# `vllm bench latency`. Runs the maintainer's two benchmark commands on an environment that
# already has everything installed. It never installs, syncs, rebuilds or checks anything
# out: it checks the environment, reports what deviates from the recipe, and stops for
# anything the user has not accepted.
#   https://github.com/torch-spyre/spyre-inference/issues/1102#issuecomment-6001115830
#
# Usage: run_1102.sh [options]
#   --python PATH       interpreter of the environment to benchmark (default:
#                       $VIRTUAL_ENV/bin/python, else $UV_PROJECT_ENVIRONMENT/bin/python)
#   --hf-python PATH    interpreter for the hf arm only (default: --python)
#   --env-script FILE   source FILE first: the host's Spyre runtime environment, e.g. the
#                       venv's bin/activate (sourcing only sets shell variables)
#   --model PATH        model dir or HF id (default /models/google/gemma-4-26B-A4B)
#   --arms "hf vllm"    arms and their order (default "hf vllm")
#   --repeat N          run the arm list N times, alternating (default 1)
#   --check             check the environment, print findings and the plan, run nothing
#   --accept-warnings   benchmark despite WARN findings (the user has reviewed them)
#   --allow-profiler    downgrade a profiler build from ERROR to WARN
#   --recompiles        TORCH_LOGS=recompiles on the vLLM arm; count post-warmup recompiles
#   --out DIR           output dir (default $HOME/gemma4-ttft-1102/<timestamp>)
# Exit: 0 ok; 1 ERROR findings; 2 usage/runtime problem; 3 WARN findings not accepted.
set -uo pipefail

SKILL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

INPUT_LEN=1984
VLLM_ARGS=(--input-len "$INPUT_LEN" --output-len 1 --batch-size 1 --num-iters-warmup 2
           --num-iters 3 --max-model-len 2048 --max-num-seqs 1)

MODEL=/models/google/gemma-4-26B-A4B
PY=""
HF_PY=""
ENV_SCRIPT=""
ARMS="hf vllm"
REPEAT=1
CHECK_ONLY=0
ACCEPT_WARNINGS=0
ALLOW_PROFILER=0
RECOMPILES=0
OUT=""
ARGS=("$@")

die() { echo "run_1102.sh: $*" >&2; exit "${2:-2}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY=$2; shift 2 ;;
    --hf-python) HF_PY=$2; shift 2 ;;
    --env-script) ENV_SCRIPT=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --arms) ARMS=$2; shift 2 ;;
    --repeat) REPEAT=$2; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    --accept-warnings) ACCEPT_WARNINGS=1; shift ;;
    --allow-profiler) ALLOW_PROFILER=1; shift ;;
    --recompiles) RECOMPILES=1; shift ;;
    --out) OUT=$2; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) die "unknown argument '$1' (try --help)" ;;
  esac
done

if [ -n "$ENV_SCRIPT" ]; then
  [ -r "$ENV_SCRIPT" ] || die "cannot read --env-script $ENV_SCRIPT"
  set +u
  # shellcheck disable=SC1090
  source "$ENV_SCRIPT"
  set -u
fi
# After --env-script, which may be what sets VIRTUAL_ENV.
if [ -z "$PY" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then PY=$VIRTUAL_ENV/bin/python
  elif [ -n "${UV_PROJECT_ENVIRONMENT:-}" ]; then PY=$UV_PROJECT_ENVIRONMENT/bin/python
  else die "no environment: pass --python, or --env-script <venv>/bin/activate"; fi
fi
[ -n "$HF_PY" ] || HF_PY=$PY
for p in "$PY" "$HF_PY"; do [ -x "$p" ] || die "no interpreter at $p"; done
for arm in $ARMS; do
  case "$arm" in hf|vllm) ;; *) die "unknown arm '$arm' (hf, vllm)" ;; esac
done
[ -n "$OUT" ] || OUT=${GEMMA4_TTFT_OUT:-$HOME/gemma4-ttft-1102}/$(date +%Y-%m-%d_%H%M%S)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

# ---------------------------------------------------------------- environment check
# One check per interpreter, from a neutral cwd so a checkout in the cwd cannot shadow
# what the benchmark itself imports.
check() {  # check <label> <python> <arms,comma-separated>
  local extra=()
  [ "$ALLOW_PROFILER" = 0 ] || extra=(--allow-profiler)
  (cd "$STAGE" && "$2" -I -B "$SKILL_DIR/scripts/check_env.py" --out "$STAGE/env_$1.json" \
    --arms "$3" --model "$MODEL" "${extra[@]+"${extra[@]}"}") > "$STAGE/check_$1.log" 2>&1
  [ -s "$STAGE/env_$1.json" ] || { echo "!! $1: check_env.py failed:"; tail -5 "$STAGE/check_$1.log"; return 1; }
}
jq_py() { "$PY" -I -B -c "$1" "${@:2}"; }

if [ "$HF_PY" = "$PY" ]; then
  check env "$PY" "$(echo "$ARMS" | tr ' ' ',')" || die "environment check could not run"
  for arm in $ARMS; do cp "$STAGE/env_env.json" "$STAGE/env_$arm.json"; done
else
  for arm in $ARMS; do
    if [ "$arm" = hf ]; then check hf "$HF_PY" hf || die "environment check could not run"
    else check vllm "$PY" vllm || die "environment check could not run"; fi
  done
fi

echo "=== environment findings (nothing was changed)"
counts=$(jq_py '
import json, sys
seen = {}
for path in sys.argv[1:]:
    for f in json.load(open(path))["findings"]:
        seen[(f["level"], f["message"])] = f
for (level, msg), f in sorted(seen.items(), key=lambda kv: ["ERROR", "WARN", "INFO"].index(kv[0][0])):
    print("  %-5s [%s] %s" % (level, f["arm"], msg))
print("COUNTS", sum(1 for k in seen if k[0] == "ERROR"), sum(1 for k in seen if k[0] == "WARN"))
' $(for arm in $ARMS; do echo "$STAGE/env_$arm.json"; done))
echo "$counts" | grep -v '^COUNTS '
read -r _ N_ERR N_WARN <<< "$(echo "$counts" | grep '^COUNTS ')"
[ "${N_ERR:-0}" = 0 ] && [ "${N_WARN:-0}" = 0 ] && echo "  (none)"

SI_DIR=""
if [[ " $ARMS " == *" vllm "* ]]; then
  SI_DIR=$(jq_py 'import json, sys; print((json.load(open(sys.argv[1])).get("spyre-inference") or {}).get("checkout") or "")' "$STAGE/env_vllm.json")
fi
VLLM_CMD=("$(dirname "$PY")/vllm")
[ -x "${VLLM_CMD[0]}" ] || VLLM_CMD=("$PY" -m vllm.entrypoints.cli.main)
echo "=== plan"
echo "  host   ${HOSTNAME:-$(uname -n)}"
echo "  model  $MODEL  input_len=$INPUT_LEN"
echo "  arms   $ARMS  x$REPEAT"
echo "  hf     OMP_NUM_THREADS=8 $HF_PY ttft.py --model $MODEL --input-len $INPUT_LEN"
echo "  vllm   (cd ${SI_DIR:-<run dir>}) ${VLLM_CMD[*]} bench latency --model $MODEL ${VLLM_ARGS[*]}"
echo "  out    $OUT"

if [ "${N_ERR:-0}" != 0 ]; then
  die "$N_ERR ERROR finding(s): this environment cannot be benchmarked as is; fix it yourself and re-run" 1
fi
if [ "${N_WARN:-0}" != 0 ] && [ "$ACCEPT_WARNINGS" = 0 ]; then
  [ "$CHECK_ONLY" = 1 ] && exit 3
  die "$N_WARN WARN finding(s) deviate from the issue's recipe; review them, then re-run with --accept-warnings" 3
fi
[ "$CHECK_ONLY" = 1 ] && exit 0
card_busy() { [ -e /dev/vfio/vfio ] && fuser /dev/vfio/vfio >/dev/null 2>&1; }
card_busy && die "/dev/vfio/vfio is held by another process; the card serves one process at a time"

# ---------------------------------------------------------------- benchmark
mkdir -p "$OUT"
for arm in $ARMS; do cp "$STAGE/env_$arm.json" "$OUT/"; done
printf '%s\n' "${ARGS[@]+"${ARGS[@]}"}" > "$OUT/args.txt"
{
  echo "host ${HOSTNAME:-$(uname -n)}  date $(date -Is)"
  echo "python $PY  hf-python $HF_PY"
  rpm -qa 2>/dev/null | grep -E '^ibm-' | sort
} > "$OUT/provenance.txt"

run_arm() {  # run_arm <arm> <rep>
  local arm=$1 rep=$2 dir log rc
  dir=$OUT/${arm}_$rep; mkdir -p "$dir"; log=$dir/run.log
  if card_busy; then
    echo "!! /dev/vfio/vfio is held by another process; not starting $arm #$rep" | tee "$log"
    echo "{\"arm\": \"$arm\", \"rep\": $rep, \"rc\": 3}" > "$dir/meta.json"
    return
  fi
  echo "=== $arm #$rep  $(date -Is)" | tee "$log"
  if [ "$arm" = hf ]; then
    (cd "$dir" && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 "$HF_PY" "$SKILL_DIR/scripts/ttft.py" --model "$MODEL" \
      --input-len "$INPUT_LEN") >> "$log" 2>&1
  else
    local extra=()
    [ "$RECOMPILES" = 0 ] || extra=(TORCH_LOGS=recompiles)
    (cd "${SI_DIR:-$dir}" && env PYTHONDONTWRITEBYTECODE=1 "${extra[@]+"${extra[@]}"}" "${VLLM_CMD[@]}" bench latency \
      --model "$MODEL" "${VLLM_ARGS[@]}" --output-json "$dir/latency.json") >> "$log" 2>&1
  fi
  rc=$?
  echo "=== $arm #$rep finished rc=$rc $(date -Is)" | tee -a "$log"
  [ "$rc" = 0 ] || tail -20 "$log" | sed 's/^/    /'
  echo "{\"arm\": \"$arm\", \"rep\": $rep, \"rc\": $rc}" > "$dir/meta.json"
}

for rep in $(seq 1 "$REPEAT"); do
  for arm in $ARMS; do
    run_arm "$arm" "$rep"
  done
done

"$PY" -I -B "$SKILL_DIR/scripts/report.py" "$OUT" | tee "$OUT/report.md"
