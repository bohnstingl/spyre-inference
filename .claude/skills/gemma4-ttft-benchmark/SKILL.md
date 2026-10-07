---
name: gemma4-ttft-benchmark
description: "Reproduce the Gemma-4 26B-A4B prefill TTFT comparison from spyre-inference#1102 on any Spyre host, on the environment that is already installed - the maintainer's ttft.py on hf-adapters (pinned PR #620, 9b075e4) against `vllm bench latency` (1984-token prompt, 1 output token, TP1) on spyre-inference. Checks the environment read-only against the issue's pins (hf-adapters commit, torch-spyre e9d31328, profiler-free build, consistent Spyre runtime libraries, behaviour-changing env vars) and flags every inconsistency to the user instead of fixing it; never installs, syncs, rebuilds or checks anything out. Then runs both arms one at a time and reports TTFT plus the vLLM/hf ratio with provenance. Use when asked to benchmark or reproduce Gemma-4 / gemma-4-26B-A4B TTFT or prefill latency, reproduce #1102, compare spyre-inference with hf-adapters, or measure an MoE change's TTFT."
---

# Gemma-4 TTFT benchmark (spyre-inference#1102)

Reproduces tdoublep's measurement from
[spyre-inference#1102](https://github.com/torch-spyre/spyre-inference/issues/1102#issuecomment-6001115830):
same model, 1984-token prompt, 1 output token, TP1. On 2026-10-05 vLLM `main` was about
1.5x slower than hf-adapters.

## The one rule: never alter the environment

This skill measures the environment it is given, and changes nothing in it. Do not run
`uv sync`, `uv pip install`, `pip`, `git checkout`/`switch`/`fetch` in a checkout, a
rebuild, or an install script, not even into a fresh scratch venv to "match the recipe".
When the check flags something, report it to the user and stop. Changing the environment
is the user's call, made with their own tooling.

## What the issue pins, and how the skill uses each pin

| issue recipe | in this skill |
|---|---|
| hf-adapters PR #620 at `9b075e46bc9689671fb7ba546ef12b3e509bbdc5` | checked; anything else is a WARN |
| hf-adapters' torch-spyre `e9d31328345f55ead94d1a65736380e6e07513bc` `[cpsat]` | checked; anything else is a WARN (it says whether the env's build contains `e9d3`) |
| `ttft.py` from the comment | [`scripts/ttft.py`](scripts/ttft.py), byte-identical (sha1 `58c92665e1258821878142ea8dcec1d803ec6b29`) |
| `OMP_NUM_THREADS=8 uv run --no-sync python ttft.py --model <model> --input-len 1984` | the same command, with the env's interpreter in place of `uv run --no-sync python` |
| `uv run --no-sync vllm bench latency --model <model> --input-len 1984 --output-len 1 --batch-size 1 --num-iters-warmup 2 --num-iters 3 --max-model-len 2048 --max-num-seqs 1` | the same command via the env's `vllm` entry point, from the spyre-inference checkout, plus `--output-json` |
| compare vLLM avg latency against `ttft.py`'s median TTFT | the report's headline ratio |

The issue's setup steps (`git clone`, `uv sync`, `uv pip install torch-spyre...`) describe
how the maintainer provisioned their host. They are **not** executed here.

## Quick start

```bash
SKILL=<this skill dir>
source <venv>/bin/activate                 # the environment to benchmark
$SKILL/scripts/run_1102.sh --check         # read-only check + plan; exit 0 / 1 / 3
$SKILL/scripts/run_1102.sh                 # benchmark (refuses on any ERROR, or on WARN without --accept-warnings)
```

On dt-inductor pods: `--env-script /scratch/virtualenv/<env>/bin/activate`, or
`--python /scratch/virtualenv/<env>/bin/python`.

## Instructions

1. **Identify the environment.** Ask the user which venv or interpreter to benchmark if it
   is not obvious; never pick one by installing. Defaults: `$VIRTUAL_ENV`, then
   `$UV_PROJECT_ENVIRONMENT`. `--env-script` sources the host's runtime environment, which
   only sets variables in the script's own shell. `--hf-python` runs the hf arm in a
   different interpreter (the issue used a separate env for it).
2. **Check:** `run_1102.sh --check [flags]`. It prints every finding and the exact commands
   it would run. Exit codes: 0 clean, 1 ERROR, 3 WARN.
3. **Flag the findings to the user, verbatim.**
   - **ERROR:** stop. Explain what is wrong and what the user would need to change, e.g.
     "torch-spyre links libaiupti: rebuild it with the profiler off". Do not change it.
   - **WARN:** list each one and ask whether to benchmark anyway. Only after a yes, re-run
     with `--accept-warnings`. The accepted WARNs are printed in the report.
   - **INFO:** mention them in the report; no question needed.
4. **Run** (after step 3): `run_1102.sh [--accept-warnings] [flags]`, in the background,
   polling its output. Nothing else may use the card meanwhile.
   - Cold cache, a vLLM run takes two to three times as long as an hf run, about half
     of it engine init; much of the hf run is its first warmup call, which compiles.
   - Use `--repeat 3` (arms alternate) for any decision; one replicate is indicative.
5. **Gate, then report.** From `report.md`:
   - a `FAILED` row: report the failure with the tail of that arm's `run.log`, no number;
   - `LEAK` (max/min > 1.10): quote the median only and say the mean is compile-contaminated;
   - with `--recompiles`: a nonzero post-warmup recompile count means warmup missed a shape.

   Report the package table, every WARN accepted, the per-arm rows and the headline ratio.

## What `--check` looks at (read-only)

`scripts/check_env.py` imports only torch and torch_spyre. It locates every other package
without importing it, and reads checkouts with `git rev-parse` / `status` /
`merge-base --is-ancestor`. It writes no bytecode (`python -B`), and neither do the benchmark runs.

| finding | level |
|---|---|
| torch / torch_spyre do not import; `_C.so` has unresolved libraries | ERROR |
| `_C.so` links `libaiupti` (profiler build) | ERROR (WARN with `--allow-profiler`; numbers labelled inflated) |
| hf-adapters or vllm / spyre-inference not installed; model `config.json` unreadable | ERROR |
| hf-adapters not at `9b075e4`; torch-spyre not `e9d3` | WARN |
| `_C.so` built from another commit than the torch-spyre checkout's HEAD | WARN |
| modified tracked files in a checkout; unmet `transformers==5.15.0` / `torch~=2.13.0` / vllm requirement | WARN |
| Spyre libraries resolving from more than one install tree (e.g. in-tree `sentient/` plus `/opt/ibm/spyre`) | WARN |
| behaviour-changing env vars (`SPYRE_*`, `VLLM_*`, `TORCHINDUCTOR_*`, `TORCH_LOGS`, `SENCORES`, `FRONTEND_POOL_ALLOCATION`, `CO_OPTIMIZING*`, `DXP_*`) | WARN |
| cache dirs, `OMP_NUM_THREADS`, `VLLM_PLUGINS`, `SPYRE_DEVICES`; stale editable metadata; spyre-inference's own torch-spyre pin vs the env's | INFO |
| model `model_type` not Gemma-4 | WARN |

The run itself also refuses a busy card (`fuser /dev/vfio/vfio`).

## Deviations from the issue

1. **One environment for both arms** by default, so the hf arm runs the env's torch-spyre
   rather than a dedicated `e9d3` build. The check flags this as a WARN, with the
   commit relation. Pass `--hf-python` to use a separate env the user built.
2. **The env's interpreter instead of `uv run --no-sync`.** Same execution, no uv involved.
3. **`ttft.py` runs from the skill directory**, not from inside the hf-adapters clone, so
   nothing is written into the checkout. With an installed `hf_adapters` the imports are
   identical.
4. **`--output-json` on `vllm bench latency`.** Output only.

## Pitfalls (measured)

- **Profiler builds.** torch-spyre's `setup.py` defaults to `USE_SPYRE_PROFILER=1`, and
  hf-adapters has no override, so the issue's literal hf-adapters install links
  `libaiupti`. spyre-inference's uv build pins the profiler off. On dt-inductor pods,
  `install-spyre-env.sh` needs `--no-profiler`. Tell the user; do not rebuild.
- **Threads.** The issue leaves `SPYRE_NUM_CPUS` unset. Without a cgroup CPU quota,
  spyre-inference then uses all physical cores (48 on the reference pod), while `ttft.py`
  pins 8. Measured irrelevant for the vLLM arm: `SPYRE_NUM_CPUS=8` moved main and the MoE
  stack by <0.5%. The report shows each vLLM run's thread count.
- **Traced MoE (spyre-inference #1154 and descendants).** It re-traces the block at the
  first real request on vLLM's lazy `moe_quant_config is None` guard, then recompiles
  prefill page attention. The timed iterations stay clean, and since #1109 there is no
  `compiled outside warmup` warning. Use `--recompiles`: the target is 0.
- **Prefix caching.** `vllm bench latency` disables it by default, which is what makes the
  repeated prompt valid. Do not enable it.
- **Wrong runtime libraries.** `libsenlib-dd2.so` not found, or an undefined `flex::...`
  symbol from `libspyre_comms.so.1`, means an older in-tree Spyre build is mixed with
  `/opt/ibm/spyre`. On dt-inductor pods, `dt-inductor2/env.sh` is stale: use the venv's
  `bin/activate`.
- **Noise.** Medians agreed to <0.5% across two sessions on the reference pod; session
  drift is about 0.7% per 2 h. Alternate arms with `--repeat`; never run them in blocks.

## Reference results

From the dt-inductor2 pod, 2026-10-06/07, both arms on one profiler-free torch-spyre
(`2ab31d08`, which contains `e9d3`). A ballpark, not a contract:

| arm | commit | median TTFT vs hf |
|---|---|---|
| hf-adapters #620 | `9b075e4` | 1.00x |
| spyre-inference main | `ad76a008` | 1.52x (1.54x issue metric) |
| MoE stack (#1058 + #1154) | `ef206a2a` | 1.14x |

## Output layout

`<out>/` (default `$HOME/gemma4-ttft-1102/<timestamp>/`):
- `report.md` / `report.json`: package table, findings, per-arm rows, ratio.
- `<arm>_<rep>/run.log`: full log; `vllm_<rep>/latency.json`: vLLM's JSON.
- `env_<arm>.json`: everything the check found, with its findings.
- `provenance.txt`: host, date, interpreters, `ibm-*` RPMs. `args.txt`: the flags used.

## Installing this skill on another host

Copy or symlink this directory into a project's `.claude/skills/`, or into
`~/.claude/skills/`. Keep the directory name `gemma4-ttft-benchmark`.
