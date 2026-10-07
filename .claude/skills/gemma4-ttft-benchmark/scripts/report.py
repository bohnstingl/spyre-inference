"""Summarize a run_1102.sh output directory as Markdown.

The headline is the issue's own comparison: `vllm bench latency` average latency (with
--output-len 1 this is effectively TTFT) against the median TTFT printed by ttft.py.

Usage: report.py <out-dir>
"""

import json
import re
import statistics
import sys
from pathlib import Path

# A timed window holding a recompile has one iteration far above the rest.
LEAK_SPREAD = 1.10
TTFT_RE = re.compile(r"input_len=\d+ TTFT median ([\d.]+) s \(min ([\d.]+), max ([\d.]+)\)")
THREADS_RE = re.compile(r"Setting each threading configuration to (\d+)")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _vllm_recompiles(log: str) -> int | None:
    if "[__recompiles]" not in log:
        return None
    _, _, timed = log.partition("\nWarming up...")
    return timed.count("[__recompiles] Recompiling")


def _row(run_dir: Path) -> dict:
    meta = _read_json(run_dir / "meta.json")
    log = (run_dir / "run.log").read_text(errors="replace") if (run_dir / "run.log").exists() else ""
    row = {"arm": meta.get("arm"), "rep": meta.get("rep"), "rc": meta.get("rc")}
    if row["arm"] == "hf":
        if m := TTFT_RE.search(log):
            median, low, high = map(float, m.groups())
            row.update(metric=median, median=median, min=low, max=high)
    else:
        data = _read_json(run_dir / "latency.json")
        if lat := data.get("latencies"):
            row.update(metric=data["avg_latency"], median=statistics.median(lat),
                       min=min(lat), max=max(lat), n=len(lat))
        if m := THREADS_RE.search(log):
            row["threads"] = int(m.group(1))
        row["recompiles"] = _vllm_recompiles(log)
    if "min" in row:
        row["spread"] = row["max"] / row["min"]
    return row


def main(out: Path) -> int:
    rows = sorted((_row(d) for d in out.glob("*_*/") if (d / "meta.json").exists()),
                  key=lambda r: (r["rep"] or 0, r["arm"] or ""))
    envs = {arm: env for arm in ("hf", "vllm") if (env := _read_json(out / f"env_{arm}.json"))}
    prov = (out / "provenance.txt").read_text().splitlines() if (out / "provenance.txt").exists() else []

    print("# Gemma-4 26B-A4B TTFT, spyre-inference#1102 recipe\n")
    print(f"Run: `{out}`\n")
    for line in prov[:2]:
        print(f"- {line}")
    rpms = [p for p in prov[2:] if p.startswith("ibm-")]
    if rpms:
        print(f"- RPMs: {', '.join(rpms)}")
    print()
    print("| arm | package | version | commit | checkout |")
    print("|---|---|---|---|---|")
    for arm, env in envs.items():
        dists = ["torch", "torch-spyre", "transformers"]
        dists += ["hf-adapters-spyre"] if arm == "hf" else ["vllm", "spyre-inference"]
        for dist in dists:
            info = env.get(dist) or {}
            print(f"| {arm} | {dist} | {info.get('version', 'MISSING')} | "
                  f"{(info.get('commit') or '')[:12]} | {info.get('checkout', '')} |")
    findings = {(f["level"], f["message"]): f for env in envs.values() for f in env.get("findings", [])}
    if findings:
        print("\nEnvironment findings (WARN = deviates from the issue's recipe, accepted for this run):\n")
        order = ["ERROR", "WARN", "INFO"]
        for (level, message), f in sorted(findings.items(), key=lambda kv: order.index(kv[0][0])):
            print(f"- **{level}** [{f['arm']}] {message}")
    print()
    print("| arm | rep | issue metric (s) | median (s) | min | max | spread | notes |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "metric" not in r:
            print(f"| {r['arm']} | {r['rep']} | FAILED rc={r['rc']} | | | | | see {r['arm']}_{r['rep']}/run.log |")
            continue
        notes = []
        if r["spread"] > LEAK_SPREAD:
            notes.append("LEAK: spread > 1.10, trust the median only")
        if r.get("threads"):
            notes.append(f"{r['threads']} threads")
        if r.get("recompiles") is not None:
            notes.append(f"{r['recompiles']} post-warmup recompiles")
        name = "TTFT median" if r["arm"] == "hf" else "avg latency"
        print(f"| {r['arm']} | {r['rep']} | {name} {r['metric']:.3f} | {r['median']:.3f} | "
              f"{r['min']:.3f} | {r['max']:.3f} | {r['spread']:.2f} | {'; '.join(notes)} |")

    by_arm = {arm: [r for r in rows if r["arm"] == arm and "metric" in r] for arm in ("hf", "vllm")}
    if by_arm["hf"] and by_arm["vllm"]:
        hf = statistics.median(r["metric"] for r in by_arm["hf"])
        v_avg = statistics.median(r["metric"] for r in by_arm["vllm"])
        v_med = statistics.median(r["median"] for r in by_arm["vllm"])
        print(f"\n**vLLM / hf-adapters: {v_avg / hf:.2f}x** (issue metric: vLLM avg latency "
              f"{v_avg:.3f} s / hf median TTFT {hf:.3f} s); median/median {v_med / hf:.2f}x")
        if len(by_arm["hf"]) < 3 or len(by_arm["vllm"]) < 3:
            print("\nFewer than 3 replicates per arm: indicative only, not an equivalence claim.")
    if any("libaiupti" in message for _, message in findings):
        print("\n**A profiler build was timed (--allow-profiler): latencies are inflated.**")
    (out / "report.json").write_text(json.dumps({"rows": rows, "envs": envs}, indent=2))
    return 0 if rows and all("metric" in r for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
