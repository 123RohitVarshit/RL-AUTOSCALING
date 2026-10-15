"""Trace-driven evaluation harness for the OS-level RealClusterEnv.

Mirrors grader/evaluate.py but drives env/real_cluster.py. Uses fewer steps
and seeds by default because real processes run in wall-clock time.
"""
from __future__ import annotations
import copy
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from grader.traces import TRACE_NAMES, get_trace
from env.real_cluster import RealClusterEnv

DEFAULT_SEEDS = (11, 42, 101)
DEFAULT_STEPS = 150
OUTPUT_DIR = Path("evaluation_results")


def _fresh_agent(agent_or_factory: Any) -> Any:
    if isinstance(agent_or_factory, type):
        return agent_or_factory()
    if callable(agent_or_factory) and not hasattr(agent_or_factory, "decide"):
        return agent_or_factory()
    return copy.deepcopy(agent_or_factory)


def evaluate_agent_real(agent_or_factory, trace_name="standard",
                        steps=DEFAULT_STEPS, seed=42) -> dict[str, float]:
    trace = get_trace(trace_name, steps=steps, seed=seed)
    env = RealClusterEnv()
    env.reset(seed=seed)
    agent = _fresh_agent(agent_or_factory)
    if hasattr(agent, "reset"):
        agent.reset()

    completed_steps = min(steps, len(trace))
    server_steps = 0.0
    try:
        for step_idx in range(completed_steps):
            env.forced_arrival_rate = float(trace[step_idx])
            state = {
                "active_servers": int(env.active_servers),
                "booting_servers": int(env.booting_servers),
                "queue": float(env.queue_length()),
                "arrival_rate": float(trace[step_idx]),
                "step": step_idx,
                "time_since_scale": int(env.time_since_scale),
            }
            action = int(agent.decide(state))
            if action not in (0, 1, 2):
                raise ValueError(f"invalid action {action}")
            env.step(action)
            server_steps += float(env.active_servers + env.booting_servers)

        metrics = env.get_metrics()
        dropped = float(metrics.get("dropped", 0.0))
        processed = float(metrics.get("processed", 0.0))
        arrival_total = float(metrics.get("arrival_total", 0.0))
        score = float(metrics.get("score", 0.0))
        return {
            "trace": trace_name,
            "seed": int(seed),
            "steps": int(completed_steps),
            "processed": processed,
            "dropped": dropped,
            "arrival_total": arrival_total,
            "drop_rate": dropped / max(arrival_total, 1e-9),
            "score": score,
            "mean_active_plus_booting_servers": server_steps / max(completed_steps, 1),
            "server_steps": server_steps,
        }
    finally:
        env.close()


def run_real_benchmark(agents, traces=TRACE_NAMES, seeds=DEFAULT_SEEDS,
                       steps=DEFAULT_STEPS, output_dir=OUTPUT_DIR):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    for agent_name, agent in agents.items():
        print(f"\nEvaluating (OS-level): {agent_name}")
        for trace_name in traces:
            for seed in seeds:
                row = evaluate_agent_real(agent, trace_name, steps, seed)
                row["agent"] = agent_name
                raw_rows.append(row)
                print(f"  trace={trace_name:18s} seed={seed:3d} "
                      f"score={row['score']:9.2f} dropped={row['dropped']:6.2f} "
                      f"drop_rate={row['drop_rate']:.3%}")

    metric_names = ("score", "processed", "dropped", "drop_rate",
                    "mean_active_plus_booting_servers", "server_steps")
    overall_rows = []
    for agent_name in agents:
        selected = [r for r in raw_rows if r["agent"] == agent_name]
        overall = {"agent": agent_name, "trials": len(selected)}
        for metric in metric_names:
            vals = [float(r[metric]) for r in selected]
            overall[f"{metric}_mean"] = statistics.mean(vals) if vals else 0.0
            overall[f"{metric}_stdev"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        overall_rows.append(overall)

    with (output_path / "real_raw_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    with (output_path / "real_results.json").open("w", encoding="utf-8") as f:
        json.dump({"raw_results": raw_rows, "summary_overall": overall_rows}, f, indent=2)

    print(f"\nOS-level results saved to: {output_path.resolve()}")
    print("\nOS-level overall summary")
    for row in overall_rows:
        print(f"  {row['agent']}: score={row['score_mean']:.2f} "
              f"+/- {row['score_stdev']:.2f}; drop_rate={row['drop_rate_mean']:.3%}; "
              f"mean_servers={row['mean_active_plus_booting_servers_mean']:.2f}")
    return raw_rows
