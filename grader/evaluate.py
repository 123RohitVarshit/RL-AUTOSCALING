"""Repeatable evaluation harness for the mathematical autoscaling simulator.

This is the deterministic, reproducible benchmark. It runs each agent on each
trace x seed combination using MathClusterEnv with forced trace injection.
No wall-clock timing, so results are identical across runs.
"""
from __future__ import annotations

import copy
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from env.math_sim import MathClusterEnv
from grader.traces import TRACE_NAMES, get_trace


DEFAULT_SEEDS = (11, 23, 42, 77, 101)
DEFAULT_STEPS = 300
OUTPUT_DIR = Path("evaluation_results")


def _fresh_agent(agent_or_factory: Any) -> Any:
    """Return a fresh agent instance from a class, a factory, or an instance."""
    if isinstance(agent_or_factory, type):
        return agent_or_factory()
    if callable(agent_or_factory) and not hasattr(agent_or_factory, "decide"):
        return agent_or_factory()
    return copy.deepcopy(agent_or_factory)


def evaluate_agent(agent_or_factory, trace_name="standard", steps=300, seed=42):
    trace = get_trace(trace_name, steps=steps, seed=seed)
    env = MathClusterEnv()
    env.reset(seed=seed)
    agent = _fresh_agent(agent_or_factory)
    if hasattr(agent, "reset"):
        try:
            agent.reset()
        except TypeError:
            pass

    completed_steps = min(steps, len(trace))
    server_steps = 0.0

    for step_idx in range(completed_steps):
        env.forced_arrival_rate = float(trace[step_idx])
        state = {
            "active_servers": int(env.active_servers),
            "booting_servers": int(env.booting_servers),
            "queue": float(env.queue),
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
    processed = float(metrics.get("processed", 0.0))
    dropped = float(metrics.get("dropped", 0.0))
    arrival_total = processed + dropped
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


def run_full_benchmark(agents, traces=TRACE_NAMES, seeds=DEFAULT_SEEDS,
                       steps=DEFAULT_STEPS, output_dir=OUTPUT_DIR):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    for agent_name, agent in agents.items():
        print(f"\nEvaluating (math sim): {agent_name}")
        for trace_name in traces:
            for seed in seeds:
                row = evaluate_agent(agent, trace_name, steps, seed)
                row["agent"] = agent_name
                raw_rows.append(row)
                print(f"  trace={trace_name:18s} seed={seed:3d} "
                      f"score={row['score']:9.2f} dropped={row['dropped']:7.2f} "
                      f"drop_rate={row['drop_rate']:.3%}")

    metric_names = ("score", "processed", "dropped", "drop_rate",
                    "mean_active_plus_booting_servers", "server_steps")

    summary_rows = []
    for agent_name in agents:
        for trace_name in traces:
            selected = [r for r in raw_rows
                        if r["agent"] == agent_name and r["trace"] == trace_name]
            summary = {"agent": agent_name, "trace": trace_name, "trials": len(selected)}
            for metric in metric_names:
                vals = [float(r[metric]) for r in selected]
                summary[f"{metric}_mean"] = statistics.mean(vals) if vals else 0.0
                summary[f"{metric}_stdev"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary_rows.append(summary)

    overall_rows = []
    for agent_name in agents:
        selected = [r for r in raw_rows if r["agent"] == agent_name]
        overall = {"agent": agent_name, "trials": len(selected)}
        for metric in metric_names:
            vals = [float(r[metric]) for r in selected]
            overall[f"{metric}_mean"] = statistics.mean(vals) if vals else 0.0
            overall[f"{metric}_stdev"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        overall_rows.append(overall)

    _write_csv(output_path / "raw_results.csv", raw_rows)
    _write_csv(output_path / "summary_by_trace.csv", summary_rows)
    _write_csv(output_path / "summary_overall.csv", overall_rows)
    with (output_path / "results.json").open("w", encoding="utf-8") as f:
        json.dump({"raw_results": raw_rows,
                   "summary_by_trace": summary_rows,
                   "summary_overall": overall_rows}, f, indent=2)

    print(f"\nResults saved to: {output_path.resolve()}")
    return raw_rows, summary_rows


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
