"""Run the OS-level benchmark. Requires rl_policy.pt and a saved LLM policy.

Run main.py first.
"""
from __future__ import annotations

import json
from pathlib import Path

from grader.evaluate_real import run_real_benchmark
from agents.heuristic import ThresholdAutoscaler
from agents.rl_agent import RLAutoscaler
from llm.generator import load_agent_from_code


def main():
    agents = {"Threshold Heuristic": ThresholdAutoscaler}

    checkpoint = Path("rl_policy.pt")
    if checkpoint.exists():
        agents["RL Agent (PPO)"] = lambda: RLAutoscaler(str(checkpoint))
        print(f"[OK] Loaded {checkpoint}")
    else:
        print("ERROR: rl_policy.pt missing. Run main.py first.")
        return

    llm_file = Path("evaluation_results/generated_llm_policy.py")
    meta_file = Path("evaluation_results/llm_meta.json")
    if llm_file.exists():
        code = llm_file.read_text(encoding="utf-8")
        source = "unknown"
        if meta_file.exists():
            source = json.loads(meta_file.read_text()).get("source", "unknown")
        label = f"LLM Agent (Groq, source={source})"
        agents[label] = lambda: load_agent_from_code(code)
        if source == "mock":
            print(f"[WARN] Loaded {llm_file}, but source={source}. "
                  "This is NOT a real LLM output; it is a hand-written fallback.")
            print("       Results for this agent will be labeled accordingly.")
        else:
            print(f"[OK] Loaded {llm_file} (source={source})")
    else:
        print("ERROR: generated_llm_policy.py missing. Run main.py first.")
        return

    print()
    run_real_benchmark(agents, steps=150, seeds=(11, 42, 101))


if __name__ == "__main__":
    main()
