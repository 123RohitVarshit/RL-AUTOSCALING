# Autoscaling Agent Benchmark

Reproducible evaluation of three autoscaling strategies on a bounded server-cluster simulator: a hand-written threshold heuristic, a PPO reinforcement-learning agent, and a policy generated zero-shot by an LLM.

## Results (deterministic simulator)

Each agent was evaluated on 3 traffic traces x 5 seeds x 300 steps. Higher score is better.

| Agent | Mean Score | ± Std | Drop Rate | Mean Servers | Trials |
|---|---|---|---|---|---|
| LLM Agent (source=groq) | +1144.77 | 766.32 | 6.0% | 2.87 | 15 |
| Threshold Heuristic | +905.45 | 1003.91 | 11.3% | 2.32 | 15 |
| RL Agent (PPO) | -307.94 | 827.09 | 23.7% | 2.00 | 15 |

Score = processed - 3.0*dropped - 0.5*server_steps - 2.0*flaps.

## Key observations

- Best mean score: **LLM Agent (source=groq)** (+1144.77 ± 766.32).
- The RL agent kept the fleet near its initial size (mean 2.00 servers), i.e. it converged to a conservative hold policy rather than learning to scale under load.
- The LLM-generated policy is high-variance (score std 766.32): strong on some traces, weak on others.

## Methodology

- Deterministic, seed-controlled traffic traces; identical trace per agent per trial.
- Five independent seeds per trace; statistics computed over all trials.
- The LLM policy is generated once from a text description of the simulator, saved to disk, and evaluated from that saved file (not regenerated per trial).
- PPO agent trained on the same simulator objective it is scored on.
- Raw per-trial results and aggregates are exported to `evaluation_results/` as CSV/JSON.

## Limitations and honest notes

- Headline results use the deterministic simulator so they are reproducible.
- An OS-level tier using real `multiprocessing` workers and IPC was also implemented, but its throughput depends on wall-clock scheduling, so it is not reproducible run-to-run. It is retained as a robustness check, not as the ranking source.
- The RL agent's conservative collapse is a genuine finding about this reward/task combination, not a harness artifact; addressing it is future work.
