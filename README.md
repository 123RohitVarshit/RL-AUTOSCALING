# AI Autoscaling Policy Benchmark

A dual-tier evaluation harness comparing three approaches to server-cluster autoscaling:
a hand-written threshold heuristic, a Proximal Policy Optimization (PPO) agent, and
a policy generated zero-shot by a large language model.

This project was built to answer a specific research question:

> **When given identical information about a queueing system, which AI approach
> generates better autoscaling decision logic?**

The Tier 1 benchmark is designed to be **reproducible and fair**. All agents face the
same deterministic traffic traces, the same scoring function, and the same
environmental constraints. Results are reported honestly, including failure modes
and limitations.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Agents Under Test](#agents-under-test)
- [Tier 1: Mathematical Simulator (Reproducible)](#tier-1-mathematical-simulator-reproducible)
- [Tier 2: OS-Level Simulator (Non-Reproducible)](#tier-2-os-level-simulator-non-reproducible)
- [Key Findings](#key-findings)
- [Methodology](#methodology)
- [Limitations and Honest Notes](#limitations-and-honest-notes)
- [Reproducibility](#reproducibility)
- [Project Structure](#project-structure)

---

## Problem Statement

Autoscaling is a control-plane problem with three conflicting objectives:

1. **Minimize cost**: Running idle servers wastes resources.
2. **Maximize throughput**: Process as many requests as possible.
3. **Maintain SLOs**: Avoid dropping requests or causing excessive latency.

The challenge is compounded by:

- **Cold-start delay**: New servers take time to become active (5 steps in this benchmark).
- **Non-stationary traffic**: Arrival rates vary sinusoidally with sudden spikes.
- **Cost-sensitive decisions**: Every server-step incurs a penalty; every dropped request incurs a larger penalty.

This benchmark evaluates how well three different AI approaches can balance these tradeoffs.

---

## System Architecture

The benchmark uses a **dual-tier architecture** to separate reproducible research
from real-world robustness testing.

### Tier 1: Mathematical Simulator (Reproducible)

- **Location**: `env/math_sim.py`
- **Purpose**: Deterministic, seed-controlled evaluation for fair comparison.
- **Model**: Queueing system with bounded capacity.
  - Each active server processes up to 3 requests per step.
  - Queue capacity is 50; excess requests are dropped.
  - Cold-start delay is 5 steps.
- **Determinism**: Same trace + same seed + same agent artifacts = identical results every time.
- **Use case**: Headline ranking and quantitative comparison.

### Tier 2: OS-Level Simulator (Non-Reproducible)

- **Location**: `env/real_cluster.py`
- **Purpose**: Robustness check under real process and IPC constraints.
- **Model**: Real `multiprocessing.Process` workers with real IPC queues.
  - Workers are spawned and terminated as real OS processes.
  - Requests flow through `multiprocessing.Queue`.
  - Cold-start involves real process creation overhead.
- **Non-determinism**: Wall-clock timing, OS scheduling, and IPC latency vary run-to-run.
- **Use case**: Qualitative robustness check, NOT the ranking source.

### Why Two Tiers?

The mathematical simulator is **reproducible and fair**, making it suitable for ranking.
The OS-level simulator is **realistic but non-reproducible**, making it suitable only
for verifying that policies don't break under real constraints.

A production validation system would require real microservices, real HTTP traffic,
and real cloud metrics, which is outside the scope of this research benchmark.

---

## Agents Under Test

### 1. Threshold Heuristic

- **Type**: Hand-written rule-based policy.
- **Logic**:
  - Scale up if queue length exceeds threshold.
  - Scale down if queue length is below threshold and cooldown has passed.
  - Cooldown of 5 steps to avoid flapping.
- **Strengths**: Simple, interpretable, no training required.
- **Weaknesses**: Reactive; only responds after queue has already grown.

### 2. RL Agent (PPO)

- **Type**: Deep reinforcement learning policy.
- **Architecture**: 2-layer MLP (64 → 64 → 3 actions).
- **Training**: 400 episodes × 300 steps on the Tier 1 simulator's default traffic, entropy annealing (0.02 to 0.005), best-checkpoint selection (greedy evaluation every 50 episodes).
- **Objective**: The PPO reward uses the same drop penalty (3.0) as the evaluation score.
- **Strengths**: Can learn complex, non-linear decision boundaries.
- **Weaknesses**: In this run, the selected checkpoint's greedy policy never scales (see Key Findings).

### 3. LLM Agent (Groq / Qwen)

- **Type**: Zero-shot code generation.
- **Model**: `qwen/qwen3.8-27b` via Groq API.
- **Prompt**: Textual description of environment rules and scoring function.
- **Output**: Python class `AutoscalerAgent` with `decide(state) -> int` method.
- **Strengths**: Can synthesize control logic from natural-language descriptions.
- **Weaknesses**: One policy is generated per benchmark, so variance across generations is not measured.

#### LLM Policy Provenance

- `main.py` requests the policy from the Groq API and validates the response format.
- A 10-step runtime smoke test follows. If the generated policy crashes or returns an invalid action (for example `None`), or if the API call fails, `main.py` substitutes a mock policy and records `mock_runtime_fallback` or `mock` in `evaluation_results/llm_meta.json`.
- Both evaluation scripts print the policy source in the agent name (for example `LLM Agent (Groq, source=groq)`). The Tier 1 and Tier 2 results below were produced with `source=groq`.
- The saved files (`evaluation_results/generated_llm_policy.py`, `llm_prompt.txt`, `llm_meta.json`) are what the evaluation loads. No API call is made during evaluation.

---

## Tier 1: Mathematical Simulator (Reproducible)

### Configuration

- **Traces**: `standard`, `thundering_herd`, `gradual_ramp`
- **Seeds**: 11, 23, 42, 77, 101
- **Steps per episode**: 300
- **Total trials**: 3 agents × 3 traces × 5 seeds = 45 trials

### Scoring Function

```
score = processed
      - 3.0 × dropped
      - 0.5 × server_steps
      - 2.0 × flaps
```

where:

- `processed`: total requests successfully processed
- `dropped`: total requests dropped due to queue overflow
- `server_steps`: sum of (active_servers + booting_servers) across all steps
- `flaps`: number of scale reversals within the cooldown window

### Results

| Agent | Mean Score | ± Std Dev | Drop Rate | Mean Servers | Trials |
|-------|-----------|-----------|-----------|--------------|--------|
| **LLM Agent (Groq)** | **+1144.77** | 766.32 | **6.0%** | 2.87 | 15 |
| Threshold Heuristic | +905.45 | 1003.91 | 11.3% | 2.32 | 15 |
| RL Agent (PPO) | -307.94 | 827.09 | 23.7% | 2.00 | 15 |

Std Dev is the population standard deviation (n) over all 15 trials, so it includes the
spread between trace types, not only seed-to-seed variation.

### Per-Trace Breakdown

#### Standard Trace (sinusoidal with random spikes)

| Agent | Mean Score | Drop Rate |
|-------|-----------|-----------|
| LLM Agent | +1211.00 | 2.95% |
| Threshold | +1035.26 | 5.74% |
| RL Agent | +657.84 | 11.55% |

#### Thundering Herd Trace (sudden 25x spikes)

| Agent | Mean Score | Drop Rate |
|-------|-----------|-----------|
| LLM Agent | +188.27 | 15.19% |
| Threshold | -373.93 | 27.85% |
| RL Agent | -247.93 | 24.09% |

#### Gradual Ramp Trace (steady increase from 2 to 15 req/step)

| Agent | Mean Score | Drop Rate |
|-------|-----------|-----------|
| LLM Agent | +2035.03 | 0.0% |
| Threshold | +2055.04 | 0.21% |
| RL Agent | -1333.74 | 35.38% |

### Paired Comparisons

Each trial is a (trace, seed) pair that every agent faces, so agents can be compared per trial.

| Comparison | Standard | Thundering Herd | Gradual Ramp | Total |
|------------|----------|-----------------|--------------|-------|
| LLM beats Threshold | 5 of 5 | 5 of 5 | 1 of 5 | 11 of 15 |
| LLM beats RL | 5 of 5 | 5 of 5 | 5 of 5 | 15 of 15 |
| RL beats Threshold | 0 of 5 | 5 of 5 | 0 of 5 | 5 of 15 |

On gradual_ramp, Threshold leads the LLM policy in 4 of 5 trials by 14 to 55 points.

### Observations

1. **LLM Agent has the best mean score and lowest drop rate**, and beats Threshold in 11 of 15 paired trials and PPO in 15 of 15. It uses more servers on average than Threshold (2.87 vs 2.32).
2. **Threshold Heuristic is competitive** on gradual ramps (it is slightly ahead there) but struggles with sudden spikes.
3. **RL Agent never scales up** (mean servers = 2.00), which costs it heavily on gradual_ramp.
4. **Most of the score spread is between trace types, not seeds.** Within-trace sample standard deviation across seeds averages about 100 points per agent (LLM 96, Threshold 119, PPO 116), while per-trace means differ by thousands.

---

## Tier 2: OS-Level Simulator (Non-Reproducible)

### Configuration

- **Traces**: `standard`, `thundering_herd`, `gradual_ramp`
- **Seeds**: 11, 42, 101 (reduced from 5 to 3 due to runtime)
- **Steps per episode**: 150 (reduced from 300 due to wall-clock overhead)
- **Total trials**: 3 agents × 3 traces × 3 seeds = 27 trials
- **Runs**: results below are from a single run of `run_real_eval.py`.

### Scoring Function

Same as Tier 1, except that flaps are not penalized:

```
score = processed - 3.0 × dropped - 0.5 × server_steps
```

### Results

| Agent | Mean Score | ± Std Dev | Drop Rate | Mean Servers | Trials |
|-------|-----------|-----------|-----------|--------------|--------|
| **LLM Agent (Groq)** | **-178.67** | 1106.09 | **23.9%** | 3.76 | 9 |
| RL Agent (PPO) | -333.78 | 400.54 | 28.2% | 2.00 | 9 |
| Threshold Heuristic | -1028.39 | 1410.28 | 43.1% | 3.50 | 9 |

Std Dev in this tier is the sample standard deviation (n-1) over the 9 trials.
Tier 1 reports the population standard deviation (n).

### Per-Trace Breakdown

#### Standard Trace

| Agent | Mean Score | Drop Rate |
|-------|-----------|-----------|
| LLM Agent | -798.00 | 35.72% |
| RL Agent | +140.00 | 16.82% |
| Threshold | -932.33 | 40.30% |

#### Thundering Herd Trace

| Agent | Mean Score | Drop Rate |
|-------|-----------|-----------|
| LLM Agent | -754.50 | 35.85% |
| RL Agent | -551.67 | 35.13% |
| Threshold | -1786.17 | 62.31% |

#### Gradual Ramp Trace

| Agent | Mean Score | Drop Rate |
|-------|-----------|-----------|
| LLM Agent | +1016.50 | 0.0% |
| RL Agent | -589.67 | 32.68% |
| Threshold | -366.67 | 26.56% |

### Observations

1. **By mean score the order is LLM > RL > Threshold in this run**, which differs from Tier 1 (LLM > Threshold > RL). The gaps are small relative to the trial-to-trial spread: in paired trials, each of the three pairwise comparisons (LLM vs Threshold, LLM vs RL, RL vs Threshold) is won by the same agent in only 6 of 9 trials.
2. **Overall mean scores are negative for all three agents**, but individual trials range from about +1030 to about -3160, and every agent has at least one positive trial.
3. **Within-trace spread is much larger than in Tier 1.** The average within-trace sample standard deviation is about 605 (LLM) and 1107 (Threshold), versus about 96 and 119 in Tier 1. Wall-clock timing and IPC effects are the candidate cause; this was not isolated.
4. **LLM Agent had 0% drops on all three gradual_ramp trials.**
5. **Threshold on gradual_ramp scored about +1030 for seeds 11 and 42 and -3163 (79.7% drops) for seed 101**, while the same policy drops almost nothing on gradual_ramp in Tier 1.

### Why OS-Level Results Are Not the Ranking Source

The OS-level simulator uses real `multiprocessing.Process` workers and real IPC queues.
This is expected to introduce wall-clock timing variance:

- Process spawn latency varies by 10-100ms.
- IPC queue throughput depends on OS scheduling.
- Worker completion times are non-deterministic.

As a result, the same agent + same trace + same seed is **not expected to produce
identical results on different runs**. Repeat-run variance has not been quantified yet,
and all Tier 2 numbers here come from one run. This makes the OS tier unsuitable for
ranking but useful as a robustness check.

---

## Key Findings

### 1. One LLM-Generated Policy Scored Highest in Tier 1

The LLM-generated policy achieved the best mean score on the reproducible Tier 1
benchmark (+1144.77 vs +905.45 for Threshold and -307.94 for PPO). It beat Threshold in
11 of 15 paired trials (all 10 on standard and thundering_herd, 1 of 5 on gradual_ramp)
and PPO in all 15.

This shows that a zero-shot generated policy can outperform a hand-written heuristic under
this scoring function. It does not show that the method does so reliably: a single policy
was evaluated, the prompt included the scoring function, and the policy uses more servers
than Threshold on average.

### 2. The Selected PPO Checkpoint Never Scales

The RL agent as evaluated never scales (mean servers = 2.00 on every trial). The training
logs point to a degenerate greedy policy rather than a learned preference for holding:

- Stochastic training rollouts used all three actions (for example 137 hold, 157 scale-up,
  6 scale-down in one 300-step episode).
- Greedy evaluation of successive checkpoints alternated between always-hold and
  always-scale-up (evaluation action counts of [900, 0, 0] or [0, 900, 0] over three
  300-step episodes).
- Checkpoint selection kept an always-hold checkpoint (episode 100) because it outscored
  the always-scale-up checkpoints on the selection episodes (+797 vs +495).

The cause has not been isolated. Candidate explanations are argmax decoding of a
near-uniform policy, checkpoint selection on only three evaluation episodes, and training
on a single traffic distribution. This finding characterizes this training pipeline, not
PPO in general.

### 3. Score Spread Is Mostly Between Trace Types (Tier 1)

In Tier 1, the within-trace sample standard deviation across seeds averages about 100
points per agent, while per-trace mean scores differ by thousands (for example, the LLM
policy averages +188 on thundering_herd and +2035 on gradual_ramp). The ± values in the
overall tables therefore mostly reflect differences between trace types. Variance caused
by regenerating the LLM policy was not measured.

### 4. The OS-Tier Ordering Differs From Tier 1 but Is Not Established

In the reported Tier 2 run, RL scored above Threshold, the reverse of Tier 1. With 9 trials
per agent and 6-of-9 paired wins for each pairwise comparison, this is not enough to
conclude that OS-level realism changes the ranking. Quantifying it requires repeated runs.

### 5. Reproducibility Requires Determinism

The Tier 1 simulator is deterministic: the same saved artifacts, traces, and seeds give
the same results. The OS-level simulator is not. This is a fundamental tradeoff:
**reproducibility requires determinism, but realism introduces non-determinism**.
Retraining PPO or regenerating the LLM policy with `main.py` does not recreate the
reported policies exactly, because the API output varies.

---

## Methodology

### Traffic Traces

Three trace families are used, each generated with a seed-controlled random number
generator:

1. **Standard**: Sinusoidal base (5 + 3·sin(0.05t)) with 2-5 random spikes.
2. **Thundering Herd**: Low baseline (3 req/step) with two sudden 25x spikes.
3. **Gradual Ramp**: Linear increase from 2 to 15 req/step with Gaussian noise.

Each trace is generated once per seed and reused across all agents to ensure fair
comparison. PPO is trained on the simulator's default traffic (`main.py`), not on these
trace families.

### Scoring

Within each tier, all agents are scored using the same objective function.

Tier 1:

```
score = processed - 3.0 × dropped - 0.5 × server_steps - 2.0 × flaps
```

Tier 2 omits the flaps term. PPO is trained on the Tier 1 simulator with the same 3.0
drop penalty used in evaluation.

### Agent Instantiation

- **Threshold Heuristic**: Instantiated fresh for each trial.
- **RL Agent**: Loaded from the best checkpoint selected during training (`rl_policy.pt`).
- **LLM Agent**: Generated via the Groq API in `main.py`, saved to disk, and loaded for each trial.

The LLM policy is **not regenerated per trial** to ensure reproducibility.

### Statistical Aggregation

- **Mean and standard deviation** computed over all trials for each agent (population
  standard deviation in Tier 1, sample standard deviation in Tier 2).
- **Drop rate** computed per trial as `dropped / (processed + dropped)` and averaged over trials.
- **Mean servers** computed as the average of (active + booting) across all steps.

---

## Limitations and Honest Notes

### 1. This Is a Research Benchmark, Not Production Validation

The mathematical simulator is a simplified queueing model. It does not capture:

- Real network latency
- Real CPU contention
- Real memory pressure
- Real garbage collection pauses
- Real cloud billing

**This benchmark answers "which AI approach writes better decision logic?",
not "which policy should I deploy to production?"**

### 2. OS-Level Tier Is Non-Reproducible and Single-Run

The OS-level simulator uses real processes and wall-clock timing, which introduces
run-to-run variance. Results from this tier should be interpreted as **robustness
checks**, not as quantitative rankings. The reported numbers come from one run with 9
trials per agent, and repeat-run variance has not been measured. Tier 2 also omits the
flaps penalty and uses 150 steps and 3 seeds.

### 3. PPO Training Is Underexplored

The RL agent was trained for 400 episodes with a simple 2-layer MLP, using discounted
returns with a learned value baseline (no GAE). More sophisticated architectures (LSTM,
attention), longer training, training on all three trace families, or sampling actions at
evaluation instead of argmax might change the result. The current "never scales" outcome
is a finding about this pipeline, not a definitive statement about RL's capabilities.

Checkpoint selection used greedy evaluation on seeds 7, 11 and 42. Seeds 11 and 42 are
also benchmark seeds, so the selection episodes overlap the evaluation seeds for PPO.

### 4. LLM Prompt Is Zero-Shot and a Single Sample

The LLM policy was generated from a single prompt with no examples. `main.py` validates
the response format and smoke-tests the policy, but does not use benchmark scores to
refine it. Only one generated policy is evaluated, so variance across generations is
unknown. Few-shot prompting, chain-of-thought, iterative debugging, or repeated sampling
might change performance.

### 5. Scoring Function Is Arbitrary

The penalty weights (3.0 for drops, 0.5 for server_steps, 2.0 for flaps) are
chosen to reflect a cost-sensitive production environment. Different weights would
produce different rankings. The current weights prioritize drop avoidance over cost
minimization.

### 6. Small Samples and No Significance Testing

Tier 1 has 15 trials per agent and Tier 2 has 9, drawn from only 3 trace families. Paired
win counts are reported, but no formal significance tests or confidence intervals are.

---

## Reproducibility

### Prerequisites

```bash
pip install -r requirements.txt
# or: pip install numpy torch gymnasium openai
export GROQ_API_KEY="your-key-here"   # only needed to regenerate the LLM policy
```

### Run the Benchmark

```bash
# Optional: retrain PPO and regenerate the LLM policy.
# This overwrites rl_policy.pt and the saved LLM policy, and the new LLM policy
# will differ from the one reported here.
python main.py

# Tier 1: Mathematical simulator (reproducible, ~1 minute)
python final_eval.py

# Tier 2: OS-level simulator (non-reproducible, ~10-20 minutes)
python run_real_eval.py
```

The reported numbers come from the saved `rl_policy.pt` and
`evaluation_results/generated_llm_policy.py`. Keep both in the repository and skip
`main.py` to re-evaluate them.

### Expected Outputs

- `evaluation_results/raw_results.csv`: Per-trial results for all agents.
- `evaluation_results/summary_by_trace.csv`: Aggregated results by trace.
- `evaluation_results/summary_overall.csv`: Overall aggregated results.
- `evaluation_results/results.json`: Full results in JSON format.
- `README.md`: Auto-generated results summary.

### Determinism

Tier 1 evaluation of the saved artifacts is **fully deterministic**. Running
`final_eval.py` multiple times will produce identical results.

Tier 2 results are **non-deterministic** due to wall-clock timing. Running
`run_real_eval.py` multiple times is expected to produce different results.

---

## Project Structure

```
mechanize-autoscaler-bench/
├── main.py                    # Training + LLM generation orchestrator
├── final_eval.py              # Tier 1 reproducible benchmark
├── run_real_eval.py           # Tier 2 OS-level benchmark
├── requirements.txt           # Python dependencies
├── rl_policy.pt               # Selected PPO checkpoint (copied from checkpoints/best.pt)
├── env/
│   ├── math_sim.py            # Deterministic queueing simulator
│   └── real_cluster.py        # Real-process simulator
├── agents/
│   ├── base.py                # Abstract agent interface
│   ├── heuristic.py           # Threshold heuristic
│   └── rl_agent.py            # PPO policy wrapper
├── llm/
│   ├── templates.py           # Prompt templates
│   └── generator.py           # Groq API integration
├── grader/
│   ├── traces.py              # Seed-controlled traffic generation
│   ├── evaluate.py            # Tier 1 evaluation harness
│   └── evaluate_real.py       # Tier 2 evaluation harness
├── evaluation_results/        # CSV/JSON outputs, saved LLM policy, prompt, metadata
└── checkpoints/               # PPO training checkpoints
```

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{autoscaling_benchmark_2026,
  title = {AI Autoscaling Policy Benchmark},
  author = {Rohit Chiluka},
  year = {2026},
  howpublished = {\url{https://github.com/123RohitVarshit/RL-AUTOSCALING}}
}
```

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgments

- Groq API for LLM code generation.
- Gymnasium for the environment interface.
- PyTorch for PPO training.

---
