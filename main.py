"""Train PPO, generate LLM policy (with runtime fallback), and verify."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.math_sim import MathClusterEnv
from agents.rl_agent import PolicyNetwork, RLAutoscaler, state_to_obs

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation_results"
CKPT_DIR = ROOT / "checkpoints"

SEED           = 42
EPISODES       = 400
STEPS          = 300
MINIBATCH      = 256
EPOCHS         = 4
GAMMA          = 0.99
CLIP           = 0.2
ENT_COEF_INIT  = 0.02
ENT_COEF_FINAL = 0.005
VF_COEF        = 0.5
LR             = 3e-4
EVAL_EVERY     = 50


def state_from_env(env, step_idx):
    return {
        "active_servers": int(env.active_servers),
        "booting_servers": int(env.booting_servers),
        "queue": float(env.queue),
        "arrival_rate": float(env.arrival_rate),
        "step": int(step_idx),
        "time_since_scale": int(env.time_since_scale),
    }


def verify_alignment():
    env = MathClusterEnv()
    obs, _ = env.reset(seed=999)
    probe = state_to_obs(state_from_env(env, 0))
    if not np.allclose(probe, obs, atol=1e-6):
        raise AssertionError(f"Normalization mismatch: {probe} != {obs}")
    print("[OK] state_to_obs matches MathClusterEnv._obs exactly.")


def deterministic_eval(policy, seeds=(7, 11, 42), steps=300):
    """Deterministic evaluation with H/U/D tracking."""
    policy.eval()
    scores = []
    drops_list = []
    total_counts = np.zeros(3, dtype=int)
    
    for seed in seeds:
        env = MathClusterEnv()
        env.reset(seed=seed)
        counts = np.zeros(3, dtype=int)
        for t in range(steps):
            s = state_from_env(env, t)
            obs = torch.from_numpy(state_to_obs(s)).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy(obs)
            a = int(torch.argmax(logits, dim=-1).item())
            counts[a] += 1
            env.step(a)
        m = env.get_metrics()
        scores.append(m["score"])
        drops_list.append(m["dropped"])
        total_counts += counts
        
    policy.train()
    return float(np.mean(scores)), float(np.mean(drops_list)), total_counts.tolist()


def train_rl():
    print("=" * 60)
    print("PHASE 1: Training PPO with best-checkpoint selection")
    print("=" * 60)

    torch.manual_seed(SEED)
    env = MathClusterEnv()
    policy = PolicyNetwork(obs_size=5, act_size=3)
    opt = optim.Adam(policy.parameters(), lr=LR)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_score = -1e9
    best_ep = -1

    for ep in range(EPISODES):
        alpha = ep / max(1, EPISODES - 1)
        ent_coef = ENT_COEF_INIT * (1 - alpha) + ENT_COEF_FINAL * alpha

        obs, _ = env.reset(seed=1000 + ep)
        obs_buf, act_buf, logp_buf, val_buf, rew_buf = [], [], [], [], []
        counts = np.zeros(3, dtype=int)

        for _ in range(STEPS):
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            logits, value = policy(obs_t.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits[0])
            action = dist.sample()

            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            counts[int(action.item())] += 1

            obs_buf.append(obs_t)
            act_buf.append(action)
            logp_buf.append(dist.log_prob(action))
            val_buf.append(value[0, 0])
            rew_buf.append(reward)
            obs = next_obs
            if terminated or truncated:
                break

        n = len(rew_buf)
        rewards = torch.tensor(rew_buf, dtype=torch.float32)
        returns = torch.zeros(n, dtype=torch.float32)
        running = 0.0
        for t in reversed(range(n)):
            running = rewards[t].item() + GAMMA * running
            returns[t] = running
        values = torch.stack(val_buf).detach()
        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_batch = torch.stack(obs_buf)
        act_batch = torch.stack(act_buf)
        old_logp = torch.stack(logp_buf).detach()

        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for start in range(0, n, MINIBATCH):
                mb = perm[start:start + MINIBATCH]
                logits, v = policy(obs_batch[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_batch[mb])
                ratio = (new_logp - old_logp[mb]).exp()

                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[mb]
                p_loss = -torch.min(s1, s2).mean()
                v_loss = ((v[:, 0] - returns[mb]) ** 2).mean()
                loss = p_loss + VF_COEF * v_loss - ent_coef * dist.entropy().mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()

        if (ep + 1) % EVAL_EVERY == 0 or (ep + 1) == EPISODES:
            mean_score, mean_drops, eval_hud = deterministic_eval(policy)
            marker = ""
            if mean_score > best_score:
                best_score = mean_score
                best_ep = ep + 1
                torch.save(policy.state_dict(), CKPT_DIR / "best.pt")
                marker = " ★ new best"
            print(f"ep {ep+1:3d} | trn_rew {rewards.sum().item():+8.2f} "
                  f"trn_drop {env.total_dropped:6.1f} "
                  f"H/U/D {counts.tolist()} | "
                  f"eval_score {mean_score:+9.2f} "
                  f"eval_drop {mean_drops:6.1f} "
                  f"eval_H/U/D {eval_hud} "
                  f"ent {ent_coef:.4f}{marker}")

    shutil.copy(CKPT_DIR / "best.pt", ROOT / "rl_policy.pt")
    print(f"\n[OK] Best checkpoint: episode {best_ep}, eval_score={best_score:+.2f}")
    print(f"[OK] Saved {ROOT / 'rl_policy.pt'}")


def verify_rl():
    agent = RLAutoscaler(str(ROOT / "rl_policy.pt"))
    env = MathClusterEnv()
    env.reset(seed=2024)
    counts = np.zeros(3, dtype=int)
    for t in range(STEPS):
        a = agent.decide(state_from_env(env, t))
        if a not in (0, 1, 2):
            raise AssertionError(f"Invalid action: {a}")
        counts[a] += 1
        env.step(a)
    m = env.get_metrics()
    print(f"[OK] Best-checkpoint rollout (seed 2024): score={m['score']:+.2f}, dropped={m['dropped']:.1f}, H/U/D={counts.tolist()}")


def generate_llm():
    print("\n" + "=" * 60)
    print("PHASE 2: LLM code generation (Groq)")
    print("=" * 60)

    from llm.generator import generate_agent_code, load_agent_from_code, MOCK_CODE
    from llm.templates import SYSTEM_PROMPT, USER_PROMPT

    code, source = generate_agent_code()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    policy_path = EVAL_DIR / "generated_llm_policy.py"
    policy_path.write_text(code, encoding="utf-8")
    (EVAL_DIR / "llm_prompt.txt").write_text(SYSTEM_PROMPT + "\n\n" + USER_PROMPT, encoding="utf-8")
    
    meta = {"source": source, "size_bytes": policy_path.stat().st_size}
    (EVAL_DIR / "llm_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[OK] Saved {policy_path} ({meta['size_bytes']} bytes)")
    print(f"[OK] Source: {source}")
    if source == "mock":
        print("    ^^^ This is the MOCK fallback. It is NOT an LLM output.")

    # RUNTIME SMOKE TEST: Catches logical bugs like returning None
    agent = load_agent_from_code(code)
    env = MathClusterEnv()
    env.reset(seed=7)
    smoke_test_passed = True
    
    for t in range(10):
        try:
            a = agent.decide(state_from_env(env, t))
            if a not in (0, 1, 2):
                print(f"[WARN] Policy returned invalid action: {a!r} on step {t}. Falling back to mock.")
                smoke_test_passed = False
                break
            env.step(a)
        except Exception as e:
            print(f"[WARN] Policy crashed during smoke test: {e}. Falling back to mock.")
            smoke_test_passed = False
            break

    if not smoke_test_passed:
        code = MOCK_CODE
        source = "mock_runtime_fallback"
        policy_path.write_text(code, encoding="utf-8")
        meta["source"] = source
        meta["size_bytes"] = policy_path.stat().st_size
        (EVAL_DIR / "llm_meta.json").write_text(json.dumps(meta, indent=2))
        print("[OK] Reverted to mock policy. Benchmark will continue safely.")
        
        agent = load_agent_from_code(code)
        env.reset(seed=7)
        for t in range(10):
            a = agent.decide(state_from_env(env, t))
            env.step(a)

    print("[OK] Policy loaded and returned valid actions for 10 steps.")


def main():
    verify_alignment()
    train_rl()
    verify_rl()
    generate_llm()

    print("\n" + "=" * 60)
    print("ARTIFACTS READY")
    print("=" * 60)
    for rel in ["rl_policy.pt", "evaluation_results/generated_llm_policy.py", "evaluation_results/llm_meta.json"]:
        p = ROOT / rel
        status = f"OK ({p.stat().st_size} bytes)" if p.exists() else "MISSING"
        print(f"  {rel}: {status}")
    print("\nNext: python run_real_eval.py")


if __name__ == "__main__":
    main()
