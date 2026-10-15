import numpy as np

TRACE_NAMES = ("standard", "thundering_herd", "gradual_ramp")


def _rand_int(rng, low, high):
    low = max(0, int(low))
    high = max(low + 1, int(high))
    return int(rng.integers(low, high))


def get_trace(name: str = "standard", steps: int = 300, seed: int = 42) -> np.ndarray:
    """Return a reproducible workload trace. Different seeds produce different traces."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps, dtype=np.float32)

    if name == "standard":
        rates = 5.0 + 3.0 * np.sin(t * 0.05)

        num_spikes = int(rng.integers(2, 5))
        high = max(1, steps - 15)
        spike_starts = rng.integers(0, high, size=num_spikes)

        for start in spike_starts:
            start = int(start)
            end = min(start + 15, steps)
            rates[start:end] *= float(rng.uniform(2.0, 4.0))

        return np.clip(rates, 0.0, 25.0).astype(np.float32)

    if name == "thundering_herd":
        rates = np.full(steps, 3.0, dtype=np.float32)

        h1 = _rand_int(rng, 0.25 * steps, 0.45 * steps)
        h2 = _rand_int(rng, 0.65 * steps, min(steps - 5, 0.85 * steps))

        h1_end = min(h1 + 20, steps)
        h2_end = min(h2 + 10, steps)

        rates[h1:h1_end] = float(rng.uniform(20.0, 25.0))
        rates[h2:h2_end] = float(rng.uniform(15.0, 20.0))

        return np.clip(rates, 0.0, 25.0).astype(np.float32)

    if name == "gradual_ramp":
        rates = np.linspace(2.0, 15.0, steps, dtype=np.float32)
        rates += rng.normal(0.0, 1.0, size=steps).astype(np.float32)
        return np.clip(rates, 0.0, 25.0).astype(np.float32)

    raise ValueError(f"Unknown trace {name!r}. Choose from {TRACE_NAMES}.")
