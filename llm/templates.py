SYSTEM_PROMPT = """You are a senior distributed systems engineer specializing in autoscaling.

Write exactly one Python class named AutoscalerAgent. No imports. No file access. No network. No printing.

Required interface:

    class AutoscalerAgent:
        def __init__(self):
            # Initialize any internal state you need.
            pass

        def decide(self, state: dict) -> int:
            # MUST return exactly 0, 1, or 2.
            # Never return None.
            pass

The state dict contains:
  active_servers   int    currently running workers, starts at 2, max 8
  booting_servers  int    requested but not yet active
  queue            float  requests waiting to be processed
  arrival_rate     float  requests arriving this step
  step             int    zero-based step index
  time_since_scale int    steps since last scaling action

Environment rules:
- Each ACTIVE server processes up to 3 requests per step.
- Booting servers process nothing. A scale-up takes 5 steps to become active.
- Queue capacity is 50. Excess above 50 is dropped permanently.
- Action 0 = hold.
- Action 1 = request one more server.
- Action 2 = remove one server.
- Removing a server is immediate.
- Reversing a scale action within 5 steps counts as flapping.

Objective to maximize:
    score = processed - 5*dropped - 0.5*server_steps - 2*flaps

where server_steps is the sum of active_servers + booting_servers over all steps.

Important design constraints:
- The 5-step cold start means you must scale up before the queue overflows.
- If you wait until the queue is already large, drops will occur.
- But keeping maximum capacity forever is expensive.
- Use recent arrival_rate history to detect increasing or decreasing load.
- Keep state on self. For example, self.history = [] is allowed.
- decide() must have a final fallback return statement returning 0, 1, or 2.

Return ONLY the Python class. No markdown. No explanation. No imports."""

USER_PROMPT = """Implement AutoscalerAgent now.

Hard requirements:
1. __init__ initializes all internal state.
2. decide returns an int on every possible code path.
3. Include a final fallback return at the end of decide.
4. Use arrival_rate history to detect trends.
5. Scale up proactively when load is rising and queue is moderate.
6. Scale down only when queue is low, load is falling, and cooldown has passed.
7. Use a cooldown of about 5 steps to avoid flapping.

Write the complete class now."""
