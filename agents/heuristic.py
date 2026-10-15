from agents.base import AutoscalerAgent


class ThresholdAutoscaler(AutoscalerAgent):
    def __init__(self, scale_up_threshold=10, scale_down_threshold=3, cooldown=10):
        self.scale_up = scale_up_threshold
        self.scale_down = scale_down_threshold
        self.cooldown = cooldown
        self.last_scale_step = -100

    def decide(self, state: dict) -> int:
        queue = state.get("queue", 0)
        step = state.get("step", 0)
        if step - self.last_scale_step < self.cooldown:
            return 0
        if queue > self.scale_up:
            self.last_scale_step = step
            return 1
        elif queue < self.scale_down:
            self.last_scale_step = step
            return 2
        return 0
