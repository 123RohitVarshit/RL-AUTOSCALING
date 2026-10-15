class AutoscalerAgent:
    def __init__(self):
        self.history = []
        self.last_scale_step = -100
        self.last_action = 0

    def decide(self, state: dict) -> int:
        active = state.get('active_servers', 2)
        booting = state.get('booting_servers', 0)
        queue = state.get('queue', 0.0)
        arrival = state.get('arrival_rate', 0.0)
        step = state.get('step', 0)
        time_since_scale = state.get('time_since_scale', 100)

        self.history.append(arrival)
        if len(self.history) > 10:
            self.history.pop(0)

        total_capacity = active + booting
        cooldown_ok = time_since_scale >= 5

        # Calculate trend
        trend = 0.0
        if len(self.history) >= 3:
            recent = self.history[-3:]
            older = self.history[-6:-3] if len(self.history) >= 6 else [0.0] * (3 - len(self.history[-6:-3]))
            # Simple linear regression slope or just difference of averages
            avg_recent = sum(recent) / len(recent)
            avg_older = sum(older) / len(older) if older else 0.0
            trend = avg_recent - avg_older

        # Predicted load: current queue + expected arrivals vs capacity
        # Capacity per step is 3 * active
        capacity = active * 3
        
        # If we are already at max capacity, we can only hold or scale down
        if total_capacity >= 8:
            # Only scale down if queue is very low and load is decreasing
            if queue < 5 and trend < 0 and cooldown_ok and active > 2:
                self.last_scale_step = step
                self.last_action = 2
                return 2
            return 0

        # Scale up logic
        # 1. If queue is high, scale up immediately (if not in cooldown or if critical)
        # 2. If trend is strongly positive and queue is moderate, scale up proactively
        
        # Thresholds
        # If queue > 15, we are in danger. Scale up if possible.
        # If trend > 2.0 (arrivals increasing by 2 per step on average), scale up if queue > 5.
        
        should_scale_up = False
        
        # Critical: Queue is high
        if queue > 15:
            should_scale_up = True
        # Proactive: Trend is rising and queue is moderate
        elif trend > 1.5 and queue > 5:
            should_scale_up = True
        # Proactive: Trend is rising and capacity is low relative to arrival rate
        elif trend > 0.5 and arrival > capacity * 0.8 and queue > 2:
            should_scale_up = True

        if should_scale_up and cooldown_ok and total_capacity < 8:
            self.last_scale_step = step
            self.last_action = 1
            return 1

        # Scale down logic
        # Only scale down if:
        # 1. Queue is very low (< 3)
        # 2. Trend is negative or flat (not rising)
        # 3. Cooldown has passed
        # 4. We have more than 2 active servers (keep at least 2)
        # 5. No booting servers (don't scale down while scaling up)
        
        if queue < 3 and trend <= 0.5 and cooldown_ok and active > 2 and booting == 0:
            # Additional check: ensure we aren't about to get hit by a spike
            # If arrival rate is still high, don't scale down
            if arrival < capacity * 0.5:
                self.last_scale_step = step
                self.last_action = 2
                return 2

        # Default: Hold
        return 0