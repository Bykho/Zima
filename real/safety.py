"""Proprioceptive watchdog: ramp, limits, load, tracking error, estop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyConfig:
    max_delta_ticks: int = 12
    max_tracking_error: int = 120
    max_load: int = 850
    fault_streak: int = 6

    @classmethod
    def from_dict(cls, d: dict | None) -> SafetyConfig:
        d = d or {}
        return cls(
            max_delta_ticks=int(d.get("max_delta_ticks", 12)),
            max_tracking_error=int(d.get("max_tracking_error", 120)),
            max_load=int(d.get("max_load", 850)),
            fault_streak=int(d.get("fault_streak", 6)),
        )


class Watchdog:
    def __init__(self, cfg: SafetyConfig, limits: dict[str, tuple[int, int]]):
        self.cfg = cfg
        self.limits = limits
        self.last_cmd: dict[str, int] = {}
        self.streak = 0

    def seed(self, present: dict[str, int]):
        self.last_cmd = dict(present)

    def filter(self, desired: dict[str, int]) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, target in desired.items():
            lo, hi = self.limits.get(name, (0, 4095))
            t = max(lo, min(hi, int(target)))
            if name in self.last_cmd:
                delta = t - self.last_cmd[name]
                cap = self.cfg.max_delta_ticks
                if delta > cap:
                    t = self.last_cmd[name] + cap
                elif delta < -cap:
                    t = self.last_cmd[name] - cap
            out[name] = t
        self.last_cmd = dict(out)
        return out

    def check(
        self,
        cmd: dict[str, int],
        present: dict[str, int],
        load: dict[str, int],
    ) -> str | None:
        for name in cmd:
            err = abs(present.get(name, cmd[name]) - cmd[name])
            if err > self.cfg.max_tracking_error:
                return f"{name}: tracking {err} ticks (goal {cmd[name]}, present {present.get(name)})"
            lv = abs(int(load.get(name, 0)))
            if lv > self.cfg.max_load:
                return f"{name}: load {lv} (max {self.cfg.max_load})"
        return None

    def note(self, fault: str | None) -> bool:
        """Return True when fault streak hits estop threshold."""
        if fault:
            self.streak += 1
            if self.streak >= self.cfg.fault_streak:
                return True
        else:
            self.streak = 0
        return False

    @property
    def last_fault(self) -> str | None:
        return getattr(self, "_last_fault", None)

    def set_fault(self, msg: str):
        self._last_fault = msg
