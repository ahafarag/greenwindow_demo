"""Deadline-aware carbon scheduler.

A job has a duration (hours), an average power draw (kW), a release time and
a deadline. GreenWindow picks the start hour inside [release, deadline - duration]
whose forecasted carbon intensity over the run is lowest. Deadlines are hard
constraints: a job is never scheduled to finish after its deadline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Job:
    duration: int        # hours
    power_kw: float      # average draw
    slack: int           # hours between release and deadline beyond the duration

    @property
    def window(self) -> int:
        return self.duration + self.slack


def best_start(curve: np.ndarray, duration: int, latest_start: int) -> int:
    """Offset (0..latest_start) minimising the sum of `curve` over the run."""
    cs = np.concatenate([[0.0], np.cumsum(curve)])
    sums = cs[duration:duration + latest_start + 1] - cs[0:latest_start + 1]
    return int(np.argmin(sums))


def best_hours(curve: np.ndarray, duration: int, window: int) -> np.ndarray:
    """Interruptible jobs (checkpointed training): the `duration` cheapest hours."""
    return np.sort(np.argsort(curve[:window])[:duration])


def emissions_kg(actual: np.ndarray, hours: np.ndarray, power_kw: float) -> float:
    """actual in gCO2/kWh, one value per hour -> kg CO2."""
    return float(actual[hours].sum() * power_kw / 1000.0)
