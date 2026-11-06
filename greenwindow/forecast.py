"""Carbon-intensity forecasting for the Spanish grid.

Direct multi-horizon model: one gradient-boosted regressor predicts the
intensity at T = t + h for every horizon h in 1..H, using only information
available at issue time t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZON = 48


def _features(y: pd.Series, t_idx: np.ndarray, h: int) -> np.ndarray:
    """Features for target index t+h, built only from y[:t+1]."""
    v = y.to_numpy()
    idx = y.index
    T = t_idx + h
    k = int(np.ceil(h / 24))  # most recent same-hour observation that is <= t
    same_hour = v[T - 24 * k]
    week_ago = v[T - 168]
    last = v[t_idx]
    d1 = last - v[t_idx - 1]
    cs = np.concatenate([[0.0], np.cumsum(v)])
    mean24 = (cs[t_idx + 1] - cs[t_idx - 23]) / 24
    mean168 = (cs[t_idx + 1] - cs[t_idx - 167]) / 168
    ts = idx[np.minimum(T, len(idx) - 1)]
    # hour of target in local Spanish time (solar/demand patterns follow local clock)
    local = ts.tz_convert("Europe/Madrid")
    hod = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    doy = local.dayofyear.to_numpy()
    # level-free features: everything relative to the trailing 24h mean, so the
    # model transfers across seasons when the overall grid mix shifts
    return np.column_stack([
        np.full(len(t_idx), h), hod, dow,
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        last / mean24, d1 / mean24, mean168 / mean24,
        same_hour / mean24, week_ago / mean24,
    ]), mean24


def build_dataset(y: pd.Series, issue_idx: np.ndarray, horizon: int = HORIZON):
    X, Y, meta = [], [], []
    for h in range(1, horizon + 1):
        ok = issue_idx[issue_idx + h < len(y)]
        f, lvl = _features(y, ok, h)
        X.append(f)
        Y.append(y.to_numpy()[ok + h] / lvl)
        meta.append(np.column_stack([ok, np.full(len(ok), h)]))
    return np.vstack(X), np.concatenate(Y), np.vstack(meta)


class CarbonForecaster:
    """Hybrid forecaster: gradient-boosted correction blended with the
    seasonal-naive (same hour, last observed day) forecast. In backtests the
    seasonal-naive signal is very strong on the solar-driven Spanish grid, so the
    blend weight keeps it as an anchor."""

    def __init__(self, horizon: int = HORIZON, blend: float = 0.5):
        self.horizon = horizon
        self.blend = blend
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=400, learning_rate=0.05,
            max_leaf_nodes=31, categorical_features=[2], random_state=0)

    def fit(self, y: pd.Series, issue_idx: np.ndarray):
        X, Y, _ = build_dataset(y, issue_idx, self.horizon)
        self.model.fit(X, Y)
        return self

    def predict_from(self, y: pd.Series, t: int) -> np.ndarray:
        """Forecast y[t+1 .. t+H] given history up to and including index t."""
        out = np.empty(self.horizon)
        for h in range(1, self.horizon + 1):
            f, lvl = _features(y, np.array([t]), h)
            out[h - 1] = self.model.predict(f)[0] * lvl[0]
        naive = persistence_forecast(y, np.array([t]), self.horizon)[0]
        return self.blend * out + (1 - self.blend) * naive

    def predict_many(self, y: pd.Series, issues: np.ndarray) -> np.ndarray:
        """Vectorised forecasts, shape (len(issues), H)."""
        out = np.empty((len(issues), self.horizon))
        for h in range(1, self.horizon + 1):
            f, lvl = _features(y, issues, h)
            out[:, h - 1] = self.model.predict(f) * lvl
        naive = persistence_forecast(y, issues, self.horizon)
        return self.blend * out + (1 - self.blend) * naive


def persistence_forecast(y: pd.Series, issues: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """Baseline: same hour on the most recent observed day."""
    v = y.to_numpy()
    out = np.empty((len(issues), horizon))
    for h in range(1, horizon + 1):
        k = int(np.ceil(h / 24))
        out[:, h - 1] = v[issues + h - 24 * k]
    return out
