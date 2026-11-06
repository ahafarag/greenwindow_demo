"""Backtest GreenWindow on a year of Spanish grid data.

Train the forecaster on the first ~9 months, then replay a held-out period:
every job is planned with the forecast available at its release time and
charged with the carbon intensity that actually happened.
"""
import json
import sys

import numpy as np
import pandas as pd

from greenwindow.forecast import HORIZON, CarbonForecaster, persistence_forecast
from greenwindow.scheduler import best_hours, best_start, emissions_kg

SPLIT = pd.Timestamp("2026-08-01", tz="UTC")
rng = np.random.default_rng(42)

y = pd.read_csv("data/es_carbon_intensity_hourly.csv", parse_dates=["time_utc"],
                index_col="time_utc")["gco2_per_kwh"]
y = y.asfreq("h").interpolate(limit=6)
assert y.isna().sum() == 0, "unfilled gaps"

split = y.index.searchsorted(SPLIT)
train_issues = np.arange(170, split - HORIZON)
test_issues = np.arange(split, len(y) - HORIZON)

fc = CarbonForecaster().fit(y, train_issues)
pred = fc.predict_many(y, test_issues)
pers = persistence_forecast(y, test_issues)
actual = np.stack([y.to_numpy()[test_issues + h] for h in range(1, HORIZON + 1)], axis=1)

mae_model = np.abs(pred - actual).mean(axis=0)
mae_pers = np.abs(pers - actual).mean(axis=0)
print(f"Forecast MAE 1-48h: GreenWindow {mae_model.mean():.1f}  persistence {mae_pers.mean():.1f} gCO2/kWh")

# ---- job replay -------------------------------------------------------------
N = 4000
durations = rng.choice([1, 2, 4, 8], N, p=[.25, .3, .3, .15])
slacks = rng.choice([4, 8, 12, 24, 36], N)
slacks = np.minimum(slacks, HORIZON - durations)
rel = rng.integers(0, len(test_issues), N)
POWER = 10.0  # kW, roughly one 8-GPU training node

rows = []
for d, s, r in zip(durations, slacks, rel):
    t = test_issues[r]
    win = d + s
    act = y.to_numpy()[t + 1:t + 1 + win]
    curves = {"greenwindow": pred[r, :win], "persistence": pers[r, :win], "oracle": act}
    row = {"duration": int(d), "slack": int(s),
           "run_now": emissions_kg(act, np.arange(d), POWER)}
    for name, c in curves.items():
        st = best_start(c, d, s)
        row[name] = emissions_kg(act, np.arange(st, st + d), POWER)
        row[name + "_interruptible"] = emissions_kg(act, best_hours(c, d, win), POWER)
    rows.append(row)
jobs = pd.DataFrame(rows)

def saving(col):
    return 100 * (1 - jobs[col].sum() / jobs["run_now"].sum())

summary = {
    "test_period": [str(y.index[split].date()), str(y.index[-1].date())],
    "train_hours": int(split), "test_hours": int(len(y) - split),
    "mean_intensity_test": round(float(y.iloc[split:].mean()), 1),
    "forecast_mae_greenwindow": round(float(mae_model.mean()), 1),
    "forecast_mae_persistence": round(float(mae_pers.mean()), 1),
    "forecast_mae_h6": [round(float(mae_model[:6].mean()),1), round(float(mae_pers[:6].mean()),1)],
    "forecast_mae_by_h_greenwindow": [round(float(x), 1) for x in mae_model],
    "forecast_mae_by_h_persistence": [round(float(x), 1) for x in mae_pers],
    "jobs": N, "power_kw": POWER, "deadline_misses": 0,
    "run_now_t": round(jobs["run_now"].sum() / 1000, 2),
}
for col in ["persistence", "greenwindow", "oracle"]:
    summary[f"saving_pct_{col}"] = round(saving(col), 1)
    summary[f"saving_pct_{col}_interruptible"] = round(saving(col + "_interruptible"), 1)
summary["share_of_oracle"] = round(saving("greenwindow") / saving("oracle") * 100, 1)
by_slack = jobs.groupby("slack").apply(
    lambda g: pd.Series({"greenwindow": 100 * (1 - g.greenwindow.sum() / g.run_now.sum()),
                         "oracle": 100 * (1 - g.oracle.sum() / g.run_now.sum()),
                         "persistence": 100 * (1 - g.persistence.sum() / g.run_now.sum()),
                         "n": len(g)}), include_groups=False).round(1)
summary["saving_by_slack"] = by_slack.reset_index().to_dict(orient="records")
print(json.dumps({k: v for k, v in summary.items() if "by_h" not in k}, indent=1))

json.dump(summary, open("results/summary.json", "w"), indent=1)
jobs.to_csv("results/jobs.csv", index=False)
pd.DataFrame({"issue_time": y.index[test_issues], **{f"h{h}": pred[:, h - 1] for h in (1, 6, 24, 48)}}).to_csv(
    "results/forecasts_sample.csv", index=False)
import joblib; joblib.dump(fc, "results/forecaster.joblib")
