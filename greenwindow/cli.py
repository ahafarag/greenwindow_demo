"""greenwindow plan --duration 4 --deadline-hours 24 [--power-kw 10]

Prints the lowest-carbon start time for a job, using the trained forecaster
and the latest observed grid data.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from greenwindow.scheduler import best_start


def main():
    ap = argparse.ArgumentParser(prog="greenwindow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--duration", type=int, required=True, help="job length in hours")
    p.add_argument("--deadline-hours", type=int, required=True, help="must finish within N hours")
    p.add_argument("--power-kw", type=float, default=10.0)
    p.add_argument("--data", default="data/es_carbon_intensity_hourly.csv")
    p.add_argument("--model", default="results/forecaster.joblib")
    a = ap.parse_args()

    y = pd.read_csv(a.data, parse_dates=["time_utc"], index_col="time_utc")["gco2_per_kwh"]
    y = y.asfreq("h").interpolate(limit=6).dropna()
    fc = joblib.load(a.model)
    t = len(y) - 1
    curve = fc.predict_from(y, t)[: a.deadline_hours]
    slack = a.deadline_hours - a.duration
    if slack < 0:
        raise SystemExit("deadline shorter than job duration")
    s = best_start(curve, a.duration, slack)
    now_kg = curve[: a.duration].sum() * a.power_kw / 1000
    best_kg = curve[s: s + a.duration].sum() * a.power_kw / 1000
    start = (y.index[t] + pd.Timedelta(hours=s + 1)).tz_convert("Europe/Madrid")
    print(f"Data up to:      {y.index[t].tz_convert('Europe/Madrid'):%Y-%m-%d %H:%M} (Madrid)")
    print(f"Best start:      {start:%Y-%m-%d %H:%M} (Madrid), in {s + 1} h")
    print(f"Forecast CO2:    {best_kg:.1f} kg vs {now_kg:.1f} kg if started now "
          f"({100 * (1 - best_kg / now_kg):.0f}% less)")


if __name__ == "__main__":
    main()
