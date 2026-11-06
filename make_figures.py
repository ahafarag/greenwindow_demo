"""Figures for the README / Devpost gallery (3:2, 1500x1000)."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from greenwindow.scheduler import best_start

GREEN, GREY, DARK, RED, BLUE = "#1f9d55", "#9aa5b1", "#1f2933", "#d64545", "#3a6ea5"
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 19, "figure.dpi": 100})

y = pd.read_csv("data/es_carbon_intensity_hourly.csv", parse_dates=["time_utc"],
                index_col="time_utc")["gco2_per_kwh"].asfreq("h").interpolate(limit=6)
S = json.load(open("results/summary.json"))
fc = joblib.load("results/forecaster.joblib")
yl = y.tz_convert("Europe/Madrid")

# 1. hero: last 5 days + 48h forecast with the chosen window
t = len(y) - 1
f = fc.predict_from(y, t)
fut = yl.index[t] + pd.to_timedelta(np.arange(1, 49), "h")
dur, dl = 6, 36
s = best_start(f[:dl], dur, dl - dur)
fig, ax = plt.subplots(figsize=(15, 10))
hist = yl.iloc[-24 * 5:]
ax.plot(hist.index, hist.values, color=DARK, lw=2.2, label="Observed grid intensity")
ax.plot(fut, f, color=GREEN, lw=2.6, ls="--", label="GreenWindow 48h forecast")
ax.axvspan(fut[s] - pd.Timedelta("30min"), fut[s + dur - 1] + pd.Timedelta("30min"),
           color=GREEN, alpha=.18, label=f"Chosen window for a {dur}h job (deadline {dl}h)")
ax.axvspan(fut[0] - pd.Timedelta("30min"), fut[dur - 1] + pd.Timedelta("30min"),
           color=RED, alpha=.10, label="Run-now window")
ax.axvline(yl.index[t], color=GREY, lw=1)
ax.set_ylabel("gCO₂ / kWh")
ax.set_title("Spain's grid: carbon intensity swings 2-3x within a day")
ax.legend(loc="upper left", frameon=False, fontsize=13)
fig.tight_layout(); fig.savefig("docs/fig1_forecast_window.png"); plt.close(fig)

# 2. savings by slack
sl = pd.DataFrame(S["saving_by_slack"])
fig, ax = plt.subplots(figsize=(15, 10))
x = np.arange(len(sl)); w = .27
ax.bar(x - w, sl.persistence, w, color=GREY, label="Seasonal-naive forecast")
ax.bar(x, sl.greenwindow, w, color=GREEN, label="GreenWindow forecast")
ax.bar(x + w, sl.oracle, w, color=DARK, alpha=.35, label="Perfect foresight (upper bound)")
for i, v in enumerate(sl.greenwindow):
    ax.text(i, v + .8, f"{v:.0f}%", ha="center", color=GREEN, fontweight="bold")
ax.set_xticks(x, [f"{int(v)} h" for v in sl.slack])
ax.set_xlabel("Deadline slack (hours a job may wait)")
ax.set_ylabel("CO₂ saved vs. running immediately (%)")
ax.set_title(f"4,000 replayed jobs, Aug-Nov 2026: {S['saving_pct_greenwindow']:.0f}% less CO₂, zero missed deadlines")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("docs/fig2_savings_by_slack.png"); plt.close(fig)

# 3. heatmap hour x month
hm = yl.groupby([yl.index.month, yl.index.hour]).mean().unstack()
order = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
hm = hm.reindex(order)
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(hm.values, aspect="auto", cmap="RdYlGn_r")
ax.set_yticks(range(12), ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"])
ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
ax.set_xlabel("Hour of day (Madrid)")
ax.set_title("Average carbon intensity, Nov 2025 - Nov 2026 (gCO₂/kWh)")
fig.colorbar(im, ax=ax, fraction=.03)
fig.tight_layout(); fig.savefig("docs/fig3_heatmap.png"); plt.close(fig)

# 4. MAE by horizon
fig, ax = plt.subplots(figsize=(15, 10))
h = np.arange(1, 49)
ax.plot(h, S["forecast_mae_by_h_persistence"], color=GREY, lw=2.5, label=f"Seasonal-naive (MAE {S['forecast_mae_persistence']})")
ax.plot(h, S["forecast_mae_by_h_greenwindow"], color=GREEN, lw=3, label=f"GreenWindow hybrid (MAE {S['forecast_mae_greenwindow']})")
ax.set_xlabel("Forecast horizon (hours ahead)"); ax.set_ylabel("Mean absolute error (gCO₂/kWh)")
ax.set_title("Held-out forecast error, Aug-Nov 2026")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("docs/fig4_forecast_error.png"); plt.close(fig)
print("figures written")
