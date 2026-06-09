"""
make_figure.py  --  Render the headline result to assets/results.png.

Uses a fixed pre-tournament snapshot (June 2026) of the four venues' P(advance to R32)
for the five teams where Polymarket and the recreational book (DraftKings) diverged most.
Snapshot, not a live pull, so the figure is deterministic and reproducible.

  python3 make_figure.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- captured snapshot: P(advance from group / reach R32), percent ----
TEAMS = ["Saudi Arabia", "Qatar", "DR Congo", "Ghana", "New Zealand"]
MODEL = np.array([29.7, 17.1, 39.9, 49.9, 31.3])   # internal Poisson MC (derived from PM)
PM    = np.array([34.8, 21.2, 43.4, 51.4, 32.3])   # Polymarket (devigged to 32)
DK    = np.array([47.6, 30.7, 51.9, 57.8, 38.0])   # DraftKings, recreational (devigged to 32)
PIN   = np.array([35.4, 22.6, 44.3, 49.6, 35.6])   # Pinnacle, true sharp (two-way devig)

C = {"model": "#9ca3af", "pm": "#2563eb", "dk": "#ea580c", "pin": "#16a34a"}

def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2.1, 1]})

    # ---- Panel A: grouped bars per team ----
    x = np.arange(len(TEAMS)); w = 0.2
    axL.bar(x - 1.5*w, MODEL, w, label="Internal model", color=C["model"])
    axL.bar(x - 0.5*w, PM,    w, label="Polymarket",     color=C["pm"])
    axL.bar(x + 0.5*w, DK,    w, label="DraftKings (recreational)", color=C["dk"])
    axL.bar(x + 1.5*w, PIN,   w, label="Pinnacle (sharp)", color=C["pin"])
    axL.set_xticks(x); axL.set_xticklabels(TEAMS, rotation=15, ha="right")
    axL.set_ylabel("P(advance to Round of 32)  [%]")
    axL.set_title("Four venues, same market: the five most-divergent underdogs", fontsize=11, weight="bold")
    axL.legend(frameon=False, fontsize=9, ncol=2)
    axL.grid(axis="y", alpha=0.25); axL.set_axisbelow(True)
    axL.spines[["top", "right"]].set_visible(False)

    # ---- Panel B: mean absolute distance from the sharp (Pinnacle) ----
    md = {"Polymarket": np.mean(np.abs(PM - PIN)),
          "Internal\nmodel": np.mean(np.abs(MODEL - PIN)),
          "DraftKings": np.mean(np.abs(DK - PIN))}
    names = list(md); vals = [md[k] for k in names]
    cols = [C["pm"], C["model"], C["dk"]]
    bars = axR.bar(names, vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        axR.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.1f}", ha="center", fontsize=10, weight="bold")
    axR.set_ylabel("mean |venue − Pinnacle|  [pp]")
    axR.set_title("Distance from the sharp", fontsize=11, weight="bold")
    axR.grid(axis="y", alpha=0.25); axR.set_axisbelow(True)
    axR.spines[["top", "right"]].set_visible(False)
    axR.set_ylim(0, max(vals) * 1.25)

    fig.suptitle("Polymarket sits on top of the sharp; DraftKings carries the favourite-longshot bias",
                 fontsize=13, weight="bold")
    fig.text(0.5, 0.005,
             "Pinnacle (true sharp) agrees with Polymarket to 1.6pp and disagrees with the recreational "
             "book by 7.7pp → Polymarket's advance market is fair → no edge.",
             ha="center", fontsize=9, color="#444")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs("assets", exist_ok=True)
    out = os.path.join("assets", "results.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
