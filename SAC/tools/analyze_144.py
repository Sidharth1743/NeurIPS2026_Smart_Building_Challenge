#!/usr/bin/env python3
"""Analyse the 144-scenario scoring of the r14 checkpoints from their recordings.

Reads, for every checkpoint and building:
  results/<run>/scored{90,54}/<step>_b<b>.json   the scored month per scenario
  results/<run>/record{90,54}/<step>/building_train_<b>/parts.bin  per-step parts
  results/<run>/record{90,54}/<step>/building_train_<b>/rew.bin    per-step reward
and writes a markdown report: score decomposition, per building, per weather, the
scenarios that carry the sum, the time-of-day bins and the fuel split.

    SAC/.venv/bin/python SAC/tools/analyze_144.py --run r14_structured_200m_s75 \
        --steps 49972800 89951040 119934720 159912960 --out <md>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SAC = Path(__file__).resolve().parents[1]
ROOT = SAC.parent
PARTS = ["productivity", "energy_cost", "carbon", "comfort_dev_K", "worst_dev_K",
         "out_of_band_frac", "elec_kwh", "gas_kwh"]
SETS = {"90": ("eval5_heldout18", (1, 2, 3, 6, 8)), "54": ("eval3_heldout18", (4, 5, 7))}
BINS = [("night 19-06", 19, 6), ("morning 06-12", 6, 12), ("afternoon 12-16", 12, 16),
        ("evening 16-19", 16, 19)]


def minute_of_week(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Episode starts Sunday 06:00; one step is 5 minutes. Returns (weekday, minute)."""
    m = (t.astype(np.int64) * 5 + 6 * 60) % (7 * 1440)
    return m // 1440, m % 1440          # weekday 0 = Sunday


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    run = SAC / "results" / a.run

    rows, per_building, per_weather, per_bin, carriers, fuel = [], {}, {}, {}, {}, {}
    for step in a.steps:
        agg = {k: [] for k in ("score", "prod", "en", "carb", "cmf", "worst", "elec", "gas")}
        for kind, (setname, buildings) in SETS.items():
            for b in buildings:
                jf = run / f"scored{kind}" / f"{step}_b{b}.json"
                if not jf.is_file():
                    continue
                d = json.loads(jf.read_text())
                names = sorted(d["scenarios"])
                rec = run / f"record{kind}" / str(step) / f"building_train_{b}"
                p = np.fromfile(rec / "parts.bin", dtype=np.float64).reshape(-1, len(names), 8) \
                    if (rec / "parts.bin").is_file() else None
                for i, n in enumerate(names):
                    s = d["scenarios"][n]
                    rows.append((step, kind, b, n, s["mean_reward"], s["productivity"],
                                 s["energy_cost"], s["carbon"]))
                    per_building.setdefault(step, {}).setdefault(b, []).append(abs(s["mean_reward"]))
                    w = n.split("__")[1].rsplit("_", 1)[0]
                    per_weather.setdefault(step, {}).setdefault(w, []).append(abs(s["mean_reward"]))
                    if p is not None:
                        pa = p[:, i, :]
                        agg["prod"].append(pa[:, 0].mean()); agg["en"].append(pa[:, 1].mean())
                        agg["carb"].append(pa[:, 2].mean()); agg["cmf"].append(pa[:, 3].mean())
                        agg["worst"].append(pa[:, 4].max())
                        agg["elec"].append(pa[:, 6].sum()); agg["gas"].append(pa[:, 7].sum())
                        t = np.arange(p.shape[0])
                        wd, mn = minute_of_week(t)
                        weekend = wd >= 5
                        for name, lo, hi in BINS:
                            sel = ((mn >= lo) & (mn < hi)) if lo < hi else ((mn >= lo) | (mn < hi))
                            if not sel.any():
                                continue
                            per_bin.setdefault(step, {}).setdefault(name, []).append(
                                (pa[sel, 0].mean(), pa[sel, 1].mean(), pa[sel, 2].mean()))
                        sel = weekend & (mn >= 6 * 60) & (mn < 19 * 60)
                        if sel.any():
                            per_bin.setdefault(step, {}).setdefault("weekend 06-19", []).append(
                                (pa[sel, 0].mean(), pa[sel, 1].mean(), pa[sel, 2].mean()))
        sc = [abs(r[4]) for r in rows if r[0] == step]
        agg["score"] = float(np.mean(sc))
        fuel[step] = (float(np.mean(agg["elec"])) if agg["elec"] else float("nan"),
                      float(np.mean(agg["gas"])) if agg["gas"] else float("nan"))
        carriers[step] = agg

    out = [f"# {a.run} — 144-scenario analysis (from the recordings)\n",
           "Score = mean over scenarios of the return; lower is better. Terms are the fleet means "
           "of the per-step parts over all 8640 steps x agents.\n"]
    out.append("## Headline\n")
    out.append("| step | 144 score | productivity | energy | carbon | comfort dev K | worst zone K | elec kWh | gas kWh |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for step in a.steps:
        c = carriers[step]
        e, g = fuel[step]
        out.append(f"| {step:,} | **{c['score']:.6f}** | {np.mean(c['prod']):+.4f} | "
                   f"{np.mean(c['en']):.4f} | {np.mean(c['carb']):.4f} | {np.mean(c['cmf']):.3f} | "
                   f"{np.max(c['worst']):.1f} | {e:.0f} | {g:.0f} |")

    out.append("\n## Per building\n")
    bs = sorted({b for s in per_building.values() for b in s})
    out.append("| step | " + " | ".join(f"train_{b}" for b in bs) + " |")
    out.append("|---|" + "---|" * len(bs))
    for step in a.steps:
        out.append(f"| {step:,} | " + " | ".join(
            f"{np.mean(per_building[step][b]):.4f}" if b in per_building[step] else "-"
            for b in bs) + " |")

    out.append("\n## Per weather (mean over the buildings/weathers available)\n")
    ws = sorted({w for s in per_weather.values() for w in s})
    out.append("| step | " + " | ".join(ws) + " |")
    out.append("|---|" + "---|" * len(ws))
    for step in a.steps:
        out.append(f"| {step:,} | " + " | ".join(
            f"{np.mean(per_weather[step][w]):.4f}" if w in per_weather[step] else "-" for w in ws) + " |")

    out.append("\n## Scenarios that carry the sum (top 10 at the final step)\n")
    last = a.steps[-1]
    worst = sorted([r for r in rows if r[0] == last], key=lambda r: -abs(r[4]))[:10]
    out.append("| scenario | mean reward | productivity | energy | carbon |")
    out.append("|---|---|---|---|---|")
    for r in worst:
        out.append(f"| {r[3]} | {abs(r[4]):.4f} | {r[5]:+.4f} | {r[6]:.4f} | {r[7]:.4f} |")

    out.append("\n## Time of day (fleet means of the per-step parts)\n")
    out.append("| bin | " + " | ".join(f"{step:,}" for step in a.steps) + " |")
    out.append("|---|" + "---|" * len(a.steps))
    for name in [b[0] for b in BINS] + ["weekend 06-19"]:
        cells = []
        for step in a.steps:
            v = per_bin.get(step, {}).get(name)
            cells.append(f"{0.6*np.mean([x[0] for x in v]) + 0.2*np.mean([x[1] for x in v]) + 0.2*np.mean([x[2] for x in v]):+.4f}"
                         if v else "-")
        out.append(f"| {name} | " + " | ".join(cells) + " |")

    out.append("\n(cell value = 0.6*productivity + 0.2*energy + 0.2*carbon for that bin, "
               "so it is the score's arithmetic on the fleet means, not a per-scenario mean.)\n")
    a.out.write_text("\n".join(out) + "\n")
    print(f"wrote {a.out}  ({len(rows)} scenario rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
