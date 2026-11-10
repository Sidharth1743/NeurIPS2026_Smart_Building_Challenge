#!/usr/bin/env python3
"""Summarize training runs: agents, blobs, envs/scenario, solver mode, epoch SPS, us/env-step, VRAM.

usage: tools/run_stats.py results/runs/r4_rot_s0*  (directories or .out prefixes)
Reads args.txt (the ./puffer args), train.log (the env's startup line + VRAM) and tb/ (SPS series).
The last SPS epoch is dropped: a run's final epoch is cut off and reports a bogus rate.
"""
import re
import statistics
import sys
from pathlib import Path

TBREAD = Path(__file__).resolve().parent / "tbread.py"


def sps_series(tb: Path) -> list[float]:
    """Epoch SPS values from TensorBoard events (tbread.py owns the format)."""
    import subprocess
    try:
        out = subprocess.run([sys.executable, str(TBREAD), str(tb), "SPS"], capture_output=True, text=True).stdout
    except Exception:
        return []
    vals = []
    for tok in out.split(":", 1)[-1].split():
        if ":" in tok:
            try:
                vals.append(float(tok.split(":")[1]))
            except ValueError:
                pass
    return vals


def stats(d: Path) -> dict:
    a = (d / "args.txt").read_text() if (d / "args.txt").exists() else ""
    log = (d / "train.log").read_text(errors="ignore").replace("\r", "\n") if (d / "train.log").exists() else ""
    g = lambda pat, s=log: (re.search(pat, s).group(1) if re.search(pat, s) else "")
    agents = int(m.group(1)) if (m := re.search(r"total_agents=(\d+)", a)) else 0
    blobs = len(list(re.findall(r"weather \d+: ", log))) or 0
    mode = g(r"value mode (\d)") or ("2" if "weather per slab" in log else "?")
    rpl = g(r"(\d+) env\(s\)/lane")
    vram = g(r"VRAM:\s*([\d.]+)/24G")
    per_env = g(r"(\d+) envs?, episode length") or g(r": (\d+) envs,")
    vals = [v for v in sps_series(d / "tb") if v > 0]
    vals = vals[:-1] if len(vals) > 1 else vals
    sps = statistics.median(vals) if vals else 0.0
    return {
        "run": d.name, "set": g(r"blob_dir=/art/sets/(\w+)") or g(r"blob=/art/blobs/(\S+?)\.sbb"),
        "agents": agents, "scenarios": blobs, "per_scenario": per_env, "mode": mode, "rpl": rpl,
        "sps": sps, "ms_step": (agents * 1e3 / sps if sps else 0.0), "us_env_step": (1e6 / sps if sps else 0.0),
        "vram": vram, "sps_series": vals,
    }


def main() -> int:
    rows = [stats(Path(p)) for p in sys.argv[1:]]
    hdr = f"{'run':22s} {'set':10s} {'agents':>7s} {'scen':>5s} {'env/scen':>8s} {'mode':>4s} {'rpl':>4s} {'SPS':>8s} {'ms/step':>8s} {'us/env-step':>11s} {'VRAM':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:22s} {str(r['set'])[:10]:10s} {r['agents']:7d} {r['scenarios']:5d} "
              f"{str(r['per_scenario']):>8s} {str(r['mode']):>4s} {str(r['rpl']):>4s} {r['sps']:8.0f} "
              f"{r['ms_step']:8.1f} {r['us_env_step']:11.1f} {str(r['vram']):>5s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
