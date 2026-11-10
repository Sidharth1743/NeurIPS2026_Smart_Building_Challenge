#!/usr/bin/env python3
"""Per-step kernel composition and stream overlap of a gate run (nsys `cuda_gpu_trace` CSV).

Why: `tools/bench.sh --nsys <blob> <nenv>` writes a kernel *summary* (total time per kernel over
the whole run).  What matters for the step's critical path is the *per-step* average and whether the
stream2 convection really hides behind the main stream, which the summary cannot show.  This script
assigns every kernel to a step window (period detected from the kernel that repeats most regularly,
i.e. the step loop) and reports:

  * main-stream busy time (union of kernel intervals, per step),
  * stream2 (convection) time and how much of it overlaps main-stream kernels,
  * per-kernel average per step, sorted.

Device/host copies are reported separately: the gate compares against the oracle every step, so its
D2H copy (~25 ms at 256 envs) is a gate artifact, not trainer work.  Measurement only, no GPU needed.

Usage:
  tools/bench.sh --nsys building_train_1__fairbanks_1 256          # writes results/perf/nsys_*.nsys-rep
  docker run --rm -v "$PWD":/cp -w /cp pufferai/puffertank:5.0 \
      bash -lc "nsys stats -r cuda_gpu_trace --format csv /cp/results/perf/nsys_<...>.nsys-rep" > t.csv
  tools/timeline.py t.csv [warmup_steps]
"""
import csv
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        hdr = None
        for row in rd:  # nsys stats prints progress lines before the CSV header
            if row and any("Duration (ns)" in c for c in row):
                hdr = row
                break
        if hdr is None:
            raise SystemExit("timeline.py: no CSV header found (is this a cuda_gpu_trace report?)")
        li = {h.strip().lower(): i for i, h in enumerate(hdr)}
        cs, cd, cn = li["start (ns)"], li["duration (ns)"], li["name"]
        for r in rd:
            if len(r) <= cn:
                continue
            try:
                s, d = int(r[cs]), int(r[cd])
            except ValueError:
                continue
            if d > 0:
                rows.append((s, d, r[cn]))
    rows.sort()
    return rows


def union(iv):
    iv = sorted(iv)
    tot, cs_, ce = 0, None, 0
    for s, e in iv:
        if cs_ is None:
            cs_, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs_
            cs_, ce = s, e
    return tot + (ce - cs_ if cs_ is not None else 0)


def overlap(a, b):
    tot = 0
    for s, e in a:
        for ms, me in b:
            if me <= s:
                continue
            if ms >= e:
                break
            tot += min(e, me) - max(s, ms)
    return tot


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = load(sys.argv[1])
    skip = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    starts = defaultdict(list)
    for s, d, n in rows:
        starts[n].append(s)
    best = None                     # (period, name, count): the most regular once-per-step kernel
    for n, t in starts.items():
        if len(t) < 20:
            continue
        t.sort()
        gaps = [b - a for a, b in zip(t, t[1:])]
        if max(gaps) - min(gaps) > 0.05 * min(gaps):
            continue
        if best is None or len(t) > best[2]:
            best = ((min(gaps) + max(gaps)) // 2, n, len(t))
    if best is None:
        raise SystemExit("timeline.py: no regular per-step kernel found (not a step loop trace?)")
    period, marker, cnt = best
    print(f"step period {period/1e6:.3f} ms from {cnt} x {marker}\n")

    t0 = min(starts[marker])
    n_steps = int((max(s + d for s, d, n in rows) - t0) / period)
    is_conv = lambda n: "convect2" in n
    is_copy = lambda n: "memcpy" in n.lower() or "memset" in n.lower()

    steps = []
    for i in range(n_steps):
        a, b = t0 + i * period, t0 + (i + 1) * period
        per_k, iv_main, iv_conv, cp = defaultdict(int), [], [], 0
        for s, d, n in rows:
            if s >= b or s + d <= a:
                continue
            ss, ee = max(s, a), min(s + d, b)
            if is_copy(n):
                cp += ee - ss
            else:
                per_k[n] += ee - ss
                (iv_conv if is_conv(n) else iv_main).append((ss, ee))
        if per_k:
            steps.append((per_k, iv_main, iv_conv, cp))
    st = steps[skip:] or steps
    avg = lambda xs: sum(xs) / max(1, len(xs))

    main_t = [union(x[1]) for x in st]
    conv_t = [union(x[2]) for x in st]
    both_t = [overlap(sorted(x[1]), sorted(x[2])) for x in st]
    print(f"{'main-stream busy (kernels, union)':38s} {avg(main_t)/1e6:8.3f} ms")
    print(f"{'stream2 (convection)':38s} {avg(conv_t)/1e6:8.3f} ms")
    print(f"{'  of it overlapping main-stream kernels':38s} {avg(both_t)/1e6:8.3f} ms"
          f" ({100*avg(both_t)/max(1e-9, avg(conv_t)):.0f}%)")
    print(f"{'GPU busy (main + conv - overlap)':38s} {(avg(main_t)+avg(conv_t)-avg(both_t))/1e6:8.3f} ms")
    print(f"{'copies per step (gate oracle compare)':38s} {avg([x[3] for x in st])/1e6:8.3f} ms\n")
    print(f"per-kernel average per step (ms, {len(st)} steps):")
    tot = defaultdict(list)
    for per_k, *_ in st:
        for k, v in per_k.items():
            tot[k].append(v)
    for k, v in sorted(tot.items(), key=lambda kv: -sum(kv[1])):
        if avg(v) / 1e6 < 0.02:
            continue
        print(f"  {avg(v)/1e6:7.3f}  {k[:76]}{' [stream2]' if is_conv(k) else ''}")


if __name__ == "__main__":
    main()
