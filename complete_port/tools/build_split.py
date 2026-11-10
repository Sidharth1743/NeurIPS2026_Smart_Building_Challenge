#!/usr/bin/env python3
"""Build the training/evaluation split the r8 chain trains and scores on.

Option A (80/20): 36 training buildings, 9 evaluation buildings.  Evaluation buildings never appear
in any training set (enforced by tools/check_split.py before every chain segment), and the sealed
seed range (30000-30019) is excluded everywhere.

Sources (first hit wins per building/weather), all under complete_port/artifacts/:
  train/                     the 30 shipped scenarios (bake_training_set.sh)
  splits/gen-train/blobs/    generated training pool
  splits/gen-val/blobs/      generated validation pool (option A: eval9 only; the rest trains)
  splits/gen-val-s/blobs/    generated small-building validation pool (same)
  blobs/                     single scenarios (shipped shapes, 300xx pool, 05xx pool)
  sets/*/                    fallback

Writes:
  research/splits/train36.json   manifest
  research/splits/eval9.json     manifest
  artifacts/sets/tr36_s1..s6     six balanced training segments (relative symlinks)
  artifacts/sets/evalcore3       per-segment score set (eval9 subset, one large + one small + 20003)
  artifacts/sets/eval9           full evaluation set
  artifacts/sets/eval9a, eval9b  the two halves of the final score (one per GPU)

The set dirs hold relative symlinks into the canonical blob locations; both live under artifacts/,
so they resolve inside the container where artifacts/ is mounted as /art.  Idempotent: each set dir
is wiped and rebuilt.
"""
import json
import os
import sys
from datetime import datetime

CP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # complete_port/
REPO = os.path.dirname(CP)
ART = os.path.join(CP, "artifacts")
SETS = os.path.join(ART, "sets")
OUTDIR = os.path.join(REPO, "research", "splits")

# The two decisions of option A.  Everything else follows from the usable blob inventory.
EVAL9 = ["building_20001", "building_20003", "building_20004",
         "building_20009", "building_20011", "building_20012",
         "building_25005", "building_25011", "building_25014"]
CORE3 = ["building_20003", "building_20009", "building_25014"]
SEALED = range(30000, 30020)                     # holdout seeds: never in the repo's pipelines
UNLOADABLE = {"30028", "30021", "0500", "30025"}  # rejected by the loader (see README)
NSEG = 6

SRC_ORDER = [
    ("train", os.path.join(ART, "train")),
    ("gen-train", os.path.join(ART, "splits", "gen-train", "blobs")),
    ("gen-val", os.path.join(ART, "splits", "gen-val", "blobs")),
    ("gen-val-s", os.path.join(ART, "splits", "gen-val-s", "blobs")),
    ("blobs", os.path.join(ART, "blobs")),
]


def building_of(stem):
    return stem.split("__", 1)[0]


def is_sealed(b):
    tail = b[len("building_"):] if b.startswith("building_") else b
    return tail.isdigit() and int(tail) in SEALED


def inventory():
    """(building, weather) -> (blob_path, source_name), first source hit wins."""
    inv, srcs = {}, []
    for name, root in SRC_ORDER:
        if not os.path.isdir(root):
            continue
        srcs.append((name, root))
        for f in sorted(os.listdir(root)):
            if not f.endswith(".sbb"):
                continue
            stem = f[:-4]
            if "__" not in stem:
                continue
            b, w = stem.split("__", 1)
            inv.setdefault((b, w), (os.path.join(root, f), name))
    # fallback: any set dir (never overrides a canonical source)
    if os.path.isdir(SETS):
        for d in sorted(os.listdir(SETS)):
            root = os.path.join(SETS, d)
            if not os.path.isdir(root):
                continue
            for f in sorted(os.listdir(root)):
                if not f.endswith(".sbb"):
                    continue
                stem = f[:-4]
                if "__" not in stem:
                    continue
                b, w = stem.split("__", 1)
                p = os.path.join(root, f)
                inv.setdefault((b, w), (os.path.realpath(p), "sets/" + d))
    return inv, srcs


def main():
    inv, srcs = inventory()
    buildings = {}
    for (b, w), (p, src) in inv.items():
        buildings.setdefault(b, {})[w] = (p, src)

    usable, dropped = [], {}
    for b in sorted(buildings):
        tail = b[len("building_"):] if b.startswith("building_") else b
        if is_sealed(b):
            dropped[b] = "sealed"
        elif tail in UNLOADABLE:
            dropped[b] = "unloadable"
        else:
            usable.append(b)

    missing = [b for b in EVAL9 if b not in buildings]
    if missing:
        sys.exit("eval9 buildings without blobs: %s" % ", ".join(missing))
    train36 = [b for b in usable if b not in EVAL9]
    if len(usable) != 45 or len(train36) != 36:
        sys.exit("expected 45 usable / 36 training buildings, found %d / %d\nusable: %s"
                 % (len(usable), len(train36), ", ".join(usable)))
    if dropped:
        print("dropped:", ", ".join("%s(%s)" % (b, r) for b, r in sorted(dropped.items())))

    def scen(b):
        return [w for w in sorted(buildings[b])]

    def size(b):
        return sum(os.path.getsize(buildings[b][w][0]) for w in scen(b))

    # six balanced training segments: snake draft over buildings sorted by total blob bytes
    order = sorted(train36, key=lambda b: (-size(b), b))
    segs = [[] for _ in range(NSEG)]
    for i, b in enumerate(order):
        row, col = divmod(i, NSEG)
        k = col if row % 2 == 0 else NSEG - 1 - col
        segs[k].append(b)
    segs = [sorted(s) for s in segs]

    # eval sets: full, per-segment core, and the two final-score halves (built for GPU 0 / GPU 1)
    a = sorted(EVAL9, key=lambda b: (-size(b), b))
    eval9a, eval9b = a[:5], a[5:]

    sets = {}
    for i, s in enumerate(segs):
        sets["tr36_s%d" % (i + 1)] = s
    sets["evalcore3"] = CORE3
    sets["eval9"] = sorted(EVAL9)
    sets["eval9a"] = sorted(eval9a)
    sets["eval9b"] = sorted(eval9b)

    def linkname(dst, target):
        rel = os.path.relpath(target, os.path.dirname(dst))
        os.symlink(rel, dst)

    summary = []
    for name, bs in sets.items():
        d = os.path.join(SETS, name)
        if os.path.isdir(d):
            for f in os.listdir(d):
                os.unlink(os.path.join(d, f))
        else:
            os.makedirs(d)
        n = 0
        bysrc = {}
        for b in bs:
            for w in scen(b):
                p, src = buildings[b][w]
                linkname(os.path.join(d, "%s__%s.sbb" % (b, w)), p)
                n += 1
                bysrc[src] = bysrc.get(src, 0) + 1
        summary.append((name, len(bs), n, ", ".join("%s:%d" % kv for kv in sorted(bysrc.items()))))

    def manifest(path, title, bs):
        blob_get = lambda b, w: buildings[b][w][0]
        doc = {
            "split": title,
            "created": datetime.now().isoformat(timespec="seconds"),
            "n": sum(len(scen(b)) for b in bs),
            "buildings": bs,
            "scenarios": [
                {"name": "%s/%s" % (b, w), "blob": blob_get(b, w)}
                for b in bs for w in scen(b)
            ],
        }
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")

    os.makedirs(OUTDIR, exist_ok=True)
    manifest(os.path.join(OUTDIR, "train36.json"), "train36", train36)
    manifest(os.path.join(OUTDIR, "eval9.json"), "eval9", sorted(EVAL9))

    print("sources:")
    for name, root in srcs:
        print("  %-10s %s" % (name, os.path.relpath(root, CP)))
    print("\nsets (buildings, scenarios, sources):")
    for name, nb, ns, bysrc in summary:
        print("  %-10s %2d buildings %3d scenarios  [%s]" % (name, nb, ns, bysrc))
    print("\ntraining buildings per segment:")
    for i, s in enumerate(segs):
        print("  tr36_s%d: %s" % (i + 1, " ".join(x.replace("building_", "") for x in s)))
        for b in s:
            print("      %-14s %s" % (b, ",".join(scen(b))))
    print("\neval buildings (eval9; size order):")
    for b in a:
        print("  %-14s %s  %.0f MiB" % (b, ",".join(scen(b)), size(b) / 1048576))
    print("\nmanifests: %s" % ", ".join(os.path.relpath(p, REPO) for p in
          (os.path.join(OUTDIR, "train36.json"), os.path.join(OUTDIR, "eval9.json"))))


if __name__ == "__main__":
    main()
