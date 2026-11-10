#!/usr/bin/env python3
"""Split leak checker: no training blob may be an evaluation blob, and the sealed seed range
(30000-30019) must not appear anywhere.  Run before every chain segment.

    tools/check_split.py [--deep] [--quiet] [SET ...]

With no arguments it checks the chain's sets: every directory under artifacts/sets whose name
starts with "tr" is training, every "eval..." directory is evaluation.  Explicit names are looked up
under artifacts/sets (a path with a / is used as is).

Training x evaluation checks:
  inode    the same file (a hardlink is exactly the r7 contamination: artifacts/train/* pointed at
           the gen-val blobs, so an eval building sat inside the training set);
  name     the same (building, weather) from two different files;
  content  --deep: same size + same sha256 of the first and last 1 MiB (identical copies under
           other names); the .sbb header/metadata differs per scenario, so equal head+tail means
           equal scenario for every blob produced by the bake/gen tools.
Everywhere: a building id inside the sealed range 30000-30019 is a violation, and so is a set with
no .sbb files or an entry that does not resolve.
"""
import hashlib
import os
import sys

CP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # complete_port/
SETS = os.path.join(CP, "artifacts", "sets")
SEALED = range(30000, 30020)


def entries(d):
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".sbb"):
            continue
        p = os.path.join(d, f)
        if not os.path.exists(p):
            out.append((f[:-4], None, None, None))
            continue
        st = os.stat(p)
        out.append((f[:-4], os.path.realpath(p), (st.st_dev, st.st_ino), st.st_size))
    return out


def building_of(name):
    b = name.split("__", 1)[0]
    return b[len("building_"):] if b.startswith("building_") else b


def sealed_of(name):
    t = building_of(name)
    return t.isdigit() and int(t) in SEALED


def hash_parts(path, size):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1048576))
        if size > 2097152:
            f.seek(-1048576, os.SEEK_END)
            h.update(f.read())
    return h.hexdigest()


def main():
    args = list(sys.argv[1:])
    deep = "--deep" in args
    quiet = "--quiet" in args
    names = [a for a in args if not a.startswith("--")]
    if names:
        dirs = [n if "/" in n else os.path.join(SETS, n) for n in names]
    else:
        if not os.path.isdir(SETS):
            sys.exit("no %s (run tools/build_split.py first)" % SETS)
        dirs = [os.path.join(SETS, d) for d in sorted(os.listdir(SETS))
                if os.path.isdir(os.path.join(SETS, d))]
    train_d = sorted(d for d in dirs if os.path.basename(d).startswith("tr"))
    eval_d = sorted(d for d in dirs if os.path.basename(d).startswith("eval"))
    if not train_d or not eval_d:
        sys.exit("need at least one training and one evaluation set (got %d / %d)"
                 % (len(train_d), len(eval_d)))

    fails, n_train, n_eval = [], 0, 0
    train = {}          # name -> (set, path)
    train_inodes = {}   # (dev, ino) -> (set, path, name)
    train_sizes = {}    # size -> [name]
    for d in train_d + eval_d:
        es = entries(d)
        if not es:
            fails.append("empty set: %s" % os.path.relpath(d, SETS))
        for name, path, ino, size in es:
            label = os.path.relpath(d, CP)
            if path is None:
                fails.append("broken entry: %s/%s.sbb" % (label, name))
                continue
            if sealed_of(name):
                fails.append("sealed building in %s: %s" % (label, name))
            if os.path.basename(d).startswith("tr"):
                n_train += 1
                if name in train:
                    fails.append("duplicate scenario across training sets: %s (%s, %s)"
                                 % (name, train[name][0], label))
                train[name] = (label, path)
                train_inodes[ino] = (label, path, name)
                train_sizes.setdefault(size, []).append(name)
            else:
                n_eval += 1
                other = train_inodes.get(ino)
                if other:
                    fails.append("SAME FILE in training and evaluation: %s (%s) == %s (%s)"
                                 % (name, label, other[2], other[0]))
                if name in train:
                    fails.append("SAME SCENARIO in training and evaluation: %s (%s, %s)"
                                 % (name, label, train[name][0]))
                if deep and size in train_sizes:
                    h = hash_parts(path, size)
                    for cand in train_sizes[size]:
                        cp = train[cand][1]
                        if hash_parts(cp, size) == h:
                            fails.append("IDENTICAL CONTENT: %s (%s) == %s (%s)"
                                         % (name, label, cand, train[cand][0]))

    if not quiet:
        print("check_split: %d training scenarios in %d sets, %d evaluation scenarios in %d sets%s"
              % (n_train, len(train_d), n_eval, len(eval_d), ", deep" if deep else ""))
        for d in train_d:
            print("  train    %-10s %3d" % (os.path.basename(d), len(entries(d))))
        for d in eval_d:
            print("  eval     %-10s %3d" % (os.path.basename(d), len(entries(d))))
    if fails:
        print("check_split: LEAK")
        for f in fails:
            print("  " + f)
        return 1
    print("check_split: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
