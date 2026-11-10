#!/usr/bin/env python3
"""Merge per-GPU evaluation JSONs into one score file.

    tools/merge_eval_json.py OUT.json IN.json [IN.json ...]

Each input is a results/eval/*.json written by tools/eval_policy.sh: per-scenario mean per-step
reward and reward terms under "scenarios", score = mean |mean_reward| over scenarios (lower is
better).  The final chain score is one full-episode evaluation per scenario, split across the two
GPUs as eval9a + eval9b; this concatenates them, recomputes the score over all 45 scenarios and
refuses overlapping scenarios or an incomplete run.
"""
import json
import os
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out, ins = sys.argv[1], sys.argv[2:]
    merged, source, expect, complete = {}, {}, 0, True
    for p in ins:
        with open(p) as f:
            d = json.load(f)
        for k, v in d["scenarios"].items():
            if k not in merged:
                merged[k] = v
                source[k] = os.path.basename(p)
        expect += d.get("n_scenarios", 0)
        complete = complete and d.get("complete", False)
    if len(merged) != expect:
        sys.exit("merge: %d unique scenarios but the inputs claim %d" % (len(merged), expect))
    score = sum(abs(v["mean_reward"]) for v in merged.values()) / len(merged)
    doc = {
        "scenarios": dict(sorted(merged.items())),
        "n_scenarios_done": len(merged),
        "n_scenarios": len(merged),
        "complete": complete,
        "score": score,
        "sources": source,
    }
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("merge_eval_json: %d scenarios from %s -> %s (score %.6f, complete %s)"
          % (len(merged), ", ".join(os.path.basename(p) for p in ins), out, score, complete))


if __name__ == "__main__":
    main()
