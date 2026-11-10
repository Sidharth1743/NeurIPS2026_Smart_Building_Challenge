#!/usr/bin/env python3
"""Print the decode-relevant flags of a run, one per line.

A trained checkpoint is only readable by a binary built for the same network, so evaluating one
needs the recipe flags it was trained with. This prints exactly the tokens that decide the shape and
the decoding, from the run's `args.txt`, or from `flags.txt` when the launch arguments were not
archived. `score_eval90.py` uses the same filter.

    decode_flags.py <run directory>
"""
from __future__ import annotations

import pathlib
import shlex
import sys

PREFIXES = (
    "--policy.", "--env.canonical=", "--env.residual=", "--env.zone_memory=",
    "--env.zone_critic=", "--env.reward_scale=", "--env.comfort_shape=",
    "--train.zone_critic=", "--train.valve_zone=", "--train.zone_adv=",
    "--train.room_value=", "--train.component_values=", "--train.freeze_return=",
    "--train.role_adam=",
)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run = pathlib.Path(sys.argv[1])
    src = run / "args.txt"
    if not src.exists():
        src = run / "flags.txt"
    if not src.exists():
        print(f"no args.txt or flags.txt in {run}", file=sys.stderr)
        return 1
    seen, out = set(), []
    for token in shlex.split(src.read_text()):
        key = token.split("=", 1)[0]
        if token.startswith(PREFIXES) and key not in seen:
            seen.add(key)
            out.append(token)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
