#!/usr/bin/env python3
"""Dump every TensorBoard scalar in a run directory to one CSV, without TensorFlow.

usage:
    tb_to_csv.py <tb dir> <out.csv>

Writes `tag,step,value`, one row per scalar event, sorted by tag then step. This is
the stored form of the training curves in this release: the event files themselves
are 20 MB to 130 MB per run and are not shipped.

The event-file parser is the one in `complete_port/tools/tbread.py`, which reads the
protobuf wire format directly; only the output shape differs.
"""
from __future__ import annotations

import csv
import glob
import struct
import sys


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        c = buf[i]
        i += 1
        result |= (c & 0x7F) << shift
        shift += 7
        if c < 0x80:
            return result, i


def _fields(buf: bytes):
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value = buf[i:i + 8]
            i += 8
        elif wire == 5:
            value = buf[i:i + 4]
            i += 4
        else:
            length, i = _varint(buf, i)
            value = buf[i:i + length]
            i += length
        yield field, value


def read(tb_dir: str) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for path in sorted(glob.glob(tb_dir.rstrip("/") + "/events.out*")):
        with open(path, "rb") as fh:
            buf = fh.read()
        i = 0
        while i < len(buf):
            (length,) = struct.unpack("<Q", buf[i:i + 8])
            event = buf[i + 12:i + 12 + length]
            i += 12 + length + 4
            step = 0
            summary = None
            for field, value in _fields(event):
                if field == 2:
                    step = value
                elif field == 5:
                    summary = value
            if not summary:
                continue
            for _, entry in _fields(summary):
                tag = None
                for field, value in _fields(entry):
                    if field == 1:
                        tag = value.decode()
                    elif field == 2:
                        (scalar,) = struct.unpack("<f", value)
                        if tag is not None:
                            series.setdefault(tag, []).append((step, scalar))
    return series


SCALAR_TAGS = {"SPS", "agent_steps", "epoch", "uptime", "importance"}


def keep(tag: str) -> bool:
    """Epoch-resolution series the release ships.

    Per-scenario diagnostics are 30 scenarios times 59 tags per epoch and dominate the file, so
    only the per-scenario score series is kept; the aggregate series are kept whole.
    """
    if tag in SCALAR_TAGS:
        return True
    head = tag.split("/")[0]
    if head in {"env", "loss", "perf", "util", "live"}:
        return True
    if head == "scenario":
        return tag.rsplit("/", 1)[-1] in {"score", "episode_score", "episode_return",
                                          "productivity", "energy_cost_norm", "carbon_norm"}
    return False


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    tb_dir, out_csv = sys.argv[1], sys.argv[2]
    series = read(tb_dir)
    rows = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tag", "step", "value"])
        for tag in sorted(series):
            if not keep(tag):
                continue
            for step, value in series[tag]:
                writer.writerow([tag, step, repr(value)])
                rows += 1
    print(f"{out_csv}: {len(series)} tags, {rows} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
