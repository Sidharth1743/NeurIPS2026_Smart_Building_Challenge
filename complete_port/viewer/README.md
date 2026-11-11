# viewer

Replays a simulator snapshot in a window or in the terminal. It only reads a file, so it never slows
the simulator and can attach to a run that is already going.

The simulator writes the snapshot; this tool does not. `tools/train.sh` does it with
`SB_SNAPSHOT=<file>`, `tools/watch.sh` does it for a live replay, and the gate
`viewer: play replay writes a snapshot` writes one into `results/verify_snap.sbs`.

```bash
python sb_viewer.py results/verify_snap.sbs                      # window
python sb_viewer.py results/verify_snap.sbs --terminal           # table in the terminal
python sb_viewer.py results/verify_snap.sbs --terminal --once    # one frame, for a log
python sb_viewer.py results/verify_snap.sbs --screenshot out.png # headless render
```

| flag | meaning |
|---|---|
| `<snapshot>` | the `.sbs` file to read |
| `--terminal` | print a live table instead of opening a window |
| `--once` | with `--terminal`, print one frame and exit |
| `--screenshot <path>` | render one frame to a PNG and exit, no display needed |
| `--interval <s>` | terminal refresh period |
| `--fps <n>` | window refresh rate |
| `--wait <s>` | how long to wait for the first snapshot to appear |

It needs `numpy` and, for the window, `pygame`; `./run.sh setup` builds `viewer/.venv` with both.
