#!/usr/bin/env python3
"""Live viewer for a Smart Buildings GPU env snapshot (pygame window or terminal table).

The simulator writes one env's state to a file (SB_SNAPSHOT=<file>, see README "Viewer");
this tool only reads that file, so it never slows the simulator and can attach/detach
at any time.  Geometry (rooms, zone names, setpoint schedules) comes from the baked blob
named in the snapshot.

  sb_viewer.py <snapshot>                 pygame window
  sb_viewer.py <snapshot> --terminal      live table in the terminal
  sb_viewer.py <snapshot> --once --terminal            print one frame and exit
  sb_viewer.py <snapshot> --once --screenshot out.png  render one frame headless (SDL dummy)

Window keys: SPACE freeze/unfreeze the display, M map mode (temperature / deviation from the
comfort band), L zone labels, [ ] move the colour range, - = widen/narrow it, S save
screenshot, Q/Esc quit.  (Freeze only stops the display; the simulator keeps running.)
"""
import argparse
import datetime as dt
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pysim"))
import sbb  # noqa: E402

PART_NAMES = ["productivity", "energy_cost_norm", "carbon_norm", "mean_dev_K", "worst_dev_K",
              "out_of_band_frac", "elec_kWh_step", "gas_kWh_step"]


# ------------------------------------------------------------------ snapshot
class Snap:
    pass


def read_snapshot(path):
    """Parse one snapshot file; returns None if it is missing/partial."""
    try:
        b = open(path, "rb").read()
    except OSError:
        return None
    if len(b) < 4 + 48 + 96 + 512 or b[:4] != b"SBS1":
        return None
    s = Snap()
    ints = struct.unpack_from("<12i", b, 4)
    (ver, s.step, s.max_steps, s.env, s.nz, s.nvav, s.nahu, s.rows, s.cols, s.ncv,
     s.n_action, _) = ints
    if ver != 1:
        return None
    sc = struct.unpack_from("<12d", b, 52)
    s.reward, s.ep_ret, s.boiler_sp, s.boiler_ret = sc[:4]
    s.parts = dict(zip(PART_NAMES, sc[4:]))
    off = 52 + 96
    s.blob = b[off:off + 512].split(b"\0")[0].decode()
    off += 512
    need = off + 4 * s.ncv + 8 * (4 * s.nz + 4 * s.nvav + 2 * s.nahu + s.n_action) + 4 * s.nahu
    if len(b) != need:
        return None

    def take(dtype, n):
        nonlocal off
        a = np.frombuffer(b, dtype=dtype, count=n, offset=off)
        off += a.nbytes
        return a

    s.field = take("<f4", s.ncv).reshape(s.rows, s.cols)
    s.zone_t, s.occ, s.zo_sp, s.zo_act = (take("<f8", s.nz) for _ in range(4))
    s.vav_heatsp, s.vav_coolsp, s.vav_damper, s.vav_reheat = (take("<f8", s.nvav) for _ in range(4))
    s.ahu_sat, s.ahu_flow = take("<f8", s.nahu), take("<f8", s.nahu)
    s.ahu_run = take("<i4", s.nahu)
    s.native = take("<f8", s.n_action)
    return s


# ---------------------------------------------------------------- geometry
def resolve_blob(path):
    """The simulator runs in a container (/cp = complete_port, /art = artifacts): map back to the host."""
    root = os.path.join(HERE, "..")
    for cand in (path, path.replace("/cp/", root + "/", 1), path.replace("/art/", os.path.join(root, "artifacts") + "/", 1),
                 os.path.join(root, "artifacts", "blobs", os.path.basename(path)), os.path.join(os.getcwd(), os.path.basename(path))):
        if os.path.isfile(cand):
            return cand
    raise SystemExit(f"blob {path} not found; copy it next to where you run the viewer")


class Geom:
    """Static data from the blob: room map, zone names, schedules."""

    def __init__(self, blob_path):
        r = sbb.BlobReader(resolve_blob(blob_path))
        self.rows, self.cols = (int(v) for v in r.get("grid"))
        ro = r.get("room_offsets")
        rc = r.get("room_cells").reshape(-1, 2)
        self.room_names = r.get("room_names").tobytes().decode().split("\n")
        self.kind = np.full(self.rows * self.cols, -1, np.int32)   # room index per cell
        self.centroid = {}
        for q in range(len(ro) - 1):
            cells = rc[ro[q]:ro[q + 1]]
            self.kind[cells[:, 0] * self.cols + cells[:, 1]] = q
            if len(cells):
                self.centroid[q] = (cells[:, 0].mean(), cells[:, 1].mean())
        self.kind = self.kind.reshape(self.rows, self.cols)
        self.zone_room = r.get("phys_room_idx")
        names = r.get("phys_zone_names").tobytes().decode().split("\n")
        self.zone_names = [n for n in names if n] or [f"zone{i}" for i in range(len(self.zone_room))]
        self.heat_sp, self.cool_sp = r.get("sched_heat_sp"), r.get("sched_cool_sp")
        self.ambient = r.get("weather_ambient")
        self.start = float(r.get("start_epoch")[0])
        self.dt_sec = float(r.get("obs_meta")[4]) if len(r.get("obs_meta")) > 4 else 300.0
        try:
            import zoneinfo
            self.tz = zoneinfo.ZoneInfo(r.get("time_zone").tobytes().decode())
        except Exception:
            self.tz = dt.timezone.utc
        self.zone_label = {}
        for z, q in enumerate(self.zone_room):
            if int(q) in self.centroid:
                self.zone_label[z] = self.centroid[int(q)]

    def clock(self, step):
        t = dt.datetime.fromtimestamp(self.start + step * self.dt_sec, self.tz)
        return t.strftime("%a %Y-%m-%d %H:%M")

    def band(self, step):
        i = min(max(step, 0), len(self.heat_sp) - 1)
        return float(self.heat_sp[i]), float(self.cool_sp[i])

    def amb(self, step):
        return float(self.ambient[min(max(step, 0), len(self.ambient) - 1)])


def zone_dev(s, g):
    h, c = g.band(s.step)
    return np.maximum(np.maximum(s.zone_t - c, h - s.zone_t), 0.0)


# ---------------------------------------------------------------- terminal
def frame_text(s, g, max_rows):
    h, c = g.band(s.step)
    dev = zone_dev(s, g)
    p = s.parts
    L = [f"\x1b[1mSmart Buildings GPU env {s.env}\x1b[0m  step {s.step}/{s.max_steps}  {g.clock(s.step)}  "
         f"ambient {g.amb(s.step) - 273.15:.1f} C  band {h - 273.15:.1f}..{c - 273.15:.1f} C",
         f"reward {s.reward:+.5f}  episode return {s.ep_ret:+.3f}  productivity {p['productivity']:+.3f}  "
         f"energy {p['energy_cost_norm']:.4f}  carbon {p['carbon_norm']:.4f}  "
         f"elec {p['elec_kWh_step']:.2f} kWh  gas {p['gas_kWh_step']:.2f} kWh (this step)",
         f"comfort: mean dev {p['mean_dev_K']:.2f} K  worst {p['worst_dev_K']:.2f} K  "
         f"zones out of band {100 * p['out_of_band_frac']:.0f}%   boiler setpoint {s.boiler_sp - 273.15:.1f} C "
         f"(return {s.boiler_ret - 273.15:.1f} C)",
         "AHU  " + "  ".join(f"#{i}:{'on' if s.ahu_run[i] else 'off'} sat {s.ahu_sat[i] - 273.15:.1f}C "
                              f"flow {s.ahu_flow[i]:.2f}" for i in range(s.nahu)),
         "", f"{'zone':<14}{'T [C]':>8}{'dev K':>8}{'occ':>7}{'flow sp':>9}{'flow':>8}"]
    order = np.argsort(-dev)
    for z in order[:max(max_rows - len(L) - 1, 3)]:
        col = "\x1b[31m" if dev[z] > 0 else "\x1b[32m"
        name = g.zone_names[z] if z < len(g.zone_names) else str(z)
        L.append(f"{name[:13]:<14}{col}{s.zone_t[z] - 273.15:8.2f}{dev[z]:8.2f}\x1b[0m"
                 f"{s.occ[z]:7.1f}{s.zo_sp[z]:9.3f}{s.zo_act[z]:8.3f}")
    if s.nz > len(L):
        L.append(f"... {s.nz} zones, sorted by comfort deviation")
    return "\n".join(L)


def run_terminal(args):
    g = None
    last = -1
    while True:
        s = read_snapshot(args.snapshot)
        if s is not None and (s.step != last or args.once):
            if g is None or g.rows != s.rows:
                g = Geom(s.blob)
            rows = os.get_terminal_size().lines if sys.stdout.isatty() else 40
            sys.stdout.write(("\x1b[H\x1b[2J" if sys.stdout.isatty() and not args.once else "") +
                             frame_text(s, g, rows) + "\n")
            sys.stdout.flush()
            last = s.step
        if args.once and s is not None:
            return 0
        if args.once and time.time() - t0 > args.wait:
            print("no snapshot", file=sys.stderr)
            return 1
        time.sleep(args.interval)


# ------------------------------------------------------------------ window
def make_lut():
    """blue -> white -> red, 256 entries."""
    x = np.linspace(0.0, 1.0, 256)
    r = np.clip(2 * x, 0, 1)
    b = np.clip(2 * (1 - x), 0, 1)
    gch = 1 - np.abs(2 * x - 1)
    return (np.stack([r, 0.25 + 0.75 * gch, b], 1) * 255).astype(np.uint8)


def render_map(s, g, lo, hi, mode, lut):
    """RGB array (cols, rows, 3) for pygame.surfarray."""
    if mode == 0:
        v = (s.field - lo) / max(hi - lo, 1e-6)
    else:   # deviation of each room's cells from the comfort band, zones only
        h, c = g.band(s.step)
        d = np.maximum(np.maximum(s.field - c, h - s.field), 0.0)
        v = 0.5 + 0.5 * np.clip(np.where(s.field > 0.5 * (h + c), d, -d) / max(hi - lo, 1e-6) * 4, -1, 1)
    img = lut[np.clip((v * 255).astype(np.int32), 0, 255)].astype(np.float32)
    img[g.kind == -1] = (18, 18, 24)
    img[g.kind == 0] = (30, 30, 36)
    wall = g.kind == 1
    img[wall] *= 0.45
    return np.ascontiguousarray(img.astype(np.uint8).transpose(1, 0, 2))


def run_window(args):
    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    W, H = 1500, 900
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Smart Buildings GPU env")
    font = pygame.font.SysFont("monospace", 15)
    small = pygame.font.SysFont("monospace", 12)
    lut = make_lut()
    g = None
    s = None
    frozen = False
    mode, labels = 0, True
    lo, hi = 285.0, 305.0
    clock = pygame.time.Clock()
    t0 = time.time()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k == pygame.K_SPACE:
                    frozen = not frozen
                elif k == pygame.K_m:
                    mode = 1 - mode
                elif k == pygame.K_l:
                    labels = not labels
                elif k == pygame.K_LEFTBRACKET:
                    lo, hi = lo - 1, hi - 1
                elif k == pygame.K_RIGHTBRACKET:
                    lo, hi = lo + 1, hi + 1
                elif k == pygame.K_MINUS:
                    lo, hi = lo - 1, hi + 1
                elif k == pygame.K_EQUALS:
                    lo, hi = min(lo + 1, hi - 2), max(hi - 1, lo + 2)
                elif k == pygame.K_s:
                    pygame.image.save(screen, f"sb_view_{int(time.time())}.png")
        if not frozen:
            n = read_snapshot(args.snapshot)
            if n is not None:
                s = n
                if g is None or g.rows != s.rows or g.cols != s.cols:
                    g = Geom(s.blob)
        screen.fill((12, 12, 16))
        if s is None:
            screen.blit(font.render(f"waiting for {args.snapshot} ...", True, (220, 220, 220)), (20, 20))
        else:
            mw = W - 520
            surf = pygame.surfarray.make_surface(render_map(s, g, lo, hi, mode, lut))
            sc = min(mw / s.cols, (H - 40) / s.rows)
            size = (max(int(s.cols * sc), 1), max(int(s.rows * sc), 1))
            surf = pygame.transform.smoothscale(surf, size)
            ox, oy = 10, 20
            screen.blit(surf, (ox, oy))
            dev = zone_dev(s, g)
            if labels:
                for z, (ry, cx) in g.zone_label.items():
                    if z < s.nz:
                        t = small.render(f"{s.zone_t[z] - 273.15:.1f}", True,
                                         (255, 255, 255) if dev[z] <= 0 else (255, 215, 0))
                        screen.blit(t, (ox + cx * sc - t.get_width() / 2, oy + ry * sc - t.get_height() / 2))
            x0 = ox + size[0] + 20
            h, c = g.band(s.step)
            p = s.parts
            rows = [
                (f"step {s.step}/{s.max_steps}   {g.clock(s.step)}", (255, 255, 255)),
                (f"ambient {g.amb(s.step) - 273.15:6.1f} C   band {h - 273.15:.1f}..{c - 273.15:.1f} C", (200, 200, 200)),
                (f"reward {s.reward:+.5f}   return {s.ep_ret:+.3f}", (255, 255, 255)),
                (f"productivity {p['productivity']:+.3f}  energy {p['energy_cost_norm']:.4f}  carbon {p['carbon_norm']:.4f}", (200, 200, 200)),
                (f"elec {p['elec_kWh_step']:.2f} kWh/step   gas {p['gas_kWh_step']:.2f} kWh/step", (200, 200, 200)),
                (f"comfort dev mean {p['mean_dev_K']:.2f} K  worst {p['worst_dev_K']:.2f} K  out {100 * p['out_of_band_frac']:.0f}%",
                 (255, 215, 0) if p["worst_dev_K"] > 0 else (120, 220, 120)),
                (f"boiler sp {s.boiler_sp - 273.15:.1f} C (return {s.boiler_ret - 273.15:.1f} C)", (200, 200, 200)),
            ]
            for i in range(s.nahu):
                rows.append((f"AHU{i} {'on ' if s.ahu_run[i] else 'off'} sat {s.ahu_sat[i] - 273.15:5.1f} C  flow {s.ahu_flow[i]:6.2f}", (170, 200, 255)))
            y = 20
            for text, col in rows:
                screen.blit(font.render(text, True, col), (x0, y))
                y += 19
            y += 8
            screen.blit(font.render(f"{'zone':<12}{'T C':>7}{'dev':>6}{'occ':>6}{'flow':>7}", True, (150, 150, 150)), (x0, y))
            y += 19
            for z in np.argsort(-dev):
                if y > H - 60:
                    break
                name = g.zone_names[z] if z < len(g.zone_names) else str(z)
                col = (255, 120, 100) if dev[z] > 0 else (120, 220, 120)
                screen.blit(small.render(f"{name[:11]:<12}{s.zone_t[z] - 273.15:7.2f}{dev[z]:6.2f}{s.occ[z]:6.1f}{s.zo_act[z]:7.3f}", True, col), (x0, y))
                y += 15
            # colour bar: temperature range, K shown as C
            bar = pygame.surfarray.make_surface(np.ascontiguousarray(lut[np.linspace(0, 255, 200).astype(int)][:, None, :].repeat(14, 1)))
            screen.blit(bar, (x0, H - 48))
            screen.blit(small.render(f"{lo - 273.15:.0f} C", True, (200, 200, 200)), (x0, H - 30))
            hs = small.render(f"{hi - 273.15:.0f} C", True, (200, 200, 200))
            screen.blit(hs, (x0 + 200 - hs.get_width(), H - 30))
            mt = "temperature" if mode == 0 else "band deviation"
            screen.blit(small.render(f"map: {mt}   {'FROZEN' if frozen else 'live'}", True, (200, 200, 200)), (x0 + 230, H - 48))
            screen.blit(small.render("SPACE freeze  M map  L labels  [ ] shift  - = range  S shot  Q quit", True, (130, 130, 130)), (x0 + 230, H - 30))
        pygame.display.flip()
        if args.screenshot and (s is not None or time.time() - t0 > args.wait):
            pygame.image.save(screen, args.screenshot)
            print(f"saved {args.screenshot}" if s is not None else "no snapshot")
            return 0 if s is not None else 1
        clock.tick(args.fps)
    pygame.quit()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--terminal", action="store_true", help="print a live table instead of opening a window")
    ap.add_argument("--once", action="store_true", help="(terminal) print one frame and exit")
    ap.add_argument("--screenshot", help="render one frame headless to this PNG and exit")
    ap.add_argument("--interval", type=float, default=0.25, help="terminal refresh seconds")
    ap.add_argument("--fps", type=int, default=30, help="window refresh rate")
    ap.add_argument("--wait", type=float, default=30.0, help="seconds to wait for the first snapshot with --once/--screenshot")
    a = ap.parse_args()
    t0 = time.time()
    sys.exit(run_terminal(a) if a.terminal else run_window(a))
