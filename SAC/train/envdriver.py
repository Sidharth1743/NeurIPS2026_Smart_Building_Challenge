"""Rollout client for SAC/env/sb_env (the standalone GPU env driver).

The driver runs inside pufferai/puffertank:5.0 with GPU 0; this process is the
learner. Data moves through a POSIX shm file that both processes map: the driver
writes actions in and reads obs/reward/terminal out (the env's own pinned
buffers). The control socket only carries 1-byte opcodes, so the per-step cost is
one round trip, not a copy.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
import socket
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAC = Path(__file__).resolve().parents[1]
# Shared-memory ring + control sockets go in SAC_WORK (default SAC/run). Inside an
# image (Modal) /sac is read-only, so the launcher points this at local scratch.
WORK = Path(os.environ.get("SAC_WORK", str(SAC / "run")))
ROOT = SAC.parent
IMAGE = os.environ.get("SB_IMAGE", "pufferai/puffertank:5.0")
PUFFERLIB = os.environ.get("PUFFERLIB", "/data/puffer-work/plverify")
WITH_GPU = ROOT / "complete_port/tools/with_gpu.sh"

HDR_BYTES = 64  # Hdr is 64 bytes: 8+8*4? -> keep in sync with sb_env.cu (sizeof rounded to 64)
OP_STEP, OP_LIVE, OP_PARTS, OP_RESET, OP_QUIT = b"S", b"L", b"P", b"R", b"Q"


@dataclass
class Header:
    n_agents: int
    obs_dim: int
    obs_stride: int
    act_dim: int
    act_stride: int
    mask_width: int
    ep_len: int
    zm_dim: int
    masks: np.ndarray  # [n_unique, mask_width] uint8
    scen: np.ndarray   # [n_agents] int32 -> index into masks

    @property
    def mask_of_agent(self) -> np.ndarray:
        return self.masks[self.scen]  # [n_agents, mask_width]


def _wait(path: Path, timeout: float, proc: subprocess.Popen) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists():
            return
        if proc.poll() is not None:
            raise RuntimeError(f"driver exited early (rc={proc.returncode})")
        time.sleep(0.05)
    raise TimeoutError(f"{path} did not appear in {timeout:.0f}s")


class EnvDriver:
    def __init__(
        self,
        blob_dir: str | None,
        groups: int,
        agents: int,
        run_dir: Path,
        name: str = "env",
        graph: int = 1,
        desync: int = 1,
        rew_norm: int = 0,
        max_steps: int = 0,
        eval_out: str | None = None,
        gpu: int = 0,
        log_path: Path | None = None,
        extra: list[str] | None = None,
        blob: str | None = None,
        freeze_rew_norm_at: int = 0,
        zone_memory: int = 0,
        env_vars: dict[str, str] | None = None,
        launch: str = "docker",
        binary: str = "bin/sb_env",
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.shm_path = run_dir / f"{name}.shm"
        self.hdr_path = run_dir / f"{name}.json"
        # AF_UNIX paths are capped at 108 bytes: keep the socket in a short dir
        sockdir = WORK / "r"
        sockdir.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha1(f"{name}-{os.getpid()}".encode()).hexdigest()[:12]
        self.ctrl_path = sockdir / f"{tag}.sock"
        self.log_path = log_path or (run_dir / f"{name}.driver.log")
        for p in (self.shm_path, self.ctrl_path, self.hdr_path):
            if p.exists():
                p.unlink()
        self.shm_path.touch()

        self._cname = f"{name}-{os.getpid()}"
        self._launch = launch

        def _c(p: Path) -> str:
            return "/w/" + str(p.resolve().relative_to(ROOT))

        if launch not in ("docker", "direct"):
            raise ValueError(f"launch must be docker|direct, got {launch}")
        direct = launch == "direct"

        if direct:
            # same container as the learner (Modal, CI): no docker, no path translation
            c_shm, c_ctrl, c_hdr = str(self.shm_path), str(self.ctrl_path), str(self.hdr_path)
            c_blob, c_one = (blob_dir or ""), (blob or "")
            sb_args = [
                str((SAC / binary) if not str(binary).startswith("/") else Path(binary)),
                "--blob_dir", c_blob, "--blob", c_one, "--groups", str(groups), "--agents", str(agents),
                "--shm", c_shm, "--ctrl", c_ctrl, "--header", c_hdr,
                "--graph", str(graph), "--desync", str(desync), "--rew_norm", str(rew_norm),
                "--max_steps", str(max_steps),
            ]
            args = sb_args
        else:
            c_shm, c_ctrl, c_hdr = (_c(self.shm_path), _c(self.ctrl_path), _c(self.hdr_path))
            def cpath(p: str | None) -> str:
                if p is None:
                    return ""
                try:
                    return "/w/" + str(Path(p).resolve().relative_to(ROOT))
                except ValueError:
                    return p  # already a container path

            c_blob = cpath(blob_dir)
            c_one = cpath(blob)
            args = [
                str(WITH_GPU), str(gpu), "--",
                "docker", "run", "--rm", "--name", f"sac-{self._cname}",
                "--gpus", f"device={gpu}",
                "-v", f"{ROOT}:/w", "-v", f"{PUFFERLIB}:/pl:ro", "-w", "/w/SAC",
                *[a for k, v in (env_vars or {}).items() for a in ("-e", f"{k}={v}")],
                IMAGE, f"./{binary}",
                "--blob_dir", c_blob, "--blob", c_one, "--groups", str(groups), "--agents", str(agents),
                "--shm", c_shm, "--ctrl", c_ctrl, "--header", c_hdr,
                "--graph", str(graph), "--desync", str(desync), "--rew_norm", str(rew_norm),
                "--max_steps", str(max_steps),
            ]
        if freeze_rew_norm_at > 0:
            args += ["--freeze_rew_norm_at", str(freeze_rew_norm_at)]
        if zone_memory:
            args += ["--zone_memory", "1"]
        if eval_out:
            args += ["--eval_out", eval_out]
        if extra:
            args += extra
        self.log = open(self.log_path, "wb")
        penv = dict(os.environ)
        if direct:
            penv.update(env_vars or {})
        self.proc = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT, env=penv)
        try:
            _wait(self.hdr_path, 600.0, self.proc)
            _wait(self.ctrl_path, 60.0, self.proc)
            self.h = self._read_header()
        except BaseException:
            self._kill()
            raise

        self._f = open(self.shm_path, "r+b")
        n_a = self.h.n_agents * self.h.act_stride
        n_o = self.h.n_agents * self.h.obs_stride
        n_r = self.h.n_agents
        n_z = self.h.n_agents * 90 * self.h.zm_dim
        total = HDR_BYTES + 4 * (n_a + n_o + 2 * n_r) + 4 * n_z
        self._mm = mmap.mmap(self._f.fileno(), total)
        base = HDR_BYTES
        self._act = np.frombuffer(self._mm, dtype=np.float32, count=n_a, offset=base).reshape(
            self.h.n_agents, self.h.act_stride
        )
        base += 4 * n_a
        self._obs = np.frombuffer(self._mm, dtype=np.float32, count=n_o, offset=base).reshape(
            self.h.n_agents, self.h.obs_stride
        )
        base += 4 * n_o
        self._rew = np.frombuffer(self._mm, dtype=np.float32, count=n_r, offset=base)
        base += 4 * n_r
        self._term = np.frombuffer(self._mm, dtype=np.float32, count=n_r, offset=base)
        base += 4 * n_r
        self._zm = (np.frombuffer(self._mm, dtype=np.float32, count=n_z, offset=base)
                    .reshape(self.h.n_agents, 90, self.h.zm_dim) if n_z else None)

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        t0 = time.time()
        while True:
            try:
                self.sock.connect(str(self.ctrl_path))
                break
            except (ConnectionRefusedError, FileNotFoundError):
                if time.time() - t0 > 120:
                    raise
                time.sleep(0.05)

        self.obs = self._obs[:, : self.h.obs_dim]
        self.zm = self._zm      # [n, 90, 11] when env.zone_memory = 1, else None
        self.mask = self.h.mask_of_agent[:, : self.h.act_dim].astype(np.float32)  # [n, act_dim]
        self.n_active = self.mask.sum(1)                                          # [n]
        self._r1 = b""
        self.steps = 0

    def _kill(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:
            pass
        if getattr(self, "_launch", "docker") == "docker":
            subprocess.run(["docker", "rm", "-f", f"sac-{self._cname}"], capture_output=True)
        self.log.close()

    def _read_header(self) -> Header:
        t0 = time.time()
        while True:
            try:
                d = json.loads(self.hdr_path.read_text())
                break
            except (json.JSONDecodeError, FileNotFoundError):
                if time.time() - t0 > 30:
                    raise
                time.sleep(0.02)
        return Header(
            n_agents=d["n_agents"], obs_dim=d["obs_dim"], obs_stride=d["obs_stride"],
            act_dim=d["act_dim"], act_stride=d["act_stride"], mask_width=d["mask_width"],
            ep_len=d["ep_len"], zm_dim=d.get("zm_dim", 0), masks=np.asarray(d["masks"], dtype=np.uint8),
            scen=np.asarray(d["scen"], dtype=np.int32),
        )

    # ------------------------------------------------------------------ api --
    def reset(self) -> np.ndarray:
        self._cmd(OP_RESET)
        return self.obs.copy()

    def _cmd(self, op: bytes) -> None:
        self.sock.sendall(op)
        r = self._recvn(1)
        if r != b"\x00":
            raise RuntimeError(f"driver replied {r!r}")

    def _recvn(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            k = self.sock.recv(n - len(buf))
            if not k:
                raise RuntimeError("driver closed the control socket")
            buf += k
        return bytes(buf)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """actions: float32 [n, act_dim] in [-1,1]; inactive heads are zeroed here."""
        a = self._act
        a[:, : self.h.act_dim] = actions * self.mask
        if self.h.act_stride > self.h.act_dim:
            a[:, self.h.act_dim :] = 0.0
        self.sock.sendall(OP_STEP)
        if self._recvn(1) != b"\x00":
            raise RuntimeError("step failed")
        self.steps += 1
        return self.obs.copy(), self._rew.copy(), self._term.copy()

    def parts(self) -> np.ndarray:
        """Per-step reward parts of the step just taken, float64 [n_agents, 8].

        The core's SB_RECORD companion (`sb_rec_parts`): productivity, energy_cost, carbon,
        comfort_dev_K, worst_dev_K, out_of_band_frac, elec_kwh, gas_kwh. Empty if the driver
        cannot serve it (more than one group in the blob factorization).
        """
        self.sock.sendall(OP_PARTS)
        (cnt,) = struct.unpack("<I", self._recvn(4))
        if not cnt:
            return np.zeros((0, 8), np.float64)
        buf = self._recvn(8 * cnt)
        return np.frombuffer(buf, dtype=np.float64).reshape(-1, 8)

    def live(self) -> dict[str, float]:
        self.sock.sendall(OP_LIVE)
        (n,) = struct.unpack("<I", self._recvn(4))
        txt = self._recvn(n).decode() if n else ""
        out: dict[str, float] = {}
        for line in txt.splitlines():
            if "\t" not in line:
                continue
            k, v = line.split("\t", 1)
            try:
                out[k] = float(v)
            except ValueError:
                pass
        return out

    def close(self) -> None:
        try:
            self.sock.sendall(OP_QUIT)
        except OSError:
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._kill()
        self.sock.close()
        for attr in ("obs", "zm", "mask", "_act", "_obs", "_rew", "_term", "_zm"):
            if hasattr(self, attr):
                delattr(self, attr)
        try:
            self._mm.close()
        except BufferError:
            pass  # numpy views keep exported pointers; the OS reclaims at exit
        self._f.close()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
