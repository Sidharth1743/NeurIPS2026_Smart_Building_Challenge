#!/usr/bin/env python3
"""CleanRL SAC on the competition contract, 50,135,040 env steps.

Everything the algorithm does is CleanRL `sac_continuous_action.py` with the
defaults kept; see SAC/protocol.json for the four declared adaptations (masked
heads, per-sample entropy target, per-scenario reward normalisation, terminal at
truncation).

Usage:
  SAC/.venv/bin/python SAC/train/train_sac.py --run r01_cleanrl_sac_s75 \
      --out SAC/results/r01_cleanrl_sac_s75 --gpu 0
"""
from __future__ import annotations

import argparse
import hashlib
import queue
import threading
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SAC = HERE.parent
ROOT = SAC.parent
sys.path.insert(0, str(HERE))

from envdriver import EnvDriver  # noqa: E402
from model import Actor, SoftQNetwork, StructuredActor  # noqa: E402
from replay import Replay  # noqa: E402

DEFAULT_BLOB = ROOT / "complete_port/artifacts/sets/r85_four"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_or_none(p: Path) -> str | None:
    """Provenance hashes are optional: inside an image the repo sources may be absent
    (the launcher hashes the frozen runtime and the dataset instead)."""
    try:
        return sha256(p)
    except OSError:
        return None


class TB:
    def __init__(self, path: Path):
        from torch.utils.tensorboard import SummaryWriter

        self.w = SummaryWriter(str(path))

    def add(self, k: str, v: float, step: int) -> None:
        self.w.add_scalar(k, v, step)

    def close(self) -> None:
        self.w.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--config", type=Path, default=SAC / "config/sac50m.json")
    ap.add_argument("--port", type=int, default=0, help="driver control/dir suffix")
    ap.add_argument("--binary", default=None, help="driver binary (absolute, or relative to SAC/)")
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    out = a.out.resolve()
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "tb").mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("SAC_WORK", str(SAC / "run")))
    run_dir = work / f"{a.run}-{a.port}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a", buffering=1)

    def say(msg: str) -> None:
        line = f"{now()} {msg}"
        print(line, flush=True)
        log.write(line + "\n")

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    device = "cuda"

    blas = Path(cfg.get("blob_dir", str(DEFAULT_BLOB)))
    if not blas.is_absolute():
        blas = ROOT / blas
    say(f"batch {blas}  groups {cfg['groups']}  agents {cfg['agents']}")
    env = EnvDriver(
        str(blas), int(cfg["groups"]), int(cfg["agents"]), run_dir,
        name=f"{a.run}-{a.port}", graph=1, desync=1, rew_norm=1,
        gpu=a.gpu, log_path=out / "driver.log",
        freeze_rew_norm_at=int(int(cfg.get("freeze_rew_norm_at_steps", 0)) // int(cfg["agents"])),
        zone_memory=int(cfg.get("zone_memory", 0)),
        # the env's memory trajectory buffer defaults to 1024 steps, which does not
        # fit a 960-agent batch in 24 GB (its own comment says so); one rollout is enough
        env_vars=({"SB_ZM_TRAJ": str(int(cfg.get("zm_traj", 288)))} if int(cfg.get("zone_memory", 0)) else {})
        | ({"CUDA_VISIBLE_DEVICES": str(a.gpu)} if str(cfg.get("launch", "docker")) == "direct" else {}),
        launch=str(cfg.get("launch", "docker")),
        binary=str(a.binary or cfg.get("binary", "bin/sb_env")),
    )
    h = env.h
    n, obs_dim, act_dim = h.n_agents, h.obs_dim, h.act_dim
    say(f"env: {n} agents, obs {obs_dim}, act {act_dim}, ep_len {h.ep_len}, "
        f"active {sorted({int(env.n_active[i]) for i in range(n)})}")

    # ---- CleanRL SAC hyperparameters (defaults) ----
    total_timesteps = int(cfg["total_timesteps"])
    buffer_size = int(cfg["buffer_size"])
    gamma = float(cfg["gamma"])
    tau = float(cfg["tau"])
    batch_size = int(cfg["batch_size"])
    learning_starts = int(cfg["learning_starts"])
    policy_lr = float(cfg["policy_lr"])
    q_lr = float(cfg["q_lr"])
    policy_frequency = int(cfg["policy_frequency"])
    target_network_frequency = int(cfg["target_network_frequency"])
    hidden = int(cfg["hidden"])
    ckpt_every = int(cfg["checkpoint_every_vector_steps"])
    utd = int(cfg.get("utd", 1))
    anneal_lr = int(cfg.get("anneal_lr", 0))
    anneal_steps = int(cfg.get("anneal_steps", 0)) or total_timesteps
    q_ln = bool(int(cfg.get("q_layernorm", 0)))
    atype = str(cfg.get("actor", "flat"))
    structured = atype == "structured"
    head = str(cfg.get("head", "gauss"))
    beta_floor = float(cfg.get("beta_floor", 0.5))
    beta_conc_max = float(cfg.get("beta_conc_max", 12.0))
    head_kw = ({"head": head, "beta_floor": beta_floor, "beta_conc_max": beta_conc_max}
               if head == "beta" else {})

    vec_steps = total_timesteps // n
    warm_steps = max(1, learning_starts // n)
    fz = int(cfg.get("freeze_rew_norm_at_steps", 0))
    say(f"reward normalizer: {'frozen after %d env steps' % fz if fz else 'running EMA'}")
    say(f"budget {total_timesteps} env steps = {vec_steps} vector steps, warmup {warm_steps}")

    # ---- agent ----
    actor = (StructuredActor(obs_dim, act_dim, hidden, **head_kw) if structured
             else Actor(obs_dim, act_dim, hidden, **head_kw)).to(device)
    if structured:
        eager = actor.featurize
        if int(cfg.get("compile_features", 0)):
            class _MaybeCompiled:
                """torch.compile with a permanent fallback: any shape-specific
                inductor failure drops to eager instead of killing the run."""
                def __init__(self, fn):
                    self.eager, self.fast = fn, None
                    try:
                        self.fast = torch.compile(fn, dynamic=False)
                    except Exception as e:
                        say(f"features: torch.compile unavailable ({type(e).__name__}); eager")
                def __call__(self, *a):
                    if self.fast is None:
                        return self.eager(*a)
                    try:
                        return self.fast(*a)
                    except Exception as e:
                        say(f"features: compile failed at runtime ({type(e).__name__}: {e}); eager")
                        self.fast = None
                        return self.eager(*a)
            feat = _MaybeCompiled(eager)
        else:
            feat = eager
    else:
        feat = (lambda x, zm=None: x)
    qf1 = SoftQNetwork(obs_dim, act_dim, hidden, q_ln).to(device)
    qf2 = SoftQNetwork(obs_dim, act_dim, hidden, q_ln).to(device)
    qf1_target = SoftQNetwork(obs_dim, act_dim, hidden, q_ln).to(device)
    qf2_target = SoftQNetwork(obs_dim, act_dim, hidden, q_ln).to(device)
    say(f"Q networks: layernorm={q_ln}   actor: {atype}   "
        f"batch={batch_size} utd={utd} passes={utd * batch_size / n:.2f} "
        f"head={head}{'' if head != 'beta' else f'(floor {beta_floor} cap {beta_conc_max})'} "
        f"anneal_lr={anneal_lr} over {anneal_steps:,} "
        f"compile_features={int(cfg.get('compile_features', 0))}")
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr)
    actor_optimizer = torch.optim.Adam(list(actor.parameters()), lr=policy_lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = torch.optim.Adam([log_alpha], lr=q_lr)

    mask_table = torch.as_tensor(env.h.masks[:, :act_dim], dtype=torch.float32, device=device)  # [S, A]
    scen_np = env.h.scen.astype(np.int64)
    scen_t = torch.as_tensor(scen_np, dtype=torch.long, device=device)
    mask_np = env.mask  # [n, A] float32

    zm_dim = env.h.zm_dim
    say(f"zone memory: env {zm_dim} features per zone (0 = off)")
    rb = Replay(buffer_size, n, obs_dim, act_dim, device, zm_dim=zm_dim, zm_zones=90)
    tb = TB(out / "tb")
    (out / "args.json").write_text(json.dumps(vars(a) | {"config": str(a.config)}, indent=2, default=str) + "\n")
    (out / "manifest.json").write_text(json.dumps({
        "run": a.run, "started": now(), "seed": seed,
        "driver_binary": str(a.binary or cfg.get("binary", "bin/sb_env")),
        "driver_sha256": sha256_or_none(Path(str(a.binary or cfg.get("binary", "bin/sb_env")))),
        "config": cfg,
        "env_source_sha256": sha256_or_none(ROOT / "complete_port/gpu/smart_buildings.cu"),
        "blas": str(blas),
    }, indent=2) + "\n")

    def save(step: int) -> None:
        p = out / "ckpt" / f"{step:012d}.pt"
        torch.save({
            "actor": actor.state_dict(), "qf1": qf1.state_dict(), "qf2": qf2.state_dict(),
            "qf1_target": qf1_target.state_dict(), "qf2_target": qf2_target.state_dict(),
            "log_alpha": log_alpha.detach().cpu(),
            "q_optimizer": q_optimizer.state_dict(), "actor_optimizer": actor_optimizer.state_dict(),
            "a_optimizer": a_optimizer.state_dict(),
            "step": step, "obs_dim": obs_dim, "act_dim": act_dim, "hidden": hidden, "q_ln": q_ln,
            "actor_type": atype, "head": head, "beta_floor": beta_floor,
            "beta_conc_max": beta_conc_max,
            "anneal_lr": anneal_lr, "anneal_steps": anneal_steps,
            "masks": env.h.masks[:, :act_dim].tolist(),
        }, p)
        say(f"saved {p.name}")

    # ------------------------------------------------------------------ async --
    # Cleanba-style overlap (base.async = 1 in the PPO config): the env process
    # steps vector step t+1 while this thread runs the UTD updates for step t.
    # The rollout actor is a weight snapshot taken after each update, so the env
    # thread never reads weights an optimizer is writing.
    rollout = (StructuredActor(obs_dim, act_dim, hidden, **head_kw) if structured
               else Actor(obs_dim, act_dim, hidden, **head_kw)).to(device)
    rollout.load_state_dict(actor.state_dict())
    for q_ in rollout.parameters():
        q_.requires_grad_(False)
    lock = threading.Lock()
    q: "queue.Queue" = queue.Queue(maxsize=int(cfg.get("queue_depth", 2)))
    SENTINEL = object()
    env_err: list[BaseException] = []
    async_ = int(cfg.get("async", 1))

    def env_loop() -> None:
        try:
            obs = torch.as_tensor(env.reset(), device=device)
            zm = torch.as_tensor(env.zm, device=device) if zm_dim else None
            for vs in range(vec_steps):
                if vs < warm_steps:
                    a_np = np.random.uniform(-1.0, 1.0, size=(n, act_dim)).astype(np.float32) * mask_np
                    act = torch.as_tensor(a_np, device=device)
                else:
                    with lock:
                        with torch.no_grad():
                            act, _, _ = rollout.get_action(feat(obs, zm), mask_table[scen_t])
                nxt, r_np, t_np = env.step(act.cpu().numpy())
                live = env.live() if (vs + 1) % 288 == 0 else None
                q.put((obs, act, torch.as_tensor(r_np, device=device),
                       torch.as_tensor(t_np, device=device), live, zm))
                obs = torch.as_tensor(nxt, device=device)
                if zm_dim:
                    zm = torch.as_tensor(env.zm, device=device)
        except BaseException as e:  # noqa: BLE001 - re-raised on the main thread
            env_err.append(e)
        finally:
            q.put(SENTINEL)

    def run_learner() -> None:
        nonlocal alpha
        start = time.time()
        ep_ret = torch.zeros(n, device=device)
        ep_len = torch.zeros(n, device=device)
        ep_hist: list[float] = []
        ret_hist: list[float] = []
        score_seen: set[str] = set()
        update_no = 0
        log_ctr = 0
        ckpt_at = ckpt_every * n
        while True:
            rec = q.get()
            if rec is SENTINEL:
                break
            obs_b, act_b, rew_b, term_b, live, zm_b = rec
            rb.add(obs_b, act_b, rew_b, term_b, scen_t, zm_b)
            step_now = rb.w

            ep_ret += rew_b
            ep_len += 1
            term_np_ = term_b.cpu().numpy()
            for i in np.nonzero(term_np_ > 0)[0]:
                ep_hist.append(float(ep_len[i].item()))
                ret_hist.append(float(ep_ret[i].item()))
                if len(ret_hist) > 256:
                    ret_hist.pop(0)
                    ep_hist.pop(0)
            ep_ret = ep_ret * (1.0 - term_b)
            ep_len = ep_len * (1.0 - term_b)

            if anneal_lr:   # linear to zero over the real budget (r139's stopping term)
                frac = min(1.0, step_now / float(anneal_steps))
                lr_a, lr_q = policy_lr * (1.0 - frac), q_lr * (1.0 - frac)
                for g in actor_optimizer.param_groups:
                    g["lr"] = lr_a
                for g in q_optimizer.param_groups:
                    g["lr"] = lr_q
                for g in a_optimizer.param_groups:
                    g["lr"] = lr_q

            if step_now > learning_starts and rb.ready():
                for _u in range(utd):
                    data = rb.sample(batch_size)
                    m = mask_table[data["scen"]]
                    if structured:   # one feature pass for obs and obs2 (launch-bound)
                        if zm_dim:
                            # zm pairs with obs; the target needs the NEXT transition's block
                            zmb = data["zm"].reshape(-1, 90, zm_dim)
                            zmb2 = data["zm2"].reshape(-1, 90, zm_dim)
                            both = feat(torch.cat([data["obs"], data["obs2"]], 0),
                                        torch.cat([zmb, zmb2], 0))
                        else:
                            both = feat(torch.cat([data["obs"], data["obs2"]], 0))
                        nb = data["obs"].shape[0]
                        fb = {k: v[:nb] for k, v in both.items()}
                        fb2 = {k: v[nb:] for k, v in both.items()}
                    else:
                        fb, fb2 = data["obs"], data["obs2"]
                    with torch.no_grad():
                        next_act, next_log_pi, _ = actor.get_action(fb2, m)
                        q1n = qf1_target(data["obs2"], next_act)
                        q2n = qf2_target(data["obs2"], next_act)
                        min_qn = torch.min(q1n, q2n) - alpha * next_log_pi
                        next_q = data["rew"].flatten() + (1.0 - data["done"].flatten()) * gamma * min_qn.view(-1)
                    q1 = qf1(data["obs"], data["act"]).view(-1)
                    q2 = qf2(data["obs"], data["act"]).view(-1)
                    qf1_loss = F.mse_loss(q1, next_q)
                    qf2_loss = F.mse_loss(q2, next_q)
                    qf_loss = qf1_loss + qf2_loss
                    q_optimizer.zero_grad(set_to_none=True)
                    qf_loss.backward()
                    q_optimizer.step()

                    if update_no % policy_frequency == 0:
                        for _ in range(policy_frequency):
                            pi, log_pi_a, _ = actor.get_action(fb, m)
                            qf1_pi = qf1(data["obs"], pi)
                            qf2_pi = qf2(data["obs"], pi)
                            min_qf_pi = torch.min(qf1_pi, qf2_pi)
                            actor_loss = ((alpha * log_pi_a) - min_qf_pi).mean()
                            actor_optimizer.zero_grad(set_to_none=True)
                            actor_loss.backward()
                            actor_optimizer.step()
                            target_entropy = -m.sum(1, keepdim=True)
                            with torch.no_grad():
                                _, log_pi_c, _ = actor.get_action(fb, m)
                            alpha_loss = (-log_alpha.exp() * (log_pi_c + target_entropy)).mean()
                            a_optimizer.zero_grad(set_to_none=True)
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                    if update_no % target_network_frequency == 0:
                        for p_, tp in zip(qf1.parameters(), qf1_target.parameters()):
                            tp.data.copy_(tau * p_.data + (1.0 - tau) * tp.data)
                        for p_, tp in zip(qf2.parameters(), qf2_target.parameters()):
                            tp.data.copy_(tau * p_.data + (1.0 - tau) * tp.data)
                    update_no += 1

                with lock:
                    rollout.load_state_dict(actor.state_dict())

                log_ctr += 1
                if log_ctr % 10 == 0:
                    tb.add("losses/qf1_values", q1.mean().item(), step_now)
                    tb.add("losses/qf2_values", q2.mean().item(), step_now)
                    tb.add("losses/qf1_loss", qf1_loss.item(), step_now)
                    tb.add("losses/qf2_loss", qf2_loss.item(), step_now)
                    tb.add("losses/actor_loss", actor_loss.item(), step_now)
                    tb.add("losses/alpha", alpha, step_now)
                    tb.add("losses/alpha_loss", alpha_loss.item(), step_now)
                    tb.add("losses/log_pi", log_pi_a.mean().item(), step_now)
                    tb.add("losses/entropy_per_head", -log_pi_a.mean().item() / float(env.n_active.mean()), step_now)

            if live is not None:
                for k, v in live.items():
                    if "episode_score" in k and v == 0.0 and k not in score_seen:
                        continue
                    if "episode_score" in k:
                        score_seen.add(k)
                    if k == "env/score":
                        tb.add("env/mean_reward", v, step_now)
                        tb.add("env/score_pos", -v, step_now)
                        continue
                    tb.add(k, v, step_now)
                el = time.time() - start
                tb.add("perf/sps", step_now / max(el, 1e-9), step_now)
                if ret_hist:
                    tb.add("train/ep_return_z_mean", float(np.mean(ret_hist)), step_now)
                    tb.add("train/ep_length_mean", float(np.mean(ep_hist)), step_now)
                tb.add("perf/lr_actor", actor_optimizer.param_groups[0]["lr"], step_now)
                tb.add("perf/lr_q", q_optimizer.param_groups[0]["lr"], step_now)
                say(f"steps {step_now} ({step_now // n}/{vec_steps}) sps {step_now / max(el, 1e-9):.0f} "
                    f"lr {actor_optimizer.param_groups[0]['lr']:.2e} "
                    f"alpha {alpha:.4f} q {q1.mean().item() if update_no else 0:.3f} "
                    f"ret {np.mean(ret_hist) if ret_hist else float('nan'):.3f}")

            if step_now >= ckpt_at:
                save(step_now)
                ckpt_at += ckpt_every * n

    try:
        th = threading.Thread(target=env_loop, daemon=True, name="env")
        th.start()
        run_learner()
        th.join(timeout=120)
        if env_err:
            raise env_err[0]
        save(rb.w)
        say(f"done ({'async' if async_ else 'sync'} queue) steps {rb.w}")
    finally:
        env.close()
        tb.close()
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
