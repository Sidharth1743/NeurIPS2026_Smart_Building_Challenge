#!/usr/bin/env python3
"""Generate gpu/sb_group_state.inc: the list of file-scope variables that make up ONE scenario group.

The env keeps a scenario's host state (pointers to its device buffers, shapes, streams, ...) in
file-scope statics.  A multi-scenario batch swaps them per group by byte copy (sb_group_save /
sb_group_load).  Missing a variable would silently mix two groups, so the list is generated from
the sources and `--check` (run by verify_all.sh) fails if the committed file is stale.

  gen_group_state.py          rewrite gpu/sb_group_state.inc
  gen_group_state.py --check  exit 1 if it is out of date
"""
import os
import re
import sys

GPU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpu")
SOURCES = ["smart_buildings.cu", "sb_gpu_kernels.cuh"]
# not per group: shared by every group, or a one-shot helper
EXCLUDE = {
    "stream": "the framework's rollout stream, bound once by puf_bind_stream",
    "g_gexec": "gate-only CUDA graph handle",
    "g_test_ep_len": "gate-only restart length",
    "g_test_start": "gate-only shifted reset offset",
    "g_blob_path": "scratch for the vec constructor's path parse",
    "g_groups": "the group table itself",
    "g_fork": "event that forks the group streams off the rollout stream (shared)",
    "g_scen_names": "scenario names of the whole batch (per-scenario metric keys)",
    "g_last_sc": "last episode score per scenario of the whole batch",
    "g_rew_scale": "reward scale of the whole batch (env.reward_scale)",
    "g_rew_norm": "rew_norm flag of the whole batch (env.rew_norm)",
    "g_rew_center": "reward-centering flag of the whole batch (env.rew_center)",
    "g_center_beta": "reward-centering step size of the whole batch (env.center_beta)",
    "d_rew_norm": "per-scenario reward-normalization stats, batch-wide (env.rew_norm)",
    "g_rn_mu": "host EMA state of d_rew_norm (not a device buffer)",
    "g_rn_var": "host EMA state of d_rew_norm (not a device buffer)",
    "g_canon": "canonical layout flag of the whole batch",
    "g_random_start": "random-start flag of the whole batch",
    "g_zone_command": "zone-command flag of the whole batch (env.zone_command)",
    "g_probe": "probe mode of the whole batch (env.probe, diagnostic)",
    "g_probe_water": "probe water action of the whole batch (env.probe_water, diagnostic)",
    "g_plant_pair": "plant-pair flag of the whole batch (env.plant_pair)",
    "g_plant_eps": "plant-pair setpoint shift of the whole batch (env.plant_eps)",
    "g_zone_memory": "zone-memory flag of the whole batch (env.zone_memory)",
    "g_interleave": "weather interleave flag of the whole batch (env.interleave)",
    "g_prestart": "pre-dawn exploration flag of the whole batch (env.prestart_explore)",
    "g_resid": "residual flag of the whole batch",
    "g_res_scale": "residual scale of the whole batch",
    "g_res_dscale": "residual run-bit scale of the whole batch",
    "g_guard_d0": "comfort guard threshold of the whole batch",
    "g_guard_d1": "comfort guard width of the whole batch",
    "g_guard_hot": "comfort guard side of the whole batch",
    "g_comfort_shape": "comfort-shaping scale of the whole batch (env.comfort_shape)",
    "g_shape_gamma": "comfort-shaping gamma of the whole batch (env.shape_gamma)",
    "g_fork2": "phase fork event of the fused multi-group step (shared)",
    "g_fork3": "phase fork event of the fused multi-group step (shared)",
    "g_mg_on": "fused multi-group solve: batch-wide",
    "g_mg_force": "fused multi-group solve: batch-wide",
    "g_mg_mw": "fused multi-group solve: batch-wide",
    "g_mg_rpl": "fused multi-group solve: batch-wide",
    "d_mg_ordF": "fused multi-group solve: batch-wide",
    "d_mg_ordB": "fused multi-group solve: batch-wide",
    "n_mgF": "fused multi-group solve: batch-wide",
    "n_mgB": "fused multi-group solve: batch-wide",
    "d_mg_ep": "fused multi-group solve: batch-wide",
    "d_mg_tk": "fused multi-group solve: batch-wide",
}
ONLY_TRAINER = {   # declared under #ifndef SB_TEST
    "g_envs", "g_rew_fw", "g_term_fw", "d_canon_obs", "d_entact", "d_cmdact", "d_probeact",
    "d_rew_learn", "d_term_learn",
}


def strip_comments(t):
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
    return re.sub(r"//[^\n]*", "", t)


def names():
    out = []
    for f in SOURCES:
        t = strip_comments(open(os.path.join(GPU, f)).read())
        for line in (x.rstrip() for x in t.split("\n")):
            if not re.match(r"^static\s", line) or not line.rstrip().endswith(";"):
                continue
            decl = line.split("=")[0] if "=" in line and "(" not in line.split("=")[0] else line
            if "(" in decl:      # function
                continue
            body = re.sub(r"^static\s+", "", line.rstrip(";"))
            # one declarator per comma-separated part: the name is the last identifier before an
            # initializer or array bound (the first part also carries the type)
            parts = body.split(",")
            got = 0
            for part in parts:
                part = part.split("=")[0].strip()
                part = re.sub(r"\[[^\]]*\]\s*$", "", part).strip()
                m = re.search(r"([A-Za-z_][A-Za-z_0-9]*)$", part)
                if m:
                    out.append(m.group(1))
                    got += 1
            if got != len(parts):      # a declarator the parser could not read: never skip silently
                sys.exit(f"gen_group_state.py: cannot parse declaration: {line}")
    return out


def render():
    ns = [n for n in names() if n not in EXCLUDE]
    seen = set()
    lines = ["/* GENERATED by tools/gen_group_state.py -- do not edit.  One entry per file-scope variable of a scenario group. */"]
    for n in ns:
        if n in seen:
            continue
        seen.add(n)
        if n in ONLY_TRAINER:
            lines += ["#ifndef SB_TEST", f"SBG({n})", "#endif"]
        else:
            lines.append(f"SBG({n})")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    path = os.path.join(GPU, "sb_group_state.inc")
    new = render()
    if "--check" in sys.argv:
        old = open(path).read() if os.path.exists(path) else ""
        if old != new:
            print("sb_group_state.inc is stale: run tools/gen_group_state.py")
            sys.exit(1)
        print(f"sb_group_state.inc up to date ({new.count('SBG(')} variables)")
    else:
        open(path, "w").write(new)
        print(f"wrote {path}: {new.count('SBG(')} variables")
