#!/usr/bin/env python3
"""Assemble SAC/kaggle/submission.py from two proven sources.

  * results/r139_submission/submission.py : canonical-observation builder (gateway obs ->
    the 983-wide vector), the metadata layout, the 219-head -> gateway action decode, the
    weights finder.  Proven on the Kaggle leaderboard.
  * SAC/train/features.py : the numpy reference for the per-role feature blocks. Already
    gated against the torch trainer at 2.2e-16 by SAC/tools/check_features.py.

The only new code is the structured actor's forward pass (7 linear layers) and its weight
parser.  Nothing here runs a controller: the action is tanh(mu), the deterministic mean the
scorer uses.

    python3 SAC/kaggle/build_submission.py
"""
from __future__ import annotations

from pathlib import Path

SAC = Path(__file__).resolve().parents[1]
ROOT = SAC.parent
SRC = ROOT / "results/r139_submission/submission.py"
FEA = SAC / "train/features.py"
OUT = Path(__file__).resolve().parent / "submission.py"


def lines(path: Path) -> list[str]:
    return path.read_text().splitlines(keepends=True)


def take(path: Path, first: int, last: int) -> str:
    """1-based inclusive line range."""
    return "".join(lines(path)[first - 1:last])


HEADER = '''"""Kaggle policy: the structured SAC actor trained on the 30-scenario batch.

Deterministic decode: every head is tanh(mu) of a learned affine readout, masked by the
contract. There is no controller, no sequencer, no guard -- the same forward pass the
training eval (`SAC/tools/score_set.py`, tanh(mean)) ran.

Inputs are built exactly as the trainer builds them, from the gateway observation only:

  pool        80 values   (zone/AHU/plant statistics, presence, demand aggregates)
  zone local  68 per zone slot  [own sensors 0..5 | its AHU 6..13 | cold deficit 14,15 |
                                 1 | memory 11 (env.zone_memory=0 -> zeros) | siblings 40]
  AHU local   32 per AHU
  plant local 16

The canonical 983-value observation and the 219-head -> gateway action mapping are the
proven r139 submission code; the feature blocks and the actor are this repository's SAC
implementation. Weights are a flat float32 dump of the 12 actor tensors.

Assembly: SAC/kaggle/build_submission.py. Do not hand-edit.
"""
'''
# noqa: E501

MINE_CONST = '''
# ---------------------------------------------------------------- SAC actor --
# SAC/train/model.py StructuredActor: Linear(80->256) trunk, three per-role readouts and
# the small nonlinear zone branch. 12 tensors, 36,518 floats (the 438 mask buffers are
# not weights).
POOL, ZONE_L, AHU_L, PLANT_L = 80, 68, 32, 16
SAC_H = 256
ZM_H = 32
N_ACTOR = (POOL * SAC_H + SAC_H                                   # trunk
           + (SAC_H + ZONE_L) * 4 + 4                             # r0
           + (SAC_H + AHU_L) * 8 + 8                              # r1
           + (SAC_H + PLANT_L) * 6 + 6                            # r2
           + (SAC_H + ZONE_L) * ZM_H + ZM_H                       # zm1
           + ZM_H * 4 + 4)                                        # zm2
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
'''

MINE_INIT = '''
    """One policy instance per scenario; the gateway rebuilds it on every first turn."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.fallback = False
        path = _find_weights()
        self.w = _parse_actor(path)
        meta = _sidecar(path)
        self.head = meta.get("head", "gauss")
        self.beta_floor = float(meta.get("beta_floor", 0.5))
        self.beta_conc_max = float(meta.get("beta_conc_max", 12.0))
        print(f"head {self.head} floor {self.beta_floor} cap {self.beta_conc_max}", flush=True)
        try:
            self._layout(building_meta_data)
        except DoesNotFit as exc:
            print(f"canonical slots do not fit this building: {exc}", flush=True)
            self.fallback = True
'''

MINE_ACTOR = '''
def _beta_ab(p, q, floor, cap):
    """softplus rows -> (a, b); total concentration capped at `cap`, mean preserved."""
    a = np.logaddexp(0.0, p).astype(f32) + f32(floor)
    b = np.logaddexp(0.0, q).astype(f32) + f32(floor)
    if cap and cap > 0:
        total = a + b
        k = np.where(total > cap, f32(cap) / total, f32(1.0)).astype(f32)
        a = (a * k).astype(f32)
        b = (b * k).astype(f32)
    return a, b


def _actor_forward(w, o, head="gauss", floor=0.5, cap=12.0):
    """983-wide canonical observation -> 219-head executed action.

    The readouts emit two rows per head: (mean, log_std) for the Gaussian and (u, v) for
    the Beta. Gaussian: tanh(mean). Beta, the competition decode: a = softplus(u) + floor,
    b = softplus(v) + floor, a + b capped at `cap` with the mean preserved, action =
    2 a / (a + b) - 1.
    """
    pool = np_pool(o[None, :])[0]
    loc0 = np_zone_local(o[None, :])[0]
    loc1 = np_ahu_local(o[None, :])[0]
    loc2 = np_plant_local(o[None, :])[0][0]
    h = (pool @ w["trunk_w"].T + w["trunk_b"]).astype(f32)
    x0 = np.concatenate([np.broadcast_to(h, (CN_ZS, SAC_H)), loc0], -1)
    o0 = (x0 @ w["r0_w"].T + w["r0_b"]).astype(f32)
    zn = np.tanh((x0 @ w["zm1_w"].T + w["zm1_b"]).astype(f32))
    o0 = (o0 + (zn @ w["zm2_w"].T + w["zm2_b"]).astype(f32)).astype(f32)
    x1 = np.concatenate([np.broadcast_to(h, (CN_K, SAC_H)), loc1], -1)
    o1 = (x1 @ w["r1_w"].T + w["r1_b"]).astype(f32)
    x2 = np.concatenate([h, loc2])[None, :]
    o2 = (x2 @ w["r2_w"].T + w["r2_b"]).astype(f32)
    p1 = np.zeros(CN_ACT, f32)
    p2 = np.zeros(CN_ACT, f32)
    p1[0:2 * CN_ZS] = o0[:, 0:2].reshape(-1)
    p2[0:2 * CN_ZS] = o0[:, 2:4].reshape(-1)
    p1[CN_AA:CN_AA + 4 * CN_K] = o1[:, 0:4].reshape(-1)
    p2[CN_AA:CN_AA + 4 * CN_K] = o1[:, 4:8].reshape(-1)
    p1[CN_AH:CN_AH + 3] = o2[0, 0:3]
    p2[CN_AH:CN_AH + 3] = o2[0, 3:6]
    if head == "beta":
        a, b = _beta_ab(p1, p2, floor, cap)
        return (f32(2.0) * a / (a + b) - f32(1.0)).astype(f32)
    return np.tanh(p1).astype(f32)

'''

MINE_FORWARD = '''
    # --- SAC actor ------------------------------------------------------------
    def _forward(self, o):
        return _actor_forward(self.w, o, self.head, self.beta_floor, self.beta_conc_max)

'''

MINE_TAIL = '''

    def get_action(self, obs):
        if self.fallback:
            return self._neutral()
        try:
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            canon = self._canon(obs)
            executed = self._forward(canon)
            return self._decode(executed)
        except Exception:
            traceback.print_exc()
            raise


def _sidecar(path):
    """weights.json next to weights.bin (head, floor, cap); {} when absent."""
    p = Path(path).with_name("weights.json")
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"weights.json unreadable ({type(exc).__name__}: {exc}); assuming gauss", flush=True)
        return {}


def _parse_actor(path):
    """Flat float32 dump -> the 12 actor tensors, in the order SAC/.../export_weights.py wrote."""
    w = np.fromfile(path, dtype=np.float32)
    print(f"loaded weights {path} floats {w.size}", flush=True)
    if w.size != N_ACTOR:
        raise ValueError(f"weights has {w.size} floats, the SAC actor has {N_ACTOR}")
    out, o = {}, 0

    def take(name, rows, cols):
        nonlocal o
        n = rows * cols
        out[name + "_w"] = w[o:o + n].reshape(rows, cols).copy()
        o += n

    def bias(name, n):
        nonlocal o
        out[name + "_b"] = w[o:o + n].copy()
        o += n

    take("trunk", SAC_H, POOL); bias("trunk", SAC_H)
    take("r0", 4, SAC_H + ZONE_L); bias("r0", 4)
    take("r1", 8, SAC_H + AHU_L); bias("r1", 8)
    take("r2", 6, SAC_H + PLANT_L); bias("r2", 6)
    take("zm1", ZM_H, SAC_H + ZONE_L); bias("zm1", ZM_H)
    take("zm2", 4, ZM_H); bias("zm2", 4)
    if o != N_ACTOR:
        raise ValueError(f"weight cursor {o} stopped before {N_ACTOR}")
    return out


Policy = StructuredSacPolicy
SacPolicy = StructuredSacPolicy

if __name__ == "__main__":
    from kaggle_evaluation.smart_buildings_inference_server import SmartBuildingsInferenceServer
    SmartBuildingsInferenceServer(StructuredSacPolicy).run()
'''


def main() -> int:
    # 1. r139 constants + helpers (48..376): canonical layout, field names, metadata helpers,
    #    weight finder, decode helpers.  Same 219-head contract this actor was trained on.
    block = take(SRC, 48, 376)
    # the r139 PPO weight parser is replaced below by _parse_actor; drop it so no dead
    # reference to N_PARAMS survives.
    start = block.index("def _parse_weights(")
    block = block[:start]

    parts = [
        HEADER,
        "from __future__ import annotations\n\nimport json\nimport math\nimport os\nimport traceback\nfrom pathlib import Path\n\nimport numpy as np\n\n"
        "try:\n    from kaggle_evaluation.smart_buildings_inference_server import Policy as _Base\n"
        "except Exception:  # noqa: BLE001\n    class _Base:  # type: ignore[no-redef]\n"
        "        def __init__(self, building_meta_data, action_spec):\n"
        "            self.building_meta_data = building_meta_data\n            self.action_spec = action_spec\n\n",
        block,
        MINE_CONST,
        "\n# ---------------------------------------------- numpy feature blocks (SAC) --\n",
        take(FEA, 26, 158),          # constants, index helpers and the numpy feature blocks
        MINE_ACTOR,
        "\n\nclass StructuredSacPolicy(_Base):\n",
        MINE_INIT,
        take(SRC, 391, 536),          # _layout
        "\n", take(SRC, 548, 551),     # _nat
        "\n", take(SRC, 553, 595),     # _canon
        MINE_FORWARD,
        "\n", take(SRC, 756, 759),     # _neutral
        "\n", take(SRC, 761, 772),     # _decode
        MINE_TAIL,
    ]
    OUT.write_text("".join(parts))
    print(f"wrote {OUT}  {len(OUT.read_text().splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
