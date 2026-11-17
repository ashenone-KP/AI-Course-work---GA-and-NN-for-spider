"""Spider-like gait from foot paths + the per-leg IK network (option E put to work).

    cd spider_cw
    python -m part2_nn.ik.foot_gait                    # sweep stride / lift / duty, walk each in Webots
    python -m part2_nn.ik.foot_gait 0.4 0.24 0.5       # one gait: stride (m), lift (m), duty [, wave]
    add --motor to use the motor-range per-leg network instead of the pose-range one

A real spider plants a foot and keeps it still on the ground while the body passes over it, then lifts
it and swings it forward; legs L1 L3 R2 R4 and L2 L4 R1 R3 take turns (alternating tetrapod). The GA's
joint-sine gaits only approximate that (pi0's feet slide 7.7 m in 4 cycles). Here each foot's path is
written down directly, in the body frame (x forward, z up), over one 300-frame cycle:
    stance (fraction `duty` of the cycle): straight back along x by `stride`, at ground height
    swing  (the rest): forward again by `stride`, lifted by up to `lift` (half a sine)
`wave` > 0 starts each leg's step that fraction of a cycle earlier per leg position from the front, so
the hind legs lead (a back-to-front ripple, as in real spiders).
Feet sit, on average, where pi0's feet are (its fan and reach walk without self-collision), at HEIGHT
below the body centre. The per-leg network turns every foot target into its leg's 3 angles.
Writes outputs/spider2_<model>_s<stride>_h<lift>_d<duty>[_w<wave>]/best_gait.csv and info.json, and the
sweep table outputs/nn_ik_motor/spider2_sweep_<model>.txt (model: "pose" or "motor" range per-leg network).
The first version (unmatched touch-down, see foot_paths) is kept as spider_<model>_... / spider_sweep_<model>.txt.
"""
import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from part1_ga.body import collision_fraction
from part1_ga.gait import FRAMES, MOTOR_NAMES
from part2_nn import nn
from part2_nn.ik.ik_data import MOTOR_HI, MOTOR_LO, feet_of, foot_error_cm, from_per_leg, per_leg, unscale
from part2_nn.ik.train import LEG_MODEL_PATH, LEG_MOTOR_MODEL_PATH

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # spider_cw
GROUP_B = np.array([0, 1, 0, 1, 0, 1, 0, 1])          # L1 L2 L3 L4 R4 R3 R2 R1: L1 L3 R4 R2 step together
POSITION = np.array([0, 1, 2, 3, 3, 2, 1, 0])         # 0 = front pair .. 3 = hind pair
SWEEP = {"stride": (0.2, 0.3, 0.4), "lift": (0.06, 0.12, 0.18, 0.24), "duty": (0.5, 0.65)}
# Ground height in the body frame (m). pi0's lowest foot is at -0.80, but only with the foot pulled in: at
# pi0's average foot position a leg reaches no lower than -0.75 (knee at its limit). A grid search over all
# legal angles found every target of a 0.4 m stride reachable within ~1 cm at -0.60, missing by 3 / 7 / 11 cm
# at -0.65 / -0.70 / -0.75.
HEIGHT = -0.60


def neutral_feet(height=HEIGHT, reach=1.0, spread=1.0):
    """Average foot position of each leg over pi0's cycle, x scaled by `spread` (front/hind feet further
    forward/back) and y by `reach` (feet further out), at ground height `height`: (8, 3)."""
    frames = np.loadtxt(os.path.join(ROOT, "outputs", "pi0", "best_gait.csv"), delimiter=",", skiprows=1)
    xy = feet_of(frames).reshape(-1, 8, 3)[..., :2].mean(0) * [spread, reach]
    return np.column_stack([xy, np.full(8, height)])


def foot_paths(stride, lift, duty, wave=0.0, height=HEIGHT, reach=1.0, spread=1.0, matched=True):
    """Foot targets (FRAMES, 24) in the body frame for one cycle.
    matched=True: the swing leaves and lands with the stance's backward speed (a cubic Hermite curve), so
    the foot is still over the ground at lift-off and touch-down. matched=False is the first version (a
    half-cosine that stops relative to the body, so the foot landed moving forward at body speed, and slid)."""
    u = (np.arange(FRAMES)[:, None] / FRAMES + 0.5 * GROUP_B + wave * POSITION) % 1.0      # (T, 8)
    stance = u < duty
    s = np.where(stance, u / duty, (u - duty) / (1 - duty))                                 # 0..1 in each part
    if matched:
        m = -stride * (1 - duty) / duty               # stance speed dx/ds, in swing time units
        swing = (-stride / 2 * (2 * s**3 - 3 * s**2 + 1) + stride / 2 * (-2 * s**3 + 3 * s**2)
                 + m * (s**3 - 2 * s**2 + s) + m * (s**3 - s**2))
    else:
        swing = -stride / 2 + stride * (1 - np.cos(np.pi * s)) / 2
    dx = np.where(stance, stride / 2 - stride * s, swing)
    dz = np.where(stance, 0.0, lift * np.sin(np.pi * s))
    feet = np.broadcast_to(neutral_feet(height, reach, spread), (FRAMES, 8, 3)).copy()
    feet[..., 0] += dx
    feet[..., 2] += dz
    return feet.reshape(FRAMES, 24)


def load_leg_model(motor=False):
    """The pose-range per-leg network (default), or the motor-range one. Every foot target here lies in the
    pose range, where the pose-range network is 4x more precise (0.26 vs 1.1 cm): its best gait walked
    2.46 m straight, the motor-range one's 2.08 m turning 5.5 deg (docs/ik_comparison.md)."""
    path = LEG_MOTOR_MODEL_PATH if motor else LEG_MODEL_PATH
    model, extra = nn.load(path)
    return model, extra, os.path.basename(path)


def angles_for(feet, leg_model=None):
    """Foot targets (n, 24) -> motor angles (n, 24) through the per-leg network."""
    model, extra, _ = leg_model or load_leg_model()
    f, _ = per_leg(feet, np.zeros_like(feet))
    return from_per_leg(unscale(model.forward((f - extra["mean"]) / extra["std"]), MOTOR_LO[:3], MOTOR_HI[:3]))


# GA encoding "foot" (ga_trainer encoding=foot): the GA searches these 7 foot-path settings, the per-leg
# network turns them into joint angles. Bounds: around the sweep's best (0.4 / 0.24 / 0.5 at -0.60), which
# sat on the edge of the grid; heights below -0.68 are out of reach (see HEIGHT).
FOOT_GENES = ("stride", "lift", "duty", "height", "reach", "spread", "wave")
FOOT_LOW = np.array([0.10, 0.04, 0.45, -0.68, 0.85, 0.85, 0.00])
FOOT_HIGH = np.array([0.60, 0.30, 0.80, -0.50, 1.10, 1.15, 0.15])
FOOT_WRAP = np.zeros(len(FOOT_GENES), bool)
# encoding=footwide: the foot_t14 runs ended with stride, lift, height and spread at their upper bounds and
# reach at its lower one. Widened as far as the per-leg network stays accurate on the best gait's
# neighbours: reach < 0.78 brings feet near the hip axis (15+ cm IK error at 0.65) and > 1.10 is out of
# the legs' reach (7+ cm at 1.15); spread 1.40 is past the accurate zone (22 cm worst at 1.45), so the GA
# pays in physics for going there. Long strides collide unless reach/spread grow too: the GA trades them.
FOOT_WIDE_LOW = np.array([0.10, 0.04, 0.35, -0.68, 0.78, 0.85, 0.00])
FOOT_WIDE_HIGH = np.array([0.90, 0.45, 0.85, -0.38, 1.10, 1.40, 0.25])
_MODEL = []                                           # the per-leg network, loaded once per process


def decode(chrom):
    """7 foot-path genes (FOOT_GENES order) -> (FRAMES, 24) motor angles: the GA's decode for encoding=foot."""
    if not _MODEL:
        _MODEL.append(load_leg_model())
    stride, lift, duty, height, reach, spread, wave = np.asarray(chrom, float)
    angles = angles_for(foot_paths(stride, lift, duty, wave, height, reach, spread), _MODEL[0])
    return np.clip(angles, MOTOR_LO, MOTOR_HI)                 # like gait.decode: always legal motor angles


def make(stride, lift, duty, wave=0.0, leg_model=None):
    """Build, check and save one gait. Returns (run name, info)."""
    leg_model = leg_model or load_leg_model()
    feet = foot_paths(stride, lift, duty, wave)
    angles = angles_for(feet, leg_model)
    err = foot_error_cm(angles, feet)
    info = {"stride": stride, "lift": lift, "duty": duty, "wave": wave, "model": leg_model[2],
            "ik_foot_cm_mean": float(err.mean()), "ik_foot_cm_max": float(err.max()),
            "self_collision": float(collision_fraction(angles)),
            "ideal_m_per_cycle": stride / duty}                     # body speed if no foot slips
    tag = "motor" if leg_model[2] == os.path.basename(LEG_MOTOR_MODEL_PATH) else "pose"   # keep both models' gaits
    # spider2_: matched touch-down (default); the first version's gaits are kept as spider_<tag>_...
    run = f"spider2_{tag}_s{stride:.2f}_h{lift:.2f}_d{duty:.2f}" + (f"_w{wave:.2f}" if wave else "")
    out = os.path.join(ROOT, "outputs", run)
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, "best_gait.csv"), angles, delimiter=",", fmt="%.6f",
               header=",".join(MOTOR_NAMES), comments="")
    with open(os.path.join(out, "info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    return run, info


def table(made, walked):
    lines = [f"{'run':32s} {'IK err':>7s} {'coll':>5s} {'ideal':>6s} {'forward':>8s} {'slide':>6s} "
             f"{'heading':>8s} {'pitch':>6s} fell"]
    for run, info in made:
        w = walked[run]
        lines.append(f"{run:32s} {info['ik_foot_cm_mean']:5.2f}cm {info['self_collision']:5.2f} "
                     f"{info['ideal_m_per_cycle'] * 4:5.2f}m {w['forward']:7.2f}m {w.get('slide', np.nan):5.2f}m "
                     f"{w['heading_deg']:+7.1f}d {w.get('pitch', np.nan):6.3f} {w.get('fell', np.nan):4.0f}")
    return lines


def main():
    from part2_nn.ik.gait_test import walk           # here, not at the top: the Webots GA imports this module
    args = [a for a in sys.argv[1:] if a != "--motor"]
    leg_model = load_leg_model(motor="--motor" in sys.argv)
    if args:
        made = [make(*map(float, args[:4]), leg_model=leg_model)]
    else:
        made = [make(s, h, d, leg_model=leg_model)
                for s, h, d in itertools.product(SWEEP["stride"], SWEEP["lift"], SWEEP["duty"])]
    runs = [r for r, _ in made] + ["pi0"]
    with ThreadPoolExecutor(6) as pool:                              # Webots ports 1234.. limit ~11
        walked = dict(zip(runs, pool.map(walk, runs)))
    lines = [f"Spider foot-path gaits through the per-leg network ({leg_model[2]}), 4 cycles in Webots",
             "ideal = stride / duty x 4 cycles (no slipping); slide = all 8 feet's sliding on the floor", ""]
    lines += table(made, walked)
    p = walked["pi0"]
    lines += ["", f"{'pi0 (GA, for comparison)':32s} {'':7s} {'':5s} {'':6s} {p['forward']:7.2f}m "
              f"{p.get('slide', np.nan):5.2f}m {p['heading_deg']:+7.1f}d {p.get('pitch', np.nan):6.3f} "
              f"{p.get('fell', np.nan):4.0f}"]
    text = "\n".join(lines)
    out = os.path.join(ROOT, "outputs", "nn_ik_motor")
    os.makedirs(out, exist_ok=True)
    if len(made) > 1:
        prefix, tag = made[0][0].split("_")[:2]                    # spider2_sweep_pose.txt etc.
        with open(os.path.join(out, f"{prefix}_sweep_{tag}.txt"), "w") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    feet = foot_paths(0.3, 0.1, 0.6)
    for matched, worst in ((True, 0.15), (False, None)):                     # one-frame differences, L1
        x = foot_paths(0.3, 0.1, 0.6, matched=matched).reshape(FRAMES, 8, 3)[:, 0, 0]
        v = x[1] - x[0]                                                       # stance speed (stance from frame 0)
        gaps = [abs((x[0] - x[-1]) - v) / abs(v), abs((x[181] - x[180]) - v) / abs(v)]  # touch-down, lift-off
        assert max(gaps) < worst if matched else min(gaps) > 0.8, (matched, gaps)
    stance = feet.reshape(FRAMES, 8, 3)[..., 2] == neutral_feet()[:, 2]
    assert np.allclose(stance.mean(0), 0.6, atol=1 / FRAMES)                  # each foot is down 60% of the time
    assert (stance[:, [0, 2, 4, 6]].all(1) | ~stance[:, [0, 2, 4, 6]].any(1)).all()  # a tetrapod moves as one
    xs = feet.reshape(FRAMES, 8, 3)[..., 0]
    assert np.allclose(np.ptp(np.where(stance, xs, np.nan)[:, 0][stance[:, 0]]), 0.3, atol=0.002)  # stance = stride
    assert np.ptp(xs, axis=0).max() <= 0.3 * 1.1          # the matched swing overshoots back only a little
    main()
