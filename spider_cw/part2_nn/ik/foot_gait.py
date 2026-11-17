"""Spider-like gait from foot paths + the per-leg IK network (option E put to work).

    cd spider_cw
    python -m part2_nn.ik.foot_gait                    # sweep stride / lift / duty, walk each in Webots
    python -m part2_nn.ik.foot_gait 0.3 0.08 0.6       # one gait: stride (m), lift (m), duty [, wave]

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
Writes outputs/spider_<model>_s<stride>_h<lift>_d<duty>[_w<wave>]/best_gait.csv and info.json, and the
sweep table outputs/nn_ik_motor/spider_sweep_<model>.txt (model: "pose" or "motor" range per-leg network).
"""
import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from part1_ga.body import collision_fraction
from part1_ga.gait import FRAMES, MOTOR_NAMES
from part1_ga.operator_experiments import ROOT
from part2_nn import nn
from part2_nn.ik.gait_test import walk
from part2_nn.ik.ik_data import MOTOR_HI, MOTOR_LO, feet_of, foot_error_cm, from_per_leg, per_leg, unscale
from part2_nn.ik.train import LEG_MODEL_PATH, LEG_MOTOR_MODEL_PATH

GROUP_B = np.array([0, 1, 0, 1, 0, 1, 0, 1])          # L1 L2 L3 L4 R4 R3 R2 R1: L1 L3 R4 R2 step together
POSITION = np.array([0, 1, 2, 3, 3, 2, 1, 0])         # 0 = front pair .. 3 = hind pair
SWEEP = {"stride": (0.2, 0.3, 0.4), "lift": (0.06, 0.12, 0.18, 0.24), "duty": (0.5, 0.65)}
# Ground height in the body frame (m). pi0's lowest foot is at -0.80, but only with the foot pulled in: at
# pi0's average foot position a leg reaches no lower than -0.75 (knee at its limit). A grid search over all
# legal angles found every target of a 0.4 m stride reachable within ~1 cm at -0.60, missing by 3 / 7 / 11 cm
# at -0.65 / -0.70 / -0.75.
HEIGHT = -0.60


def neutral_feet():
    """Average foot position of each leg over pi0's cycle (x, y), at ground height HEIGHT: (8, 3)."""
    frames = np.loadtxt(os.path.join(ROOT, "outputs", "pi0", "best_gait.csv"), delimiter=",", skiprows=1)
    f = feet_of(frames).reshape(-1, 8, 3)
    return np.column_stack([f[..., :2].mean(0), np.full(8, HEIGHT)])


def foot_paths(stride, lift, duty, wave=0.0):
    """Foot targets (FRAMES, 24) in the body frame for one cycle."""
    u = (np.arange(FRAMES)[:, None] / FRAMES + 0.5 * GROUP_B + wave * POSITION) % 1.0      # (T, 8)
    stance = u < duty
    s = np.where(stance, u / duty, (u - duty) / (1 - duty))                                 # 0..1 in each part
    dx = np.where(stance, stride / 2 - stride * s, -stride / 2 + stride * (1 - np.cos(np.pi * s)) / 2)
    dz = np.where(stance, 0.0, lift * np.sin(np.pi * s))
    feet = np.broadcast_to(neutral_feet(), (FRAMES, 8, 3)).copy()
    feet[..., 0] += dx
    feet[..., 2] += dz
    return feet.reshape(FRAMES, 24)


def load_leg_model():
    """The motor-range per-leg network if it has been trained, else the pose-range one."""
    path = LEG_MOTOR_MODEL_PATH if os.path.exists(LEG_MOTOR_MODEL_PATH) else LEG_MODEL_PATH
    model, extra = nn.load(path)
    return model, extra, os.path.basename(path)


def angles_for(feet, leg_model=None):
    """Foot targets (n, 24) -> motor angles (n, 24) through the per-leg network."""
    model, extra, _ = leg_model or load_leg_model()
    f, _ = per_leg(feet, np.zeros_like(feet))
    return from_per_leg(unscale(model.forward((f - extra["mean"]) / extra["std"]), MOTOR_LO[:3], MOTOR_HI[:3]))


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
    run = f"spider_{tag}_s{stride:.2f}_h{lift:.2f}_d{duty:.2f}" + (f"_w{wave:.2f}" if wave else "")
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
    leg_model = load_leg_model()
    if len(sys.argv) > 1:
        made = [make(*map(float, sys.argv[1:5]), leg_model=leg_model)]
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
        with open(os.path.join(out, f"spider_sweep_{made[0][0].split('_')[1]}.txt"), "w") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    feet = foot_paths(0.3, 0.1, 0.6)
    stance = feet.reshape(FRAMES, 8, 3)[..., 2] == neutral_feet()[:, 2]
    assert np.allclose(stance.mean(0), 0.6, atol=1 / FRAMES)                  # each foot is down 60% of the time
    assert (stance[:, [0, 2, 4, 6]].all(1) | ~stance[:, [0, 2, 4, 6]].any(1)).all()  # a tetrapod moves as one
    assert np.ptp(feet.reshape(FRAMES, 8, 3)[..., 0], axis=0).max() <= 0.3 + 1e-9    # stride respected
    main()
