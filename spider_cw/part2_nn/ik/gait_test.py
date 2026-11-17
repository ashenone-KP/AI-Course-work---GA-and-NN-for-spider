"""Which IK network is better on REAL gaits? (run after part2_nn.ik.train)

    cd spider_cw
    python -m part2_nn.ik.gait_test

The networks are trained and tested on random poses. Here each GA gait (300 frames) is turned into
foot positions with the forward kinematics, fed through both networks, and the predicted angles are
scored two ways:
  1. offline: foot error (cm) and angle RMSE against the GA's own angles;
  2. in Webots: the rebuilt gait is walked for 4 cycles (ga_trainer measure=), next to the original.
Rebuilt gaits are saved as outputs/ik_body_<gait>/ and outputs/ik_leg_<gait>/ (playable by gait_player).
Writes outputs/nn_ik/gait_test.txt.
"""
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from part1_ga.gait import MOTOR_NAMES, POSE_HI, POSE_LO
from part1_ga.operator_experiments import ROOT, WEBOTS, WORLDS
from part2_nn import nn
from part2_nn.ik.ik_data import (MOTOR_HI, MOTOR_LO, SIGN, feet_of, foot_error_cm, from_per_leg, per_leg,
                                 unscale)
from part2_nn.ik.train import LEG_MODEL_PATH, MODEL_PATH, OUT

GAITS = ("pi0", "ref10_2", "shared1", "straight045_tuned")
CYCLES = 4


def predict(feet):
    """Foot positions (n, 24) -> {"body": angles, "leg": angles}, both (n, 24) motor angles."""
    model, extra = nn.load(MODEL_PATH)
    leg_model, leg_extra = nn.load(LEG_MODEL_PATH)
    f, _ = per_leg(feet, np.zeros_like(feet))
    return {"body": unscale(model.forward((feet - extra["mean"]) / extra["std"])),
            "leg": from_per_leg(unscale(leg_model.forward((f - leg_extra["mean"]) / leg_extra["std"]),
                                        MOTOR_LO[:3], MOTOR_HI[:3]))}


def outside_training_box(angles):
    """True where an angle lies outside the ranges the training poses were drawn from."""
    a = angles * SIGN
    return (a < np.tile(POSE_LO[:3], 8) - 1e-9) | (a > np.tile(POSE_HI[:3], 8) + 1e-9)


def walk(run):
    """Walk outputs/<run>/best_gait.csv in Webots for CYCLES cycles -> {"forward", "fell", ...}."""
    with open(os.path.join(WORLDS, "spider_world_1.wbt")) as f:
        world = f.read()
    world = re.sub(r'^  controller ".*"$', '  controller "ga_trainer"', world, flags=re.M)
    world = re.sub(r"^  controllerArgs .*$", f'  controllerArgs ["measure={run}" "cycles={CYCLES}" "show=0"]',
                   world, flags=re.M)
    path = os.path.join(WORLDS, "_ik_" + run + ".wbt")
    with open(path, "w") as f:
        f.write(world)
    track = os.path.join(ROOT, "outputs", run, "track.csv")
    kept = open(track, "rb").read() if os.path.exists(track) else None   # measure= overwrites it (GA runs keep a 20-cycle track)
    try:
        out = subprocess.run([WEBOTS, "--batch", "--mode=fast", "--stdout", "--stderr", "--minimize",
                              "--no-rendering", path], capture_output=True, text=True, timeout=900).stdout
    finally:
        os.remove(path)
        if kept is not None:
            with open(track, "wb") as f:
                f.write(kept)
    line = next((l for l in out.splitlines() if f"measure {run}:" in l), None)
    if line is None:
        raise RuntimeError(f"{run}: no measure line from Webots:\n{out[-2000:]}")
    m = {k: float(v) for k, v in re.findall(r"(\w+) ([+-][\d.]+)", line.split(":", 1)[1])}
    head = re.search(rf"path {re.escape(run)}:.*heading ([+-][\d.]+) deg", out)
    m["heading_deg"] = float(head.group(1)) if head else np.nan
    return m


def main():
    rows, runs = [], []
    for g in GAITS:
        frames = np.loadtxt(os.path.join(ROOT, "outputs", g, "best_gait.csv"), delimiter=",", skiprows=1)
        feet = feet_of(frames)
        outside = outside_training_box(frames)
        frame_out = outside.reshape(-1, 8, 3).any(-1)                     # (300, 8): leg has an angle outside
        runs.append(g)
        for net, pred in predict(feet).items():
            err = foot_error_cm(pred, feet)                               # (300, 8)
            ang = np.degrees(pred - frames)
            rows.append({"gait": g, "net": net, "foot_cm": err.mean(), "foot_cm_p95": np.percentile(err, 95),
                         "foot_cm_in": err[~frame_out].mean(), "foot_cm_out": err[frame_out].mean() if frame_out.any() else np.nan,
                         "angle_rmse": np.sqrt(np.mean(ang ** 2)), "outside": outside.mean()})
            run = f"ik_{net}_{g}"
            os.makedirs(os.path.join(ROOT, "outputs", run), exist_ok=True)
            np.savetxt(os.path.join(ROOT, "outputs", run, "best_gait.csv"), pred, delimiter=",", fmt="%.6f",
                       header=",".join(MOTOR_NAMES), comments="")
            runs.append(run)

    with ThreadPoolExecutor(6) as pool:                                   # Webots ports 1234.. limit ~11
        walked = dict(zip(runs, pool.map(walk, runs)))

    lines = [f"IK networks on real GA gaits: 300 frames x 8 feet each, walked {CYCLES} cycles in Webots", "",
             f"{'gait':18s} {'network':8s} {'foot err':>9s} {'95%':>6s} {'in box':>7s} {'out box':>8s} "
             f"{'angle RMSE':>10s}   {'forward':>8s} {'GA':>6s} {'kept':>5s} {'heading':>8s} fell"]
    for r in rows:
        w, ga = walked[f"ik_{r['net']}_{r['gait']}"], walked[r["gait"]]
        r.update(forward=w["forward"], ga_forward=ga["forward"], fell=w.get("fell", np.nan),
                 heading=w["heading_deg"])
        lines.append(f"{r['gait']:18s} {r['net']:8s} {r['foot_cm']:7.2f}cm {r['foot_cm_p95']:6.2f} "
                     f"{r['foot_cm_in']:7.2f} {r['foot_cm_out']:8.2f} {r['angle_rmse']:8.2f}deg   "
                     f"{r['forward']:7.2f}m {r['ga_forward']:5.2f}m {r['forward'] / ga['forward']:5.0%} "
                     f"{r['heading']:+7.1f}d {r['fell']:4.0f}")
    lines += ["", "in box / out box: mean foot error on legs whose angles all lie inside / partly outside "
              "the ranges the training poses were drawn from (part2_nn/ik/ik_data.py).",
              "share of GA angles outside those ranges: "
              + ", ".join(f"{g} {next(r['outside'] for r in rows if r['gait'] == g):.0%}" for g in GAITS)]
    for net in ("body", "leg"):
        sel = [r for r in rows if r["net"] == net]
        lines.append(f"{net:5s} mean over gaits: foot error {np.mean([r['foot_cm'] for r in sel]):.2f} cm, "
                     f"distance kept {np.mean([r['forward'] / r['ga_forward'] for r in sel]):.0%}")
    text = "\n".join(lines)
    with open(os.path.join(OUT, "gait_test.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
