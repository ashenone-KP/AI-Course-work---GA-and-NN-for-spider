"""Option A dataset: [sin phase, cos phase, speed] -> 24 Webots motor angles.

Source: a family of 9 GA gaits (Part 1 trainer, target= mode), all refined from ref10_2 so that
neighbouring speeds are similar gaits (mean joint-angle change 1-2 deg between neighbours), which
is what makes interpolating between them meaningful. Each gait is 300 frames = one cycle.

Inputs
    sin(phase), cos(phase): phase = 2 pi k / 300 for frame k. Two numbers instead of one so the
        input is continuous around the cycle (frame 299 is next to frame 0, not 299 units away).
    speed: the gait's MEASURED speed in Webots (metres per cycle, from ga_trainer measure=<run>
        cycles=4), not its GA target: the network should learn what the gait actually does.
        Normalised with the training speeds' mean and std so all inputs have a similar scale.
Outputs
    the 24 angles scaled per joint to [-1/1.1, 1/1.1] using the motor limits:
        y = (angle - mid) / (1.1 * half)
    A tanh output layer then can only produce angles inside the limits (+10%, clipped in
    unscale()), i.e. valid joint angles by construction; the 10% margin keeps targets that sit
    exactly on a limit from needing tanh = +-1 (infinite pre-activation).
Splits
    test: one whole speed (HELD_OUT = speed060, the middle of the family) never seen in training.
        Frames of one gait are near-copies of each other, so a random frame split would leak and
        hide whether the network can produce a NEW speed.
    validation: by default a WHOLE other speed (val_run), used for early stopping and for choosing
        hyperparameters (leave-one-speed-out). A random 15% of the training frames was tried first:
        configurations that fitted those best were the worst on a new speed (they overfit the
        speeds they had seen), so random-frame validation chose the wrong models.
    The speed normalisation uses all non-test speeds, so it is the same in every fold.
"""
import os

import numpy as np

from part1_ga.gait import FRAMES, MOTOR_HI, MOTOR_LO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# measured forward distance over 4 cycles / 4  (ga_trainer measure=<run> cycles=4)
# speed055/065/070 were added to fill the gap 0.24-0.45 m/cycle around the test speed (the first
# network's error was largest there). speed050 (target 0.50) is left out: the GA could not slow it
# down from ref10_2 and it measured 0.449, a second, different gait at speed075's speed.
# That family (speed015-090) curved left, 17-180 m turning radius over 20 cycles, and so did the
# network trained on it (kept as models/speed_model_curved.npz, outputs/nn_speed_curved). Each gait
# was then re-evolved with a steer gene (encoding=steer, 4-cycle evaluation, turnw=5: straight0xx)
# and its steer gene tuned for zero steady turn (ga_trainer tune=, straight0xx_tuned): 66 m to
# >1 km radius. These are the tuned gaits; speeds re-measured over 4 cycles.
RUNS = {"straight015_tuned": 0.485 / 4, "straight030_tuned": 0.656 / 4, "straight045_tuned": 0.970 / 4,
        "straight055_tuned": 1.248 / 4, "straight060_tuned": 1.465 / 4, "straight065_tuned": 1.518 / 4,
        "straight070_tuned": 1.621 / 4, "straight075_tuned": 1.833 / 4, "straight090_tuned": 1.909 / 4}
HELD_OUT = "straight060_tuned"
MID = (MOTOR_HI + MOTOR_LO) / 2
HALF = 1.1 * (MOTOR_HI - MOTOR_LO) / 2


def load_frames(run):
    """outputs/<run>/best_gait.csv (header of motor names + 300 rows) -> (300, 24) radians."""
    return np.loadtxt(os.path.join(ROOT, "outputs", run, "best_gait.csv"), delimiter=",", skiprows=1)


def scale(angles):
    return (np.asarray(angles) - MID) / HALF


def unscale(y):
    """Network output -> motor angles, clipped to the motor limits."""
    return np.clip(MID + HALF * np.asarray(y), MOTOR_LO, MOTOR_HI)


def inputs(speed, speed_mean, speed_std, n=FRAMES):
    """One cycle of network inputs (n, 3) for a speed in metres per cycle."""
    phase = 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.sin(phase), np.cos(phase), np.full(n, (speed - speed_mean) / speed_std)])


def neighbours(run, exclude=()):
    """The nearest slower and faster gaits to `run` (skipping `exclude`), for the blend baseline."""
    others = sorted((r for r in RUNS if r != run and r not in exclude), key=RUNS.get)
    lower = [r for r in others if RUNS[r] < RUNS[run]]
    upper = [r for r in others if RUNS[r] > RUNS[run]]
    return lower[-1], upper[0]


def build(held_out=HELD_OUT, val_run=None, val_frac=0.15, seed=0):
    """-> dict with X/Y for train, val, test, the speed normalisation and the run names.
    val_run: a speed to hold out for validation; None = a random val_frac of the training frames."""
    non_test = [r for r in RUNS if r != held_out]
    speeds = np.array([RUNS[r] for r in non_test])
    mean, std = float(speeds.mean()), float(speeds.std())
    train_runs = [r for r in non_test if r != val_run]
    X = np.vstack([inputs(RUNS[r], mean, std) for r in train_runs])
    Y = np.vstack([scale(load_frames(r)) for r in train_runs])
    if val_run is None:
        idx = np.random.default_rng(seed).permutation(len(X))
        n_val = int(val_frac * len(X))
        val, tr = idx[:n_val], idx[n_val:]
        X_val, Y_val, X, Y = X[val], Y[val], X[tr], Y[tr]
    else:
        X_val, Y_val = inputs(RUNS[val_run], mean, std), scale(load_frames(val_run))
    return {"X_train": X, "Y_train": Y, "X_val": X_val, "Y_val": Y_val,
            "X_test": inputs(RUNS[held_out], mean, std), "Y_test": scale(load_frames(held_out)),
            "speed_mean": mean, "speed_std": std, "train_runs": train_runs, "held_out": held_out}


if __name__ == "__main__":
    d = build()
    assert d["X_train"].shape[1] == 3 and d["Y_train"].shape[1] == 24
    assert len(d["X_train"]) + len(d["X_val"]) == (len(RUNS) - 1) * FRAMES and d["X_test"].shape == (FRAMES, 3)
    assert np.abs(d["Y_train"]).max() <= 1 / 1.1 + 1e-9                     # targets inside tanh's range
    f = load_frames("straight045_tuned")
    assert np.allclose(unscale(scale(f)), f)                                  # scaling round trip
    assert not np.any(np.isclose(d["X_train"][:, 2], d["X_test"][0, 2]))      # held-out speed never in training
    f2 = build(val_run="straight045_tuned")
    assert len(f2["X_train"]) == (len(RUNS) - 2) * FRAMES and np.isclose(f2["X_val"][0, 2], (RUNS["straight045_tuned"] - d["speed_mean"]) / d["speed_std"])
    print(f"speed_data self-check passed: {len(d['X_train'])} train, {len(d['X_val'])} val, "
          f"{len(d['X_test'])} test frames; train speeds {[round(RUNS[r], 3) for r in d['train_runs']]}, "
          f"held out {d['held_out']} = {RUNS[d['held_out']]:.3f} m/cycle")
