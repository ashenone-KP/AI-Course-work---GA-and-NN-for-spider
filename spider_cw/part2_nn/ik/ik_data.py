"""Option E data: random joint angles -> foot positions, for the inverse-kinematics network.

    cd spider_cw
    python -m part2_nn.ik.ik_data      # self-check

The network learns feet -> angles:
    input   24 numbers, (x, y, z) of the 8 feet in the body frame (metres), leg order L1..L4, R4..R1
    output  24 Webots motor angles (hip, thigh, knee per leg)
Training pairs are made with the forward kinematics in part1_ga/body.py, run on random angles.
FK is exact and fast, so we can make as many examples as we like, and every target is reachable.

Angles are drawn from the same ranges the GA uses for a leg's base pose (gait.POSE_LO/HI):
left hip [-1.2, 1.2], thigh [-1.5, 0] (leg below the body), knee [-0.5, 1]. Right legs mirror the
left ones (R = -L), as in gait.py. Each leg is drawn independently, so the set covers far more
poses than the GA gaits do.
"""
import numpy as np

from part1_ga.body import HIP_POS, SIDE, forward_kinematics
from part1_ga.gait import MOTOR_HI, MOTOR_LO, POSE_HI, POSE_LO

LEFT_LO, LEFT_HI = POSE_LO[:3], POSE_HI[:3]
# Left-leg angle ranges the training poses are drawn from. "pose": the GA's base-pose ranges (first
# models). "motor": every angle a left motor and a mirrored right one can both reach (right knee
# [-2, 1] mirrors to [-1, 2]), leg below the body: real GA gaits push hips to 1.46 and thighs to -1.57.
BOXES = {"pose": (LEFT_LO, LEFT_HI),
         "motor": (np.array([MOTOR_LO[0], MOTOR_LO[1], -1.0]), np.array([MOTOR_HI[0], 0.0, 1.0]))}
SIGN = np.repeat(SIDE, 3)                    # +1 for left-leg columns, -1 for right-leg columns
N_SAMPLES = 50_000
N_TRAIN, N_VAL = 40_000, 5_000               # the remaining 5,000 are the test set


def random_angles(n, rng, box="pose"):
    """n random poses (n, 24) inside the left-leg ranges, mirrored for the right legs."""
    lo, hi = BOXES[box]
    return rng.uniform(np.tile(lo, 8), np.tile(hi, 8), size=(n, 24)) * SIGN


def feet_of(angles):
    """Angles (..., 24) -> foot positions (..., 24)."""
    feet = forward_kinematics(angles)[..., 3, :]
    return feet.reshape(*feet.shape[:-2], 24)


def scale(angles, lo=MOTOR_LO, hi=MOTOR_HI):
    """Motor angles -> [-1, 1] using the motor limits, so a tanh output is always a legal angle."""
    return 2 * (angles - lo) / (hi - lo) - 1


def unscale(y, lo=MOTOR_LO, hi=MOTOR_HI):
    return lo + (y + 1) / 2 * (hi - lo)


def build(seed=0, n=N_SAMPLES, box="pose"):
    """The whole dataset, split 40k / 5k / 5k. Inputs are standardised per coordinate with the
    mean and std of the TRAINING feet only (the test set must not leak into anything we fit)."""
    rng = np.random.default_rng(seed)
    angles = random_angles(n, rng, box)
    feet = feet_of(angles)
    tr, va = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_VAL)
    te = slice(N_TRAIN + N_VAL, n)
    mean, std = feet[tr].mean(0), feet[tr].std(0)
    d = {"mean": mean, "std": std}
    for name, part in (("train", tr), ("val", va), ("test", te)):
        d[name] = {"X": (feet[part] - mean) / std, "Y": scale(angles[part]),
                   "feet": feet[part], "angles": angles[part]}
    return d


def foot_error_cm(pred_angles, target_feet):
    """How far (cm) each foot ends up from its target when the predicted angles are played: (n, 8).
    This is the error that matters on the robot, not the angle error."""
    diff = feet_of(pred_angles) - target_feet
    return 100 * np.linalg.norm(diff.reshape(-1, 8, 3), axis=-1)


def per_leg(feet, angles):
    """Split whole-body samples into one sample per leg, all in left-leg form: the foot relative
    to its own hip, with y flipped for right legs, and the right legs' angles negated.
    (n, 24), (n, 24) -> (8n, 3), (8n, 3). Lets one small 3 -> 3 network serve all 8 legs."""
    f = feet.reshape(-1, 8, 3) - HIP_POS
    f[..., 1] *= SIDE
    a = angles.reshape(-1, 8, 3) * SIDE[:, None]
    return f.reshape(-1, 3), a.reshape(-1, 3)


def from_per_leg(leg_angles):
    """Inverse of the angle part of per_leg: (8n, 3) left-form angles -> (n, 24) motor angles."""
    return (leg_angles.reshape(-1, 8, 3) * SIDE[:, None]).reshape(-1, 24)


HIP_AXIS_LEFT = np.array([0.0, -1.0, 1.0]) / np.sqrt(2)       # left hip axis (proto), right legs mirror it


def axis_distance(feet):
    """Distance (m) of each foot from its own hip's rotation axis: (n, 24) -> (n, 8).
    Near 0 the hip angle hardly moves the foot, so many hip angles give the same foot (singular)."""
    f = feet.reshape(-1, 8, 3) - HIP_POS
    f[..., 1] *= SIDE                                            # left-leg form
    along = f @ HIP_AXIS_LEFT
    return np.linalg.norm(f - along[..., None] * HIP_AXIS_LEFT, axis=-1)


def ambiguity_check(n=3000, near_cm=2.0, seed=1):
    """Can two clearly different angle sets put a foot in the same place (inside our ranges)?
    For one leg: draw n poses, find each one's nearest neighbour by foot position, and look at
    how different the angles of pairs closer than near_cm are. If the IK has one answer, close
    feet always mean close angles. Returns (pairs found, largest angle difference in degrees).
    Result: the knee-up/knee-down second solution needs |knee| > pi - 1, outside the motor limits,
    so it never appears. The pairs that do differ are hip angles of a tucked leg whose foot is
    close to the hip axis (see axis_distance)."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n, 24))
    a[:, :3] = rng.uniform(LEFT_LO, LEFT_HI, size=(n, 3))
    f = feet_of(a)[:, :3]
    d = np.linalg.norm(f[:, None, :] - f[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    j = d.argmin(1)
    close = 100 * d[np.arange(n), j] < near_cm
    gap = np.degrees(np.abs(a[close, :3] - a[j[close], :3]).max(1))
    return int(close.sum()), float(gap.max())


if __name__ == "__main__":
    d = build()
    assert d["train"]["X"].shape == (N_TRAIN, 24) and d["test"]["Y"].shape == (N_SAMPLES - N_TRAIN - N_VAL, 24)
    assert np.abs(d["train"]["X"].mean(0)).max() < 1e-9 and np.allclose(d["train"]["X"].std(0), 1)
    for part in ("train", "val", "test"):
        assert np.abs(d[part]["Y"]).max() <= 1                       # every target fits a tanh output
        assert np.allclose(unscale(d[part]["Y"]), d[part]["angles"])
        assert foot_error_cm(d[part]["angles"], d[part]["feet"]).max() < 1e-9   # exact angles: 0 error

    f, a = per_leg(d["test"]["feet"], d["test"]["angles"])
    assert np.allclose(from_per_leg(a), d["test"]["angles"])
    # mirroring must be exact: a right leg with angles -a has the same relative foot as a left leg with a
    left = np.zeros((1, 24))
    left[0, :3] = [0.4, -0.8, 0.3]                                   # L1
    right = np.zeros((1, 24))
    right[0, 21:] = [-0.4, 0.8, -0.3]                                # R1, mirrored
    fl, _ = per_leg(feet_of(left), left)
    fr, _ = per_leg(feet_of(right), right)
    assert np.allclose(fl[0], fr[7]), (fl[0], fr[7])

    pairs, worst = ambiguity_check()
    print(f"ambiguity check: {pairs} nearest-neighbour pairs with feet < 2 cm apart, "
          f"largest joint angle difference {worst:.1f} deg")
    near = axis_distance(d["train"]["feet"])
    print(f"feet within 10 cm of their hip axis: {100 * (near < 0.10).mean():.1f}% "
          f"(within 5 cm: {100 * (near < 0.05).mean():.1f}%)")
    print(f"feet range (m): x {d['train']['feet'][:, 0::3].min():.2f}..{d['train']['feet'][:, 0::3].max():.2f}, "
          f"z {d['train']['feet'][:, 2::3].min():.2f}..{d['train']['feet'][:, 2::3].max():.2f}")
    print("ik_data self-check passed")
