"""Chromosome encoding: 26 real genes -> one 300-frame gait for the Webots spider (300 x 24 angles).

Angles are Webots motor angles (SpiderOctopod.proto), so a decoded gait plays on the robot as-is.
Each frame is a 1x24 pose in the MATLAB leg order the spec uses: L1,L2,L3,L4,R4,R3,R2,R1, and per
leg (hip, thigh, knee). MOTOR_NAMES gives the Webots device for each of the 24 columns.

Gene layout (lessons from the first version, see docs/encoding.txt):
    chrom[0:12]   base pose of the LEFT legs L1..L4 (hip, thigh, knee). Right legs mirror them:
                  R_k = -L_k for all three joints (the proto's right joints turn the other way;
                  the stock standing pose is L thigh -1.07 / knee 0.58, R 1.07 / -0.58).
    chrom[12:24]  swing amplitude of the same 12 joints, mirrored the same way.
    chrom[24]     femur - phase of every thigh relative to its own leg's hip.
    chrom[25]     tibia - phase of every knee relative to its own leg's hip.

    leg_phase = LR * is_right + FB * position     (position L1..L4 = 0..3, R4..R1 = 3..0)
    angle_j(t) = clip(offset_j + amp_j * sin(2*pi*t/300 + leg_phase + joint_phase_j), motor limits)

LR = FB = pi are fixed: the alternating tetrapod of real spiders, {L1,L3,R2,R4} then {L2,L4,R1,R3}.
With LR as a gene (version 1) the GA got stuck on left/right pairs moving together ("rowing").
With FB as a gene the first physics run chose fb = 0.52 pi, a quarter-cycle wave that looked out
of sync, with the front legs pulling and the 3rd pair dragged.
"""
import numpy as np

FRAMES = 300
N_JOINTS = 24
N_POSE = 12
N_PHASES = 2
N_GENES = 2 * N_POSE + N_PHASES
LR = np.pi                                                # right side half a cycle behind
FB = np.pi                                                # neighbouring legs on a side alternate

LEGS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
MOTOR_NAMES = [f"{leg}_{j}_joint" for leg in LEGS for j in ("hip", "thigh", "knee")]
MOTOR_LO = np.tile([-1.5708, -1.5708, -2.0], 8)          # from SpiderOctopod.proto
MOTOR_HI = np.tile([1.5708, 1.5708, 1.0], 8)

# Left-leg pose bounds. The hips are only 0.325 m apart and the legs point straight out, so
# neighbouring legs swinging towards each other collide unless the legs fan out (front forward,
# hind back): measured collision-free hip swing 0.15 rad with no fan, 0.35 at +-0.6, 0.55 at +-1.2.
# With hip +-0.6 the GA kept hip swing near 0 and stepped in place, so hips get +-1.2.
# Thigh [-1.5, 0]: for a left leg negative = down, so the legs stay below the body (standing
# is -1.07). Knee [-0.5, 1]: a right knee is the negated left knee and must fit [-2, 1].
POSE_LO = np.tile([-1.2, -1.5, -0.5], 4)
POSE_HI = np.tile([1.2, 0.0, 1.0], 4)
AMP_HI = np.tile([0.6, 0.5, 0.5], 4)

LOW = np.concatenate([POSE_LO, np.zeros(N_POSE), np.zeros(N_PHASES)])
HIGH = np.concatenate([POSE_HI, AMP_HI, np.full(N_PHASES, 2 * np.pi)])
WRAP = np.arange(N_GENES) >= 2 * N_POSE                   # phases are angles: wrap, don't clip

# Shared-motion encoding, 11 genes: every leg makes the same step (same thigh/knee pose, same swing,
# same joint timing); only the hip direction differs per leg, because the legs must fan out
# (front forward, hind back) to swing without colliding. Mirrored and phased as above.
#   s[0:4] hip pose of L1..L4   s[4] thigh pose   s[5] knee pose
#   s[6:9] hip, thigh, knee swing amplitude       s[9:11] thigh, knee phase
SHARED_GENES = 11
SHARED_LOW = np.concatenate([POSE_LO[0::3], POSE_LO[1:3], np.zeros(3), np.zeros(N_PHASES)])
SHARED_HIGH = np.concatenate([POSE_HI[0::3], POSE_HI[1:3], AMP_HI[:3], np.full(N_PHASES, 2 * np.pi)])
SHARED_WRAP = np.arange(SHARED_GENES) >= 9

# Push-pull encoding, 12 genes = shared motion + p[11]: knee phase offset of the hind two pairs
# (L3, L4, R4, R3) relative to the front two. The knee sets how far the foot reaches out. Front
# legs point forward, so drawing the foot in while it is down PULLS the body forward; hind legs
# point backward, so the same motion pulls it backward. An offset of pi makes the hind legs
# extend while the front ones draw in: front legs pull, hind legs push (the user's suggestion).
# Left as a gene so the GA shows whether pi is what works.
PUSHPULL_GENES = 12
PUSHPULL_LOW = np.append(SHARED_LOW, 0.0)
PUSHPULL_HIGH = np.append(SHARED_HIGH, 2 * np.pi)
PUSHPULL_WRAP = np.append(SHARED_WRAP, True)

# Steer encoding, 13 genes = push-pull + p[12]: left hips swing (1 + steer) x the hip amplitude,
# right hips (1 - steer), so one side takes longer strides. Mirroring makes the gait left/right
# symmetric by design, yet every push-pull gait curved left in physics (ref10_2: 21 m turning radius
# over 20 cycles, and the speed family + the network trained on it inherited that); without this
# gene the GA had no way to correct it.
STEER_GENES = 13
STEER_MAX = 0.3
_STEER = np.where(np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 3) == 1, -1.0, 1.0) * (np.tile([0, 1, 2], 8) == 0)

_T = 2 * np.pi * np.arange(FRAMES) / FRAMES
_SOURCE = np.concatenate([np.arange(12), np.arange(12).reshape(4, 3)[::-1].ravel()])  # R4..R1 <- L4..L1
_RIGHT = np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 3)
_MIRROR = np.where(_RIGHT == 1, -1.0, 1.0)
_POSITION = np.repeat([0, 1, 2, 3, 3, 2, 1, 0], 3)
_JOINT = np.tile([0, 1, 2], 8)                            # 0 hip, 1 thigh, 2 knee
_HIND_KNEE = ((_JOINT == 2) & (_POSITION >= 2)).astype(float)


def phases(chrom):
    """chrom (..., 26) -> per-joint phase (..., 24)."""
    c = np.asarray(chrom, dtype=float)
    rel = np.concatenate([np.zeros_like(c[..., :1]), c[..., 24:26]], axis=-1)[..., _JOINT]
    return LR * _RIGHT + FB * _POSITION + rel


def expand_shared(chrom):
    """Shared-motion chromosome (..., 11) -> the equivalent per-leg chromosome (..., 26)."""
    s = np.asarray(chrom, dtype=float)
    thigh_knee = np.broadcast_to(s[..., None, 4:6], (*s.shape[:-1], 4, 2))
    pose = np.concatenate([s[..., 0:4, None], thigh_knee], axis=-1).reshape(*s.shape[:-1], 12)
    amp = np.concatenate([s[..., 6:9]] * 4, axis=-1)
    return np.concatenate([pose, amp, s[..., 9:11]], axis=-1)


FOOT_GENES = 7     # encoding=foot: foot-path settings, decoded by the per-leg IK network (Part 2)


def decode(chrom):
    """chrom (..., 26) per-leg, (..., 11) shared-motion, (..., 12) push-pull or (..., 13) push-pull
    + steer -> motor angles (..., 300, 24), clipped to the motor limits. A single (7,) foot-path
    chromosome goes to part2_nn/ik/foot_gait.decode (imported only then: it loads the network)."""
    c = np.asarray(chrom, dtype=float)
    if c.shape == (FOOT_GENES,):
        from part2_nn.ik import foot_gait
        return foot_gait.decode(c)
    offset, steer = 0.0, 1.0
    if c.shape[-1] == STEER_GENES:
        steer = 1.0 + c[..., 12:13] * _STEER
        c = c[..., :PUSHPULL_GENES]
    if c.shape[-1] == PUSHPULL_GENES:
        offset = c[..., 11:12] * _HIND_KNEE
        c = c[..., :SHARED_GENES]
    if c.shape[-1] == SHARED_GENES:
        c = expand_shared(c)
    pose = _MIRROR * c[..., _SOURCE]
    amp = _MIRROR * c[..., N_POSE + _SOURCE] * steer
    angles = pose[..., None, :] + amp[..., None, :] * np.sin(_T[:, None] + (phases(c) + offset)[..., None, :])
    return np.clip(angles, MOTOR_LO, MOTOR_HI)


def random_population(n, rng):
    return rng.uniform(LOW, HIGH, size=(n, N_GENES))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pop = random_population(5, rng)
    g = decode(pop)
    assert g.shape == (5, FRAMES, N_JOINTS) and np.allclose(decode(pop[0]), g[0])
    assert (g >= MOTOR_LO).all() and (g <= MOTOR_HI).all()
    c = pop[0].copy()
    c[:12] = np.tile([0.0, -1.07, 0.58], 4)
    c[12:24] = 0
    stand = decode(c)[0]
    assert np.allclose(stand[:3], [0, -1.07, 0.58]) and np.allclose(stand[21:], [0, 1.07, -0.58])
    c[12:24] = 0.3
    c[24:] = [0, 0]
    f = decode(c)
    i = {name: k for k, name in enumerate(MOTOR_NAMES)}
    # R1 is mirrored (-) AND half a cycle behind (-): same motor angle as L1, i.e. it swings the
    # opposite way in the world, so L1 and R1 alternate
    assert np.allclose(f[:, i["L1_hip_joint"]], f[:, i["R1_hip_joint"]])
    assert np.allclose(f[:, i["L1_hip_joint"]], f[:, i["L3_hip_joint"]])        # L1, L3 in step
    assert np.allclose(f[:, i["L1_hip_joint"]], -f[:, i["L2_hip_joint"]])       # L1, L2 alternate
    sh = rng.uniform(SHARED_LOW, SHARED_HIGH, size=(3, SHARED_GENES))
    full = expand_shared(sh)
    assert full.shape == (3, N_GENES) and np.allclose(decode(sh), decode(full))
    assert np.allclose(full[:, 12:24].reshape(3, 4, 3), sh[:, None, 6:9])         # every leg: same swing
    assert np.allclose(full[:, :12].reshape(3, 4, 3)[..., 0], sh[:, :4])            # own hip direction
    assert np.allclose(full[:, :12].reshape(3, 4, 3)[..., 1:], sh[:, None, 4:6])    # same thigh/knee pose
    mid = (SHARED_LOW + SHARED_HIGH) / 2
    mid[6:9] = 0.2                                                               # small swing: nothing clipped
    fp, fs = decode(np.append(mid, np.pi)), decode(mid)
    rest = decode(np.append(mid[:6], np.zeros(5)))[0]                            # zero swing = resting pose
    knee = np.array([MOTOR_NAMES.index(f"{leg}_knee_joint") for leg in LEGS])
    hind = np.isin(np.array(LEGS), ["L3", "L4", "R4", "R3"])
    assert np.allclose(fp[:, knee[~hind]], fs[:, knee[~hind]])                 # front knees unchanged
    assert np.allclose(fp[:, knee[hind]] + fs[:, knee[hind]], 2 * rest[knee[hind]])  # hind knee swing reversed
    assert np.allclose(decode(np.append(mid, 0.0)), fs)                          # offset 0 = shared motion
    pp = np.append(mid, np.pi)
    assert np.allclose(decode(np.append(pp, 0.0)), fp)                           # steer 0 = push-pull
    st = decode(np.append(pp, 0.2))
    swing = lambda f, leg: np.ptp(f[:, MOTOR_NAMES.index(f"{leg}_hip_joint")])
    assert np.isclose(swing(st, "L2"), 1.2 * swing(fp, "L2")) and np.isclose(swing(st, "R2"), 0.8 * swing(fp, "R2"))
    assert np.allclose(np.delete(st, [MOTOR_NAMES.index(f"{l}_hip_joint") for l in LEGS], axis=1),
                       np.delete(fp, [MOTOR_NAMES.index(f"{l}_hip_joint") for l in LEGS], axis=1))  # only hips change
    print("gait self-check passed")
