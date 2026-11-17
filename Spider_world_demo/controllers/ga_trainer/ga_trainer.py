"""Evolve a walking gait for the Webots spider, scoring every gait in Webots physics.

Set the robot's controller to "ga_trainer" (selfCollision TRUE) and press play. Options go in
controllerArgs as key=value, e.g.  [ "pop=24" "gens=40" "seed=0" "name=run0" ]:
    pop 24, gens 40, seed 0   GA population, generations, random seed
    period 2.0                seconds per gait cycle (300 frames)
    cycles 2                  gait cycles walked per evaluation
    show 1                    after each generation, replay the best gait in real time (0 = never)
    encoding shared           "shared": every leg makes the same step, 11 genes; "pushpull": shared plus a
                              hind-knee timing gene, 12 genes; "pushpi": push-pull with that gene fixed
                              at pi (front legs pull, hind legs push); "steer": pushpi plus a gene that
                              swings one side's hips more than the other's, 13 genes; "legs": per-leg, 26 genes;
                              "foot": 7 foot-path genes (stride, lift, duty, height, reach, spread, wave) turned
                              into joint angles by the per-leg IK network (part2_nn/ik/foot_gait.py)
    turnw 0.5                 penalty per rad of heading change over the evaluation
    pitch 1                   1: penalise body tilt beyond `tilt`; 0: no tilt term
    tilt 0.07                 body tilt (rad, mean) that is free; the default PITCH_FREE keeps the old runs
                              reproducible, 0.14 (8 deg) allows a slight lean
    slip 0.3                  penalty per metre of foot sliding (SLIP_PENALTY)
    selection tournament      GA operators (part1_ga/ga.py): selection tournament|roulette,
    crossover blend           crossover uniform|two_point|blend, adaptive 1|0 (adaptive or fixed
    adaptive 1, mrate 0.05    mutation sigma), mrate = per-gene mutation probability
    target <m>                walk this far per evaluation instead of as far as possible: the forward
                              term becomes -2 x |forward - target| (makes a family of gaits at set
                              speeds, the training data for the Part 2 neural network)
    start <run>               refine a saved run: the population starts as that run's best gait plus
                              small variations (3% of each gene's range), mutation sigma 0.03
    name run<seed>            output folder spider_cw/outputs/<name>/
    selftest 1                only check that physics is repeatable, then stop
    measure <run>             walk a saved gait `cycles` cycles; print body, per-leg and path figures
    tune <run>                line search on a steer run's steer gene for zero steady turn (see tune())
    replay <run>              walk the best gait of each improving generation in real time, captioned,
                              and record outputs/<run>/learning.mp4 (needs rendering, not --no-rendering)

How a gait is scored (fitness(), all lengths in metres over the evaluation):
    1. The robot is put back in the saved start state (standing still at the start position).
    2. It moves to the gait's first pose (SETTLE_S), then walks `cycles` cycles.
    3. score = forward distance - 0.5 |sideways drift| - turnw |turn (rad)|
               - LOW_PENALTY x fraction of the walk with the body below MIN_BODY_HEIGHT
               - 2 x (how far the body dipped below it)
               - FALL_PENALTY if it tipped over (evaluation stops there)
               - COLLISION_PENALTY x fraction of frames whose commanded pose makes legs collide
               + STEP_REWARD x mean foot lift (capped at LIFT_CAP) of the legs that step
               - IDLE_PENALTY per leg that does not step (touch down AND lift MIN_LIFT)
               - SLIP_PENALTY x metres the feet slide while on the floor
               - PITCH_PENALTY x mean body tilt beyond PITCH_FREE (rad, nose up or down)
    Feet are measured, not commanded: joint sensors + body pose each step -> body.forward_kinematics.
    (Added after the first live run: front legs pulled, the 3rd pair never lifted and was dragged.)
    Gaits that collide in more than SKIP_COLLISION of their frames are not simulated at all.
The GA (part1_ga/ga.py: tournament selection, blend crossover, adaptive Gaussian mutation,
elitism) only sees the scores. Physics is deterministic, so scores are cached per chromosome.

After every generation the run is saved to spider_cw/outputs/<name>/: history.json, best_chrom.json
and best_gait.csv (300 x 24 motor angles, columns = gait.MOTOR_NAMES). Plot with
    python -m part1_ga.plots outputs/<name>
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "spider_cw"))
sys.path.insert(0, ROOT)

import numpy as np                                      # noqa: E402 (needs the path above)
from controller import Supervisor                       # noqa: E402

from part1_ga import body, gait                         # noqa: E402
from part1_ga.ga import GAConfig, run                   # noqa: E402

SETTLE_S = 1.5              # time to move from the start pose into the gait's first pose
MIN_BODY_HEIGHT = 0.25      # body centre height (m); the box is 0.3 tall, so 0.15 = belly on the floor
LOW_PENALTY = 5.0           # first runs slid on their bellies (min height 0.14) when this was only a 2x dip penalty
FALL_PENALTY = 5.0
COLLISION_PENALTY = 3.0
SKIP_COLLISION = 0.2
STAND = np.tile([0.0, -1.07, 0.58], 8) * np.repeat(body.SIDE, 3)   # stock standing pose
CONTACT_Z = 0.04            # a foot tip (tibia end, radius 0.02) below this height is on the floor
LIFT_CAP = 0.10             # foot lift above this earns nothing more
MIN_LIFT = 0.05             # a leg "steps" only if its foot touches down AND rises at least this high
STEP_REWARD = 0.5           # x mean capped lift of the legs that step (0..1)
IDLE_PENALTY = 0.4          # per leg that does not step (still, held up, or dragged along the floor)
PITCH_PENALTY = 3.0         # per rad of mean body tilt beyond PITCH_FREE; shared1 walked nose-down
                            # 0.21 rad (12 deg) the whole time, which looked wrong in Webots
PITCH_FREE = 0.07           # 4 deg of tilt is free: penalising all tilt levelled the walk to ~1 deg but
                            # halved its speed (level0-2: 0.05-1.24 m over 4 cycles vs shared1 2.48 m)
SLIP_PENALTY = 0.3          # per metre feet slide while on the floor. Was 0.1: run1 then slid 11 m for
                            # 3.8 m forward. Not higher: sine joint motion always skates a little, and a
                            # big weight would make standing still the best option


def foot_bounds():
    """encoding=foot gene bounds (gait.decode sends 7-gene chromosomes to the per-leg IK network)."""
    from part2_nn.ik import foot_gait
    return foot_gait.FOOT_LOW, foot_gait.FOOT_HIGH, foot_gait.FOOT_WRAP


def leg_metrics(pos, rot, angles):
    """Real foot motion from a physics run: body position (T,3), orientation (T,3,3) and measured
    joint angles (T,24) -> per-leg peak lift, touched-down flag, stepping flag and slide distance."""
    feet = pos[:, None, :] + np.einsum("tij,tlj->tli", rot, body.forward_kinematics(angles)[:, :, 3, :])
    z, xy = feet[..., 2], feet[..., :2]
    contact = z < CONTACT_Z
    lift = z.max(axis=0)
    stepping = contact.any(axis=0) & (~contact).any(axis=0) & (lift >= MIN_LIFT)
    both = contact[1:] & contact[:-1]
    slide = (np.linalg.norm(np.diff(xy, axis=0), axis=-1) * both).sum(axis=0)
    return {"lift": lift, "touch": contact.mean(axis=0), "stepping": stepping, "slide": slide}


def parse_args(argv):
    opts = {"pop": 24, "gens": 40, "seed": 0, "period": 2.0, "cycles": 2, "show": 1, "name": None, "selftest": 0,
            "measure": None, "encoding": "shared", "pitch": 1, "slip": SLIP_PENALTY, "start": None, "target": None,
            "selection": "tournament", "crossover": "blend", "adaptive": 1, "mrate": 0.05, "turnw": 0.5, "tune": None, "replay": None,
            "tilt": PITCH_FREE}
    for a in argv:
        key, _, value = a.partition("=")
        if key not in opts:
            raise SystemExit(f"ga_trainer: unknown option {key!r}; known: {', '.join(opts)}")
        opts[key] = (value if key in ("name", "measure", "tune", "replay", "encoding", "start", "selection", "crossover")
                     else float(value) if key in ("period", "slip", "target", "mrate", "turnw", "tilt") else int(value))
    if opts["encoding"] not in ("shared", "pushpull", "pushpi", "steer", "legs", "foot"):
        raise SystemExit("ga_trainer: encoding must be shared, pushpull, pushpi, steer, legs or foot")
    if opts["selection"] not in ("tournament", "roulette") or opts["crossover"] not in ("uniform", "two_point", "blend"):
        raise SystemExit("ga_trainer: selection = tournament|roulette, crossover = uniform|two_point|blend")
    opts["name"] = opts["name"] or f"run{opts['seed']}"
    return opts


class Trainer:
    def __init__(self, opts):
        self.opts = opts
        self.sup = Supervisor()
        self.dt = int(self.sup.getBasicTimeStep())
        self.robot = self.sup.getSelf()
        self.motors = [self.sup.getDevice(n) for n in gait.MOTOR_NAMES]
        self.sensors = [self.sup.getDevice(n + "_sensor") for n in gait.MOTOR_NAMES]
        for sensor in self.sensors:
            sensor.enable(self.dt)
        self.cache, self.history, self.gen = {}, [], 0
        self.out = os.path.join(ROOT, "outputs", opts["name"])
        self.command(STAND)                              # stand up, then remember this as the start
        self.wait(2.0)
        self.robot.saveState("start")

    # ------------------------------------------------------------------ simulation helpers
    def command(self, angles):
        for m, a in zip(self.motors, angles):
            m.setPosition(float(a))

    def wait(self, seconds):
        for _ in range(int(seconds * 1000 / self.dt)):
            if self.sup.step(self.dt) == -1:
                raise SystemExit(0)

    def pose(self):
        """(x, y, z, yaw, up_z) of the body."""
        x, y, z = self.robot.getPosition()
        r = self.robot.getOrientation()                  # 3x3 row-major
        return x, y, z, math.atan2(r[3], r[0]), r[8]

    def walk(self, frames, cycles):
        """Restore the start state, settle into frame 0, play `cycles` cycles. Returns measurements."""
        self.robot.loadState("start")
        self.robot.resetPhysics()
        self.command(frames[0])
        self.wait(SETTLE_S)
        x0, y0, _, yaw0, _ = self.pose()
        x, y, yaw = x0, y0, yaw0
        n, period = len(frames), self.opts["period"]
        min_z, low, fell, track = math.inf, 0, False, []
        for s in range(int(cycles * period * 1000 / self.dt)):
            phase = s * self.dt / 1000 / period * n
            k, w = int(phase), phase - int(phase)
            self.command(frames[k % n] * (1 - w) + frames[(k + 1) % n] * w)
            if self.sup.step(self.dt) == -1:
                raise SystemExit(0)
            x, y, z, yaw, up = self.pose()
            track.append((self.robot.getPosition(), self.robot.getOrientation(),
                          [sensor.getValue() for sensor in self.sensors]))
            min_z = min(min_z, z)
            low += z < MIN_BODY_HEIGHT
            if up < 0.5:                                 # tilted past 60 degrees: fallen over
                fell = True
                break
        dx, dy = x - x0, y - y0
        c, s_ = math.cos(yaw0), math.sin(yaw0)           # distances in the start heading's frame
        rot = np.array([t[1] for t in track]).reshape(-1, 3, 3)
        legs = leg_metrics(np.array([t[0] for t in track]), rot, np.array([t[2] for t in track]))
        pitch = -np.arcsin(np.clip(rot[:, 2, 0], -1, 1))    # body x axis below horizontal = nose down (rad)
        rel = np.array([t[0][:2] for t in track]) - (x0, y0)
        path = np.column_stack([rel @ (c, s_), rel @ (-s_, c),       # along / sideways from the start heading,
                                np.unwrap(np.arctan2(rot[:, 1, 0], rot[:, 0, 0])) - yaw0])  # heading change (rad)
        return {"forward": dx * c + dy * s_, "drift": -dx * s_ + dy * c,
                "turn": (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi,
                "min_height": min_z, "low_time": low / max(s + 1, 1), "fell": float(fell),
                "lift": float((np.minimum(legs["lift"], LIFT_CAP) * legs["stepping"]).mean() / LIFT_CAP),
                "idle_legs": float((~legs["stepping"]).sum()), "slide": float(legs["slide"].sum()),
                "pitch": float(np.abs(pitch).mean()), "nose_down": float(pitch.mean()), "legs": legs, "path": path}

    # ------------------------------------------------------------------ fitness
    def fitness(self, chrom):
        key = np.round(chrom, 12).tobytes()
        if key in self.cache:
            return self.cache[key]
        frames = gait.decode(chrom)
        coll = float(body.collision_fraction(frames))
        if coll > SKIP_COLLISION:
            m = {"forward": 0.0, "drift": 0.0, "turn": 0.0, "min_height": 0.0, "low_time": 0.0, "fell": 0.0,
                 "lift": 0.0, "idle_legs": 8.0, "slide": 0.0, "pitch": 0.0, "nose_down": 0.0, "skipped": 1.0}
        else:
            m = self.walk(frames, self.opts["cycles"])
            m.pop("legs")
            m.pop("path")
            m["skipped"] = 0.0
        m["collision"] = coll
        goal = m["forward"] if self.opts["target"] is None else -2 * abs(m["forward"] - self.opts["target"])
        score = (goal - 0.5 * abs(m["drift"]) - self.opts["turnw"] * abs(m["turn"])
                 - (LOW_PENALTY * m["low_time"] + 2 * max(0.0, MIN_BODY_HEIGHT - m["min_height"])) * (1 - m["skipped"])
                 - FALL_PENALTY * m["fell"] - COLLISION_PENALTY * coll
                 + STEP_REWARD * m["lift"] - IDLE_PENALTY * m["idle_legs"] - self.opts["slip"] * m["slide"]
                 - PITCH_PENALTY * max(0.0, m["pitch"] - self.opts["tilt"]) * self.opts["pitch"])
        self.cache[key] = (score, m)
        return score, m

    def evaluate_population(self, pop):
        scores, comps = [], []
        for i, chrom in enumerate(pop):
            self.label(f"generation {self.gen}: evaluating {i + 1}/{len(pop)}", 1)
            s, m = self.fitness(chrom)
            scores.append(s)
            comps.append(m)
        return np.array(scores), {k: np.array([c[k] for c in comps]) for k in comps[0]}

    # ------------------------------------------------------------------ progress display + saving
    def label(self, text, line):
        self.sup.setLabel(line, text, 0.01, 0.01 + 0.05 * line, 0.07, 0xFFFFFF, 0, "Arial")

    def on_generation(self, rec):
        self.history.append(rec)
        c = rec["components"]
        print(f"gen {rec['gen']:3d}  best {rec['best']:+.3f}  mean {rec['mean']:+.3f}  "
              f"forward {c['forward']:+.2f} m  drift {c['drift']:+.2f}  turn {c['turn']:+.2f}  "
              f"min height {c['min_height']:.2f}  low {c['low_time']:.2f}  lift {c['lift']:.2f}  "
              f"idle legs {c['idle_legs']:.0f}  slide {c['slide']:.2f}  pitch {c['pitch']:.2f}  fell {c['fell']:.0f}  collision {c['collision']:.2f}", flush=True)
        self.save()
        self.gen = rec["gen"] + 1
        if self.opts["show"]:
            self.label(f"generation {rec['gen']} best: {c['forward']:+.2f} m forward, score {rec['best']:+.2f}", 0)
            self.label("replaying the best gait in real time", 1)
            self.sup.simulationSetMode(Supervisor.SIMULATION_MODE_REAL_TIME)
            self.walk(gait.decode(rec["best_chrom"]), self.opts["cycles"])
            self.sup.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

    def save(self):
        os.makedirs(self.out, exist_ok=True)
        hist = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in self.history]
        with open(os.path.join(self.out, "history.json"), "w") as f:
            json.dump(hist, f)
        best = max(self.history, key=lambda r: r["best"])
        with open(os.path.join(self.out, "best_chrom.json"), "w") as f:
            json.dump({"fitness": best["best"], "generation": best["gen"], "components": best["components"],
                       "options": self.opts, "chromosome": best["best_chrom"].tolist()}, f, indent=1)
        np.savetxt(os.path.join(self.out, "best_gait.csv"), gait.decode(best["best_chrom"]),
                   delimiter=",", fmt="%.6f", header=",".join(gait.MOTOR_NAMES), comments="")

    def train(self):
        o = self.opts
        cfg = GAConfig(pop_size=o["pop"], generations=o["gens"], stall_generations=o["gens"], seed=o["seed"],
                       selection=o["selection"], crossover=o["crossover"], adaptive_mutation=bool(o["adaptive"]),
                       mutation_rate=o["mrate"])
        bounds = foot_bounds() if o["encoding"] == "foot" else None
        low, high, wrap = bounds or {"shared": (gait.SHARED_LOW, gait.SHARED_HIGH, gait.SHARED_WRAP),
                           "pushpull": (gait.PUSHPULL_LOW, gait.PUSHPULL_HIGH, gait.PUSHPULL_WRAP),
                           "pushpi": (np.append(gait.SHARED_LOW, np.pi), np.append(gait.SHARED_HIGH, np.pi),
                                      np.append(gait.SHARED_WRAP, False)),     # zero range: the GA never moves it
                           "steer": (np.append(gait.SHARED_LOW, [np.pi, -gait.STEER_MAX]),
                                     np.append(gait.SHARED_HIGH, [np.pi, gait.STEER_MAX]),
                                     np.append(gait.SHARED_WRAP, [False, False])),
                           "legs": (gait.LOW, gait.HIGH, gait.WRAP)}[o["encoding"]]
        initial = None
        if o["start"]:                                   # refine a known gait instead of starting at random
            with open(os.path.join(ROOT, "outputs", o["start"], "best_chrom.json")) as f:
                base = np.array(json.load(f)["chromosome"])
            if o["encoding"] == "steer" and len(base) == gait.PUSHPULL_GENES:
                base = np.append(base, 0.0)              # a push-pull gait is a steer gait with steer 0
            if len(base) != len(low):
                raise SystemExit(f"ga_trainer: {o['start']} has {len(base)} genes, encoding {o['encoding']} needs {len(low)}")
            noise = np.random.default_rng(o["seed"]).normal(0, 0.03, (o["pop"] - 1, len(base))) * (high - low)
            initial = np.vstack([base, base + noise])    # the start gait itself stays in the population
            cfg.mutation_sigma = 0.03
        self.sup.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
        best, score, _ = run(self.evaluate_population, low, high, cfg, wrap=wrap, on_generation=self.on_generation,
                             initial=initial)
        print(f"done: best score {score:+.3f}; saved to {self.out}", flush=True)
        if not o["show"]:                                # batch runs: close Webots
            self.sup.simulationQuit(0)
            return
        self.label(f"finished: best score {score:+.2f}. Looping the best gait.", 1)
        self.sup.simulationSetMode(Supervisor.SIMULATION_MODE_REAL_TIME)
        while True:
            self.walk(gait.decode(best), 4)

    def selftest(self):
        """The score cache is only valid if physics is repeatable: walk one gait three times."""
        chrom = gait.random_population(1, np.random.default_rng(1))[0]
        chrom[:12] = np.tile([0.0, -1.0, 0.5], 4)        # a sane pose so the test gait gets simulated
        runs = [self.walk(gait.decode(chrom), 1) for _ in range(3)]
        print("selftest:", [{k: round(v, 6) for k, v in r.items() if k not in ("legs", "path")} for r in runs], flush=True)
        assert all(abs(r["forward"] - runs[0]["forward"]) < 1e-6 for r in runs), "physics not repeatable"
        print("ga_trainer self-check passed", flush=True)


def measure(trainer, run):
    """Walk a saved gait (outputs/<run>/best_gait.csv: GA runs, or neural-network gaits written by
    part2_nn.predict, which have no chromosome) and print what each leg really did."""
    frames = np.loadtxt(os.path.join(ROOT, "outputs", run, "best_gait.csv"), delimiter=",", skiprows=1)
    m = trainer.walk(frames, trainer.opts["cycles"])
    legs, path = m.pop("legs"), m.pop("path")
    print(f"measure {run}: " + "  ".join(f"{k} {v:+.3f}" for k, v in m.items()), flush=True)
    # straightness, from the body position at the end of each cycle (removes the sway within a step)
    per_cycle = path[len(path) // trainer.opts["cycles"] - 1::len(path) // trainer.opts["cycles"]]
    pts = np.vstack([[0, 0, 0], per_cycle])
    length = np.hypot(*np.diff(pts[:, :2], axis=0).T).sum()
    chord = pts[-1, :2] / max(np.hypot(*pts[-1, :2]), 1e-9)
    off_chord = np.abs(pts[:, 0] * chord[1] - pts[:, 1] * chord[0]).max()  # widest gap to the start-end line
    curv = pts[-1, 2] / max(length, 1e-9)                                    # heading change per metre walked
    print(f"path {run}: walked {length:.2f} m  sideways {pts[-1, 1]:+.2f} m (max {np.abs(pts[:, 1]).max():.2f})  "
          f"heading {np.degrees(pts[-1, 2]):+.1f} deg  curvature {np.degrees(curv):+.2f} deg/m  "
          f"radius {abs(1 / curv) if curv else np.inf:.0f} m  bow from straight {off_chord:.2f} m", flush=True)
    np.savetxt(os.path.join(ROOT, "outputs", run, "track.csv"), path, delimiter=",", fmt="%.5f",
               header="along_m,sideways_m,heading_rad", comments="")
    for i, leg in enumerate(gait.LEGS):
        print(f"  {leg}: lift {legs['lift'][i]:.3f} m  on floor {legs['touch'][i]:.0%}  "
              f"slide {legs['slide'][i]:.3f} m  stepping {bool(legs['stepping'][i])}", flush=True)


def turn_rate(path, cycles, skip=4):
    """Steady heading change (rad per cycle): slope of the heading at the end of cycles skip..cycles.
    The first cycles are a start transient: GA gaits scored on 4 cycles ended those 4 cycles facing
    straight ahead, then turned left 1-2.5 deg every cycle after."""
    n = len(path) // cycles
    h = path[n * skip - 1::n, 2]
    return float(np.polyfit(np.arange(len(h)), h, 1)[0])


def tune(trainer, run):
    """Line search on the steer gene of a saved 13-gene gait: walk it with steer values across
    [-STEER_MAX, STEER_MAX], take the steer where the steady turn rate crosses 0 (linear interpolation
    between the two grid points around the crossing) and save that gait as outputs/<run>_tuned/."""
    with open(os.path.join(ROOT, "outputs", run, "best_chrom.json")) as f:
        saved = json.load(f)
    chrom, cycles = np.array(saved["chromosome"]), trainer.opts["cycles"]
    if len(chrom) != gait.STEER_GENES:
        raise SystemExit(f"ga_trainer: tune needs a {gait.STEER_GENES}-gene (encoding=steer) run")

    def rate(steer):
        c = chrom.copy()
        c[12] = steer
        r = turn_rate(trainer.walk(gait.decode(c), cycles)["path"], cycles)
        print(f"tune {run}: steer {steer:+.3f}  turn {np.degrees(r):+.2f} deg/cycle", flush=True)
        return r

    grid = np.linspace(-gait.STEER_MAX, gait.STEER_MAX, 7)
    rates = np.array([rate(s) for s in grid])
    best, final = grid[np.argmin(np.abs(rates))], rates[np.argmin(np.abs(rates))]
    cross = np.flatnonzero(np.sign(rates[:-1]) != np.sign(rates[1:]))
    if len(cross):                                       # refine inside the bracket around the crossing
        i = cross[np.argmin(np.abs(rates[cross]))]
        (lo, rlo), (hi, rhi) = (grid[i], rates[i]), (grid[i + 1], rates[i + 1])
        for k in range(6):                               # interpolate, every other step bisect (physics is
            s = lo - rlo * (hi - lo) / (rhi - rlo) if k % 2 == 0 else (lo + hi) / 2   # not linear in steer)
            r = rate(s)
            if abs(r) < abs(final):
                best, final = s, r
            if abs(np.degrees(r)) < 0.1:
                break
            (lo, rlo), (hi, rhi) = ((s, r), (hi, rhi)) if np.sign(r) == np.sign(rlo) else ((lo, rlo), (s, r))
    chrom[12] = best
    out = os.path.join(ROOT, "outputs", run + "_tuned")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "best_chrom.json"), "w") as f:
        json.dump(dict(saved, chromosome=chrom.tolist(), tuned_from=run, steer=best,
                       turn_deg_per_cycle=float(np.degrees(final))), f, indent=1)
    np.savetxt(os.path.join(out, "best_gait.csv"), gait.decode(chrom), delimiter=",", fmt="%.6f",
               header=",".join(gait.MOTOR_NAMES), comments="")
    print(f"tuned {run}: steer {best:+.3f}  turn {np.degrees(final):+.2f} deg/cycle -> {out}", flush=True)


def replay(trainer, run):
    """Show how a saved run learned: walk the best gait of every generation where the best score
    improved (history.json), in real time with a caption, and record it to outputs/<run>/learning.mp4."""
    with open(os.path.join(ROOT, "outputs", run, "history.json")) as f:
        hist = json.load(f)
    shown, best = [], -math.inf
    for rec in hist:
        if rec["best"] > best + 1e-9:
            shown.append(rec)
            best = rec["best"]
    sup, cycles = trainer.sup, trainer.opts["cycles"]
    movie = os.path.join(ROOT, "outputs", run, "learning.mp4")
    sup.simulationSetMode(Supervisor.SIMULATION_MODE_REAL_TIME)
    sup.movieStartRecording(movie, 1280, 720, 0, 90, 1, False)
    for rec in shown:
        c = rec["components"]
        trainer.label(f"{run}: generation {rec['gen']} of {hist[-1]['gen']}   (best so far, score {rec['best']:+.2f})", 0)
        trainer.label(f"forward {c['forward']:+.2f} m in {cycles} cycles   tilt {math.degrees(c.get('pitch', 0)):.1f} deg   "
                      f"legs stepping {8 - c.get('idle_legs', 0):.0f}/8", 1)
        print(f"replay {run}: generation {rec['gen']}  score {rec['best']:+.3f}  forward {c['forward']:+.2f} m", flush=True)
        trainer.walk(gait.decode(np.array(rec["best_chrom"])), cycles)
    sup.movieStopRecording()
    while not sup.movieIsReady():
        sup.step(trainer.dt)
    print(f"replay {run}: movie {'FAILED' if sup.movieFailed() else 'saved to ' + movie}", flush=True)


if __name__ == "__main__":
    trainer = Trainer(parse_args(sys.argv[1:]))
    if trainer.opts["replay"]:
        replay(trainer, trainer.opts["replay"])
        trainer.wait(0.5)
        trainer.sup.simulationQuit(0)
    elif trainer.opts["tune"]:
        tune(trainer, trainer.opts["tune"])
        trainer.wait(0.5)
        trainer.sup.simulationQuit(0)
    elif trainer.opts["measure"]:
        measure(trainer, trainer.opts["measure"])
        trainer.wait(0.5)                                # let Webots forward the printout before quitting
        trainer.sup.simulationQuit(0)
    elif trainer.opts["selftest"]:
        trainer.selftest()
    else:
        trainer.train()
