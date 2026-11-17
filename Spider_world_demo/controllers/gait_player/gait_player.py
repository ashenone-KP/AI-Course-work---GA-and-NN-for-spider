"""Play an evolved gait on the Webots spider: spider_cw/outputs/<run>/best_gait.csv (written by
the ga_trainer controller: a header row of motor names, then 300 rows of motor angles).

controllerArgs (optional):  [ "<run name or csv path>", "<seconds per cycle>", "straight", "quit=<n>" ]
    defaults: run0, 2 s, open loop, never quit
    straight   heading feedback (see Straightener): keeps the robot on the line it starts on
    quit=<n>   stop the simulation after n cycles (needs supervisor TRUE; for headless tests)
Prints the distance walked, sideways offset and heading change after every cycle.
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "..", "..", "spider_cw", "outputs")
SETTLE_SECONDS = 1.5
STEER_MAX = 0.3             # as gait.STEER_MAX


def load_gait(path):
    """-> (motor names, frames as lists of floats)."""
    with open(path) as f:
        rows = list(csv.reader(f))
    names, frames = rows[0], [[float(v) for v in r] for r in rows[1:] if r]
    if not frames or any(len(r) != len(names) for r in frames):
        raise ValueError(f"{path}: expected a header of motor names and rows of {len(names)} angles")
    return names, frames


class Straightener:
    """Heading feedback through the steer gene's mechanism (part1_ga/gait.py): left hips swing
    (1 + s) x their amplitude around their cycle mean, right hips (1 - s); s > 0 turns right.
    Open loop, every gait curved: the GA gaits slightly, the network's up to 18 deg in 40 s, because
    a joint-angle error of 1-2 deg (the network's) is as big as the steer correction itself.
    The robot has only a GPS (an InertialUnit added to the proto crashed Webots on load), so the
    heading is the direction the body travelled over the last cycle, relative to the line. The line
    is the direction travelled over the first LINE_CYCLES cycles, from the start position.
    Once per cycle: aim at the line LOOKAHEAD metres ahead, error = travel heading - that aim,
    s = KP * error + integral (KI * error per cycle; the integral learns each gait's steady curve).
    KP: tune= runs measured about -28 deg of turn per cycle per unit of s, so KP = 1 per rad removes
    roughly half the heading error each cycle."""
    KP, KI, LOOKAHEAD, LINE_CYCLES = 1.0, 0.5, 3.0, 2

    def __init__(self, names, frames):
        n = len(frames)
        self.hips = [(j, sum(f[j] for f in frames) / n, 1.0 if name.startswith("L") else -1.0)
                     for j, name in enumerate(names) if name.endswith("_hip_joint")]
        self.steer = self.integral = 0.0

    def apply(self, angles):
        a = list(angles)
        for j, mean, side in self.hips:
            a[j] = mean + (1 + self.steer * side) * (a[j] - mean)
        return a

    def update(self, heading, sideways):
        """travel heading over the last cycle (rad, + = left of the line), sideways offset (m, + = left)."""
        err = heading + math.atan2(sideways, self.LOOKAHEAD)
        self.integral = max(-STEER_MAX, min(STEER_MAX, self.integral + self.KI * err))
        self.steer = max(-STEER_MAX, min(STEER_MAX, self.KP * err + self.integral))


def main():
    args = sys.argv[1:]
    words = [a for a in args if "=" in a or a == "straight"]
    pos = [a for a in args if a not in words]
    run = pos[0] if pos else "run0"
    period = float(pos[1]) if len(pos) > 1 else 2.0
    straight = "straight" in words
    quit_after = next((int(w.split("=")[1]) for w in words if w.startswith("quit=")), None)
    path = run if run.endswith(".csv") else os.path.join(RUNS, run, "best_gait.csv")
    names, frames = load_gait(path)
    print(f"gait_player: {len(frames)} frames from {os.path.normpath(path)}, {period:.1f} s per cycle"
          f"{', heading feedback on' if straight else ''}", flush=True)

    if quit_after:
        from controller import Supervisor
        robot = Supervisor()
    else:
        from controller import Robot
        robot = Robot()
    step = int(robot.getBasicTimeStep())
    motors = [robot.getDevice(n) for n in names]
    gps = robot.getDevice("gps")
    gps.enable(step)
    keeper = Straightener(names, frames) if straight else None

    def command(angles):
        for m, a in zip(motors, angles):
            m.setPosition(a)

    n, start, prev, line, cycle = len(frames), None, None, 0.0, -1   # line: direction (rad) in the world
    while robot.step(step) != -1:
        t = robot.getTime()
        if t < SETTLE_SECONDS:
            command(frames[0])
            continue
        pos = gps.getValues()
        start = start or pos
        phase = (t - SETTLE_SECONDS) / period * n
        k, w = int(phase), phase - int(phase)
        if k // n != cycle:
            cycle = k // n
            dx, dy = pos[0] - start[0], pos[1] - start[1]
            if cycle == Straightener.LINE_CYCLES:
                line = math.atan2(dy, dx)
            fwd = dx * math.cos(line) + dy * math.sin(line)
            side = -dx * math.sin(line) + dy * math.cos(line)
            heading = 0.0
            if prev:                                      # travel direction over the last cycle
                heading = (math.atan2(pos[1] - prev[1], pos[0] - prev[0]) - line + math.pi) % (2 * math.pi) - math.pi
            prev = pos
            if keeper and cycle > Straightener.LINE_CYCLES:
                keeper.update(heading, side)
            print(f"{run}: cycle {cycle:3d}: forward {fwd:+.3f} m  sideways {side:+.3f} m  heading "
                  f"{math.degrees(heading):+.1f} deg  body height {pos[2]:.3f}"
                  + (f"  steer {keeper.steer:+.3f}" if keeper else ""), flush=True)
            if quit_after and cycle >= quit_after:
                robot.simulationQuit(0)
        a, b = frames[k % n], frames[(k + 1) % n]
        angles = [u + w * (v - u) for u, v in zip(a, b)]
        command(keeper.apply(angles) if keeper else angles)


if __name__ == "__main__":
    main()
