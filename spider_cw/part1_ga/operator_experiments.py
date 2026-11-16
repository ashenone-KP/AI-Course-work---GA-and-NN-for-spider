"""GA operator experiments in Webots physics: which selection, crossover, mutation and population size
work best on the real (physics-scored) gait problem.

    cd spider_cw
    python -m part1_ga.operator_experiments run         # all variants x seeds, headless, in parallel
    python -m part1_ga.operator_experiments summarise   # plots + table from the saved runs

Every variant changes ONE setting of BASE (the reference setup: push-pull encoding with pi fixed,
the full fitness incl. the tilt rule, 40 x 60 generations) and is run with SEEDS, so a difference
is not just one lucky seed. Each run is the ga_trainer controller in its own headless Webots
(--batch --mode=fast --no-rendering) on a temporary copy of the world with its controllerArgs; the
copy is deleted afterwards. Runs land in outputs/operators/<variant>_s<seed>/ like any trainer run.
The summary compares the best fitness per generation (mean +- std over seeds, part1_ga.plots.
compare_runs) and the final best fitness and forward distance per variant.
Webots: set WEBOTS_EXE, otherwise the default per-user install path is used.
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from part1_ga.plots import compare_runs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLDS = os.path.join(ROOT, "..", "Spider_world_demo", "worlds")
OUT = os.path.join(ROOT, "outputs", "operators")
WEBOTS = os.environ.get("WEBOTS_EXE", os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Webots",
                                                   "msys64", "mingw64", "bin", "webots.exe"))
BASE = {"pop": 40, "gens": 60, "encoding": "pushpi"}
SEEDS = (0, 1, 2)
EXPERIMENTS = {
    "selection": {"tournament": {}, "roulette": {"selection": "roulette"}},
    "crossover": {"blend": {}, "uniform": {"crossover": "uniform"}, "two_point": {"crossover": "two_point"}},
    "mutation": {"adaptive, rate 0.05": {}, "fixed sigma": {"adaptive": 0},
                 "rate 0.01": {"mrate": 0.01}, "rate 0.2": {"mrate": 0.2}},
    "population": {"pop 20": {"pop": 20}, "pop 40": {}, "pop 80": {"pop": 80}},
}


def run_name(change, seed):
    tag = "base" if not change else "_".join(f"{k}{v}" for k, v in sorted(change.items()))
    return f"operators/{tag}_s{seed}"


def jobs():
    """Unique (name, controllerArgs) pairs: the base setup appears in every experiment but runs once."""
    out = {}
    for variants in EXPERIMENTS.values():
        for change in variants.values():
            for seed in SEEDS:
                opts = dict(BASE, **change, seed=seed, show=0, name=run_name(change, seed))
                out[opts["name"]] = [f"{k}={v}" for k, v in opts.items()]
    return out


def run_one(name, args):
    if os.path.exists(os.path.join(ROOT, "outputs", name, "history.json")):
        return f"{name}: already done"
    with open(os.path.join(WORLDS, "spider_world_1.wbt")) as f:
        world = f.read()
    world = re.sub(r'^  controller ".*"$', '  controller "ga_trainer"', world, flags=re.M)
    world = re.sub(r"^  controllerArgs .*$", "  controllerArgs [" + " ".join(f'"{a}"' for a in args) + "]",
                   world, flags=re.M)
    path = os.path.join(WORLDS, "_op_" + name.replace("/", "_") + ".wbt")
    with open(path, "w") as f:
        f.write(world)
    try:
        subprocess.run([WEBOTS, "--batch", "--mode=fast", "--stdout", "--stderr", "--minimize", "--no-rendering", path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4 * 3600)
    finally:
        os.remove(path)
    done = os.path.exists(os.path.join(ROOT, "outputs", name, "history.json"))
    return f"{name}: {'done' if done else 'FAILED (no history.json)'}"


def run(workers=9):
    todo = jobs()
    print(f"{len(todo)} runs, {workers} Webots at a time", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        for msg in pool.map(lambda kv: run_one(*kv), todo.items()):
            print(msg, flush=True)


def summarise():
    os.makedirs(OUT, exist_ok=True)
    lines = [f"BASE {BASE}, seeds {SEEDS}; final = best fitness after the last generation (mean +- std over seeds)"]
    for exp, variants in EXPERIMENTS.items():
        results = {}
        lines.append(f"\n{exp}")
        for label, change in variants.items():
            hists = []
            for seed in SEEDS:
                with open(os.path.join(ROOT, "outputs", run_name(change, seed), "history.json")) as f:
                    hists.append(json.load(f))
            results[label] = hists
            final = np.array([h[-1]["best"] for h in hists])
            fwd = np.array([max(h, key=lambda r: r["best"])["components"]["forward"] for h in hists])
            lines.append(f"  {label:20s} final fitness {final.mean():+.3f} +- {final.std():.3f}   "
                         f"forward of best {fwd.mean():.2f} m (seeds: {', '.join(f'{x:+.2f}' for x in final)})")
        compare_runs(results, os.path.join(OUT, f"{exp}.png"))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summarise"
    if cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 9)
    elif cmd == "summarise":
        summarise()
    else:
        print(__doc__)
