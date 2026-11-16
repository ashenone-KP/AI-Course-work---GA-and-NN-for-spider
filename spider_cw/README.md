# Spider gait GA and neural network (M33174), version 2

A genetic algorithm, written in Python, that evolves a walking gait for the Webots 8-leg spider (`Spider_world_demo/protos/SpiderOctopod.proto`). Every candidate gait is walked in Webots physics with **self-collision on**, and its score comes from what actually happened: body motion, plus each leg's real foot lift and sliding. Every leg makes the same step (shared motion), and only the direction each hip points differs. In the push-pull variant the hind legs' knees run half a cycle behind the front ones, so the front legs pull and the hind legs push. While it trains, Webots replays the best gait of each generation in real time, so you can watch evolution make progress.

Version 1 scored gaits on a no-physics model of the lecturer's MATLAB spider. That code is in `../archive/spider_cw_v1`, and what was learned from it is in `docs/design.txt`.

## Run it
1. **Train:** open `../Spider_world_demo/worlds/spider_world_1.wbt` in Webots and press play.
   - The robot's controller is `ga_trainer`, with `selfCollision TRUE` and `controllerArgs ["pop=40" "gens=100" "seed=0" "encoding=pushpi" "name=live"]`. These settings reproduce `pi0`, because the physics is repeatable.
   - The on-screen label shows the generation and "evaluating i/40". After each generation the best gait is replayed in real time.
   - Everything is saved to `outputs/<name>/` after every generation, so stopping early loses nothing.
   - The other options (`encoding=shared|pushpi|pushpull|legs`, `pitch=0` to drop the tilt rule, `start=<run>` to refine a saved gait, `slip=<weight>`, `period`, `cycles`, `show=0` for headless) are listed at the top of `controllers/ga_trainer/ga_trainer.py`.
2. **Check each leg:** set `controllerArgs ["measure=pi0" "cycles=4"]`. It walks that run's best gait and prints, per leg, the foot lift, time on the floor, slide distance and whether the leg steps.
3. **Plots:**
   ```bash
   python -m part1_ga.plots outputs/pi0
   ```
4. **Replay:** set the controller to `gait_player` with `controllerArgs ["pi0"]` (or `["shared1"]`). It prints the distance walked after every cycle.
5. **Headless:**
   ```bash
   webots.exe --batch --mode=fast --stdout --stderr --minimize --no-rendering <world>
   ```
   With `show=0` the trainer closes Webots when it's done.

Self-checks (run from this folder): `python -m part1_ga.gait`, `python -m part1_ga.body`, `python -m part1_ga.ga` and `python -m part1_ga.plots`. In Webots: `controllerArgs ["selftest=1"]` checks that physics is repeatable.

Install everything with `python -m pip install -r ../requirements.txt` (numpy, matplotlib, pillow, and torch for the PyTorch comparison). Webots must run the controllers with a Python that has numpy (here `python` on PATH is Python 3.13). If the trainer fails with `No module named numpy`, either:
- set Webots → Tools → Preferences → Python command to that Python's `python.exe`; or
- run `pip install numpy` for the Python that Webots uses.

## Files
| File | What it does |
|---|---|
| `part1_ga/gait.py` | chromosome → 300 × 24 Webots motor angles (legs L1..L4, R4..R1 × hip, thigh, knee). Shared motion (11 genes), push-pull (12) or per-leg (26 genes); alternating tetrapod timing |
| `part1_ga/ga.py` | generic real-valued GA: tournament/roulette, uniform/two-point/blend crossover, adaptive Gaussian mutation, elitism; validated on test functions |
| `part1_ga/body.py` | Webots spider geometry: forward kinematics, self-collision pre-check, 3D pose plot |
| `part1_ga/plots.py` | fitness, components, mutation σ, pose snapshots, joint angles, footfall diagram, GIF |
| `part1_ga/operator_experiments.py` | runs the GA operator comparisons headless in Webots (several at a time) and summarises them |
| `../Spider_world_demo/controllers/ga_trainer/` | the GA running in Webots: physics fitness, per-leg measurement, live replay, saving |
| `../Spider_world_demo/controllers/gait_player/` | plays a saved gait |
| `docs/design.txt` | encoding, fitness, GA and visualisation choices, with the evidence behind each |

## Outputs (`outputs/<name>/`)
| File | Content |
|---|---|
| `best_gait.csv` | header row of the 24 motor names, then 300 rows of motor angles (rad) |
| `best_chrom.json` | the best chromosome, its score and measurements, and the run options |
| `history.json` | per generation: best/mean/worst score, σ, the best chromosome and its measurements |
| `*.png`, `gait.gif` | written by `part1_ga.plots` |

## Results
Headless runs, population 40, 100 generations, 3 seeds per configuration. Distances are measured over 4 cycles (8 s). No run fell over or self-collided.

The two reference gaits:

| run | how | forward | body tilt | foot sliding |
|---|---|---|---|---|
| **pi0** | push-pull, tilt rule on | **2.05 m** | 3.4° | 7.7 m |
| **shared1** | shared motion, no tilt rule | **2.48 m** | 12° nose-down | 4.7 m |

- `pi0` is the best walk with a level body.
- `shared1` is the fastest, but leans forward.
- With the same tilt rule, the best plain shared gait (`tilt1`) reached 1.34 m.
- **`ref10_2`:** `pi0` refined with a heavier sliding penalty (`start=pi0 slip=1.0`). It slides 36% less (4.9 m against 7.7 m) for 5% less speed (1.94 m).

The table of all 8 configurations is in `docs/design.txt` §6:
- shared, push-pull (π forced or free gene), per-leg;
- with and without the tilt rule.

Every run is kept in `outputs/`.

**GA operator experiments in physics** (`python -m part1_ga.operator_experiments run`, then `summarise`; results in `outputs/operators/`, discussion in `docs/design.txt` §4). These compare selection, crossover, mutation and population size, 3 seeds each:
- tournament beats roulette;
- blend crossover beats two-point;
- a mutation rate of 0.2 beats 0.05 and 0.01 for this 12-gene encoding;
- population 80 beats 40, which beats 20.

Known weaknesses (`docs/design.txt` §7):
- Results vary a lot by seed.
- `pi0` slides a lot, and `shared1` leans.

## Part 2: neural network (`part2_nn/`)
A network written from scratch in numpy (`part2_nn/nn.py`): forward and backward passes, MSE loss, SGD with momentum, Adam, mini-batches, early stopping, weight decay, save/load and a gradient check. Its main job is option A:
- **Input:** `[sin φ, cos φ, speed]`, i.e. where the spider is in its step cycle plus how fast to walk.
- **Output:** the 24 motor angles.
- **Training data:** a family of 9 GA gaits evolved at different target speeds (trainer `target=` mode, refined from `ref10_2`). One whole speed is held out as the test.

```bash
python -m part2_nn.train_speed      # experiments + final model -> part2_nn/models/speed_model.npz
python -m part2_nn.predict 0.30     # a gait for 0.30 m per cycle -> outputs/nn_speed_0.300/best_gait.csv
python -m part2_nn.nn_plots         # loss curves, learning-rate sweep, predicted vs target, poses
python -m part2_nn.compare_pytorch  # the same network in PyTorch (needs torch)
```

**Results** (`docs/nn_design.txt` covers every item the spec requires):
- **Speed control:** network gaits walk in Webots within 0.4–3% of the requested speed (7.5% at the slowest). That includes speeds no gait existed for (0.30 → 0.298, 0.40 → 0.394 m per cycle) and the held-out test speed (0.357 → 0.369).
- **Joint angles:** on the held-out speed they are 1.14° RMSE from the GA gait, close to simply blending the two nearest gaits (1.04°).
- **Training speeds:** the first version (5 speeds) was 3.5° off. Adding 3 gaits in the gap around the test speed fixed it, and that version is kept for comparison.
- **Against PyTorch:** gradients agree to 4×10⁻¹⁷, and both reach the same final loss (5.560229×10⁻⁵).

Option E (inverse kinematics) is a teammate's part; their brief is `docs/nn_teammate_prompt.md`.
