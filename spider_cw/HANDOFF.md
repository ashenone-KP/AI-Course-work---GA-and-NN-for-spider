# Handoff: Spider GA (M33174), version 2

**Coursework:** `Info/M33174_cw_item1_.pdf`, deadline 2 Dec 2026, 13:00.
- Part 1: a GA evolves a spider gait.
- Part 2: a from-scratch neural network, plus a PyTorch comparison.

## Where things stand
- Part 1 was rebuilt at the user's request: "a reboot from scratch with everything we learned. We want to do python and show progress in webot with self collision on."
- The GA runs inside Webots (`Spider_world_demo/controllers/ga_trainer`) and scores each gait by walking it in physics, with `selfCollision TRUE`. See `README.md` for how to run it and `docs/design.txt` for the reasons and results.
- Version 1 (a no-physics MATLAB-model GA, its docs and its operator experiments) is in `../archive/spider_cw_v1`. Its gaits didn't transfer to Webots.
- Two reference gaits are kept (user's choice):
  - `outputs/pi0`: push-pull, the hind knees fixed half a cycle behind the front ones (the user's idea). 2.05 m over 4 cycles, nearly level (3°), but slides a lot.
  - `outputs/shared1`: shared motion. 2.48 m, but leans forward 12°.
  - `outputs/ref10_2`: pi0 refined with a heavier sliding penalty. 1.94 m, with 36% less sliding. The user likes it too.
- All runs are kept for now (user's choice). Part 1 is paused; Part 2 (the neural network) is in progress, using options A and E (see below).
- `docs/design.txt` explains every fitness term and encoding choice, each added after a failure seen in Webots:
  - bodies sliding on their bellies;
  - front legs pulling while the third pair was dragged;
  - stepping in place when the hip swing was too limited;
  - idle legs with a separate swing per leg;
  - heavy foot sliding;
  - the forward lean;
  - front and hind legs working against each other.
- The world is set to train live: `ga_trainer`, with `encoding=pushpi`, seed 0, population 40 and 100 generations, saving to `outputs/live`. These settings reproduce pi0.

## Next steps
1. **Optional:** reduce pi0's foot sliding. Run more seeds per configuration, because results vary a lot by seed (see design.txt §8).
2. ~~Operator experiments in physics~~ Done: `outputs/operators/`, `docs/design.txt` §4. They suggest a mutation rate of about 0.2 for the 12-gene encoding.
3. **Optional:** draw the footfall diagram from measured contacts (the trainer's `measure=` already prints them) instead of commanded angles.
4. **Part 2 (in progress):** options A + E, sharing the from-scratch numpy core `part2_nn/nn.py`. Its self-check passes: gradient check, SGD and Adam convergence, save/load.
   - **A (main):** `[sin φ, cos φ, speed]` → 24 angles. The training gaits come from the trainer's `target=` mode, refined from `ref10_2` at 6 target distances (`outputs/speed0*`).
   - **E:** inverse kinematics, 8 foot positions → 24 angles. It's handed to a teammate with the brief in `docs/nn_teammate_prompt.md`.
   - **Option A status:** done. Its write-up is `docs/nn_design.txt`, the model `part2_nn/models/speed_model.npz`, and the figures are in `outputs/nn_speed/`.
     - It's trained on 8 GA speeds made straight (`straight0xx_tuned`; see design.txt §7). Network gaits walk within 1–2% of the requested speed in Webots (6–7% at 0.117 and 0.300).
     - Joint error on the held-out speed is 1.86°, against 1.90° for a simple blend of neighbouring gaits.
     - Open loop the gaits still curve (up to 18° in 40 s). `gait_player ... "straight"` keeps them within 5 cm of their line.
     - Earlier rounds are kept: 5 speeds (`nn_speed_5speeds`), and 8 curved speeds (`nn_speed_curved`, 1.14°).
   - **PyTorch:** installed (CPU, user approved). The comparison is done: gradients agree to 6e-17.
   - **Done:** 3 more training speeds between 0.24 and 0.45, which cut the test error from 3.5° to 1.1°.
   - **Done:** walking straight. The user saw network gaits curve. The fix was a steer gene, then a steer line search (`tune=`), then retraining, then heading feedback in `gait_player`.
5. The group's contribution sheet.
6. **Visuals:**
   - `worlds/all_gaits.wbt`: 13 gaits side by side, with feedback.
   - `ga_trainer replay=<run>`: replays how a run learned and records `outputs/<run>/learning.mp4`. The pi0 video exists locally but isn't committed (208 MB).

## Environment notes
- Windows. Python 3.13 with numpy and matplotlib; Webots R2025a in `%LOCALAPPDATA%\Programs\Webots`.
- Headless run: `webots.exe --batch --mode=fast --stdout --stderr --minimize --no-rendering <world>`. With `show=0` the trainer stops at the end, but Webots stays open.
- A "Fact-Forcing Gate" hook asks for facts before the first write to each file and before destructive commands. State them and retry the same call.
