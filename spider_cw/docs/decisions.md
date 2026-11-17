# Every decision, why we made it, and what it changed

This is the project's decision log, roughly in the order things happened. Each row gives the decision, the reason (usually something that went wrong in Webots), and the measured effect. Distances are over 4 gait cycles (8 s) unless stated. The detailed evidence is in `design.txt` (Part 1), `nn_design.txt` (Part 2, option A) and `ik_comparison.md` (option E and the foot-path gaits). `build_guide.md` turns the final choices into a recipe.

**Where it ended up:**

| | best | result |
|---|---|---|
| Joint-sine GA gait | `pi0` (push-pull, 12 genes) | 2.05 m, level (3.4°), 7.75 m of foot sliding |
| Speed network (option A) | 3-64-64-24 tanh, Adam 0.03 | 1.86° joint error on an unseen speed; walks within 1–2% of the requested speed (6–7% at two speeds) |
| IK network (option E) | per-leg 3-64-64-3 tanh | feet within 0.50 cm on test poses, 2.8 cm when rebuilding real GA gaits |
| **Best gait overall** | **`footwide_s2`: GA-tuned foot paths through the IK network** | **6.55 m, 2.46 m of sliding, 33.3 m in 40 s within 1° of its heading** |

---

## 1. Overall approach

| Decision | Why | Effect |
|---|---|---|
| **Rebuild Part 1 from scratch (version 2), in Python, scored in Webots physics with self-collision on** | Version 1 scored gaits on a fast no-physics model of the lecturer's MATLAB spider. Its gaits walked in the model, not in Webots: different leg lengths and hip axes put the feet elsewhere, and 10–44% of joint targets were clamped. Even a model of the Webots geometry was exploited (steps that barely lifted the feet scraped in physics). | Every score now comes from what actually happened. The lesson that shaped everything after: **whatever the fitness doesn't measure, the GA exploits.** |
| **Keep Python, not MATLAB** | The spec allows any language. Everything (trainer, network, plots) shares one numpy codebase that runs inside Webots. | A port would mean rewriting and rerunning all results. The teammate's MATLAB branch shows a MATLAB GA is possible, but it isn't the main pipeline. |
| **Cache scores per chromosome** | The self-test walks one gait 3 times and gets identical results (to 1e-6), and the live run reproduced the headless run exactly. | Elites and duplicates cost nothing to re-score. |
| **Save everything after every generation; keep every run** | Long runs, and the user wanted to compare them later. | Stopping early never loses work. Every configuration in the docs can be reopened. |
| **Show progress live** (label, real-time replay of each generation's best, later `replay=` video) | The point of version 2 was to *see* the GA work in Webots. | Watching it is how most of the problems below were found. |

## 2. Part 1: how a gait is encoded

| Decision | Why | Effect |
|---|---|---|
| **One sine per joint** (`pose + amp·sin(2πk/300 + phase)`) instead of 7200 free numbers | 300 × 24 values can't be evolved directly. A sine is smooth and periodic. | A search space of 11–26 genes that converges. |
| **Fixed alternating-tetrapod timing** (phase = π·is_right + π·position) | Version 1: with 24 free phases every seed stood still, and 2000 random phase sets never beat standing. Left/right offset as a gene made pairs "row"; front/back offset as a gene (version 2) chose a quarter-cycle wave that looked out of sync and dragged the 3rd pair. | Every chromosome is a coordinated gait. |
| **Mirror symmetry** (right leg = −left) | Version 1: 0 of 5 seeds walked without it, 5 of 8 with it. | Halves the genes and gives a left/right-balanced walk. |
| **Shared motion** (every leg makes the same step; only its hip direction differs) | With a separate swing per leg, the GA let some legs idle: one pair touched the floor only 6–24% of the time. The user's observation: "all the legs are the same, so match their movements". | Every leg on the floor 51–82% of the time. Best 2.48 m against 1.66 m per-leg, and less sliding per metre. |
| **Push-pull: hind knees half a cycle behind** (the user's idea) | With shared motion, front knees pull the body forward but hind knees, pointing backward, pull it backward: the two ends fight. | As a **free** gene the GA drifted back to 0 in 4 of 6 runs (best 1.86 m). **Fixed at π** (`pushpi`) it gave `pi0`: 2.05 m, level, against 1.34 m for shared motion under the same tilt rule. |
| **Hip fan bounds ±1.2 rad, hip swing ≤ 0.6** | Hips are only 0.325 m apart. With ±0.6 the legs collided unless the hip swing was about 0. | With ±0.6 the robot stepped in place (0.15–0.37 m in 2 cycles); with ±1.2 the best run walked 1.87 m in 2 cycles. |
| **Thigh [−1.5, 0], knee [−0.5, 1]** | Legs stay below the body, and the mirrored right knee stays inside its motor range. | No illegal angles after mirroring. |
| **Steer gene** (13th gene: left hips swing ×(1+s), right ×(1−s)) | Mirroring makes every gait symmetric, so the GA had no way to correct a curve that the physics adds. | Made the straight speed family possible (§5). With s = 0 it decodes exactly to the push-pull gait. |
| **Foot-path encoding** (`encoding=foot`, 7 genes; §8) | Joint sines can't draw a flat stance stroke, so every sine gait slides (pi0: 7.75 m). | 4.10 m in 4 cycles with 3.30 m of sliding, then 6.55 m with wider bounds. |

## 3. Part 1: how a gait is scored

| Decision | Why | Effect |
|---|---|---|
| **Score the walk in physics: forward − drift − turn** | The real goal is distance walked. | The baseline every term below was added to. |
| **Body-height penalty** (5 × time below 0.25 m, plus 2 × the dip) | With only a dip penalty, the first runs slid forward on their bellies (lowest point 0.14 m; the box is 0.3 m tall). | All runs keep the body at 0.25 m or above. |
| **Per-leg terms** (lift reward capped at 10 cm, idle-leg penalty) | The next run walked 0.85 m per cycle, but only the front pair pulled. The 3rd pair never lifted and was dragged 3.5–4 m. | All 8 legs step in every run since. |
| **Feet measured from the *measured* joint angles** | Commanded angles don't show legs blocked by self-collision. | Lift, contact and sliding are real. |
| **Slide penalty 0.3 per metre** | At 0.1, run1 slid 11 m for 3.8 m forward. Much higher and standing still wins, because sine motion always skates a little. | Shared gaits slide about 1.9 m per metre forward instead of 3. |
| **Tilt rule: 4° free, then 3 per rad** | shared1 walked 12° nose-down. Penalising *all* tilt halved the speed (best 1.24 m against 2.48 m): the lean was helping it push. | Level walking at a modest cost. Push-pull then recovered the speed (pi0 2.05 m at 3.4°). |
| **Self-collision pre-check** (skip if > 20% of frames collide, else 3 × the fraction) | Webots enforces collisions physically, but simulating hopeless gaits wastes time. | Faster runs. No result in the docs self-collides. |
| **Score 2 cycles** (joint-sine runs) | Speed: 100 generations × 40 gaits. | Hid slow curves. Over 40 s ref10_2 turned 29° and the best operator run 64° (§5). |
| **Score 4 cycles** (straight family and foot-path runs) | A start-up wobble can fake a straight walk or a good distance. | Fairer scores. The steady curve still needed its own tuning step (§5). |
| **`tilt=` option; 8° free for foot-path runs** (the user: "a slight tilt is okay") | Allow a slight lean where it helps. | The foot-path gaits stayed level anyway (pitch 0.010–0.015 rad). The default stays 4° so old runs reproduce. |

## 4. Part 1: GA operators

| Decision | Why | Effect (physics experiments, 3 seeds, 60 generations) |
|---|---|---|
| **Tournament selection (k = 3)** | It compares ranks only, so it works with negative scores and keeps constant pressure. | 0.76 m best gait, against 0.45 m for roulette, whose pressure fades late. |
| **Blend crossover (BLX-0.5, circular for phases)** | Children land between and around two good parents and stay coherent. | Two-point was clearly worst (−0.09 ± 0.57 fitness): it cuts apart genes that only work together. Blend's best gait walked twice as far as uniform's. |
| **Gaussian mutation scaled by each gene's range, adaptive sigma** | One sigma suits genes in radians and in phase units alike. Big steps early, small ones late. | Works, but in 60-generation runs a fixed sigma did slightly better (the adaptive one shrinks too early). |
| **Mutation rate 0.05 for the reference gaits; 0.2 later** | 0.05 came from version 1 (28 genes, 1.4 mutations per child). With 12 genes it mutates only 0.6 genes per child. | 0.2 was best and most consistent (+0.51 ± 0.09, 0.96 m); 0.01 stalled. **Used for the foot-path runs.** |
| **Elitism 2; wrap phases instead of clipping** | The best gait can never be lost. 2π + 0.1 is the same gait as 0.1. | Best-so-far never drops. |
| **Population 40** | Time. 80 was best (+0.68) but costs twice as much simulation. | The reference gaits weren't rerun with 80. |

## 5. Walking straight

| Decision | Why | Effect |
|---|---|---|
| **Measure 20-cycle paths** (curvature, radius, bow) | Watching all gaits side by side showed some walking into the next lane. | Curves made visible: ref10_2 +28.6° (21 m radius), pop80_s2 −63.7°, pi0 −3.7°. |
| **Re-evolve the speed family with the steer gene over 4 cycles** | First fix attempt. | **A trap:** every gait turned under 0.6° in the 4 scored cycles but 3–33° over 20. The GA found gaits whose start-up wobble happens to end straight at cycle 4. |
| **Tune the steer gene alone against the *steady* turn rate** (`tune=`: slope of the heading over cycles 4–12) | Leave out the transient the GA exploited. | Turning radius went from 17–180 m to 66 m – 6 km. These became the network's training data. |
| **Heading feedback in the player** (PI on the steer mechanism, aim 3 m ahead) | The network's joint error (~1.9°) is as big as the steer correction itself, so open loop can't be straight. | Every network gait ends within 5 cm of its line after 40 s, against up to 1.05 m without. Fails only for pop80_s2 (curves more than steer can cancel). |
| **GPS-only feedback** | An InertialUnit added to the robot crashed Webots on load. | The heading is taken from the direction of travel. |

## 6. Part 2, option A: speed-conditioned gait network

| Decision | Why | Effect |
|---|---|---|
| **Options A (main) and E (teammate)** | A builds on the GA's gaits; E is the teammate's part. They share one from-scratch core (`nn.py`). | Two working Part 2 networks. |
| **Input `[sin φ, cos φ, speed]`, output 24 angles** | The gait is a loop, so sin/cos put the last frame next to the first. Speed is a goal input, so the network can make gaits for speeds it never saw. | Any speed's gait comes from one network. |
| **Training data from the GA's `target=` mode, all refined from ref10_2** | Refining from the same gait gives a smooth family (neighbours differ by 0.7–2° per joint), so interpolating means something. | 9 gaits from 0.12 to 0.48 m per cycle. |
| **Give the network the *measured* speed, not the target** | It should learn what each gait really does in physics. | The predicted gaits walk within 1–2% of the requested speed (6–7% at 0.117 and 0.300). |
| **Test on a whole held-out speed; choose settings by leave-one-speed-out** | Frames of one gait are near-copies: a random frame split leaks and rewards memorising. That happened: the best models on random frames were the worst on a new speed. | Honest error figures, and settings that generalise. |
| **tanh output, angles scaled to the motor limits with a 10% margin** | Every output is a legal angle by construction, and a target sitting on a limit doesn't need tanh = ±1. | No illegal angles. |
| **3-64-64-24, tanh hidden layers** | 16 × 16 up to 128 × 128 and 3 layers scored within 0.25° of each other. tanh interpolates smoothly; ReLU was worst in every round. | A small, smooth model. |
| **Adam, lr 0.03, batch 64, 107 epochs, final model on all 8 speeds** | Lowest leave-one-speed-out error (1.83°). 107 = its folds' mean best epoch. | Final joint error 1.86° on the unseen speed. |
| **Round 2: 3 more speeds in the widest gap** | Round 1 (5 speeds) erred most there: 3.48°, worse than blending the two nearest gaits (1.72°). | Error cut by two thirds to 1.14°. |
| **Round 3: retrain on the straightened gaits** | Round 2's gaits curved like their training data (+11 to +24° in 40 s). | 1.86°, now slightly *better* than the blend (1.90°); the straightened family is less smooth, so both errors rose. Network gaits still curve open loop, up to 18°, so feedback keeps them on line. |
| **PyTorch comparison with identical weights and batches** | The spec's 5-mark library comparison, and proof the hand-written backpropagation is right. | Gradients agree to 6e-17, identical final loss (1.634255e-4); numpy 2.3 s against torch 7.0 s. |

## 7. Part 2, option E: inverse-kinematics networks

| Decision | Why | Effect |
|---|---|---|
| **Merge our main into the teammate's `nn` branch, then copy the work back** | Their branch was our older commit plus their IK commit, so a real git merge worked with no conflicts. | One codebase; their work is credited in the commit and docs. |
| **Run the full training, not the 20-epoch test** | In the pushed quick run every network's best epoch was its last: none had finished learning. | Whole-body 7.7 → 3.67 cm; per-leg 1.9 → 0.50 cm. |
| **Judge by foot error (cm), not angle error** | What matters on the robot is where the foot lands; near the hip axis many hip angles give the same foot. | Fair comparison despite the hip ambiguity (per-leg hip error 4.3° near the axis, foot still within 0.6 cm). |
| **Test on real GA gaits rebuilt through each network and walked in Webots** (`gait_test.py`) | Random poses aren't what the robot does. | Per-leg 2.8 cm against 6.9 cm, keeping 101% of the GA distance against 89%. |
| **Per-leg network wins** | Each leg's angles depend only on its own foot, and mirroring makes all 8 legs one problem: 8× the data, 5× fewer weights. | Better on every measure. |
| **Train a third, full-motor-range model and keep all models** | 17% of real GA angles lie outside the training range (thighs at −1.57). | It fixes those (shared1's worst feet 35 cm → 2.3 cm) but is less precise inside the range (1.8 against 0.8 cm). Overall a tie: 2.91 against 2.79 cm. |
| **Use the original-range model for foot-path gaits** | Every foot-path target lies inside its range, where it is 4× more precise. | 2.46 m walking straight, against 2.08 m turning 5.5° with the motor-range model. |

## 8. Foot-path gaits (Part 1 GA + Part 2 IK network)

| Decision | Why | Effect |
|---|---|---|
| **Write gaits as foot paths, like a real spider** (planted stance, lifted swing, alternating groups) | The user wanted spider-like movement, and joint sines always slide. The IK network makes foot paths playable. | A hand-set sweep reached 2.46 m with less sliding and pitch than pi0. |
| **Ground height −0.60 m, not pi0's −0.80** | At pi0's average foot position the leg can't reach below −0.75 (knee at its limit); a grid search over all legal angles missed targets by 11 cm at −0.75 and found all reachable at −0.60. | Feet land within 0.3 cm of their targets, against 11–14 cm before. |
| **Higher swings** | In the sweep, low swings brushed the floor. | 0.4 m stride: sliding 17.2 m at 6 cm lift → 6.2 m at 24 cm. |
| **Swing lands at zero ground speed** (Hermite curve, `matched=True`) | The first swing stopped relative to the *body*, so the foot landed moving forward over the ground. | Small gain: 2.46 → 2.53 m, 6.21 → 5.91 m of sliding. Most sliding is planted feet slipping under load. |
| **Let the GA tune the foot paths** (`encoding=foot`: stride, lift, duty, height, reach, spread, ripple) | The sweep's best sat on the edge of its grid. | `foot_t14_s1`: 4.10 m, 3.30 m of sliding, level. Two seeds found almost the same genes, including a back-to-front ripple like a real spider. |
| **Mutation 0.2, 4-cycle scoring, tilt 8°** | The operator experiments (small encodings like 0.2); 4 cycles against start-up tricks; the user allowed a slight lean. | As above. |
| **Widen the bounds, but only as far as the IK network stays accurate** (`encoding=footwide`) | Stride, lift, height and spread ended at their bounds. Past reach 1.1 the feet are out of range; below 0.78 or spread past 1.4 the IK errs by 15–22 cm, and the GA would be rewarded for those mistakes. | `footwide_s2`: **6.55 m, 2.46 m of sliding, 33.3 m in 40 s.** Refining the old best stayed near it (4.40 m): the narrow bounds had hidden a better gait. |
| **Pick `footwide_s2`, not the 4% faster s0** | s0's IK errs more (worst foot 15 cm against 8.6), its legs touch in 2% of frames, and its body dips to exactly the 0.25 m limit. | A gait whose feet really follow the planned paths. |

## 9. Process decisions

| Decision | Why | Effect |
|---|---|---|
| **The spec PDF (`Info/`) and the teammate's clone are never pushed** | The repository is public. | Only project files on GitHub. |
| **No videos in git** (`*.mp4` ignored) | pi0's learning video is 208 MB. | The repository stays small; videos can be regenerated with `replay=`. |
| **Commit only when asked; noreply e-mail for commits** | The user's choice, and privacy. | Every commit was reviewed first. |
| **New options default to the old behaviour** (`tilt`, `footwide`, `--motor`, `matched`) | Earlier results must stay reproducible. | Every result in the docs can be rerun exactly. |

## What is still open
- `footwide_s2` was still improving at its last generation (50), and its stride, height and spread are at their bounds. Spread is already at the edge of the IK network's accurate zone, so going further first needs an IK network trained on those foot positions.
- Feet still slide (2.5 m in 4 cycles at best): planted feet slip under load.
- The hind feet dip about 4.5 cm below ground at the end of stance in `footwide_s2`.
- Results vary by seed; most configurations have 3 seeds.
- Still to do: the group's contribution sheet, and packaging the submission.
