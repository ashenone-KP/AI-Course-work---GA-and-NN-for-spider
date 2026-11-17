# Option E: which IK network is better?

**Verdict: the per-leg network (3 → 3, one small network shared by all 8 legs).**
It wins on every measure below while using 5× fewer weights (4,611 against 22,808).

## The two candidates (`part2_nn/ik/train.py`)
| | whole-body | per-leg |
|---|---|---|
| input → output | 8 feet (24 numbers) → 24 angles | 1 foot relative to its hip (3) → its 3 angles |
| best setup found | 128×128 tanh, Adam lr 0.001, batch 64 | 64×64 tanh, Adam lr 0.001, batch 512 |
| training samples | 40,000 poses | 320,000 legs (the same 40,000 poses × 8) |

## Evidence 1: random test poses (`outputs/nn_ik/summary.txt`, `foot_error.png`)
5,000 test poses × 8 feet, never used in training or model choice:

| | whole-body | per-leg |
|---|---|---|
| foot error, mean | 3.67 cm | **0.50 cm** |
| foot error, 95% | 8.23 cm | **1.30 cm** |
| angle RMSE | 3.49° | **0.96°** |

How: `python -m part2_nn.ik.train` (full run, about 75 min here) trains every experiment, picks the lowest
**validation** foot error, then scores it once on the test set.

## Evidence 2: real GA gaits, walked in Webots (`outputs/nn_ik/gait_test.txt`)
Random poses are not what the robot will be asked to do, so `part2_nn/ik/gait_test.py` takes four GA gaits
(300 frames each), turns their angles into foot positions with the forward kinematics, rebuilds the angles
with each network and walks the result in Webots for 4 cycles. The original gaits were walked too
(they reproduced their known distances: pi0 2.05 m, ref10_2 1.94 m, shared1 2.48 m, straight045_tuned 0.97 m).

| gait | foot error, whole-body / per-leg | distance kept, whole-body / per-leg |
|---|---|---|
| pi0 | 7.81 / **2.56 cm** | 92% / 91% |
| ref10_2 | 7.36 / **2.57 cm** | 101% / 102% |
| shared1 | 5.37 / **3.52 cm** | 92% / 103% |
| straight045_tuned | 7.24 / **2.50 cm** | 71% / **108%** |
| **mean** | 6.94 / **2.79 cm** | 89% / **101%** |

No rebuilt gait fell. The whole-body network loses 29% of the slowest gait's distance; the per-leg one
never loses more than 9%.

## Why the per-leg network wins
1. **The problem really is per leg.** A leg's three angles depend only on where its own foot is. The
   whole-body network has to discover that 21 of its 24 inputs are irrelevant to each output; the
   per-leg network is given it.
2. **Mirroring makes all 8 legs the same problem.** `ik_data.per_leg` expresses every foot relative to
   its own hip with y flipped for right legs, so one network serves all 8 legs, which means 8× the data
   for a 3-input problem.
3. **Training behaved better.** The per-leg network early-stopped at epoch 270 of 310
   (`loss.png`, right). Most whole-body runs were still improving when they hit the 500-epoch cap
   (best epoch 497 of 500), so whole-body needs far more training to close the gap, and the
   best whole-body settings still only reached 3.65 cm on validation.

## Caveats found by the real-gait test
- **The training poses miss part of the real gaits.** 16–17% of the push-pull gaits' angles lie outside the
  ranges the training poses are drawn from (`ik_data.py`: thigh [-1.5, 0], knee [-0.5, 1]): mostly
  thighs pushed to the motor limit of -1.57. Per-leg foot error is 0.3–0.9 cm on legs inside those
  ranges, but 4.6–4.9 cm on legs outside them (26 cm for shared1's few out-of-range hip angles).
  A per-leg network trained on the full motor range was tried next (below): it fixes this, but costs
  accuracy inside the range.
- **Hip angle near the hip axis.** When a foot is close to the hip's rotation axis, many hip angles put
  it in the same place, so the hip error is large there (`hip_axis.png`: per-leg hip RMSE 4.3° near the
  axis against 0.3° elsewhere). The foot still lands within 0.6 cm, which is what matters on the robot.
- **Noisy late training.** The per-leg loss spikes after epoch 150 (lr 0.001, no decay); early
  stopping keeps the best weights, but a lower final learning rate would likely help.

## A third model: per-leg, full motor range (`python -m part2_nn.ik.train --leg-motor`)
Same network and settings as the per-leg one, but its training poses are drawn from every angle a left
motor and a mirrored right motor can both reach, leg below the body (`ik_data.BOXES["motor"]`: hip ±1.57,
thigh [-1.57, 0], knee [-1, 1]), which covers the GA gaits' hips up to 1.46 and thighs down to -1.57. The
wider range only adds hip ambiguity near the hip axis (thigh and knee never differ by more than 19° for
feet within 2 cm), the same harmless case as before. Saved as `ik_leg_model_motor.npz`; the pose-range
models are kept. Early-stopped at epoch 164 (12 min): 1.22 cm mean foot error on motor-range test poses
(`outputs/nn_ik_motor/summary.txt`).

On the real GA gaits (`outputs/nn_ik/gait_test.txt`):

| gait | foot error, per-leg pose / per-leg motor | distance kept |
|---|---|---|
| pi0 | **2.56** / 3.51 cm | 91% / 94% |
| ref10_2 | **2.57** / 3.47 cm | 102% / 103% |
| shared1 | 3.52 / **1.23 cm** (95%: 34.7 / **2.3 cm**) | 103% / 98% |
| straight045_tuned | **2.50** / 3.44 cm | 108% / 103% |
| **mean** | **2.79** / 2.91 cm | 101% / 99% |

- The motor-range model removes the out-of-range failures: shared1's worst feet go from 35 cm off to 2.3 cm.
- But spreading the same 320k samples over a wider range costs accuracy inside the old range (legs in the
  pose range: 1.8 cm against 0.8 cm), so on the push-pull gaits it is ~1 cm worse. Overall they tie.
- More samples or a longer, decaying learning rate would likely give one model with both strengths.

## Putting it to work: a spider-like gait (`part2_nn/ik/foot_gait.py`)
The GA's gaits move every joint along a sine wave, so the feet go wherever the joints take them, and they
slide (pi0: 7.75 m of foot sliding in 4 cycles). A real spider does it the other way round: a planted foot
stays still on the ground while the body passes over it, then lifts and swings forward, and L1 L3 R2 R4
alternate with L2 L4 R1 R3 (alternating tetrapod). With an IK network that can be written down directly:
- **Stance** (fraction `duty` of the 2 s cycle): the foot moves straight back by `stride`, at ground height.
- **Swing** (the rest): forward again, lifted by up to `lift` (half a sine).
- Feet sit on average where pi0's feet are (its fan and reach are known not to self-collide).
- Every foot target goes through the per-leg network, giving 300 × 24 angles that play like any GA gait.

**Ground height had to come up.** At pi0's lowest foot height (-0.80 m below the body centre) the targets
are out of reach: a grid search over every legal angle missed them by 11 cm. At -0.60 m every target of a
0.4 m stride is reachable, and the network lands the feet within 0.3 cm on average (0.9 cm worst).

**Sweep, 24 gaits walked 4 cycles in Webots** (`outputs/nn_ik_motor/spider_sweep_pose.txt`; none fell):

| stride / lift / duty | forward | foot sliding | heading | pitch | legs touching |
|---|---|---|---|---|---|
| **0.40 / 0.24 / 0.50** | **2.46 m** | 6.21 m | +0.0° | 0.025 | 14% of frames |
| 0.40 / 0.24 / 0.65 | 1.97 m | 6.57 m | +0.7° | 0.028 | none |
| 0.30 / 0.24 / 0.50 | 1.85 m | 5.39 m | -0.1° | 0.023 | none |
| 0.40 / 0.06 / 0.50 | 1.82 m | 17.20 m | -0.2° | 0.022 | 11% |
| pi0 (GA) | 2.05 m | 7.75 m | -0.7° | 0.059 | none |

- The best spider gait goes **20% further than pi0 with 20% less sliding, less than half its body pitch
  and a straight heading**, as fast as shared1 (2.48 m) without shared1's 12° forward lean.
- **Lift matters most:** at every stride and duty, a higher swing cut the sliding (17.2 → 6.2 m at 0.4 m
  stride), so low swings were still brushing the floor.
- Feet still slide: gaits cover 35–80% of their ideal distance (stride / duty per cycle; the low lifts are
  the worst, 0.24 m lifts reach 77–80%), so stance feet slip in physics even though they are still
  relative to the body.
- At 0.4 m stride and duty 0.5, neighbouring legs touch in 11–14% of frames (Webots did not stop them);
  with duty 0.65 there is no contact, at 1.97 m.

**Which per-leg model for the spider gait: the pose-range one.** Every spider foot target lies inside the
pose range, where it is 4× more precise. The same 24 gaits through the motor-range model
(`outputs/nn_ik_motor/spider_sweep_motor.txt`) land the feet 1.1 cm off instead of 0.26 cm, and that
error is enough to steer the robot: the best motor-range gait (0.40 / 0.24 / 0.50) walks 2.08 m while
turning 5.5°, against 2.46 m and 0.0°; at 0.4 m stride and duty 0.5 its gaits turn 5–12°. `foot_gait`
therefore uses the pose-range model unless given `--motor`.

**Touch-down fix (`foot_paths(..., matched=True)`, runs `spider2_*`).** The first swing was a half-cosine:
the foot slowed to a stop relative to the body and landed, so over the ground it landed moving forward at
body speed and skidded. A real spider's foot is still over the ground when it lands. The swing is now a
cubic Hermite curve that leaves and lands with the stance's backward speed (it overshoots back by ~4% of
the stride just after lift-off). Same 24 gaits (`outputs/nn_ik_motor/spider2_sweep_pose.txt`):
- Best: 0.40 / 0.24 / 0.50 walks **2.53 m** with 5.91 m of sliding (was 2.46 m, 6.21 m), heading +0.1°.
- Most gaits gained 2–5% distance and lost 3–5% sliding: a real but small effect. Landing speed was not
  the main source of sliding; the stance feet slip under load.
- The faster mid-swing makes neighbouring legs touch more at 0.4 m stride and duty 0.5 (28% of frames,
  was 14%); with duty 0.65 there is still no contact.

**The GA tunes the foot paths (Part 1 `encoding=foot`, `docs/design.txt` §8).** The sweep's best sat
on the edge of its grid, so the GA searched 7 foot-path settings (stride, lift, duty, ground height,
reach, spread, back-to-front ripple), each candidate decoded through this per-leg network and walked
in Webots. Best run `outputs/foot_t14_s1`: **4.10 m in 4 cycles** (sweep best 2.53 m, pi0 2.05 m) with
3.30 m of sliding, pitch 0.015, no leg contact; 20.3 m over 20 cycles holding its heading within 2°.
Two seeds converged on nearly the same genes, including a ripple (each pair steps 0.13 of a cycle
after the pair behind it), the way real spiders walk.

## Reproduce
```bash
cd spider_cw
python -m part2_nn.ik.train       # experiments + both models -> part2_nn/ik/*.npz, outputs/nn_ik/summary.txt
python -m part2_nn.ik.ik_plots    # outputs/nn_ik/*.png
python -m part2_nn.ik.gait_test   # real GA gaits, needs Webots -> outputs/nn_ik/gait_test.txt
python -m part2_nn.ik.train --leg-motor   # per-leg network on the full motor range -> ik_leg_model_motor.npz
python -m part2_nn.ik.foot_gait   # spider foot-path sweep in Webots -> outputs/nn_ik_motor/spider_sweep_<model>.txt
```
The rebuilt gaits are in `outputs/ik_leg_<gait>/` and `outputs/ik_body_<gait>/`; play one with the
`gait_player` controller, e.g. `controllerArgs ["ik_leg_pi0"]`.
