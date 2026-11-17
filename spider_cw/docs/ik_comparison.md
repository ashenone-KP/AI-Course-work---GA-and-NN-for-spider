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
  Drawing the training poses from the full motor limits (`gait.MOTOR_LO/HI`) should fix most of this.
- **Hip angle near the hip axis.** When a foot is close to the hip's rotation axis, many hip angles put
  it in the same place, so the hip error is large there (`hip_axis.png`: per-leg hip RMSE 4.3° near the
  axis against 0.3° elsewhere). The foot still lands within 0.6 cm, which is what matters on the robot.
- **Noisy late training.** The per-leg loss spikes after epoch 150 (lr 0.001, no decay); early
  stopping keeps the best weights, but a lower final learning rate would likely help.

## Reproduce
```bash
cd spider_cw
python -m part2_nn.ik.train       # experiments + both models -> part2_nn/ik/*.npz, outputs/nn_ik/summary.txt
python -m part2_nn.ik.ik_plots    # outputs/nn_ik/*.png
python -m part2_nn.ik.gait_test   # real GA gaits, needs Webots -> outputs/nn_ik/gait_test.txt
```
The rebuilt gaits are in `outputs/ik_leg_<gait>/` and `outputs/ik_body_<gait>/`; play one with the
`gait_player` controller, e.g. `controllerArgs ["ik_leg_pi0"]`.
