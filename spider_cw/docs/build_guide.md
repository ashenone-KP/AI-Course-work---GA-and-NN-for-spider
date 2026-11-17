# Build guide: the spider GA and neural network, from scratch

This guide explains every design choice and gives the current best setup for both parts. It is detailed enough to re-implement both parts by hand. The full reasoning and evidence are in `design.txt` (Part 1), `nn_design.txt` (Part 2, option A) and `ik_comparison.md` (option E). Every decision and what it changed is listed in `decisions.md`; this guide is the recipe.

```
Part 1  chromosome (12-13 numbers) --decode--> gait: 300 frames x 24 angles --walk in Webots--> score
        GA: population of chromosomes -> select, cross, mutate -> better gaits
Part 2A GA gaits at 9 speeds --> dataset ([sin, cos, speed] -> 24 angles) --> MLP 3-64-64-24
        network(speed) --> new gait --> walk in Webots (with heading feedback)
Part 2E random poses --FK--> (foot, angles) pairs --> per-leg MLP 3-64-64-3: foot -> 3 joint angles
Best    GA chromosome (7 foot-path settings) --> 300 x 8 foot targets --per-leg IK net--> gait --> Webots
        (sections 1.8 and 2.8: the GA from Part 1 driving the network from Part 2)
```

---

## Part 1: the genetic algorithm

### 1.1 The robot and the 24 numbers
- **Robot:** the Webots `SpiderOctopod` (`Spider_world_demo/protos`), with `selfCollision TRUE`.
- **Leg order:** the spec's `[L1, L2, L3, L4, R4, R3, R2, R1]`, three joints per leg (hip, thigh, knee). That makes 24 angles in radians.
- **Motors:** `L1_hip_joint`, `L1_thigh_joint`, `L1_knee_joint`, …; each joint's sensor is the motor name plus `_sensor`.
- **Motor limits:** hip and thigh ±1.5708, knee [−2, 1].
- **Gait:** 300 frames of 24 angles make one step cycle. We play one cycle in **2 s**.
  - Each physics step (the world's default 32 ms), compute `phase = t / 2 s × 300`.
  - Command the linear interpolation between frame `⌊phase⌋ mod 300` and the next one.
- **Leg geometry** (from the proto, used by `body.py`):
  - Body box 1.3 × 0.4 × 0.3 m. Hips at x = ±0.4875, ±0.1625 and y = ±0.2.
  - Segments are 0.5 / 0.6 / 0.7 m.
  - The hip axis is tilted 45°, and the thigh and knee turn about the coxa's local x axis.
  - Standing pose: left legs (0, −1.07, 0.58); right legs are the negatives.

### 1.2 Encoding: why a gait is a formula, not 7200 numbers
A gait is 300 × 24 = 7200 numbers, which is far too many to evolve directly. Every joint therefore follows one sine wave:

```
angle_j(k) = clip( pose_j + amp_j * sin(2*pi*k/300 + phase_j), motor limits )      k = 0..299
```

The chromosome only sets `pose`, `amp` and `phase`, and those are shared heavily:

| Rule | Why (what failed without it) |
|---|---|
| **Mirror:** right leg = −(left leg) for pose and amplitude, so the right joints turn the other way in the proto. | Version 1: 0 of 5 seeds walked without symmetry, 5 of 8 with it. |
| **Fixed timing:** `phase = π·is_right + π·position + joint_phase`, where position is 0..3 for L1..L4 and R4..R1 = 3..0. This is the alternating tetrapod {L1,L3,R2,R4} / {L2,L4,R1,R3}. | With free phases every seed stood still. With the left/right offset as a gene the GA "rowed" (pairs moved together). With the front/back offset as a gene, legs went out of sync. |
| **Shared motion:** every leg makes the same step; only its hip direction (the "fan") differs. | With a separate swing per leg, some legs idled while others worked (one pair touched the floor only 6–24% of the time). |
| **Push-pull:** the hind two pairs' knee phase is +π. | With shared motion the front knees pull forward but the hind knees, pointing backward, pull backward. Offsetting the hind knees makes the hind legs push. It was the user's idea; as a free gene the GA drifted back to 0, so it is fixed at π. |
| **Hip fan bounds ±1.2 rad, hip swing ≤ 0.6** | Neighbouring hips are only 0.325 m apart. With ±0.6 the legs collided unless the hip swing was about 0, so the robot stepped in place. |

**Best encoding: `pushpi`, 12 genes** (`gait.py`, `SHARED_*` bounds plus the fixed offset):

| Gene | Meaning | Bounds | pi0's value |
|---|---|---|---|
| 0–3 | hip pose of L1, L2, L3, L4 (the fan) | [−1.2, 1.2] | −0.784, −0.004, 0.557, 0.911 |
| 4 | thigh pose, all legs | [−1.5, 0] | −1.290 |
| 5 | knee pose, all legs | [−0.5, 1] | 0.894 |
| 6 | hip swing amplitude | [0, 0.6] | 0.266 |
| 7 | thigh swing amplitude | [0, 0.5] | 0.499 |
| 8 | knee swing amplitude | [0, 0.5] | 0.198 |
| 9 | thigh phase relative to the hip (wraps) | [0, 2π) | 4.095 |
| 10 | knee phase relative to the hip (wraps) | [0, 2π) | 1.622 |
| 11 | hind-knee offset | fixed π | 3.1416 |
| *12* | *steer (only in the `steer` encoding)* | *[−0.3, 0.3]* | *straight060_tuned: 0.087* |

**Decode, by hand:**
```
for each left leg i in L1..L4 (position p = 0..3):
    pose = (hip_i, thigh, knee);  amp = (hip_amp, thigh_amp, knee_amp)
    ph   = (0, thigh_phase, knee_phase) + pi*p  (+ pi on the knee if p >= 2)
for its mirror right leg (L1<->R1, ..., L4<->R4):
    pose, amp = -pose, -amp;  ph = ph + pi
steer (13th gene): left hip amp *= (1 + s), right hip amp *= (1 - s)
frame k: angle = pose + amp * sin(2*pi*k/300 + ph), clipped to the motor limits
columns in the order L1, L2, L3, L4, R4, R3, R2, R1
```
Check: L1 and R1 give the same motor angle (mirrored *and* half a cycle apart, so they swing opposite ways in the world). L1 and L3 swing in step. L2's swing is L1's negated: with all poses at 0, L2 = −L1. `gait.py` asserts these.

### 1.3 Fitness: walk it in physics and measure what happened
Version 1 scored gaits on a no-physics model, and its gaits didn't transfer to Webots. So every gait is now walked in Webots (`ga_trainer.py`, a Supervisor controller):

1. **Once at the start:** command the standing pose, wait 2 s, then `saveState("start")`.
2. **For each chromosome:**
   - `loadState("start")` and `resetPhysics()`;
   - command frame 0 and wait 1.5 s (`SETTLE_S`);
   - play **2 cycles** (4 s).
3. **Every step, record:**
   - the body position;
   - the body orientation R (3 × 3);
   - the 24 joint sensors (the *measured* angles, which show legs blocked by collisions).
4. **Measurements:**
   - `forward` and `drift`: the displacement in the start heading's frame (yaw = atan2(R[1,0], R[0,0])).
   - `turn`: the change in yaw, wrapped to ±π.
   - Height: `min_height` and `low_time`, the fraction of steps with the body below 0.25 m.
   - `fell`: R[2,2] (the body's up-axis z) dropped below 0.5, meaning tilted more than 60°.
   - **Feet:** foot = body position + R · FK(measured angles) at the tibia tip.
     - A foot is "down" if z < 0.04 m.
     - Per leg: lift = the highest z reached; `stepping` = it touched down at some point, was up at some point, and lift ≥ 0.05; slide = the distance the foot moved while down on two consecutive steps.
     - `lift` = mean over all 8 legs of (min(lift, 0.10) if stepping else 0) / 0.10. `idle_legs` = number of legs not stepping. `slide` = total over all legs.
   - `pitch`: the mean of |asin(R[2,0])|, nose up or down.
5. **Score** (higher is better):
```
score = forward - 0.5|drift| - 0.5|turn|
        - (5*low_time + 2*max(0, 0.25 - min_height))      # sliding on the belly
        - 5*fell - 3*collision_fraction
        + 0.5*lift - 0.4*idle_legs - 0.3*slide
        - 3*max(0, pitch - tilt)                          # tilt = 0.07: 4 deg is free (option tilt=)
```
(The foot-path runs use `cycles=4` and `tilt=0.14`; see §1.8.)
6. **Self-collision pre-check** (`body.py`): run the commanded frames through forward kinematics and model each segment as a capsule.
   - A frame "collides" if a segment comes within the two radii + 2 cm of the next or next-but-one leg, or a thigh or tibia enters the body box.
   - If more than 20% of frames collide, the gait is skipped (not simulated).
7. **Cache:** physics is deterministic (the self-test walks one gait 3 times and gets identical results), so scores are cached per chromosome. This makes elites free.

**Why each term:** each one fixed a failure seen in Webots.
- Without the low-body term, the robot slid on its belly.
- Without the per-leg terms, the third pair was dragged and only the front pulled.
- The slide weight is 0.3: at 0.1 the feet slid 11 m for 3.8 m forward; much higher and standing still wins.
- The tilt rule exists because shared1 leaned 12°. Penalising all tilt halved the speed, so 4° is free.

### 1.4 The GA loop (`ga.py`)
```
pop = uniform random in [low, high]            (or a start gait + 3% noise to refine it)
for gen = 0..G:
    scores = evaluate(pop); record best/mean/worst
    stop if gen == G, or no improvement for `stall` generations, or a target is reached
    elites   = the top 2 (copied unchanged: the best can never be lost)
    parents  = tournament: n times, pick 3 at random and keep the best
    children = pair the parents; with probability 0.9 apply BLX-0.5 blend crossover, else copy:
               d = |a-b|; child ~ U[min - 0.5d, max + 0.5d]   (each gene, each child)
               circular genes: first move b to its copy nearest a (b = a + ((b-a+pi) mod 2pi) - pi)
    mutation = each gene with probability 0.05: += N(0,1) * sigma * (high-low)
               sigma = 0.1 * max((1 - 0.9*gen/G) * 0.5^(stall/stall_gens), 0.05)   (adaptive)
    repair   = clip ordinary genes, wrap phases into [0, 2pi)
    pop = elites + children
```
**Why these operators:**
- **Tournament:** it compares ranks only, so it works with negative scores and keeps constant pressure. Roulette needs shifted scores and its pressure fades late in a run.
- **Blend crossover:** it creates new values between and around the parents, and the range shrinks as the population converges. Two-point cuts apart genes that only work together, and was clearly worst in our experiments.
- **Gaussian mutation scaled by each gene's range:** one sigma then works for genes in radians and genes in phase units alike.
- **Adaptive sigma:** big steps early to explore, small steps late to fine-tune.
- **Wrap, not clip, for phases:** 2π + 0.1 is the same gait as 0.1.

### 1.5 Best joint-sine setup (Part 1)
The best gait *overall* is now the foot-path gait in §1.8; this is the best gait made purely from joint sines.

| | |
|---|---|
| **Encoding** | `pushpi` (12 genes) |
| **GA** | population 40, 100 generations, tournament k = 3, BLX-0.5 at rate 0.9, mutation rate 0.05 with adaptive sigma 0.1, elitism 2, seed 0 |
| **Fitness** | the score above, 2 cycles of 2 s |
| **Result: `pi0`** | **2.05 m in 4 cycles** (8 s), tilt 3.4°, all 8 legs stepping, no falls or collisions; 7.7 m of foot slide |
| **Run it** | Webots world `spider_world_1.wbt`, controller `ga_trainer`, args `pop=40 gens=100 seed=0 encoding=pushpi name=live` (reproduces pi0 exactly) |

**Alternatives:**
- **`ref10_2`:** pi0 refined with `start=pi0 slip=1.0`. 1.94 m, with 36% less sliding.
- **`shared1`:** 11 genes, no tilt rule. The fastest at 2.48 m, but it leans 12°.

**Operator experiments** (`outputs/operators`, 3 seeds × 60 generations) suggest better defaults for this 12-gene encoding: mutation rate 0.2 (≈ 2.4 genes per child, against 0.6 at 0.05) and population 80. The reference gaits were made before those experiments and have not been re-run with them.

### 1.6 Walking straight (added after watching all gaits side by side)
- **The problem:** scoring 2 cycles hides slow curves. Over 40 s, `ref10_2` turned 29° and the best operator run turned 64°.
- **Steer gene** (13th gene): left hip amplitude × (1 + s), right × (1 − s). Without it the mirrored encoding cannot steer.
- **Tuning** (`tune=<run>`):
  - Walk 12 cycles at s = −0.3 … 0.3 (7 values).
  - Measure the *steady* turn rate: the slope of the heading at the end of cycles 4–12. Cycles 0–3 are a start-up wobble that the GA learned to exploit.
  - Bracket the zero crossing and refine it by alternately interpolating and bisecting.
  - Result: turning radius 66 m – 6 km, against 17–180 m before.
- **Heading feedback when playing** (`gait_player`, argument `"straight"`):
```
line = direction travelled over cycles 0-1
every cycle after that:
    heading  = direction travelled over the last cycle, relative to the line
    sideways = offset from the line (+ = left)
    err      = heading + atan2(sideways, 3 m)          # aim at the line 3 m ahead
    I        = clip(I + 0.5*err, +-0.3)                # learns the gait's own steady curve
    s        = clip(1.0*err + I, +-0.3)
    hip angle = cycle-mean + (1 + s*side)*(angle - cycle-mean)   # side = +1 left, -1 right
```
  After 20 cycles every network gait ends within 5 cm of its line. Only the GPS is needed.

### 1.7 Visualisation (what to build so you can see it working)
- **Fitness:** best, mean and worst per generation, plus each score component of the best gait.
- **Mutation sigma** per generation.
- **Poses:** 3D renders at 4 points in the run.
- **Joint angles** per leg, a footfall diagram and a GIF (`plots.py`).
- **Live:** the Webots label and a real-time replay of each generation's best gait.
- **`replay=<run>`:** a captioned video of the improving generations.
- **`measure=<run>`:** per-leg lift, slide and stepping, plus path curvature.
- **`all_gaits.wbt`:** all main gaits side by side.

### 1.8 Best setup: foot-path gaits (the GA tunes foot paths, the IK network makes angles)
**Why:** a joint sine can't draw a flat stance stroke, so every sine gait's feet slide (pi0: 7.75 m in 4 cycles). A real spider does it the other way round: a planted foot stays still on the ground while the body passes over it, then lifts and swings forward. With the per-leg IK network (§2.8) that can be written down directly, as foot paths, and the GA only has to tune 7 numbers that each mean something physical.

**Neutral feet** (each leg's average foot position over pi0's cycle, body frame: x forward, y left, metres):

| leg | L1 | L2 | L3 | L4 | R4 | R3 | R2 | R1 |
|---|---|---|---|---|---|---|---|---|
| x̄ | 0.8163 | 0.2033 | −0.3628 | −0.8117 | −0.8223 | −0.3699 | 0.2036 | 0.8247 |
| ȳ | 1.2283 | 1.3356 | 1.2876 | 1.2211 | −1.2258 | −1.2945 | −1.3485 | −1.2387 |

**Decode a 7-gene chromosome** `(stride S, lift H, duty d, height h, reach r, spread p, wave w)`:
```
group    = (0, 1, 0, 1, 0, 1, 0, 1)        for L1 L2 L3 L4 R4 R3 R2 R1  (L1 L3 R4 R2 step together)
position = (0, 1, 2, 3, 3, 2, 1, 0)        front pair = 0 .. hind pair = 3
for frame k = 0..299, leg i:
    u = (k/300 + 0.5*group_i + w*position_i) mod 1          # where this leg is in its own step
    if u < d:                                                # stance: foot on the ground
        s  = u/d;            dx = S/2 - S*s;   dz = 0       # straight back at constant speed
    else:                                                    # swing: foot in the air
        s  = (u-d)/(1-d);    m = -S*(1-d)/d                  # m = stance speed in swing-time units
        dx = -S/2*(2s^3 - 3s^2 + 1) + S/2*(-2s^3 + 3s^2) + m*(s^3 - 2s^2 + s) + m*(s^3 - s^2)
        dz = H*sin(pi*s)
    foot_i(k) = (p*x̄_i + dx,  r*ȳ_i,  h + dz)
angles(k) = per-leg IK network (§2.8) on every foot, clipped to the motor limits -> 300 x 24 gait
```
- **The swing curve** is a cubic Hermite curve: it starts and ends with the stance's backward speed `m`, so the foot is still over the ground at lift-off and touch-down. A half-cosine landed the foot moving forward at body speed, and it skidded.
- **The ripple:** with `w > 0`, each pair steps `w` of a cycle after the pair behind it, a back-to-front wave like a real spider's. With `w = 0` it is two rigid groups of four.
- **Ideal speed** with no slipping is `S/d` metres per cycle.

**Genes and bounds** (`encoding=footwide`; `encoding=foot` is the first, narrower version):

| gene | footwide bounds | (foot) | footwide_s2 | why the bound |
|---|---|---|---|---|
| stride S (m) | 0.10–0.90 | 0.10–0.60 | 0.900 | long strides collide unless reach/spread grow |
| lift H (m) | 0.04–0.45 | 0.04–0.30 | 0.381 | low swings brush the floor |
| duty d | 0.35–0.85 | 0.45–0.80 | 0.522 | share of the cycle each foot is down |
| height h (m) | −0.68 to −0.38 | −0.68 to −0.50 | −0.380 | lower is out of the legs' reach |
| reach r | 0.78–1.10 | 0.85–1.10 | 0.881 | > 1.10 out of reach; < 0.78 the IK errs 15+ cm |
| spread p | 0.85–1.40 | 0.85–1.15 | 1.400 | past 1.40 the IK errs (22 cm at 1.45) |
| wave w | 0.00–0.25 | 0.00–0.15 | 0.232 | |

**The bounds stop where the IK network stops being accurate:** past them the GA would be rewarded for the network's mistakes, not for better foot paths.

**GA settings for the best run:** the same GA as §1.4, with population 40, 50 generations, `cycles=4`, mutation rate 0.2 (the operator experiments' best for small encodings), `tilt=0.14`, seed 2. All genes are ordinary (clipped, none wrap). Run it with `controllerArgs ["encoding=footwide" "pop=40" "gens=50" "cycles=4" "mrate=0.2" "tilt=0.14" "seed=2" "name=footwide_s2"]`.

| | 4 cycles | foot sliding | pitch | 20 cycles (40 s) |
|---|---|---|---|---|
| **`footwide_s2`** | **6.55 m** | **2.46 m** | 0.010 | **33.3 m, heading +0.8°** |
| `foot_t14_s1` (narrow bounds) | 4.10 m | 3.30 m | 0.015 | 20.3 m, +2.0° |
| `pi0` (joint sines) | 2.05 m | 7.75 m | 0.059 | n/a |

No falls and no leg contact. `footwide_s2` was still improving at its last generation, and stride, height and spread are on their bounds.

---

## Part 2: the neural network (option A, speed-conditioned gait generator)

### 2.1 Task and input/output format
- **Input** (3): `[sin φ, cos φ, s]`.
  - `φ = 2πk/300` is the frame's place in the cycle. As sin and cos, the last frame sits next to the first.
  - `s = (speed − 0.320) / 0.1248` is the requested speed in metres per cycle, normalised with the mean and std of the 8 training speeds.
- **Output** (24): the motor angles, scaled per joint to the range a tanh can produce:
```
mid  = (limit_hi + limit_lo)/2                    hip/thigh 0, knee -0.5
half = 1.1 * (limit_hi - limit_lo)/2              hip/thigh 1.7279, knee 1.65
y = (angle - mid)/half        angle = clip(mid + half*y, limits)
```
- **Why a tanh output:** every output is then a valid joint angle by construction. The 10% margin means a target sitting exactly on a limit doesn't need tanh = ±1.

### 2.2 Training data: made by the GA
1. **Target speeds:** the trainer's `target=<m>` mode replaces `forward` with `−2·|forward − target|`; every other term stays.
2. **Refining:** starting from ref10_2 (`start=ref10_2`), it evolved one gait per target. Starting every speed from the same gait makes the 9 gaits a smooth family, so interpolating between them means something.
3. **Measuring:** each gait's real speed was measured in Webots over 4 cycles, and the network is given the *measured* speed.
4. **Straightening:** those gaits curved, so each one was re-evolved with the steer gene (4 cycles, `turnw=5`), then its steer was tuned (§1.6). These are the `straight0xx_tuned` gaits.
5. **Speeds (m per cycle):** 0.121, 0.164, 0.243, 0.312, **0.366 (test, never trained on)**, 0.380, 0.405, 0.458, 0.477.
6. **Dataset:** 8 gaits × 300 frames = 2400 samples.
   - The test set is the whole 0.366 gait. A random split of frames would leak, because frames of one gait are near-copies.

### 2.3 Network
- **Shape:** `3 → 64 → 64 → 24`, tanh on both hidden layers and on the output.
- **Initialisation:** Xavier, `W ~ N(0, 2/(n_in + n_out))` (He `2/n_in` for ReLU); biases 0.
- **Why this size:** 16 × 16 up to 128 × 128 and 3 layers all scored within 0.25° of each other. Size matters less than the learning rate. 2400 near-duplicate samples don't need a big network.
- **Why tanh:** it's smooth, so the network interpolates smoothly between speeds. ReLU (piecewise linear) was worst in every round.

### 2.4 Forward, loss and backpropagation (all hand-written in `nn.py`)
```
forward:  a0 = X;  a_l = f(a_{l-1} W_l + b_l)             keep every a_l
loss:     L = mean((a_out - Y)^2)  over all samples and all 24 outputs
          dL/da_out = 2 (a_out - Y) / (n_samples * 24)
backward: for l = last..1:
            delta_l = (dL/da_l) * f'(a_l)                   tanh' = 1 - a^2
            dW_l = a_{l-1}^T delta_l;   db_l = sum over the batch of delta_l
            dL/da_{l-1} = delta_l W_l^T
```
**Check it with finite differences:** `(L(w+ε) − L(w−ε)) / 2ε` for every weight. Our largest relative difference is below 1e−5, and PyTorch autograd agrees to 6e−17.

### 2.5 Training method
- **Optimiser:** Adam, `lr = 0.03`, β1 = 0.9, β2 = 0.999, ε = 1e−8.
```
m = b1 m + (1-b1) g;   v = b2 v + (1-b2) g^2
m^ = m/(1-b1^t);       v^ = v/(1-b2^t);       w -= lr * m^ / (sqrt(v^) + eps)
```
- **Batches:** mini-batches of 64, reshuffled every epoch.
- **Early stopping** (used while choosing settings): keep the weights from the best validation epoch, and stop after 300 epochs without improvement.
- **Choosing the settings: leave-one-speed-out.** Train 3 times, each time hiding one inner speed (0.164, 0.312, 0.405) as validation, and average its RMSE in degrees.
  - Every experiment changed one setting from a base of 64 × 64 tanh, Adam 0.001, batch 64. We tested learning rates 3e−4 … 3e−2 (Adam) and 3e−3 … 0.3 (SGD with momentum), activations, 6 architectures, batch sizes 16 / 64 / 256, and weight decay.
  - Lowest score: Adam 0.03 (1.83°).
  - **Why not a random 15% of frames:** we tried that first. The models that fitted those frames best were the *worst* on a new speed, because they memorised.
- **Final model:** the best configuration, trained on the 8 non-test speeds for its folds' mean best epoch (**107 epochs**), with no early stopping.
  - A random 15% of those frames is set aside to draw a validation curve; the other 85% are trained on.
  - The weights and the speed mean and std are saved to `part2_nn/models/speed_model.npz`.

### 2.6 Using it
```
for k in 0..299:  x_k = [sin(2*pi*k/300), cos(2*pi*k/300), (v - 0.320)/0.1248]
frames = unscale(network(x))          # 300 x 24 motor angles -> best_gait.csv
play in Webots with gait_player [<run>, 2.0, "straight"]
```

### 2.7 Current best setup (Part 2) and results
| | |
|---|---|
| **Network** | 3-64-64-24, tanh / tanh, Xavier initialisation, MSE on scaled angles |
| **Training** | Adam lr 0.03, batch 64, 107 epochs; settings chosen by leave-one-speed-out |
| **Data** | 8 straightened GA gaits, 0.12–0.48 m per cycle; 0.366 held out |
| **Joint error on the held-out speed** | 1.86° RMSE (blend of the two nearest gaits: 1.90°) |
| **Speed in Webots** | within 1–2% of the request (6–7% at 0.117 and 0.300); all legs step, no falls |
| **Straightness** | open loop up to 18° in 40 s; with feedback within 5 cm of the line |
| **PyTorch** | same initial weights and batches: identical final loss 1.634255e−4; numpy 2.3 s, torch 7.0 s |

### 2.8 Option E: the per-leg inverse-kinematics network
**Task:** given where a foot should be, output that leg's 3 joint angles. This is what turns §1.8's foot paths into a gait. It was the teammate's part (`part2_nn/ik/`); we ran the full training and chose between their two networks.

**Data** (`ik_data.py`), made with forward kinematics, so it is exact and unlimited:
1. Draw 50,000 random poses. For each leg independently: hip in [−1.2, 1.2], thigh in [−1.5, 0], knee in [−0.5, 1]. Right legs get the negated angles (mirror).
2. Compute every foot with the forward kinematics.
3. Split 40,000 for training, 5,000 for validation (early stopping, model choice) and 5,000 for the test, used once.
4. **One sample per leg:**
   - input: the foot relative to its own hip, `foot − hip`, with y negated for right legs;
   - target: that leg's 3 angles, negated for right legs.
   After mirroring, every leg is the same problem, so this gives 320,000 training samples for one small network.

**Scaling:**
```
input:  x = (rel_foot - mean) / std        mean (-0.0004, 0.9628, -0.2810), std (0.4642, 0.3086, 0.3063)
                                            (training feet only)
output: y = 2*(angle - lo)/(hi - lo) - 1     lo = (-1.5708, -1.5708, -2), hi = (1.5708, 1.5708, 1)
        angle = lo + (y + 1)/2 * (hi - lo)   tanh output -> always a legal left-leg angle
use it: for each leg, rel = foot - hip; if right: rel.y = -rel.y
        angles = unscale(net((rel - mean)/std)); if right: angles = -angles
```

**Network and training:**
- **Shape:** `3 → 64 → 64 → 3`, tanh hidden and output layers, Xavier initialisation, MSE loss.
- **Optimiser:** Adam, lr 0.001, batch 512, the same core and backpropagation as §2.4–2.5.
- **Stopping:** at most 500 epochs; stop after 40 epochs without improvement on validation and keep the best weights. Its best epoch was 270.
- **Gradient check:** relative error 4.8e−7.

**Why per-leg** rather than one 24 → 24 network for all 8 feet (best found: 128 × 128, Adam 0.001, batch 64):
- **The problem really is per leg.** A leg's angles depend only on its own foot; the whole-body network has to discover that 21 of its 24 inputs don't matter for each output.
- **More data, fewer weights:** mirroring gives one problem with 8× the data, using 5× fewer weights (4,611 against 22,808).

| | whole-body | **per-leg** |
|---|---|---|
| random test poses: mean / 95% foot error | 3.67 / 8.23 cm | **0.50 / 1.30 cm** |
| real GA gaits rebuilt: mean foot error | 6.94 cm | **2.79 cm** |
| rebuilt gaits walked in Webots: distance kept | 89% | **101%** |

**Judge by foot error, not angle error.** When a foot is near its hip's rotation axis, many hip angles put it in the same place. The per-leg hip error is 4.3° there, but the foot still lands within 0.6 cm.

**The third model** (`train --leg-motor`) draws training poses from the full motor range (hip ±1.57, thigh [−1.57, 0], knee [−1, 1]).
- **Gain:** it handles poses outside the original range (shared1's worst feet: 35 cm → 2.3 cm).
- **Cost:** it is less precise inside the range (1.1 cm against 0.26 cm on the foot paths).
- **Use:** the foot-path gaits use the original-range model, because all their targets lie inside its range.

---

## Building it yourself: order and checks
1. **`decode()`.** Check the mirror and timing rules from §1.2: L1 equals R1, L1 and L3 swing in step, and L2 is L1 negated.
2. **Forward kinematics.** Check the standing pose: all 8 feet at the same height, below the body.
3. **GA on a test function** (a sphere with a phase optimum near 0). Best-so-far must never drop with elitism, and the same seed must give the same run.
4. **A Webots supervisor:** save the start state, then walk one gait 3 times. The results must be identical, which makes caching valid.
5. **Fitness terms one at a time,** watching the best gait in Webots after each. Every term in §1.3 exists because something looked wrong.
6. **MLP plus the gradient check** before any real training.
7. **Dataset and leave-one-speed-out experiments,** then the final model, then gaits walked in Webots.
8. **IK data and the per-leg network** (§2.8). Check that the exact angles give 0 foot error through the forward kinematics, that a mirrored right leg gives the same relative foot as its left leg, and that the trained network lands test feet within about 0.5 cm.
9. **Foot paths** (§1.8). Check that each foot is down for a share `d` of the cycle, that L1, L3, R4 and R2 move together when `w = 0`, and that the swing lands at the stance's speed. Then the GA with `encoding=footwide`.
