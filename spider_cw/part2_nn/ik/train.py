"""Option E: train the inverse-kinematics network (24 foot coordinates -> 24 joint angles) and
run the hyperparameter experiments.

    cd spider_cw
    python -m part2_nn.ik.train            # every experiment in parallel + final model (~15 min)
    python -m part2_nn.ik.train --quick    # 20 epochs per run, to check the pipeline

Each experiment changes ONE setting of BASE. Every run trains on the same 40k training samples,
uses the 5k validation samples for early stopping, and is scored by the mean foot error (cm) on
the validation set. The run with the lowest score becomes the final model; only then is the test
set used, once, to report its error.

A second, per-leg network (3 -> 3, one network shared by all 8 legs) is trained for comparison.

Writes part2_nn/ik/ik_model.npz (+ ik_leg_model.npz) and outputs/nn_ik/results.json, summary.txt.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # many processes: one BLAS thread each
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from part2_nn import nn
from part2_nn.ik.ik_data import (MOTOR_HI, MOTOR_LO, ambiguity_check, axis_distance, build,
                                 foot_error_cm, from_per_leg, per_leg, scale, unscale)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # spider_cw
OUT = os.path.join(ROOT, "outputs", "nn_ik")
MODEL_PATH = os.path.join(ROOT, "part2_nn", "ik", "ik_model.npz")
LEG_MODEL_PATH = os.path.join(ROOT, "part2_nn", "ik", "ik_leg_model.npz")
LEG_MOTOR_MODEL_PATH = os.path.join(ROOT, "part2_nn", "ik", "ik_leg_model_motor.npz")   # --leg-motor

BASE = {"sizes": [64, 64], "hidden": "tanh", "opt": "adam", "lr": 1e-3, "batch": 64,
        "epochs": 500, "patience": 40, "seed": 0}
EXPERIMENTS = {
    "learning rate (Adam)": [{"lr": x} for x in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)],
    "learning rate (SGD + momentum 0.9)": [{"opt": "sgd", "lr": x} for x in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)],
    "hidden activation": [{"hidden": a} for a in ("tanh", "relu", "sigmoid")],
    "architecture": [{"sizes": s} for s in ([32, 32], [64, 64], [128, 128], [64], [128])],
    "batch size": [{"batch": b} for b in (32, 64, 256)],
}
LEG_CFG = dict(BASE, batch=512)          # 8x more samples, so 8x the batch: same updates per epoch


def label(cfg):
    return (f"{'x'.join(map(str, cfg['sizes']))} {cfg['hidden']} {cfg['opt']} "
            f"lr {cfg['lr']:g} batch {cfg['batch']}")


def make(cfg, n_in, n_out):
    model = nn.MLP([n_in, *cfg["sizes"], n_out], cfg["hidden"], "tanh", seed=cfg["seed"])
    opt = nn.Adam(cfg["lr"]) if cfg["opt"] == "adam" else nn.SGD(cfg["lr"], momentum=0.9)
    return model, opt


def run_body(cfg):
    """One 24 -> 24 run. Returns its histories, validation foot error and trained weights."""
    d = build()                                               # same seed in every process
    model, opt = make(cfg, 24, 24)
    t0 = time.time()
    h = nn.train(model, d["train"]["X"], d["train"]["Y"], d["val"]["X"], d["val"]["Y"], opt,
                 epochs=cfg["epochs"], batch_size=cfg["batch"], patience=cfg["patience"], seed=cfg["seed"])
    secs = time.time() - t0
    err = foot_error_cm(unscale(model.forward(d["val"]["X"])), d["val"]["feet"])
    return {"cfg": cfg, "train": h["train"], "val": h["val"], "best_epoch": h["best_epoch"],
            "epochs_run": len(h["train"]), "seconds": secs,
            "val_cm_mean": float(err.mean()), "val_cm_p95": float(np.percentile(err, 95))}, model.params


def leg_data(d):
    """Per-leg samples (left-leg form), inputs standardised with the training mean/std."""
    out = {}
    for part in ("train", "val", "test"):
        f, a = per_leg(d[part]["feet"], d[part]["angles"])
        out[part] = (f, scale(a, MOTOR_LO[:3], MOTOR_HI[:3]))
    mean, std = out["train"][0].mean(0), out["train"][0].std(0)
    return {k: ((f - mean) / std, y) for k, (f, y) in out.items()}, mean, std


def run_leg(cfg):
    """The 3 -> 3 network shared by all legs (320k training samples)."""
    d = build(box=cfg.get("box", "pose"))
    ld, mean, std = leg_data(d)
    model, opt = make(cfg, 3, 3)
    t0 = time.time()
    h = nn.train(model, *ld["train"], *ld["val"], opt, epochs=cfg["epochs"], batch_size=cfg["batch"],
                 patience=cfg["patience"], seed=cfg["seed"])
    secs = time.time() - t0
    pred = from_per_leg(unscale(model.forward(ld["val"][0]), MOTOR_LO[:3], MOTOR_HI[:3]))
    err = foot_error_cm(pred, d["val"]["feet"])
    return {"cfg": cfg, "train": h["train"], "val": h["val"], "best_epoch": h["best_epoch"],
            "epochs_run": len(h["train"]), "seconds": secs,
            "val_cm_mean": float(err.mean()), "val_cm_p95": float(np.percentile(err, 95))}, model.params, mean, std


def test_report(err_cm, angle_err_deg, axis_m):
    near = axis_m < 0.10
    return {"foot_cm_mean": float(err_cm.mean()), "foot_cm_p95": float(np.percentile(err_cm, 95)),
            "foot_cm_max": float(err_cm.max()),
            "angle_rmse_deg": float(np.sqrt(np.mean(angle_err_deg ** 2))),
            "angle_rmse_deg_by_joint": [float(np.sqrt(np.mean(angle_err_deg[:, j::3] ** 2))) for j in range(3)],
            "foot_cm_mean_near_axis": float(err_cm[near].mean()),
            "foot_cm_mean_far_from_axis": float(err_cm[~near].mean()),
            "hip_rmse_deg_near_axis": float(np.sqrt(np.mean(angle_err_deg[:, 0::3][near] ** 2))),
            "hip_rmse_deg_far_from_axis": float(np.sqrt(np.mean(angle_err_deg[:, 0::3][~near] ** 2)))}


def leg_motor():
    """Retrain only the per-leg network, on the full motor range (ik_data.BOXES["motor"]): real GA gaits
    put 17% of their angles outside the pose range. The pose-range models are kept as they are."""
    cfg = dict(LEG_CFG, box="motor")
    res, params, mean, std = run_leg(cfg)
    model, _ = make(cfg, 3, 3)
    for p, q in zip(model.params, params):
        p[...] = q
    nn.save(model, LEG_MOTOR_MODEL_PATH, mean=mean, std=std)
    d = build(box="motor")
    f, _ = per_leg(d["test"]["feet"], d["test"]["angles"])
    pred = from_per_leg(unscale(model.forward((f - mean) / std), MOTOR_LO[:3], MOTOR_HI[:3]))
    rep = test_report(foot_error_cm(pred, d["test"]["feet"]), np.degrees(pred - d["test"]["angles"]),
                      axis_distance(d["test"]["feet"]))
    out = os.path.join(ROOT, "outputs", "nn_ik_motor")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(dict(res, test=rep), fh)
    text = (f"per-leg 3 -> 3 network, motor range: {label(cfg)}, best epoch {res['best_epoch']} of "
            f"{res['epochs_run']}, {res['seconds']:.0f} s\n"
            f"  test (5000 motor-range poses x 8 feet): foot error mean {rep['foot_cm_mean']:.2f} cm, "
            f"95% {rep['foot_cm_p95']:.2f} cm, max {rep['foot_cm_max']:.1f} cm\n"
            f"  angle RMSE {rep['angle_rmse_deg']:.2f} deg (hip / thigh / knee "
            + " / ".join(f"{x:.2f}" for x in rep["angle_rmse_deg_by_joint"]) + ")")
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text, f"\nsaved {LEG_MOTOR_MODEL_PATH}", sep="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 epochs per run")
    ap.add_argument("--leg-motor", action="store_true", help="only the per-leg network, full motor range")
    args = ap.parse_args()
    if args.leg_motor:
        return leg_motor()
    quick = args.quick
    os.makedirs(OUT, exist_ok=True)

    configs, index = [], {}                      # each distinct config is trained once
    plan = {}
    for name, changes in EXPERIMENTS.items():
        plan[name] = []
        for ch in changes:
            cfg = dict(BASE, **ch)
            if quick:
                cfg.update(epochs=20, patience=20)
            key = json.dumps(cfg, sort_keys=True)
            if key not in index:
                index[key] = len(configs)
                configs.append(cfg)
            plan[name].append(index[key])
    leg_cfg = dict(LEG_CFG, **({"epochs": 20, "patience": 20} if quick else {}))

    print(f"training {len(configs)} body networks + 1 per-leg network in parallel ...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(configs) + 1, os.cpu_count() or 4)) as pool:
        leg_future = pool.submit(run_leg, leg_cfg)
        results = list(pool.map(run_body, configs))
        leg_res, leg_params, leg_mean, leg_std = leg_future.result()
    print(f"all runs done in {time.time() - t0:.0f} s", flush=True)
    runs = [r for r, _ in results]
    for r in runs:
        print(f"  {label(r['cfg']):42s} best epoch {r['best_epoch']:4d}  val foot error "
              f"{r['val_cm_mean']:.2f} cm (95% {r['val_cm_p95']:.2f})  {r['seconds']:.0f} s")

    # final model = lowest validation foot error; the test set is used only now, once
    best = min(range(len(runs)), key=lambda i: runs[i]["val_cm_mean"])
    model, _ = make(runs[best]["cfg"], 24, 24)
    for p, q in zip(model.params, results[best][1]):
        p[...] = q
    d = build()
    nn.save(model, MODEL_PATH, mean=d["mean"], std=d["std"])
    pred = unscale(model.forward(d["test"]["X"]))
    err = foot_error_cm(pred, d["test"]["feet"])
    axis = axis_distance(d["test"]["feet"])
    final = test_report(err, np.degrees(pred - d["test"]["angles"]), axis)

    leg_model, _ = make(leg_cfg, 3, 3)
    for p, q in zip(leg_model.params, leg_params):
        p[...] = q
    nn.save(leg_model, LEG_MODEL_PATH, mean=leg_mean, std=leg_std)
    f, _ = per_leg(d["test"]["feet"], d["test"]["angles"])
    leg_pred = from_per_leg(unscale(leg_model.forward((f - leg_mean) / leg_std), MOTOR_LO[:3], MOTOR_HI[:3]))
    leg_final = test_report(foot_error_cm(leg_pred, d["test"]["feet"]),
                            np.degrees(leg_pred - d["test"]["angles"]), axis)

    grad_err = nn.gradient_check(nn.MLP([24, 6, 5, 24], "tanh", "tanh", seed=3),
                                 d["train"]["X"][:6], d["train"]["Y"][:6])
    pairs, worst = ambiguity_check()
    n_params = lambda m: int(sum(p.size for p in m.params))

    results_json = {"base": BASE, "experiments": plan, "runs": runs, "best_run": best,
                    "final": final, "per_leg": dict(leg_res, test=leg_final),
                    "n_params": {"body": n_params(model), "per_leg": n_params(leg_model)},
                    "gradient_check": grad_err, "ambiguity": {"pairs": pairs, "max_angle_gap_deg": worst},
                    "quick": quick}
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results_json, fh)

    lines = [f"IK network results ({'QUICK RUN' if quick else 'full run'})", "",
             f"gradient check (backprop vs finite differences, 24-6-5-24 net): max relative error {grad_err:.1e}",
             f"ambiguity check: {pairs} pairs of feet < 2 cm apart, largest angle gap {worst:.0f} deg "
             f"(all near the hip axis)", "", "experiments (validation set, mean / 95th percentile foot error):"]
    for name, ids in plan.items():
        lines.append(f"  {name}")
        for i in ids:
            r = runs[i]
            lines.append(f"    {label(r['cfg']):42s} {r['val_cm_mean']:6.2f} / {r['val_cm_p95']:6.2f} cm   "
                         f"best epoch {r['best_epoch']:4d} of {r['epochs_run']:4d}   {r['seconds']:5.0f} s")
    for title, cfg, rep, n in (("final 24 -> 24 network", runs[best]["cfg"], final, n_params(model)),
                               ("per-leg 3 -> 3 network", leg_cfg, leg_final, n_params(leg_model))):
        lines += ["", f"{title}: {label(cfg)}, {n} weights (test set, 5000 poses x 8 feet)",
                  f"  foot error mean {rep['foot_cm_mean']:.2f} cm, 95% {rep['foot_cm_p95']:.2f} cm, "
                  f"max {rep['foot_cm_max']:.1f} cm",
                  f"  angle RMSE {rep['angle_rmse_deg']:.2f} deg (hip / thigh / knee "
                  + " / ".join(f"{x:.2f}" for x in rep["angle_rmse_deg_by_joint"]) + ")",
                  f"  feet within 10 cm of the hip axis: foot error {rep['foot_cm_mean_near_axis']:.2f} cm, "
                  f"hip RMSE {rep['hip_rmse_deg_near_axis']:.1f} deg; elsewhere "
                  f"{rep['foot_cm_mean_far_from_axis']:.2f} cm, {rep['hip_rmse_deg_far_from_axis']:.1f} deg"]
    lines.append(f"\nper-leg network trained in {leg_res['seconds']:.0f} s, best epoch {leg_res['best_epoch']}")
    text = "\n".join(lines)
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print("\n" + text)

    assert grad_err < 1e-5, grad_err
    assert final["foot_cm_mean"] < (20 if quick else 5), final["foot_cm_mean"]
    print(f"\nsaved {MODEL_PATH}\nsaved {LEG_MODEL_PATH}\nresults in {OUT}")


if __name__ == "__main__":
    main()
