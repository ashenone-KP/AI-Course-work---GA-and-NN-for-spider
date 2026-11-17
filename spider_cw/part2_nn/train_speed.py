"""Option A: train the speed-conditioned gait network and run the hyperparameter experiments.

    cd spider_cw
    python -m part2_nn.train_speed            # every experiment (parallel) + the final model
    python -m part2_nn.train_speed --quick    # 300 epochs per run, to check the pipeline

Model selection is leave-one-speed-out. Each configuration is trained 3 times; each time one inner
training speed (FOLDS) is hidden and used as the validation set, both for early stopping and to
score the configuration (mean joint-angle RMSE in degrees on the hidden speed). The real test
speed (speed060) is never used for any choice; its error is only reported.
Why: validating on random frames of the training gaits (first attempt) rewarded the models that
memorised the training speeds best, and those were the worst on a new speed.

Each experiment changes ONE setting of BASE. The final model is the single configuration with the
lowest leave-one-speed-out error of all trained (combining each experiment's best setting was tried
first: 128x128 + lr 0.03 + batch 16 together diverged, best epoch ~3, though each helped alone).
It is retrained on all non-test speeds for its folds' mean best epoch and saved to
part2_nn/models/speed_model.npz with its input normalisation.
Results in outputs/nn_speed/: experiments.json (all histories, for plots) and summary.txt.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from part2_nn import nn
from part2_nn.speed_data import HELD_OUT, ROOT, RUNS, build, load_frames, neighbours, scale, unscale

BASE = {"sizes": [64, 64], "hidden": "tanh", "opt": "adam", "lr": 1e-3, "batch": 64,
        "l2": 0.0, "epochs": 3000, "patience": 300, "seed": 0}
EXPERIMENTS = {
    "learning rate (Adam)": [{"lr": x} for x in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)],
    "learning rate (SGD + momentum 0.9)": [{"opt": "sgd", "lr": x} for x in (3e-3, 1e-2, 3e-2, 1e-1, 3e-1)],
    "hidden activation": [{"hidden": a} for a in ("tanh", "relu", "sigmoid")],
    "architecture": [{"sizes": s} for s in ([16, 16], [32, 32], [64, 64], [128, 128], [64], [64, 64, 64])],
    "batch size": [{"batch": b} for b in (16, 64, 256)],
    "weight decay (L2)": [{"l2": x} for x in (0.0, 1e-4, 1e-3, 1e-2)],
}
LABEL = {"learning rate (Adam)": lambda c: f"lr {c['lr']:g}",
         "learning rate (SGD + momentum 0.9)": lambda c: f"lr {c['lr']:g}",
         "hidden activation": lambda c: c["hidden"],
         "architecture": lambda c: "x".join(map(str, c["sizes"])),
         "batch size": lambda c: f"batch {c['batch']}",
         "weight decay (L2)": lambda c: f"l2 {c['l2']:g}"}
FOLDS = ("straight030_tuned", "straight055_tuned", "straight070_tuned")          # inner speeds spread over the range
OUT = os.path.join(ROOT, "outputs", "nn_speed")
MODEL_PATH = os.path.join(ROOT, "part2_nn", "models", "speed_model.npz")


def make_model(cfg):
    model = nn.MLP([3, *cfg["sizes"], 24], cfg["hidden"], "tanh", seed=cfg["seed"])
    opt = nn.Adam(cfg["lr"]) if cfg["opt"] == "adam" else nn.SGD(cfg["lr"], momentum=0.9)
    return model, opt


def rmse_deg(pred_scaled, target_scaled):
    return float(np.degrees(np.sqrt(np.mean((unscale(pred_scaled) - unscale(target_scaled)) ** 2))))


def mae_deg(pred_scaled, target_scaled):
    return float(np.degrees(np.mean(np.abs(unscale(pred_scaled) - unscale(target_scaled)))))


def fit(cfg, data, epochs=None, early_stop=True):
    model, opt = make_model(cfg)
    t0 = time.perf_counter()
    hist = nn.train(model, data["X_train"], data["Y_train"], data["X_val"], data["Y_val"], opt,
                    epochs=epochs or cfg["epochs"], batch_size=cfg["batch"],
                    patience=cfg["patience"] if early_stop else 10 ** 9, seed=cfg["seed"], l2=cfg["l2"])
    return model, hist, time.perf_counter() - t0


def run_config(cfg):
    """Leave-one-speed-out: one fit per fold. Returns the fold scores and histories, plus the
    held-out test error of the fold models (reported only, never used to choose)."""
    folds = []
    for val_run in FOLDS:
        data = build(val_run=val_run)
        model, hist, secs = fit(cfg, data)
        folds.append({"val_run": val_run, "train": hist["train"], "val": hist["val"],
                      "best_epoch": hist["best_epoch"], "val_rmse_deg": rmse_deg(model.forward(data["X_val"]), data["Y_val"]),
                      "test_rmse_deg": rmse_deg(model.forward(data["X_test"]), data["Y_test"]), "seconds": secs})
    return {"config": cfg, "folds": folds,
            "cv_rmse_deg": float(np.mean([f["val_rmse_deg"] for f in folds])),
            "test_rmse_deg": float(np.mean([f["test_rmse_deg"] for f in folds])),
            "best_epoch": int(np.mean([f["best_epoch"] for f in folds])),
            "seconds": float(sum(f["seconds"] for f in folds))}


def baseline():
    """No-learning baseline for the test speed: blend the two neighbouring gaits linearly by speed."""
    mid = HELD_OUT
    lo, hi = neighbours(mid)
    w = (RUNS[mid] - RUNS[lo]) / (RUNS[hi] - RUNS[lo])
    guess = scale((1 - w) * load_frames(lo) + w * load_frames(hi))
    return rmse_deg(guess, scale(load_frames(mid))), mae_deg(guess, scale(load_frames(mid)))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="300 epochs per run")
    args = p.parse_args(argv)
    base = dict(BASE, **({"epochs": 300, "patience": 100} if args.quick else {}))
    os.makedirs(OUT, exist_ok=True)

    jobs = [(name, dict(base, **change)) for name, changes in EXPERIMENTS.items() for change in changes]
    print(f"training {len(jobs)} configurations x {len(FOLDS)} folds in parallel...", flush=True)
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run_config, [cfg for _, cfg in jobs]))
    by_exp = {}
    for (name, _), r in zip(jobs, results):
        r["label"] = LABEL[name](r["config"])
        by_exp.setdefault(name, []).append(r)

    b_rmse, b_mae = baseline()
    lines = [f"baseline (blend of the neighbouring gaits): test RMSE {b_rmse:.2f} deg, mean abs {b_mae:.2f} deg",
             "per configuration: leave-one-speed-out RMSE (used to choose) | test-speed RMSE (reported only)"]
    for name, runs in by_exp.items():
        lines.append(f"\n{name}")
        for r in runs:
            lines.append(f"  {r['label']:10s} LOSO {r['cv_rmse_deg']:5.2f} deg | test {r['test_rmse_deg']:5.2f} deg"
                         f"  best epoch ~{r['best_epoch']:5d}  {r['seconds']:6.1f} s")
    print("\n".join(lines), flush=True)

    # final: the configuration with the lowest leave-one-speed-out error of all trained, retrained
    # on all non-test speeds for its folds' mean best epoch
    sel = min(results, key=lambda r: r["cv_rmse_deg"])
    final = sel["config"]
    epochs = max(sel["best_epoch"], 1)
    data = build()                                       # random-frame val only to plot; no early stopping
    model, hist, secs = fit(final, data, epochs=epochs, early_stop=False)
    test_rmse, test_mae = rmse_deg(model.forward(data["X_test"]), data["Y_test"]), mae_deg(model.forward(data["X_test"]), data["Y_test"])
    lines.append(f"\nfinal {final}\n  LOSO {sel['cv_rmse_deg']:.2f} deg; trained {epochs} epochs on all {len(RUNS) - 1} non-test speeds;"
                 f" test-speed RMSE {test_rmse:.2f} deg, mean abs {test_mae:.2f} deg ({secs:.1f} s)")
    print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    nn.save(model, MODEL_PATH, speed_mean=data["speed_mean"], speed_std=data["speed_std"])

    with open(os.path.join(OUT, "experiments.json"), "w") as f:
        json.dump({"base": base, "folds": FOLDS, "experiments": by_exp, "baseline_rmse_deg": b_rmse,
                   "baseline_mae_deg": b_mae, "final": {"config": final, "epochs": epochs, "loso": sel,
                                                         "train": hist["train"], "test_rmse_deg": test_rmse,
                                                         "test_mae_deg": test_mae}}, f)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {MODEL_PATH} and {OUT}", flush=True)


if __name__ == "__main__":
    main()
