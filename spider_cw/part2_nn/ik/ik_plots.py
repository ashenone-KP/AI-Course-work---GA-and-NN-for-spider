"""Plots for the inverse-kinematics network (run after part2_nn.ik.train).

    cd spider_cw
    python -m part2_nn.ik.ik_plots

Writes to outputs/nn_ik/:
    loss.png            train / validation loss per epoch (final 24->24 network and per-leg network)
    lr_sweep.png        best validation loss against learning rate, Adam and SGD + momentum
    experiments.png     validation foot error for every activation / architecture / batch-size run
    pred_vs_target.png  predicted against true angle, test set, one panel per joint type
    foot_error.png      histogram of test foot errors (cm), both networks
    hip_axis.png        hip angle error against the foot's distance from the hip axis (the ambiguity)
    poses.png           target pose vs predicted pose for a typical, a bad and the worst test pose
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from part1_ga.body import plot_pose
from part2_nn import nn
from part2_nn.ik.ik_data import MOTOR_HI, MOTOR_LO, axis_distance, build, foot_error_cm, from_per_leg, per_leg, unscale
from part2_nn.ik.train import LEG_MODEL_PATH, MODEL_PATH, OUT, label

JOINTS = ("hip", "thigh", "knee")


def predictions(d):
    model, extra = nn.load(MODEL_PATH)
    body = unscale(model.forward((d["test"]["feet"] - extra["mean"]) / extra["std"]))
    leg_model, leg_extra = nn.load(LEG_MODEL_PATH)
    f, _ = per_leg(d["test"]["feet"], d["test"]["angles"])
    leg = from_per_leg(unscale(leg_model.forward((f - leg_extra["mean"]) / leg_extra["std"]),
                               MOTOR_LO[:3], MOTOR_HI[:3]))
    return body, leg


def main():
    with open(os.path.join(OUT, "results.json")) as fh:
        res = json.load(fh)
    runs, d = res["runs"], build()
    body, leg = predictions(d)
    target = d["test"]["angles"]
    saved = []

    def save(fig, name):
        path = os.path.join(OUT, name)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # 1. loss curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, r, title in ((axes[0], runs[res["best_run"]], "24 -> 24 network: " + label(runs[res["best_run"]]["cfg"])),
                         (axes[1], res["per_leg"], "per-leg 3 -> 3 network: " + label(res["per_leg"]["cfg"]))):
        ax.semilogy(r["train"], label="train")
        ax.semilogy(r["val"], label="validation")
        ax.axvline(r["best_epoch"], color="grey", ls="--", lw=1, label=f"best epoch {r['best_epoch']}")
        ax.set(xlabel="epoch", ylabel="MSE loss (scaled angles)", title=title)
        ax.title.set_fontsize(9)
        ax.legend()
        ax.grid(alpha=0.3)
    save(fig, "loss.png")

    # 2. learning-rate sweep
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, marker in (("learning rate (Adam)", "o-"), ("learning rate (SGD + momentum 0.9)", "s-")):
        ids = res["experiments"][name]
        ax.loglog([runs[i]["cfg"]["lr"] for i in ids], [min(runs[i]["val"]) for i in ids], marker,
                  label=name.replace("learning rate ", ""))
    ax.set(xlabel="learning rate", ylabel="best validation loss", title="Learning-rate sweep (64x64 tanh, batch 64)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    save(fig, "lr_sweep.png")

    # 3. other experiments
    groups = ("hidden activation", "architecture", "batch size")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, g in zip(axes, groups):
        ids = res["experiments"][g]
        names = [label(runs[i]["cfg"]).split(" adam")[0] if g != "batch size" else f"batch {runs[i]['cfg']['batch']}"
                 for i in ids]
        ax.barh(names, [runs[i]["val_cm_mean"] for i in ids], color="tab:blue", label="mean")
        ax.barh(names, [runs[i]["val_cm_p95"] - runs[i]["val_cm_mean"] for i in ids],
                left=[runs[i]["val_cm_mean"] for i in ids], color="tab:blue", alpha=0.3, label="to 95th percentile")
        ax.set(xlabel="validation foot error (cm)", title=g)
        ax.grid(alpha=0.3, axis="x")
    axes[0].legend(loc="lower right", fontsize=8)
    save(fig, "experiments.png")

    # 4. predicted vs target angles
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for j, ax in enumerate(axes):
        t, p = np.degrees(target[:, j::3]).ravel(), np.degrees(body[:, j::3]).ravel()
        ax.scatter(t, p, s=1, alpha=0.15)
        lim = [t.min(), t.max()]
        ax.plot(lim, lim, "k--", lw=1)
        rmse = np.sqrt(np.mean((p - t) ** 2))
        ax.set(xlabel=f"true {JOINTS[j]} angle (deg)", ylabel="predicted (deg)", title=f"{JOINTS[j]}: RMSE {rmse:.1f} deg")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.suptitle("24 -> 24 network on the test set (all 8 legs)")
    save(fig, "pred_vs_target.png")

    # 5. foot error histogram
    e_body, e_leg = foot_error_cm(body, d["test"]["feet"]).ravel(), foot_error_cm(leg, d["test"]["feet"]).ravel()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, np.percentile(e_body, 99.5), 60)
    for e, name, colour in ((e_body, "24 -> 24 network", "tab:blue"), (e_leg, "per-leg 3 -> 3 network", "tab:orange")):
        ax.hist(e, bins=bins, alpha=0.6, color=colour,
                label=f"{name}: mean {e.mean():.2f}, 95% {np.percentile(e, 95):.2f} cm")
        ax.axvline(np.percentile(e, 95), color=colour, ls="--", lw=1)
    ax.set(xlabel="foot error (cm)", ylabel="feet", title="Where the predicted angles put each foot (test set)")
    ax.legend()
    save(fig, "foot_error.png")

    # 6. the ambiguity: hip error grows as the foot nears the hip axis
    axis_cm = 100 * axis_distance(d["test"]["feet"]).ravel()
    hip_err = np.degrees(np.abs(body[:, 0::3] - target[:, 0::3])).ravel()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(axis_cm, hip_err, s=1, alpha=0.2)
    ax.set(xlabel="distance of the foot from its hip axis (cm)", ylabel="hip angle error (deg)",
           title="Hip angle is ill-defined when the foot is near the hip axis", xlim=(0, 120))
    ax.grid(alpha=0.3)
    save(fig, "hip_axis.png")

    # 7. poses: typical, 95th percentile and worst test pose (by mean foot error over the 8 feet)
    per_pose = foot_error_cm(body, d["test"]["feet"]).mean(1)
    order = np.argsort(per_pose)
    picks = [("typical", order[len(order) // 2]), ("95th percentile", order[int(0.95 * len(order))]),
             ("worst", order[-1])]
    fig = plt.figure(figsize=(15, 9))
    for k, (name, i) in enumerate(picks):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        plot_pose(ax, target[i], f"{name}: target pose")
        ax = fig.add_subplot(2, 3, k + 4, projection="3d")
        plot_pose(ax, body[i], f"predicted (mean foot error {per_pose[i]:.1f} cm)")
        feet = d["test"]["feet"][i].reshape(8, 3)
        ax.scatter(*feet.T, marker="x", color="g", s=40, label="target feet")
        ax.legend(loc="upper left", fontsize=8)
    save(fig, "poses.png")

    for p in saved:
        assert os.path.getsize(p) > 0, p
    print("saved:\n  " + "\n  ".join(saved))


if __name__ == "__main__":
    main()
