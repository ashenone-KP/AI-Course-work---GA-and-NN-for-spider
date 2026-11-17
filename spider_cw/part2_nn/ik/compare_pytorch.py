"""Compare the from-scratch numpy IK network with the same network in PyTorch (marking category 8).

    cd spider_cw
    python -m part2_nn.ik.compare_pytorch      # after part2_nn.ik.train; needs torch

To make it a fair test the PyTorch model is identical to the final numpy model's setup: same layer
sizes and activations, the SAME starting weights (copied from a fresh numpy model), float64, Adam
with the same learning rate and betas, the same mini-batches in the same order (same shuffle seed),
and both compute the full train and validation loss after every epoch.
1. Gradients on one batch: our hand-written backprop vs torch autograd.
2. Both trained for EPOCHS epochs (no early stopping): loss curves, test foot error, time.
Both run on ONE CPU thread, so the times compare the code, not how many cores each grabs.
Outputs: outputs/nn_ik/pytorch_comparison.png and pytorch_comparison.txt.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"                                   # must be set before numpy/torch load

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from part2_nn import nn
from part2_nn.ik.ik_data import build, foot_error_cm, unscale
from part2_nn.ik.train import OUT, label, make

EPOCHS = 100
torch.set_num_threads(1)
ACT ={"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "sigmoid": torch.nn.Sigmoid, "linear": torch.nn.Identity}


def torch_copy(model):
    """torch.nn.Sequential with the same layers, activations and weights as a numpy MLP."""
    layers = []
    for i, act in enumerate(model._acts()):
        W, b = model.params[2 * i], model.params[2 * i + 1]
        lin = torch.nn.Linear(*W.shape).double()
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(W.T.copy()))      # torch stores (out, in)
            lin.bias.copy_(torch.from_numpy(b.copy()))
        layers += [lin, ACT[act]()]
    return torch.nn.Sequential(*layers)


def train_torch(net, X, Y, Xv, Yv, lr, epochs, batch, seed=0):
    """Same loop as nn.train without early stopping: shuffle with the same numpy generator."""
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    X, Y, Xv, Yv = (torch.from_numpy(a) for a in (X, Y, Xv, Yv))
    rng, hist = np.random.default_rng(seed), {"train": [], "val": []}
    for _ in range(epochs):
        order = torch.from_numpy(rng.permutation(len(X)))
        for start in range(0, len(X), batch):
            idx = order[start:start + batch]
            opt.zero_grad()
            loss = torch.mean((net(X[idx]) - Y[idx]) ** 2)
            loss.backward()
            opt.step()
        with torch.no_grad():
            hist["train"].append(float(torch.mean((net(X) - Y) ** 2)))
            hist["val"].append(float(torch.mean((net(Xv) - Yv) ** 2)))
    return hist


def main():
    with open(os.path.join(OUT, "results.json")) as fh:
        res = json.load(fh)
    cfg = dict(res["runs"][res["best_run"]]["cfg"], epochs=EPOCHS, patience=EPOCHS)
    d = build()
    tr, va, te = d["train"], d["val"], d["test"]

    # 1. gradients on one batch, from identical weights
    ours, _ = make(cfg, 24, 24)
    net = torch_copy(ours)
    xb, yb = tr["X"][:64], tr["Y"][:64]
    _, dpred = nn.mse(ours.forward(xb), yb)
    g_ours = [g.copy() for g in ours.backward(dpred)]
    loss = torch.mean((net(torch.from_numpy(xb)) - torch.from_numpy(yb)) ** 2)
    loss.backward()
    g_torch = [p.grad.numpy().T if p.grad.ndim == 2 else p.grad.numpy() for p in net.parameters()]
    grad_diff = max(float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12)) for a, b in zip(g_ours, g_torch))

    # 2. training, both from the same starting weights
    ours, opt = make(cfg, 24, 24)
    net = torch_copy(ours)
    t0 = time.time()
    h_ours = nn.train(ours, tr["X"], tr["Y"], va["X"], va["Y"], opt, epochs=EPOCHS,
                      batch_size=cfg["batch"], patience=EPOCHS, seed=0)
    t_ours = time.time() - t0
    t0 = time.time()
    h_torch = train_torch(net, tr["X"], tr["Y"], va["X"], va["Y"], cfg["lr"], EPOCHS, cfg["batch"])
    t_torch = time.time() - t0

    with torch.no_grad():
        pred_torch = net(torch.from_numpy(te["X"])).numpy()
    e_ours = foot_error_cm(unscale(ours.forward(te["X"])), te["feet"])
    e_torch = foot_error_cm(unscale(pred_torch), te["feet"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(h_ours["train"], label="numpy (ours) train")
    ax.semilogy(h_ours["val"], label="numpy (ours) validation")
    ax.semilogy(h_torch["train"], "--", label="PyTorch train")
    ax.semilogy(h_torch["val"], "--", label="PyTorch validation")
    ax.set(xlabel="epoch", ylabel="MSE loss", title=f"numpy vs PyTorch, {label(cfg)}")
    ax.title.set_fontsize(9)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "pytorch_comparison.png"), dpi=120, bbox_inches="tight")

    text = "\n".join([
        f"numpy (from scratch) vs PyTorch {torch.__version__}, {label(cfg)}, {EPOCHS} epochs, float64",
        f"same starting weights, same batches, both compute train+val loss every epoch, one CPU thread each",
        "",
        f"gradient on one batch from identical weights: largest relative difference {grad_diff:.1e}",
        "",
        f"{'':22s}{'numpy (ours)':>14s}{'PyTorch':>14s}",
        f"{'final train loss':22s}{h_ours['train'][-1]:14.6f}{h_torch['train'][-1]:14.6f}",
        f"{'final val loss':22s}{h_ours['val'][-1]:14.6f}{h_torch['val'][-1]:14.6f}",
        f"{'test foot error mean':22s}{e_ours.mean():12.2f}cm{e_torch.mean():12.2f}cm",
        f"{'test foot error 95%':22s}{np.percentile(e_ours, 95):12.2f}cm{np.percentile(e_torch, 95):12.2f}cm",
        f"{'training time':22s}{t_ours:13.1f}s{t_torch:13.1f}s",
        f"{'time per epoch':22s}{t_ours / EPOCHS:13.2f}s{t_torch / EPOCHS:13.2f}s",
        f"val loss after epoch 1: numpy {h_ours['val'][0]:.6f}, PyTorch {h_torch['val'][0]:.6f}",
    ])
    with open(os.path.join(OUT, "pytorch_comparison.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    assert grad_diff < 1e-9, grad_diff
    print("\npytorch comparison done")


if __name__ == "__main__":
    main()
