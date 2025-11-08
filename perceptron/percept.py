from single_layer import Perceptron
from gen_data import *
import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse

def plot_boundary_and_points(Xs, y, model, title, out_path):
    # mesh over standardized feature space
    x1_min, x1_max = Xs[:,0].min() - 0.5, Xs[:,0].max() + 0.5
    x2_min, x2_max = Xs[:,1].min() - 0.5, Xs[:,1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                           np.linspace(x2_min, x2_max, 300))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = model.predict_proba(grid).reshape(xx1.shape)

    plt.figure()
    m0, m1 = (y == 0), (y == 1)
    plt.scatter(Xs[m0,0], Xs[m0,1], alpha=0.6, label="class 0")
    plt.scatter(Xs[m1,0], Xs[m1,1], alpha=0.6, label="class 1")
    cs = plt.contour(xx1, xx2, probs, levels=[0.5])
    plt.clabel(cs, inline=True, fmt="p=0.5")
    plt.title(title)
    plt.xlabel("x1 (standardized)"); plt.ylabel("x2 (standardized)"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def plot_loss(loss_history, title, out_path):
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.title(title)
    plt.xlabel("Epoch"); plt.ylabel("Binary Cross-Entropy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Sigmoid Perceptron demo (logistic regression).")
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--dataset", type=str, default="linear",
                   help="Which synthetic dataset to use: {linear, overlap, xor}")
    g.add_argument("--csv", type=str, help="Path to CSV with columns: x1, x2, y")
    parser.add_argument("--out", type=str, default="./out_demo", help="Output directory")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0, help="Model/training seed")
    parser.add_argument("--seed-data", type=int, default=None, help="Data generation seed")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    X, y, name = get_data(args)

    Xs, mu, sigma = standardize(X)

    model = Perceptron(n=Xs.shape[1], lr=args.lr, seed=args.seed)
    loss_hist = model.train(Xs, y, epochs=args.epochs, l2=args.l2, shuffle=True)
    y_hat = model.predict_class(Xs)
    acc = (y_hat == y).mean()

    boundary_path = os.path.join(args.out, f"{name}_boundary.png")
    loss_path = os.path.join(args.out, f"{name}_loss.png")
    plot_boundary_and_points(Xs, y, model,
        f"{name} — Decision Boundary (acc={acc:.3f})", boundary_path)
    plot_loss(loss_hist, f"{name} — Training Loss (BCE)", loss_path)

    with open(os.path.join(args.out, "metrics.txt"), "w") as f:
        f.write(f"dataset: {name}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"l2: {args.l2}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"accuracy: {acc:.6f}\n")
        f.write(f"w: {model.w.tolist()}\n")
        f.write(f"b: {float(model.b)}\n")
        f.write(f"boundary_plot: {boundary_path}\n")
        f.write(f"loss_plot: {loss_path}\n")

    print(f"[OK] dataset={name}  acc={acc:.3f}")
    print(f"Saved plots to: {boundary_path}  and  {loss_path}")
    print(f"Metrics + params: {os.path.join(args.out, 'metrics.txt')}")

if __name__ == "__main__":
    main()