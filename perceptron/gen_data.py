import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_dataset(X, y, path):
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y.astype(int)})
    df.to_csv(path, index=False)
    return path

def plot_dataset(X, y, title, save_path=None):
    plt.figure()
    m0, m1 = (y == 0), (y == 1)
    plt.scatter(X[m0, 0], X[m0, 1], alpha=0.7, label="class 0")
    plt.scatter(X[m1, 0], X[m1, 1], alpha=0.7, label="class 1")
    plt.xlabel("x1"); plt.ylabel("x2"); plt.title(title); plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()

def make_linear_blobs(n_per_class=200, sep=3.0, scale=0.6, seed=10):
    r = np.random.default_rng(seed)
    mean0 = np.array([-sep/2, -sep/2])
    mean1 = np.array([ sep/2,  sep/2])
    cov = np.array([[scale, 0.0], [0.0, scale]])
    X0 = r.multivariate_normal(mean0, cov, size=n_per_class)
    X1 = r.multivariate_normal(mean1, cov, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class, dtype=int),
                   np.ones(n_per_class, dtype=int)])
    return X, y

def make_overlap_blobs(n_per_class=200, sep=1.6, seed=11):
    r = np.random.default_rng(seed)
    mean0 = np.array([-sep/2, -0.5])
    mean1 = np.array([ sep/2,  0.6])
    cov0 = np.array([[1.0, 0.2], [0.2, 0.8]])
    cov1 = np.array([[0.9, -0.1], [-0.1, 0.9]])
    X0 = r.multivariate_normal(mean0, cov0, size=n_per_class)
    X1 = r.multivariate_normal(mean1, cov1, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class, dtype=int),
                   np.ones(n_per_class, dtype=int)])
    return X, y

def make_xor(n=600, spread=0.35, seed=12):
    r = np.random.default_rng(seed)
    n_quarter = n // 4
    centers = np.array([[-1, -1], [-1,  1], [ 1, -1], [ 1,  1]])
    labels  = np.array([0, 1, 1, 0])  # XOR labeling
    X_list, y_list = [], []
    for c, lab in zip(centers, labels):
        Xc = r.normal(loc=c, scale=spread, size=(n_quarter, 2))
        yc = np.full(n_quarter, lab, dtype=int)
        X_list.append(Xc); y_list.append(yc)
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y

def standardize(X):
    mu = X.mean(axis=0, keepdims=True); sigma = X.std(axis=0, keepdims=True)
    sigma[sigma==0] = 1.0
    return (X - mu) / sigma, mu, sigma

def load_csv(path):
    df = pd.read_csv(path)
    return df[["x1","x2"]].to_numpy(float), df["y"].to_numpy(int)

def get_data(args):
    if getattr(args, "csv", None):
        X, y = load_csv(args.csv)
        return X, y, f"csv:{os.path.basename(args.csv)}"
    ds = getattr(args, "dataset", "linear").lower()
    seed = getattr(args, "seed_data", None)
    if ds == "linear":   X, y = make_linear_blobs(seed=seed)
    elif ds == "overlap":X, y = make_overlap_blobs(seed=seed)
    elif ds == "xor":    X, y = make_xor(seed=seed)
    else: raise ValueError("dataset must be {linear, overlap, xor} or provide --csv")
    return X, y, ds
    
if __name__ == "__main__":
    # 1) Linearly separable
    X_lin, y_lin = make_linear_blobs(n_per_class=250, sep=3.2, scale=0.5, seed=10)
    save_dataset(X_lin, y_lin, "ds_linear_blobs.csv")
    plot_dataset(X_lin, y_lin, "Linearly Separable Blobs", "ds_linear_blobs_plot.png")

    # 2) Moderately overlapping
    X_ovl, y_ovl = make_overlap_blobs(n_per_class=250, sep=1.8, seed=11)
    save_dataset(X_ovl, y_ovl, "ds_overlap_blobs.csv")
    plot_dataset(X_ovl, y_ovl, "Moderately Overlapping Blobs", "ds_overlap_blobs_plot.png")

    # 3) XOR (negative test for linear models)
    X_xor, y_xor = make_xor(n=600, spread=0.35, seed=12)
    save_dataset(X_xor, y_xor, "ds_xor.csv")
    plot_dataset(X_xor, y_xor, "XOR (Non-linearly Separable)", "ds_xor_plot.png")

    print("Saved files:",
          "ds_linear_blobs.csv", "ds_overlap_blobs.csv", "ds_xor.csv",
          "ds_linear_blobs_plot.png", "ds_overlap_blobs_plot.png", "ds_xor_plot.png")
