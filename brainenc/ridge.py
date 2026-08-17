
import torch

ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
VAL_FRAC = 0.15
SEED = 0


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def fit_ridge(Xtr, Ytr, alphas=ALPHAS, val_frac=VAL_FRAC, seed=SEED):
    """Primal ridge. Returns predict(X) -> tensor on CPU. Use when features are low-dimensional."""
    dev = _device()
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xn = ((Xtr - mu) / sd).to(dev).double()
    Y = Ytr.to(dev).double()

    g = torch.Generator(device=dev).manual_seed(seed)
    n = Xn.shape[0]
    nv = int(n * val_frac)
    perm = torch.randperm(n, generator=g, device=dev)
    tr, va = perm[nv:], perm[:nv]

    eye = torch.eye(Xn.shape[1], device=dev, dtype=torch.double)
    XtX, XtY = Xn[tr].t() @ Xn[tr], Xn[tr].t() @ Y[tr]
    best_a, best = None, float("inf")
    for a in alphas:
        W = torch.linalg.solve(XtX + a * eye, XtY)
        mse = ((Xn[va] @ W - Y[va]) ** 2).mean().item()
        if mse < best:
            best, best_a = mse, a

    W = torch.linalg.solve(Xn.t() @ Xn + best_a * eye, Xn.t() @ Y)
    mu_d, sd_d = mu.to(dev).double(), sd.to(dev).double()

    def predict(X):
        return (((X.to(dev).double() - mu_d) / sd_d) @ W).float().cpu()

    predict.alpha = best_a
    return predict


def fit_ridge_dual(Xtr, Ytr, alphas=ALPHAS, val_frac=VAL_FRAC, seed=SEED, chunk=2048):
    """Dual ridge, identical solution, solved in sample space. Use when d >> n (early stream)."""
    dev = _device()
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xn = ((Xtr - mu) / sd).to(dev).double()
    Y = Ytr.to(dev).double()

    n = Xn.shape[0]
    g = torch.Generator(device=dev).manual_seed(seed)
    nv = int(n * val_frac)
    perm = torch.randperm(n, generator=g, device=dev)
    tr, va = perm[nv:], perm[:nv]

    Ktr = Xn[tr] @ Xn[tr].t()
    Kva = Xn[va] @ Xn[tr].t()
    eye_tr = torch.eye(len(tr), device=dev, dtype=torch.double)
    best_a, best = None, float("inf")
    for a in alphas:
        A = torch.linalg.solve(Ktr + a * eye_tr, Y[tr])
        mse = ((Kva @ A - Y[va]) ** 2).mean().item()
        if mse < best:
            best, best_a = mse, a

    K = Xn @ Xn.t()
    A = torch.linalg.solve(K + best_a * torch.eye(n, device=dev, dtype=torch.double), Y)
    mu_d, sd_d = mu.to(dev).double(), sd.to(dev).double()

    def predict(X):
        out = []
        for i in range(0, len(X), chunk):
            xb = (X[i:i + chunk].to(dev).double() - mu_d) / sd_d
            out.append((xb @ Xn.t() @ A).float().cpu())
        return torch.cat(out)

    predict.alpha = best_a
    return predict
