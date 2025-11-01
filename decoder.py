import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def window_counts_from_voltages(Vt: np.ndarray, dt_ms: float, window_ms: float):
    """
    Vt: (steps, N) voltages (or rates)
    For spiking voltages, count upward zero-crossings per window.
    For rate mode, sum-of-rates per window as feature.
    """
    steps, N = Vt.shape
    W = int(window_ms / dt_ms)
    nwin = steps // W
    X = np.zeros((nwin, N))
    for i in range(nwin):
        seg = Vt[i*W:(i+1)*W]
        # heuristic: if values can be negative, treat as voltages; else rates
        if np.any(seg < 0):
            cross = ((seg[:-1] < 0) & (seg[1:] >= 0)).sum(axis=0)
            X[i] = cross
        else:
            X[i] = seg.sum(axis=0) * (dt_ms/1000.0)
    return X

def indexes_for_on_off(T_ms: float, dt_ms: float, window_ms: float, on_ms: float, off_ms: float):
    W = int(window_ms / dt_ms)
    on_idx = list(range(int(on_ms/dt_ms/W), int(off_ms/dt_ms/W)))
    return on_idx

def decode_stim_on_off(X: np.ndarray, on_idx: list[int], train_frac: float = 0.6):
    y = np.zeros(X.shape[0], dtype=int)
    y[on_idx] = 1
    n = X.shape[0]; ntr = max(1, int(train_frac*n))
    Xtr, Xte = X[:ntr], X[ntr:]
    ytr, yte = y[:ntr], y[ntr:]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        # not enough variation; return NaNs
        return float("nan"), float("nan")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]
    acc = accuracy_score(yte, (p>0.5).astype(int))
    try:
        auc = roc_auc_score(yte, p)
    except Exception:
        auc = float("nan")
    return acc, auc
