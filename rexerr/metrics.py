"""Evaluation metrics, bootstrap confidence intervals and prevalence analysis.

All functions take y (n, K) binary labels and p (n, K) probabilities. Classes
without positives in the evaluated set are excluded from macro averages.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def _auroc(y, s):
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _auprc(y, s):
    if y.sum() == 0:
        return np.nan
    return average_precision_score(y, s)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def as_thresholds(thr, k):
    return np.full(k, thr, dtype=float) if np.isscalar(thr) else np.asarray(thr, dtype=float)


def active_classes(y):
    return np.asarray(y).sum(axis=0) > 0


def per_class(y, p, names, thr=0.5):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    t = as_thresholds(thr, y.shape[1])
    yhat = (p >= t).astype(int)
    rows = []
    for j, name in enumerate(names):
        tp = int(((yhat[:, j] == 1) & (y[:, j] == 1)).sum())
        fp = int(((yhat[:, j] == 1) & (y[:, j] == 0)).sum())
        fn = int(((yhat[:, j] == 0) & (y[:, j] == 1)).sum())
        tn = int(((yhat[:, j] == 0) & (y[:, j] == 0)).sum())
        pr, rc, f1 = _prf(tp, fp, fn)
        rows.append({
            "class": name, "support": int(y[:, j].sum()), "precision": pr, "recall": rc, "f1": f1,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "auroc": _auroc(y[:, j], p[:, j]), "auprc": _auprc(y[:, j], p[:, j]),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": float(t[j]),
        })
    return rows


def summary(y, p, thr=0.5, mask=None):
    """Macro/micro/weighted F1, macro AUROC/AUPRC, Hamming and exact-match accuracy."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    if mask is None:
        mask = active_classes(y)
    y, p = y[:, mask], p[:, mask]
    t = as_thresholds(thr, len(mask))[mask]
    yhat = (p >= t).astype(int)
    tp = ((yhat == 1) & (y == 1)).sum(0)
    fp = ((yhat == 1) & (y == 0)).sum(0)
    fn = ((yhat == 0) & (y == 1)).sum(0)
    f1 = np.array([_prf(a, b, c)[2] for a, b, c in zip(tp, fp, fn)])
    support = y.sum(0)
    _, _, micro = _prf(tp.sum(), fp.sum(), fn.sum())
    au = np.array([_auroc(y[:, j], p[:, j]) for j in range(y.shape[1])])
    ap = np.array([_auprc(y[:, j], p[:, j]) for j in range(y.shape[1])])
    return {
        "f1_macro": float(f1.mean()),
        "f1_micro": float(micro),
        "f1_weighted": float((f1 * support).sum() / max(support.sum(), 1)),
        "auroc_macro": float(np.nanmean(au)),
        "auprc_macro": float(np.nanmean(ap)),
        "hamming_accuracy": float((yhat == y).mean()),
        "exact_match_accuracy": float((yhat == y).all(1).mean()),
        "n_classes": int(mask.sum()),
    }


def any_error_score(p):
    """Report-level probability that at least one error is present."""
    return np.asarray(p).max(axis=1)


def binary(y_any, s_any, thr=0.5):
    y_any, s_any = np.asarray(y_any).astype(int), np.asarray(s_any, dtype=float)
    yhat = (s_any >= thr).astype(int)
    tp = int(((yhat == 1) & (y_any == 1)).sum()); fp = int(((yhat == 1) & (y_any == 0)).sum())
    fn = int(((yhat == 0) & (y_any == 1)).sum()); tn = int(((yhat == 0) & (y_any == 0)).sum())
    pr, rc, f1 = _prf(tp, fp, fn)
    return {
        "precision": pr, "sensitivity": rc, "f1": f1,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "auroc": float(_auroc(y_any, s_any)), "auprc": float(_auprc(y_any, s_any)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": float(thr),
        "prevalence_in_set": float(y_any.mean()),
    }


def tune_thresholds(y_val, p_val, grid=None):
    """Per-class threshold maximising F1 on the validation set (0.5 if no positives)."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    y_val, p_val = np.asarray(y_val).astype(int), np.asarray(p_val, dtype=float)
    out = np.full(y_val.shape[1], 0.5)
    for j in range(y_val.shape[1]):
        if y_val[:, j].sum() == 0:
            continue
        best, best_t = -1, 0.5
        for t in grid:
            yhat = p_val[:, j] >= t
            tp = (yhat & (y_val[:, j] == 1)).sum(); fp = (yhat & (y_val[:, j] == 0)).sum()
            fn = (~yhat & (y_val[:, j] == 1)).sum()
            f = _prf(tp, fp, fn)[2]
            if f > best:
                best, best_t = f, t
        out[j] = best_t
    return out


def threshold_for_sensitivity(y_any, s_any, target):
    """Highest threshold whose sensitivity on (y_any, s_any) is >= target."""
    pos = np.sort(np.asarray(s_any)[np.asarray(y_any) == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[min(max(k, 0), len(pos) - 1)])


def prevalence_analysis(sens, spec, prevalences=(0.03, 0.05, 0.10, 0.20)):
    """Expected PPV/NPV and alert burden when the true error rate is `prevalence`."""
    rows = []
    for pi in prevalences:
        tp, fp = sens * pi, (1 - spec) * (1 - pi)
        fn, tn = (1 - sens) * pi, spec * (1 - pi)
        rows.append({
            "prevalence": pi,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "alerts_per_1000": 1000 * (tp + fp),
            "true_alerts_per_1000": 1000 * tp,
            "missed_per_1000": 1000 * fn,
        })
    return rows


def _group_index(groups):
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def bootstrap(stat_fn, arrays, groups, n_boot=1000, seed=0):
    """Cluster bootstrap (resampling `groups`, e.g. patients).

    stat_fn(*arrays_subset) -> dict of floats. Returns {key: (lo, hi)} 95% CIs.
    """
    rng = np.random.default_rng(seed)
    idx = _group_index(groups)
    samples = {}
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        sel = np.concatenate([idx[i] for i in pick])
        for k, v in stat_fn(*[a[sel] for a in arrays]).items():
            samples.setdefault(k, []).append(v)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in samples.items()}


def paired_bootstrap(stat_fn, y, p_a, p_b, groups, n_boot=1000, seed=0):
    """Difference stat(a) - stat(b) with 95% CI and two-sided bootstrap p-value."""
    rng = np.random.default_rng(seed)
    idx = _group_index(groups)
    obs = stat_fn(y, p_a) - stat_fn(y, p_b)
    diffs = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        diffs.append(stat_fn(y[sel], p_a[sel]) - stat_fn(y[sel], p_b[sel]))
    diffs = np.asarray(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(obs), "ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
            "p_value": float(min(p, 1.0))}
