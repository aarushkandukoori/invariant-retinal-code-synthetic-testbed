'''Metrics for the PROPOSAL protocol, computed on SYNTHETIC data only.

No benchmark numbers are stored anywhere; values come from whatever synthetic
data you generate.
'''
from __future__ import annotations

import numpy as np

from src.covariance import cell_means


def scene_accuracy(W, X_train, y_train, X_test, y_test) -> float:
    '''Nearest-class-mean accuracy in the projected space.

    Class means and per-dimension within-class std come from the training rows;
    test rows may come from held-out nuisance conditions.
    '''
    Ztr, Zte = X_train @ W, X_test @ W
    classes = np.unique(y_train)
    means = np.stack([Ztr[y_train == c].mean(axis=0) for c in classes])
    resid = Ztr - means[np.searchsorted(classes, y_train)]
    sd = resid.std(axis=0) + 1e-12
    d = (((Zte / sd)[:, None, :] - (means / sd)[None]) ** 2).sum(axis=2)
    pred = classes[np.argmin(d, axis=1)]
    return float(np.mean(pred == y_test))


def nuisance_leakage(W, X, scene, cond) -> float:
    '''Fraction of projected variance due to nuisance condition (within scene).'''
    Z = X @ W
    total = Z.var(axis=0).sum()
    means, _, _ = cell_means(Z, scene, cond)
    D = means - means.mean(axis=1, keepdims=True)
    return float((D**2).mean(axis=(0, 1)).sum() / (total + 1e-12))


def largest_principal_angle(W, A) -> float:
    '''Largest principal angle (radians) between col(W) and col(A).'''
    Qw, _ = np.linalg.qr(W)
    Qa, _ = np.linalg.qr(A)
    s = np.linalg.svd(Qw.T @ Qa, compute_uv=False)
    return float(np.arccos(np.clip(s.min(), -1.0, 1.0)))
