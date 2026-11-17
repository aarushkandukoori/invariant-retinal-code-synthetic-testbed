'''Classical baselines: PCA, pooled Fisher LDA, nuisance-regression + Fisher.

Reference implementations for a PROPOSAL; exercised on SYNTHETIC data only.
Each returns a read-out matrix W (N, n_components) applied to raw responses.
'''
from __future__ import annotations

import numpy as np

from src.covariance import shrink
from src.readout import generalized_eig


def pca_readout(X, n_components: int) -> np.ndarray:
    '''Top-variance directions (ignores scene labels and nuisance).'''
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:n_components].T


def fisher_readout(X, scene, n_components: int, shrinkage: float = 0.1):
    '''Fisher LDA on pooled data: nuisance variance sits inside the within-scene covariance.'''
    scene = np.asarray(scene)
    classes = np.unique(scene)
    means = np.stack([X[scene == c].mean(axis=0) for c in classes])
    Mc = means - means.mean(axis=0)
    S_B = Mc.T @ Mc / len(classes)
    R = X - means[np.searchsorted(classes, scene)]
    S_W = shrink(R.T @ R / max(len(X) - len(classes), 1), shrinkage)
    return generalized_eig(S_B, S_W, min(n_components, len(classes) - 1))[1]


def nuisance_regression_fisher(X, scene, nuis, n_components: int,
                               shrinkage: float = 0.1):
    '''Regress KNOWN nuisance features (rows, q) out of X, then run Fisher.

    W is learned on residuals but applied to raw responses. This oracle uses
    the true nuisance features, which real experiments may not have.
    '''
    design = np.hstack([np.ones((len(X), 1)), nuis])
    beta, *_ = np.linalg.lstsq(design, X, rcond=None)
    X_res = X - nuis @ beta[1:]
    return fisher_readout(X_res, scene, n_components, shrinkage)
