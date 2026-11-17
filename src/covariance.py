'''Covariance estimators for S_B, S_N and Sigma (reference implementation of a PROPOSAL).

Operates on SYNTHETIC data in this repository. Assumes a balanced grid: every
(scene, nuisance-condition) cell has at least one trial. Shrinkage is a plain
fixed-weight pull toward a scaled identity (NOT the Ledoit-Wolf formula).
S_B and S_N are computed from cell means, so they carry a small positive bias
from trial noise; this skeleton does not correct for it.
'''
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Covariances:
    S_B: np.ndarray
    S_N: np.ndarray
    Sigma: np.ndarray


def shrink(S: np.ndarray, alpha: float) -> np.ndarray:
    '''(1 - alpha) S + alpha * mean(diag S) * I.'''
    n = S.shape[0]
    return (1.0 - alpha) * S + alpha * (np.trace(S) / n) * np.eye(n)


def cell_means(X, scene, cond):
    '''Return (means (K,J,N), cond index per row, counts (K,J)).'''
    scene = np.asarray(scene)
    K = int(scene.max()) + 1
    conds, cidx = np.unique(cond, return_inverse=True)
    J, N = len(conds), X.shape[1]
    sums = np.zeros((K, J, N))
    counts = np.zeros((K, J))
    np.add.at(sums, (scene, cidx), X)
    np.add.at(counts, (scene, cidx), 1.0)
    if (counts == 0).any():
        raise ValueError('every (scene, condition) cell needs at least one trial')
    return sums / counts[..., None], cidx, counts


def estimate_covariances(X, scene, cond, shrinkage: float = 0.1) -> Covariances:
    '''Estimate S_B, S_N, Sigma from trials X (M, N).'''
    M, N = X.shape
    means, cidx, _ = cell_means(X, scene, cond)
    K, J, _ = means.shape
    fbar = means.mean(axis=1)  # (K, N) scene means over nuisance
    Fc = fbar - fbar.mean(axis=0)
    S_B = Fc.T @ Fc / K
    D = means - fbar[:, None, :]
    S_N = np.einsum('kjn,kjm->nm', D, D) / (K * J)
    R = X - means[np.asarray(scene), cidx]
    dof = max(M - K * J, 1)
    Sigma = shrink(R.T @ R / dof, shrinkage)
    return Covariances(S_B, S_N, Sigma)
