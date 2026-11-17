'''Generalized-eigenproblem read-out (reference implementation of a PROPOSAL).

Maximizes J(w) = w^T S_B w / w^T (Sigma + lam S_N) w. Stationary points solve
S_B w = mu (Sigma + lam S_N) w; we solve it by Cholesky whitening with numpy
only. This is the standard Fisher-style construction; no new result is claimed.
Used here on SYNTHETIC covariances only.
'''
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReadOut:
    W: np.ndarray  # (N, n_components), columns satisfy w^T C w = 1
    eigvalues: np.ndarray  # generalized eigenvalues, descending
    lam: float


def generalized_eig(S_B: np.ndarray, C: np.ndarray, n_components: int):
    '''Top generalized eigenpairs of (S_B, C), C symmetric positive definite.'''
    C = (C + C.T) / 2.0
    L = np.linalg.cholesky(C)
    X = np.linalg.solve(L, S_B)  # L^-1 S_B
    M = np.linalg.solve(L, X.T)  # L^-1 S_B L^-T
    M = (M + M.T) / 2.0
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1][:n_components]
    W = np.linalg.solve(L.T, vecs[:, order])
    return vals[order], W


def rayleigh_quotient(w: np.ndarray, S_B: np.ndarray, C: np.ndarray) -> float:
    '''J(w) = w^T S_B w / w^T C w.'''
    return float(w @ S_B @ w) / float(w @ C @ w)


def fit_readout(S_B, S_N, Sigma, lam: float, n_components: int) -> ReadOut:
    '''Solve the lambda-penalized problem with C = Sigma + lam * S_N.'''
    if lam < 0:
        raise ValueError('lam must be >= 0')
    vals, W = generalized_eig(S_B, Sigma + lam * S_N, n_components)
    return ReadOut(W=W, eigvalues=vals, lam=float(lam))
