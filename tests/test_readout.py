'''Unit tests on SYNTHETIC matrices: analytic checks of the eigen-solution.'''
import numpy as np

from baselines.classical import pca_readout
from baselines.contrastive import info_nce_loss
from src.readout import fit_readout, generalized_eig, rayleigh_quotient


def _spd(rng, n):
    M = rng.standard_normal((n, n))
    return M @ M.T + np.eye(n)


def test_matches_generic_solver_and_bounds_rayleigh():
    rng = np.random.default_rng(0)
    C = _spd(rng, 6)
    R = rng.standard_normal((6, 3))
    S_B = R @ R.T
    vals, W = generalized_eig(S_B, C, 6)
    ref = np.sort(np.linalg.eigvals(np.linalg.solve(C, S_B)).real)[::-1]
    assert np.allclose(vals, ref, atol=1e-8)
    assert np.isclose(rayleigh_quotient(W[:, 0], S_B, C), vals[0])
    for _ in range(50):
        w = rng.standard_normal(6)
        assert rayleigh_quotient(w, S_B, C) <= vals[0] + 1e-9


def test_rank_one_matches_closed_form():
    rng = np.random.default_rng(1)
    C = _spd(rng, 5)
    d = rng.standard_normal(5)
    vals, W = generalized_eig(np.outer(d, d), C, 1)
    w_star = np.linalg.solve(C, d)
    cos = abs(W[:, 0] @ w_star) / (np.linalg.norm(W[:, 0]) * np.linalg.norm(w_star))
    assert np.isclose(cos, 1.0)
    assert np.isclose(vals[0], d @ w_star)


def test_no_nuisance_means_lambda_is_irrelevant():
    rng = np.random.default_rng(2)
    C = _spd(rng, 5)
    R = rng.standard_normal((5, 2))
    S_B = R @ R.T
    a = fit_readout(S_B, np.zeros((5, 5)), C, 0.0, 2)
    b = fit_readout(S_B, np.zeros((5, 5)), C, 50.0, 2)
    assert np.allclose(a.eigvalues, b.eigvalues)


def test_pca_columns_orthonormal():
    X = np.random.default_rng(3).standard_normal((50, 7))
    W = pca_readout(X, 3)
    assert np.allclose(W.T @ W, np.eye(3), atol=1e-10)


def test_info_nce_prefers_aligned_pairs():
    z = np.random.default_rng(4).standard_normal((16, 8))
    assert info_nce_loss(z, z) <= info_nce_loss(z, np.roll(z, 1, axis=0))
