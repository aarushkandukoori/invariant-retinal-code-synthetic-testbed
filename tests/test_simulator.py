'''Unit tests for the SYNTHETIC simulator and covariance estimators.'''
import numpy as np

from src.covariance import estimate_covariances
from src.metrics import largest_principal_angle
from src.simulator import SimConfig, planted_covariances, simulate


def test_shapes_and_synthetic_flag():
    cfg = SimConfig(n_cells=20, n_scenes=4, n_conditions=3, n_trials=5)
    pop = simulate(cfg, seed=0)
    assert pop.responses.shape == (4 * 3 * 5, 20)
    assert pop.synthetic and 'SYNTHETIC' in pop.label


def test_zero_overlap_gives_orthogonal_subspaces():
    pop = simulate(SimConfig(overlap=0.0), seed=1)
    assert np.allclose(pop.A.T @ pop.B, 0.0, atol=1e-10)
    assert np.isclose(largest_principal_angle(pop.A, pop.B), np.pi / 2)


def test_estimators_match_planted_when_noise_is_tiny():
    cfg = SimConfig(noise_scale=1e-6, n_trials=4)
    pop = simulate(cfg, seed=2)
    cov = estimate_covariances(pop.responses, pop.scene, pop.cond)
    S_B, S_N = planted_covariances(pop)
    assert np.allclose(cov.S_B, S_B, atol=1e-6)
    assert np.allclose(cov.S_N, S_N, atol=1e-6)
