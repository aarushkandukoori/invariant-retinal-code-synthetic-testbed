'''Contrastive (InfoNCE-style) encoder baseline: STUB for a PROPOSAL.

The encoder is NOT implemented (it would need an autodiff library, which is
intentionally not a dependency). Only a numpy InfoNCE loss is provided so the
objective is concrete. No results exist for this baseline.
'''
from __future__ import annotations

import numpy as np


def info_nce_loss(z_a: np.ndarray, z_b: np.ndarray, temperature: float = 0.1) -> float:
    '''Mean InfoNCE loss; row i of z_a and z_b is a positive pair.'''
    a = z_a / (np.linalg.norm(z_a, axis=1, keepdims=True) + 1e-12)
    b = z_b / (np.linalg.norm(z_b, axis=1, keepdims=True) + 1e-12)
    logits = a @ b.T / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    log_prob = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return float(-np.mean(np.diag(log_prob)))


class ContrastiveEncoderBaseline:
    '''Placeholder: positives = same scene under different nuisance settings.'''

    def __init__(self, embed_dim: int = 8, temperature: float = 0.1):
        self.embed_dim = embed_dim
        self.temperature = temperature

    def fit(self, X, scene, cond):
        raise NotImplementedError(
            'Contrastive encoder is a stub in this skeleton; '
            'implement with an autodiff library to run the planned comparison.'
        )

    def transform(self, X):
        raise NotImplementedError('fit() is not implemented')
