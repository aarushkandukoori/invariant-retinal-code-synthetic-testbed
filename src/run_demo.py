'''End-to-end demo on SYNTHETIC data (reference implementation of a PROPOSAL).

Run from the repo root:  python -m src.run_demo
Prints a toy comparison table. These numbers come from a small synthetic
simulation, are NOT benchmark results, and say nothing about real retinas.
'''
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from baselines.classical import (  # noqa: E402
    fisher_readout, nuisance_regression_fisher, pca_readout)
from src.covariance import estimate_covariances  # noqa: E402
from src.metrics import (  # noqa: E402
    largest_principal_angle, nuisance_leakage, scene_accuracy)
from src.readout import fit_readout  # noqa: E402
from src.simulator import SYNTHETIC_LABEL, load_synthetic_population  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=SYNTHETIC_LABEL)
    ap.add_argument('--data', default=str(ROOT / 'data' / 'synthetic' / 'synthetic_population.npz'))
    ap.add_argument('--n-fit-conditions', type=int, default=4)
    ap.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 1.0, 10.0, 100.0])
    args = ap.parse_args(argv)

    path = Path(args.data)
    if not path.exists():
        from data.generate_synthetic import main as generate
        generate(['--out', str(path.parent)])
    pop = load_synthetic_population(path)
    print('*** ' + SYNTHETIC_LABEL + ' ***')
    print('*** Toy numbers from a simulation, NOT benchmark results. ***')

    fit = pop.cond < args.n_fit_conditions
    X_tr, y_tr, c_tr = pop.responses[fit], pop.scene[fit], pop.cond[fit]
    X_te, y_te, c_te = pop.responses[~fit], pop.scene[~fit], pop.cond[~fit]
    K = int(pop.scene.max()) + 1
    n = min(pop.A.shape[1], K - 1)
    cov = estimate_covariances(X_tr, y_tr, c_tr)

    readouts = {}
    for lam in args.lambdas:
        readouts[f'rayleigh lam={lam:g}'] = fit_readout(
            cov.S_B, cov.S_N, cov.Sigma, lam, n).W
    readouts['PCA'] = pca_readout(X_tr, n)
    readouts['Fisher (pooled)'] = fisher_readout(X_tr, y_tr, n)
    readouts['Nuis-regress+Fisher'] = nuisance_regression_fisher(
        X_tr, y_tr, pop.nuis_feats[c_tr], n)
    readouts['Contrastive'] = None  # stub, not implemented

    print(f'train conds={args.n_fit_conditions}, held-out conds={len(np.unique(c_te))}, '
          f'chance acc={1 / K:.3f}')
    print(f'{"method":<24}{"heldout_acc":>12}{"leakage":>10}{"max_angle":>11}')
    for name, W in readouts.items():
        if W is None:
            print(f'{name:<24}{"not implemented (stub)":>33}')
            continue
        acc = scene_accuracy(W, X_tr, y_tr, X_te, y_te)
        leak = nuisance_leakage(W, X_te, y_te, c_te)
        ang = largest_principal_angle(W, pop.A) if pop.linear else float('nan')
        print(f'{name:<24}{acc:12.3f}{leak:10.3f}{ang:11.3f}')


if __name__ == '__main__':
    main()
