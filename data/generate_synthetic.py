'''Write SYNTHETIC population data into data/synthetic/.

Reference implementation of a PROPOSAL. The output is clearly-labeled mock
data from src/simulator.py. No real recordings are used or implied. No network
access.

Usage: python data/generate_synthetic.py [--out DIR] [--seed 0] [--overlap 0.2]
'''
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simulator import SYNTHETIC_LABEL, SimConfig, save_population, simulate  # noqa: E402

FILENAME = 'synthetic_population.npz'


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=SYNTHETIC_LABEL)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'synthetic'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--overlap', type=float, default=0.2)
    ap.add_argument('--nonlinearity', default='none', choices=['none', 'softplus'])
    args = ap.parse_args(argv)

    cfg = SimConfig(overlap=args.overlap, nonlinearity=args.nonlinearity)
    pop = simulate(cfg, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / FILENAME
    save_population(pop, path)
    (out / 'SYNTHETIC_DATA_NOTICE.txt').write_text(
        SYNTHETIC_LABEL + '\nGenerated from Gaussians with planted structure. '
        'No real neural, animal or lab data.\n'
    )
    meta = {'label': SYNTHETIC_LABEL, 'seed': args.seed, 'config': asdict(cfg)}
    (out / 'synthetic_meta.json').write_text(json.dumps(meta, indent=2))
    print(f'wrote SYNTHETIC data to {path}')
    return path


if __name__ == '__main__':
    main()
