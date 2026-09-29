#!/usr/bin/env python3
"""Test configuration loading, a short CPU training run, and checkpoint reload.

This is an integration test on a small model and fixture, not retraining or
reproduction of the reported accuracy. Run from reaction_model/.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled')
    results = []
    def run(command):
        r = subprocess.run([sys.executable, *command], cwd=root, env=env, capture_output=True, text=True)
        results.append({'command': command, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr})
        if r.returncode:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({'passed': False, 'checks': results}, indent=2) + '\n')
            raise SystemExit(r.stderr)
    run(['-m', 'pip', 'check'])
    for name in ['pretrain', 'finetune']:
        run(['train.py', '--config-path', 'configs', '--config-name', name, '--cfg', 'job'])
    with tempfile.TemporaryDirectory(prefix='rxnmol-training-test-') as tmp:
        d = Path(tmp)
        for name in ['train', 'val']:
            (d/f'{name}.txt').write_text('CCO>>CCO\nCCN>>CCN\nCC(=O)O>>CC(=O)O\nC>>C\n')
        run(['train.py', '--config-path', 'configs', '--config-name', 'pretrain',
             f'data.train_path={d}/train.txt', f'data.val_path={d}/val.txt', 'data.use_hf_cache=false',
             'model.d_model=32', 'model.n_heads=4', 'model.n_kv_heads=2', 'model.n_layers=1', 'model.d_ff=64',
             'training.batch_size=2', 'training.max_steps=2', 'training.warmup_steps=0',
             'training.val_check_interval=1', 'training.precision=32-true', 'training.early_stopping_enabled=false',
             'distributed.devices=1', 'distributed.strategy=auto', 'hardware.accelerator=cpu', 'hardware.num_workers=0',
             f'logging.run_dir={d}/run', 'logging.log_every_n_steps=1'])
        run(['evaluate.py', '--checkpoint', f'{d}/run/checkpoints/last.ckpt', '--data', f'{d}/val.txt',
             '--device', 'cpu', '--beam-size', '1', '--max-len', '16', '--max-samples', '2', '--output', f'{d}/evaluation.json'])
        evaluated = json.loads((d/'evaluation.json').read_text())
        assert evaluated['metrics']['total_samples'] == 2
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    assert 'rdkit' in {k.lower() for k in packages}
    assert 'rdkit-pypi' not in {k.lower() for k in packages}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'passed': True, 'scope': 'CPU installation and small-model integration test; reported training and accuracies were not rerun.', 'python': sys.version, 'packages': packages, 'checks': results}, indent=2) + '\n')
    print('PASS: installation, both configurations, two optimizer steps, checkpoint reload and two-sample evaluation')


if __name__ == '__main__':
    main()
