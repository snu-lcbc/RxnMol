#!/usr/bin/env python3
"""Reproduce the reported record splits and 90/10 replay file.

Run from reaction_model/. Input order matters. No canonicalization,
deduplication, or route grouping is performed. Outputs are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def split(source, train, validation, probability):
    rng = random.Random(42)
    counts = [0, 0]
    with source.open(encoding='utf-8') as inp, train.open('x', encoding='utf-8') as tr, validation.open('x', encoding='utf-8') as va:
        for line in inp:
            if not line.strip():
                continue
            index = int(rng.random() < probability)
            (va if index else tr).write(line)
            counts[index] += 1
    return counts


def replay(base, synthetic, output):
    with base.open() as f:
        n = sum(bool(line.strip()) for line in f)
    target = round((0.1 / 0.9) * n)
    rng = random.Random(42)
    reservoir, seen = [], 0
    with synthetic.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            if len(reservoir) < target:
                reservoir.append(line)
            else:
                j = rng.randint(0, seen)
                if j < target:
                    reservoir[j] = line
            seen += 1
    with output.open('x', encoding='utf-8') as out, base.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.write(line)
        out.writelines(reservoir)
    return {'uspto_records': n, 'synthetic_records': len(reservoir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', type=Path, default=Path('data/raw/synthetic_rxn.txt'))
    parser.add_argument('--uspto', type=Path, default=Path('data/raw/USPTO_MIT.txt'))
    parser.add_argument('--output-dir', type=Path, default=Path('data'))
    parser.add_argument('--verify-reported', action='store_true', help='Require the archived input and output hashes')
    args = parser.parse_args()
    expected = json.loads((Path(__file__).resolve().parents[1] / 'data/manifest.json').read_text())
    sources = {'synthetic_rxn': args.synthetic, 'uspto-mit': args.uspto}
    outputs = {f'{name}/{part}_rxn.txt': args.output_dir/name/f'{part}_rxn.txt' for name in sources for part in ['train', 'val']}
    outputs['mixed/uspto_mit_with_synth10_train.txt'] = args.output_dir/'mixed/uspto_mit_with_synth10_train.txt'
    report_path = args.output_dir/'preparation_report.json'
    for p in [*outputs.values(), report_path]:
        if p.exists():
            parser.error(f'Refusing to overwrite {p}')
    input_hashes = {name: digest(path) for name, path in sources.items()}
    if args.verify_reported:
        for name, value in input_hashes.items():
            if value != expected['raw'][name]['sha256']:
                parser.error(f'Input hash mismatch for {name}')
    for p in outputs.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    report = {'seed': 42, 'input_sha256': input_hashes, 'split_counts': {}}
    for name, source in sources.items():
        report['split_counts'][name] = split(source, outputs[f'{name}/train_rxn.txt'], outputs[f'{name}/val_rxn.txt'], 0.005 if name == 'synthetic_rxn' else 0.01)
    report['replay'] = replay(outputs['uspto-mit/train_rxn.txt'], outputs['synthetic_rxn/train_rxn.txt'], outputs['mixed/uspto_mit_with_synth10_train.txt'])
    report['output_sha256'] = {name: digest(path) for name, path in outputs.items()}
    mismatches = [name for name, value in report['output_sha256'].items() if value != expected['derived'][name]['sha256']]
    report['matches_reported_files'] = not mismatches
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if args.verify_reported and mismatches:
        raise SystemExit(f'Output hash mismatch: {mismatches}')


if __name__ == '__main__':
    main()
