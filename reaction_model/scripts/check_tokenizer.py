#!/usr/bin/env python3
"""Check saved-vocabulary coverage without inferring vocabulary provenance."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tokenizer import tokenize_smiles

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('inputs', nargs='+', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error('Refusing to overwrite the report')
vocab = json.loads((ROOT/'tokenizer/tokenizer.json').read_text())['model']['vocab']
report = {'vocabulary_size': len(vocab), 'datasets': {}}
for path in args.inputs:
    unknown, affected, count = Counter(), 0, 0
    with path.open() as f:
        for line in f:
            line = line.strip().split('|')[0]
            if not line:
                continue
            count += 1
            missing = [token for side in line.split('>>') for token in tokenize_smiles(side) if token not in vocab]
            unknown.update(missing)
            affected += bool(missing)
    report['datasets'][str(path)] = {'records': count, 'records_with_unknown_tokens': affected, 'unknown_tokens': dict(unknown)}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
