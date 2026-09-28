#!/usr/bin/env python3
"""Download and verify the two training corpora and required checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', action='store_true')
    parser.add_argument('--weights', action='store_true')
    args = parser.parse_args()
    if not (args.datasets or args.weights):
        parser.error('Select --datasets, --weights, or both')
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'reference/assets.json').read_text())
    for item in manifest:
        if not (args.datasets if item['kind'] == 'datasets' else args.weights):
            continue
        destination = root/item['destination']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != item['sha256']:
                raise SystemExit(f'Existing file hash mismatch: {destination}')
        else:
            partial = destination.with_name(destination.name + '.part')
            if partial.exists():
                raise SystemExit(f'Remove incomplete download before retrying: {partial}')
            with urllib.request.urlopen(item['url']) as source, partial.open('xb') as out:
                shutil.copyfileobj(source, out)
            if digest(partial) != item['sha256']:
                raise SystemExit(f'Download hash mismatch: {partial}')
            partial.rename(destination)
        print(f'Verified {destination.name}')
        if item['kind'] == 'datasets':
            expected = json.loads((root/'data/manifest.json').read_text())['raw']
            with tarfile.open(destination, 'r:gz') as archive:
                for name, domain in [('synthetic_rxn.txt', 'synthetic_rxn'), ('USPTO_MIT.txt', 'uspto-mit')]:
                    member = archive.getmember('data/raw/' + name)
                    if not member.isfile():
                        raise SystemExit('Unexpected archive member type')
                    target = root/'data/raw'/name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        with archive.extractfile(member) as source, target.open('xb') as out:
                            shutil.copyfileobj(source, out)
                    if digest(target) != expected[domain]['sha256']:
                        raise SystemExit(f'Extracted data hash mismatch: {target}')
    if args.weights:
        target = root/'weights/tokenizer.json'
        if not target.exists():
            shutil.copyfile(root/'tokenizer/tokenizer.json', target)


if __name__ == '__main__':
    main()
