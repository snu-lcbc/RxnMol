#!/usr/bin/env python3
"""Prepare verified training assets from local files or an explicitly selected release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
from urllib.parse import quote


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
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--asset-dir', type=Path, help='Directory containing the archived asset files')
    source.add_argument('--release-tag', help='Published GitHub release tag selected by the user')
    args = parser.parse_args()
    if not (args.datasets or args.weights):
        parser.error('Select --datasets, --weights, or both')
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'reference/assets.json').read_text())
    selected = [item for item in manifest if (args.datasets if item['kind'] == 'datasets' else args.weights)]
    for item in selected:
        if not (root/item['destination']).exists():
            if args.asset_dir:
                if not (args.asset_dir/item['name']).is_file():
                    parser.error(f"Local asset is missing: {args.asset_dir/item['name']}")
            elif not args.release_tag and not item.get('url'):
                parser.error('No download URL is configured. Use --asset-dir for local files or --release-tag after publication.')
    for item in selected:
        destination = root/item['destination']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != item['sha256']:
                raise SystemExit(f'Existing file hash mismatch: {destination}')
        else:
            partial = destination.with_name(destination.name + '.part')
            if partial.exists():
                raise SystemExit(f'Remove incomplete download before retrying: {partial}')
            if args.asset_dir:
                with (args.asset_dir/item['name']).open('rb') as source, partial.open('xb') as out:
                    shutil.copyfileobj(source, out)
            else:
                url = (f"https://github.com/snu-lcbc/RxnMol/releases/download/{quote(args.release_tag, safe='')}/{quote(item['name'], safe='')}"
                       if args.release_tag else item['url'])
                with urllib.request.urlopen(url) as source, partial.open('xb') as out:
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
