#!/usr/bin/env python3
"""Prepare verified training assets from local files or an explicitly selected release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
from urllib.error import URLError
from urllib.parse import quote


DEFAULT_RELEASE_TAG = 'v1.1.0'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='With no selection flags, prepare both datasets and checkpoints.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--datasets', action='store_true', help='Prepare datasets only, unless --weights is also selected')
    parser.add_argument('--weights', action='store_true', help='Prepare checkpoints only, unless --datasets is also selected')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--local-dir', type=Path, help='Use dataset archives and checkpoints from this local directory instead of downloading')
    source.add_argument('--release-tag', default=DEFAULT_RELEASE_TAG, help='GitHub release tag to download from')
    args = parser.parse_args()
    if not (args.datasets or args.weights):
        args.datasets = args.weights = True
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'reference/assets.json').read_text())
    selected = [item for item in manifest if (args.datasets if item['kind'] == 'datasets' else args.weights)]
    for item in selected:
        if not (root/item['destination']).exists():
            if args.local_dir:
                if not (args.local_dir/item['name']).is_file():
                    parser.error(f"Local asset is missing: {args.local_dir/item['name']}")
            elif not args.release_tag and not item.get('url'):
                parser.error('No download URL is configured. Use --local-dir for local files or --release-tag to select a release.')
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
            if args.local_dir:
                with (args.local_dir/item['name']).open('rb') as source, partial.open('xb') as out:
                    shutil.copyfileobj(source, out)
            else:
                url = (f"https://github.com/snu-lcbc/RxnMol/releases/download/{quote(args.release_tag, safe='')}/{quote(item['name'], safe='')}"
                       if args.release_tag else item['url'])
                print(f"Downloading {item['name']} from {args.release_tag}...", flush=True)
                try:
                    with urllib.request.urlopen(url, timeout=60) as source, partial.open('xb') as out:
                        shutil.copyfileobj(source, out)
                except (URLError, TimeoutError) as exc:
                    raise SystemExit(
                        f"Could not download {item['name']} from {url}: {exc}. "
                        "Check that the release contains this file, or use --local-dir."
                    ) from exc
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
