#!/usr/bin/env python
"""Convert SDF/SMILES to sanitized canonical isomeric SMILES with an explicit report.

Default: preserve all valid entries, including salts, charges, and duplicate records.
Optional structural changes and filters apply only when requested. They were not
used to produce the building-block file underlying the reported RxnMol runs.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def records(path, fmt):
    if fmt == 'sdf':
        for number, mol in enumerate(Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False), 1):
            yield number, mol, ''
    else:
        with path.open() as fh:
            for number, line in enumerate(fh, 1):
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                smi = line.split()[0]
                yield number, Chem.MolFromSmiles(smi, sanitize=False), smi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--format', choices=['sdf','smi'])
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True, help='CSV with an outcome for every input record')
    parser.add_argument('--largest-fragment', action='store_true')
    parser.add_argument('--neutralize', action='store_true')
    parser.add_argument('--deduplicate', action='store_true')
    parser.add_argument('--min-mw', type=float)
    parser.add_argument('--max-mw', type=float)
    parser.add_argument('--max-abs-charge', type=int)
    parser.add_argument('--exclude-smarts', type=Path, help='One explicit exclusion SMARTS per non-comment line')
    args = parser.parse_args()
    if args.min_mw is not None and args.max_mw is not None and args.min_mw > args.max_mw:
        parser.error('--min-mw must not exceed --max-mw')
    if args.max_abs_charge is not None and args.max_abs_charge < 0:
        parser.error('--max-abs-charge must be nonnegative')
    outputs = [args.output, args.report, args.records]
    if len({p.resolve() for p in outputs + [args.input]}) != 4:
        parser.error('input and all output paths must be different')
    for path in outputs:
        if path.exists():
            parser.error(f'refusing to overwrite {path}')
    patterns = []
    if args.exclude_smarts:
        for line in args.exclude_smarts.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            pattern = Chem.MolFromSmarts(line.strip())
            if pattern is None:
                parser.error(f'invalid SMARTS: {line}')
            patterns.append(pattern)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    fmt = args.format or ('sdf' if args.input.suffix.lower() == '.sdf' else 'smi')
    chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    uncharger = rdMolStandardize.Uncharger()
    seen = set()
    counts = Counter()
    with args.output.open('w') as output, args.records.open('w', newline='') as details:
        writer = csv.DictWriter(details, fieldnames=['record','status','reason','canonical_smiles'])
        writer.writeheader()
        for number, mol, original in records(args.input, fmt):
            counts['input_records'] += 1
            reason, canonical = '', ''
            if mol is None:
                reason = 'parse_failed'
            else:
                try:
                    Chem.SanitizeMol(mol)
                    mol = Chem.RemoveHs(mol)
                    if len(Chem.GetMolFrags(mol)) > 1:
                        counts['input_multicomponent'] += 1
                    if args.largest_fragment:
                        mol = chooser.choose(mol)
                    if args.neutralize:
                        mol = uncharger.uncharge(mol)
                    Chem.SanitizeMol(mol)
                    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                    if not canonical:
                        reason = 'empty_structure'
                    elif args.min_mw is not None and Descriptors.MolWt(mol) < args.min_mw:
                        reason = 'below_min_mw'
                    elif args.max_mw is not None and Descriptors.MolWt(mol) > args.max_mw:
                        reason = 'above_max_mw'
                    elif args.max_abs_charge is not None and abs(Chem.GetFormalCharge(mol)) > args.max_abs_charge:
                        reason = 'formal_charge'
                    elif any(mol.HasSubstructMatch(p) for p in patterns):
                        reason = 'excluded_smarts'
                    elif args.deduplicate and canonical in seen:
                        reason = 'duplicate'
                except Exception as exc:
                    reason = f'sanitization_failed:{type(exc).__name__}'
            if reason:
                counts['rejected'] += 1
                counts[reason] += 1
            else:
                counts['retained'] += 1
                if original and original != canonical:
                    counts['changed_smiles_representation'] += 1
                seen.add(canonical)
                output.write(canonical + '\n')
            writer.writerow(dict(record=number,status='rejected' if reason else 'retained',reason=reason,canonical_smiles=canonical))
    report = dict(rdkit_version=rdBase.rdkitVersion, input_sha256=sha256(args.input),
                  output_sha256=sha256(args.output), script_sha256=sha256(Path(__file__)),
                  options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  counts=dict(counts), unique_output_structures=len(seen))
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
