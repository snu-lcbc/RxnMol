"""Exercise local asset installation using small checksum-verified fixtures."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


class LocalAssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'reaction_model'
        self.assets = Path(self.tmp.name)/'assets'
        self.assets.mkdir()
        for directory in ['scripts', 'reference', 'data', 'tokenizer']:
            (self.root/directory).mkdir(parents=True)
        source = Path(__file__).resolve().parents[1]/'reaction_model/scripts/download_assets.py'
        shutil.copyfile(source, self.root/'scripts/download_assets.py')
        (self.root/'tokenizer/tokenizer.json').write_text('{}')
        raw = {'synthetic_rxn.txt': b'CC>>CO\n', 'USPTO_MIT.txt': b'C>>CC\n'}
        with tarfile.open(self.assets/'reaction_datasets.tar.gz', 'w:gz') as archive:
            for name, data in raw.items():
                info = tarfile.TarInfo('data/raw/'+name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        (self.assets/'reported_finetune.ckpt').write_bytes(b'test weights')
        manifest = []
        for name, kind, destination in [('reaction_datasets.tar.gz', 'datasets', 'data/reaction_datasets.tar.gz'), ('reported_finetune.ckpt', 'weights', 'weights/reported_finetune.ckpt')]:
            manifest.append({'name': name, 'kind': kind, 'destination': destination, 'url': None,
                             'sha256': hashlib.sha256((self.assets/name).read_bytes()).hexdigest()})
        (self.root/'reference/assets.json').write_text(json.dumps(manifest))
        expected = {'raw': {key: {'sha256': hashlib.sha256(raw[name]).hexdigest()} for key, name in [('synthetic_rxn', 'synthetic_rxn.txt'), ('uspto-mit', 'USPTO_MIT.txt')]}}
        (self.root/'data/manifest.json').write_text(json.dumps(expected))

    def run_script(self, *args):
        return subprocess.run([sys.executable, str(self.root/'scripts/download_assets.py'), *args], capture_output=True, text=True)

    def test_installs_and_verifies_local_assets_and_can_recheck(self):
        result = self.run_script('--datasets', '--weights', '--asset-dir', str(self.assets))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root/'data/raw/synthetic_rxn.txt').read_bytes(), b'CC>>CO\n')
        self.assertEqual((self.root/'weights/reported_finetune.ckpt').read_bytes(), b'test weights')
        self.assertEqual((self.root/'weights/tokenizer.json').read_text(), '{}')
        recheck = self.run_script('--datasets', '--weights')
        self.assertEqual(recheck.returncode, 0, recheck.stderr)

    def test_unconfigured_download_fails_without_creating_output(self):
        result = self.run_script('--weights')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('No download URL is configured', result.stderr)
        self.assertFalse((self.root/'weights').exists())

    def test_bad_local_checksum_is_rejected(self):
        (self.assets/'reported_finetune.ckpt').write_bytes(b'wrong weights')
        result = self.run_script('--weights', '--asset-dir', str(self.assets))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('hash mismatch', result.stderr)
        self.assertFalse((self.root/'weights/reported_finetune.ckpt').exists())


if __name__ == '__main__':
    unittest.main()
