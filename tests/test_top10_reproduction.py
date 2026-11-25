import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/baselines/top10_reproduction'))
import pilot

class PilotTests(unittest.TestCase):
    def setUp(self):
        self.config = ROOT/'configs/baselines/top10_reproduction_v1'
        self.plan = json.loads((self.config/'pilot.json').read_text())

    def test_published_source_and_config(self):
        self.assertEqual(self.plan['team'], 'ns64')
        bundle = self.config/'ns64_pilot_source.zip.bin'
        self.assertEqual(pilot.digest(bundle), self.plan['source_zip_sha256'])
        with zipfile.ZipFile(bundle) as z:
            self.assertEqual(hashlib.sha256(z.read(self.plan['config'])).hexdigest(), self.plan['config_sha256'])

    def test_development_membership(self):
        rows = json.loads((ROOT/'configs/baselines/organizer_v1/development_inputs.json').read_text())
        self.assertEqual(set(self.plan['images']), {r['image'] for r in rows if r['dataset']=='ETs' and r['partition']=='dev_b'})
        self.assertEqual(len(self.plan['images']),22)

    def test_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp, patch('pilot.subprocess.check_output',side_effect=OSError('no GPU')):
            path=Path(tmp)/'result.zip'
            result=pilot.run(self.plan,Path(tmp),path)
            self.assertEqual(result['failed_stage'],'hardware')
            with zipfile.ZipFile(path) as z:
                self.assertIsNone(json.loads(z.read('result.json'))['scores'])
