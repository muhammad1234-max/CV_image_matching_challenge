import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kaggle_dataset_audit import audit, discover, FIELDS

class KaggleAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.inputs=self.base/'input';self.inputs.mkdir()
        self.root=self.inputs/'arbitrary-mount';self.root.mkdir()
        self.out=self.base/'output'

    def fixture(self):
        for d in ['a','b']:
            p=self.root/'train'/d/'s'/'same.jpg';p.parent.mkdir(parents=True);p.write_bytes(b'fixture-not-a-real-image')
        p=self.root/'test'/'x'/'image.jpg';p.parent.mkdir(parents=True);p.write_bytes(b'fixture')
        with (self.root/'train_labels.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(FIELDS)
            for d in ['a','b']:w.writerow([d,'s','same.jpg','1;0;0;0;1;0;0;0;1','0;0;0'])
        with (self.root/'sample_submission.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(FIELDS);w.writerow(['x','outliers','image.jpg',';'.join(['nan']*9),';'.join(['nan']*3)])
        (self.root/'train_thresholds.csv').write_text('dataset,scene,thresholds\na,s,0.01;0.02;0.03\n')

    def test_missing(self):
        r=audit(self.inputs,self.out);self.assertEqual(r['state'],'BLOCKED')
        self.assertTrue((self.out/'imc_dataset_audit.zip').exists())

    def test_discovery_and_ambiguity(self):
        self.fixture();self.assertEqual(discover(self.inputs)[0],self.root.resolve())
        import shutil
        shutil.copytree(self.root,self.inputs/'second')
        with self.assertRaises(ValueError):discover(self.inputs)
        self.assertEqual(discover(self.inputs,self.root)[0],self.root.resolve())

    def test_complete_audit(self):
        self.fixture();r=audit(self.inputs,self.out);self.assertEqual(r['state'],'AUDITED')
        summary=json.loads((self.out/'dataset_summary.json').read_text());self.assertEqual(summary['unique_training_images'],2)
        poses=json.loads((self.out/'pose_validation.json').read_text());self.assertEqual(poses['issues'],{})
        contract=json.loads((self.out/'submission_contract.json').read_text());self.assertEqual(contract['sample_keys_missing_visible_test_images'],0)
        thresholds=json.loads((self.out/'threshold_provenance.json').read_text());self.assertTrue(thresholds['sources'][0]['sha256'])
        with zipfile.ZipFile(self.out/'imc_dataset_audit.zip') as z:self.assertFalse(any(n.endswith(('.jpg','.sqlite')) for n in z.namelist()))

    def test_malformed(self):
        self.fixture()
        with (self.root/'train_labels.csv').open('a') as f:f.write('a,other,same.jpg,bad,0;0;0\n')
        audit(self.inputs,self.out)
        result=json.loads((self.out/'pose_validation.json').read_text())
        self.assertEqual(result['issues']['duplicate_composite_key'],1)
        self.assertEqual(result['issues']['inconsistent_scene'],1)
        self.assertEqual(result['pose_states']['malformed'],1)

    def test_archives_no_extraction(self):
        self.fixture()
        with zipfile.ZipFile(self.root/'images.zip','w') as z:z.writestr('train/a/s/extra.jpg',b'not-an-image')
        audit(self.inputs,self.out)
        inv=json.loads((self.out/'dataset_inventory.json').read_text());self.assertEqual(inv['archives'][0]['members'],1)
        self.assertFalse((self.root/'train/a/s/extra.jpg').exists())

    def test_notebook_structure(self):
        p=Path(__file__).resolve().parents[1]/'notebooks/step02b_kaggle_dataset_audit.ipynb'
        n=json.loads(p.read_text());self.assertEqual(n['nbformat'],4)
        for c in n['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
        source=''.join(n['cells'][1]['source'])
        self.assertNotIn('D:\\',source);self.assertNotIn('import kaggle',source)

    def test_outside_override(self):
        with self.assertRaises(ValueError):discover(self.inputs,self.base)
