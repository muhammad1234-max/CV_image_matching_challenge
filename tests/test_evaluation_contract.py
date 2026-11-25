"""Tiny tests of a pinned participant reference, NOT Kaggle parity tests."""
import contextlib
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import types
import unittest
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'docs/evidence/public_sources/first_metric.py.txt'
body = SOURCE.read_bytes()
assert hashlib.sha256(body).hexdigest() == 'c7ae94306ba0ae4ce9c00e9abc7e15bd99f3091fa063b5826b0dd567a6c3f261'
metric = types.ModuleType('pinned_participant_reference')
exec(compile(body, str(SOURCE), 'exec'), metric.__dict__)
HEADER = ['dataset','scene','image','rotation_matrix','translation_vector']
CENTERS = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,2,3],[2,-1,1]], dtype=float)

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tests')
        self.directory = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, centers, label='scene', outlier=False):
        path=self.directory/name
        with path.open('w',newline='') as f:
            w=csv.writer(f); w.writerow(HEADER)
            for i,c in enumerate(centers):
                w.writerow(['d',label,f'{i}.png','1;0;0;0;1;0;0;0;1',';'.join(map(str,-c))])
            if outlier:
                w.writerow(['d','outliers','out.png',';'.join(['nan']*9),';'.join(['nan']*3)])
        return path

    def score(self, centers=CENTERS, label='scene', outlier=False):
        gt=self.write('gt.csv',CENTERS,outlier=outlier)
        pred=self.write('pred.csv',centers,label,outlier)
        thresholds=self.directory/'thresholds.csv'
        thresholds.write_text('dataset,scene,thresholds\nd,scene,0.000001;0.000002;0.00001\n')
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            value,_=metric.score(gt_csv=gt,user_csv=pred,thresholds_csv=thresholds)
        return value[0]

    def test_correct_poses(self):
        self.assertAlmostEqual(self.score(),100)

    def test_label_permutation(self):
        self.assertAlmostEqual(self.score(label='arbitrary-renamed-cluster'),self.score())

    def test_similarity_alignment(self):
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        transformed=2*(CENTERS@rotation.T)+np.array([4,5,6])
        self.assertAlmostEqual(self.score(transformed),100)

    def test_wrong_pose(self):
        wrong=CENTERS.copy(); wrong[-1]+=[13,17,23]
        self.assertLess(self.score(wrong),100)

    def test_ground_truth_outlier_excluded(self):
        self.assertAlmostEqual(self.score(outlier=True),100)

    def test_predicted_outliers_lose_pose_credit(self):
        self.assertAlmostEqual(self.score(label='outliers'),0)

    def test_nan_unregistered_pose(self):
        centers=CENTERS.copy(); centers[-1]=np.nan
        self.assertLess(self.score(centers),100)

    def test_strict_threshold_and_skip(self):
        errors=np.full((6,3),0.1)
        self.assertEqual(metric.mAA_on_cameras(errors,[0.001,0.01,0.1],6,2),0)
        self.assertEqual(metric.mAA_on_cameras(errors,[0.001,0.01,0.2],6,2),1)

    def test_csv_roundtrip(self):
        p=self.write('format.csv',CENTERS,outlier=True)
        with p.open() as f:
            rows=list(csv.reader(f))
        self.assertEqual(rows[0],HEADER)
        for row in rows[1:]:
            self.assertEqual(len(row),5)
            self.assertEqual(len(row[3].split(';')),9)
            self.assertEqual(len(row[4].split(';')),3)
        parsed=metric.read_csv(p)
        np.testing.assert_allclose(parsed['d']['scene']['5.png']['c'],CENTERS[5])

    def test_malformed_rotation_rejected(self):
        p=self.write('bad.csv',CENTERS)
        p.write_text(p.read_text().replace('1;0;0;0;1;0;0;0;1','1;0'))
        with self.assertRaises(ValueError): metric.read_csv(p)

    def test_discrepancy(self):
        maa=np.array([0.9,0.1]); overlap=np.array([4,10])
        self.assertEqual(np.lexsort((-maa,-overlap))[0],1)
        self.assertEqual(np.argmax(maa),0)

    def test_clusterness_and_harmonic(self):
        self.assertAlmostEqual(metric.get_clusterness_score(np.array([4,6]),np.array([8,6])),10/14)
        self.assertEqual(metric.fuse_score(0,0,'harmonic'),0)
        self.assertAlmostEqual(metric.fuse_score(0.5,1,'harmonic'),2/3)

if __name__ == '__main__':
    unittest.main(verbosity=2)
