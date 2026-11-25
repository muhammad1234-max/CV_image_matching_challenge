import csv,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_dataset import inspect,FIELDS,inventory
from create_validation_splits import create

class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.addCleanup(self.t.cleanup);self.root=Path(self.t.name)
        self.csv=self.root/'labels.csv'
        with self.csv.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(FIELDS)
            for d in ['a','b','c','d','e']:
                for i in range(6): w.writerow([d,'scene',f'{i}.jpg','1;0;0;0;1;0;0;0;1','0;0;0'])
    def test_counts(self):
        s=inspect(self.csv,self.root/'out');self.assertEqual(s['rows'],30);self.assertEqual(len(s['datasets']),5)
    def test_duplicate_invalid_missing(self):
        with self.csv.open('a') as f:f.write('a,other,0.jpg,1;0,0;0;0\n')
        s=inspect(self.csv,self.root/'out',self.root/'images');self.assertEqual(s['issues']['duplicate_image'],1);self.assertEqual(s['issues']['inconsistent_scene'],1);self.assertEqual(s['issues']['malformed_pose'],1);self.assertEqual(s['issues']['missing_image'],31)
    def test_splits(self):
        a=create(self.csv,self.root/'s1');b=create(self.csv,self.root/'s2');self.assertEqual(a,b);self.assertEqual(set(a['assignment'].values()),{'train','development','confirmation'})
        with self.assertRaises(FileExistsError):create(self.csv,self.root/'s1')
    def test_header(self):
        self.csv.write_text('bad\n');
        with self.assertRaises(ValueError):inspect(self.csv,self.root/'out')
    def test_inventory(self):
        inventory(self.root,self.root/'inventory.csv');self.assertTrue((self.root/'inventory.csv').exists())
