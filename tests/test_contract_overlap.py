import csv,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from contract_overlap_audit import *

class ContractOverlapTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name);self.out=self.root/'out';self.out.mkdir()
    def file(self,name,fields,rows):
        p=self.root/name;write_table(p,fields,rows);return p
    def test_six_columns_order_identifiers(self):
        p=self.file('sample.csv',SUBMISSION_FIELDS,[['002','a','placeholder','same.jpg','0;0;0;0;0;0;0;0;0','0;0;0'],['001','b','x','same.jpg','nan;nan;nan;nan;nan;nan;nan;nan;nan','nan;nan;nan']])
        r=validate_submission(p,self.out,{('a','same.jpg'),('b','same.jpg')})
        self.assertEqual(r['rows'],2);self.assertFalse(r['physical_pose_validation_applied']);self.assertEqual(r['issues'],{})
        with (self.out/'sample_row_identity.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual([x['image_id'] for x in rows],['002','001']);self.assertEqual(r['repeated_basenames_across_datasets'],1)
        from inspect_dataset import inspect
        self.assertEqual(inspect(p,self.root/'inspect')['unique_image_id'],2)
    def test_duplicate_empty(self):
        row=['','a','s','i','1;0;0;0;1;0;0;0;1','0;0;0']
        p=self.file('sample.csv',SUBMISSION_FIELDS,[row,row]);r=validate_submission(p,self.out)
        self.assertEqual(r['issues']['empty_image_id'],2);self.assertEqual(r['issues']['duplicate_image_id'],1);self.assertEqual(r['issues']['duplicate_dataset_image'],1)
    def test_raw_thresholds(self):
        raw='0;-1;nan;inf;1e-3;;[1,2]'
        ts=threshold_tokens(raw);self.assertEqual([x['type'] for x in ts],['zero','negative','nonfinite','nonfinite','positive','missing','unparsed_encoding'])
        p=self.file('threshold.csv',['dataset','scene','thresholds'],[['imc2023_haiper','bike',raw]])
        r=thresholds(p,self.out);self.assertFalse(r['hash_matches_previous'])
        with (self.out/'thresholds_raw_and_parsed.csv').open() as f:self.assertEqual(next(csv.DictReader(f))['raw_thresholds'],raw)
    def test_training_crosstab(self):
        nan=';'.join(['nan']*9)
        p=self.file('train.csv',TRAIN_FIELDS,[['a','outliers','1',nan,'nan;nan;nan'],['a','s','2',nan,'nan;nan;nan'],['a','s','3','1;0;0;0;1;0;0;0;1','0;0;0']])
        pose_audit(p,self.out);r=json.loads((self.out/'pose_integrity.json').read_text());self.assertEqual(r['nonoutlier_all_nan'],1);self.assertEqual(r['outlier_not_all_nan'],0)
    def test_group_constraints(self):
        self.assertEqual(constrained_groups(['a','b','c'],[('a','b'),('b','c')]),[['a','b','c']])
    def test_missing_mount(self):
        r=run(self.root/'missing',self.out);self.assertEqual(r['state'],'BLOCKED')
    def test_exact_and_partial_overlap(self):
        for d in ['a','b']:
            p=self.root/'train'/d/'s';p.mkdir(parents=True)
            Image.new('RGB',(16,16),(100,120,140)).save(p/'same.png')
        r=overlap(self.root,self.out,seconds=5);self.assertTrue(r['complete']);self.assertEqual(r['exact_group_constraints'],[['a','b']])
        partial=overlap(self.root,self.out,seconds=0)
        self.assertFalse(partial['complete'])
    def test_embedded_notebook_end_to_end(self):
        mount=self.root/'input'/'competition';mount.mkdir(parents=True)
        p=mount/'train'/'a'/'s';p.mkdir(parents=True);Image.new('RGB',(12,12)).save(p/'one.png')
        p=mount/'test'/'a';p.mkdir(parents=True);Image.new('RGB',(12,12)).save(p/'one.png')
        write_table(mount/'train_labels.csv',TRAIN_FIELDS,[['a','s','one.png','1;0;0;0;1;0;0;0;1','0;0;0']])
        write_table(mount/'sample_submission.csv',SUBMISSION_FIELDS,[['007','a','placeholder','one.png','0;0;0;0;0;0;0;0;0','0;0;0']])
        write_table(mount/'train_thresholds.csv',['dataset','scene','thresholds'],[['a','s','0;nan;0.1']])
        nb=json.loads((Path(__file__).resolve().parents[1]/'notebooks/step02c_metadata_and_overlap_audit.ipynb').read_text())
        namespace={};exec(compile(''.join(nb['cells'][1]['source']),'embedded-audit','exec'),namespace)
        result=namespace['run'](self.root/'input',self.out,seconds=5)
        self.assertEqual(result['state'],'AUDITED')
        with zipfile.ZipFile(self.out/'imc_contract_audit.zip') as z:
            self.assertIn('pose_outlier_crosstab.csv',z.namelist())
            self.assertNotIn('sample_row_identity.csv',z.namelist());self.assertNotIn('overlap_cache.json',z.namelist())
