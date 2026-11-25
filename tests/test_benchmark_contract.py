"""Synthetic contract regressions plus explicitly labelled returned-evidence checks."""
import copy,csv,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_contract import *
from evaluate_local_reference import load_metric,prose_assignment,score_strict
from validate_benchmark import validate,CONFIG
from export_benchmark_metadata import export
from register_experiment import register
from finalize_benchmark import review

class BenchmarkTests(unittest.TestCase):
    def test_authentic_metadata_membership(self):
        members,report=review()
        self.assertEqual(len(members),1945);self.assertEqual(report['outliers'],122)
        self.assertEqual(report['finite_poses'],1823);self.assertEqual(report['non_outlier_scenes'],30)
        self.assertEqual(report['exact_duplicate_pairs_preserved'],8)
        self.assertEqual(report['sample_rows'],1945)
        self.assertEqual(canonical_hash(members),report['split_hash'])
    def test_frozen_manifest_cannot_be_replaced(self):
        from unittest.mock import patch
        import finalize_benchmark
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'benchmark.json').write_text('{"status":"FROZEN"}')
            with patch.object(finalize_benchmark,'C',p):
                with self.assertRaisesRegex(ValueError,'Already frozen'):finalize_benchmark.freeze()
            self.assertEqual(json.loads((p/'benchmark.json').read_text()),{'status':'FROZEN'})
    def test_returned_evidence_integrity(self):self.assertIn(validate()['status'],('DRAFT','FROZEN'))
    def test_real_mapping_and_merge(self):
        rows=json.loads((CONFIG/'source_groups.json').read_text())['mapping'];g={r['dataset']:r['source_group'] for r in rows}
        self.assertEqual(len(g),13);self.assertEqual(len(set(g.values())),12)
        self.assertEqual(g['imc2023_heritage'],g['imc2024_dioscuri_baalshamin'])
    def test_group_missing(self):
        with self.assertRaises(ValueError):validate_groups({'a':'a'},['a','b'],[])
    def test_duplicate_boundary(self):
        with self.assertRaises(ValueError):validate_groups({'a':'a','b':'b'},['a','b'],[('a','b')])
    def fixture(self):return [{'dataset':d,'scene':s,'image':str(i)} for i,(d,s) in enumerate([('a','outliers'),('a','scene'),('b','s'),('c','s')])]
    def members(self,records=None):return membership(records or self.fixture(),{'a':'a','b':'b','c':'c'},{'a':'dev_a','b':'dev_b','c':'confirmation'},{'a':2,'b':1,'c':1},{('a','outliers'):1,('a','scene'):1,('b','s'):1,('c','s'):1})
    def test_coverage_and_outliers(self):
        m=self.members();self.assertEqual(len(m),4);self.assertEqual(m[0]['scene'],'outliers')
    def test_missing_image(self):
        with self.assertRaises(ValueError):self.members(self.fixture()[1:])
    def test_duplicate_image(self):
        with self.assertRaises(ValueError):self.members(self.fixture()+self.fixture()[:1])
    def test_confirmation_isolation(self):
        r=self.members()[-1];self.assertEqual(r['fold_a_role'],'excluded_confirmation');self.assertEqual(r['fold_b_role'],'excluded_confirmation')
    def test_deterministic_hash(self):self.assertEqual(canonical_hash(self.members()),canonical_hash(self.members(list(reversed(self.fixture())))))
    def test_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f.json';save_new(p,{'v':1})
            with self.assertRaises(FileExistsError):save_new(p,{'v':2})
            self.assertEqual(json.loads(p.read_text()),{'v':1})
    def test_actual_thresholds(self):
        t=json.loads((CONFIG/'thresholds.json').read_text())['scenes'];self.assertEqual(len(t),30)
        self.assertEqual(sum(v==0 for r in t for v in r['values']),6)
        for r in t:self.assertEqual(r['values'],list(map(float,r['raw'].split(';'))))
    def test_skip_two_and_strict_threshold(self):
        m=load_metric();t=np.array([0,0,.001,.025,.05,.1]);e=np.zeros((5,6))
        self.assertEqual(m.mAA_on_cameras(e,t,5,2),1)
        self.assertAlmostEqual(m.mAA_on_cameras(e,t,5,0),4/6)
        self.assertEqual(m.mAA_on_cameras(np.tile(t,(5,1)),t,5,2),0)
    def test_assignment_difference_and_purity_tie(self):
        maa=np.array([.2,.9]);overlap=np.array([10.,8.]);sizes=np.array([20.,8.])
        self.assertEqual(np.lexsort((-maa,-overlap))[0],0);self.assertEqual(prose_assignment(maa,overlap,sizes),1)
        self.assertEqual(prose_assignment(np.array([.5,.5]),overlap,sizes),1)
        load_metric('prose_sensitivity')
    def write_csv(self,path,rows,fields=TRAIN):
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    def test_camera_center_and_perfect_score(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);rows=[]
            for i,xyz in enumerate([(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,1)]):rows.append(dict(dataset='a',scene='s',image=str(i),rotation_matrix='1;0;0;0;1;0;0;0;1',translation_vector=';'.join(map(str,xyz))))
            self.write_csv(p/'gt.csv',rows);(p/'th.csv').write_text('dataset,scene,thresholds\na,s,0;0;0.001;0.025;0.05;0.1\n')
            np.testing.assert_array_equal(load_metric().read_csv(p/'gt.csv')['a']['s']['1']['c'],[-1,0,0])
            for variant in ['participant','prose_sensitivity']:
                result=score_strict(p/'gt.csv',p/'gt.csv',p/'th.csv',variant)
                self.assertAlmostEqual(result['components']['combined_score']['overall'],1)
            self.write_csv(p/'pred.csv',rows[:-1])
            with self.assertRaises(ValueError):score_strict(p/'gt.csv',p/'pred.csv',p/'th.csv')
    def test_submission_identity_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);rows=[dict(image_id=str(i),dataset='a',scene='s',image=str(i),rotation_matrix='1;0;0;0;1;0;0;0;1',translation_vector='0;0;0') for i in range(2)]
            self.write_csv(p/'a.csv',rows,SUBMISSION);self.write_csv(p/'b.csv',rows[::-1],SUBMISSION)
            with self.assertRaises(ValueError):validate_submission_identity(p/'a.csv',p/'b.csv')
            with self.assertRaises(ValueError):load_records(p/'a.csv')
    def test_sample_placeholders_are_not_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=dict(image_id='id',dataset='a',scene='cluster0',image='i',rotation_matrix='0;0;0;0;0;0;0;0;0',translation_vector='0;0;0')
            self.write_csv(p/'sample.csv',[r],SUBMISSION)
            load_records(p/'sample.csv',SUBMISSION,validate_rotation=False)
            with self.assertRaises(ValueError):load_records(p/'sample.csv',SUBMISSION)
            r['rotation_matrix']='1;0;0;0;1;0;0;0;1';self.write_csv(p/'pred.csv',[r],SUBMISSION)
            validate_submission_identity(p/'sample.csv',p/'pred.csv')
    def experiment(self):
        b={k:'synthetic' for k in ['benchmark_version','benchmark_hash','dataset_inventory_hash','split_hash','evaluator_hash','threshold_hash']};b.update(status='FROZEN',evaluation_datasets={'dev_a':['a']})
        e=json.loads((CONFIG/'experiment_template.json').read_text());e.update({k:b[k] for k in b if k in e});e.update(role='baseline',experiment_id='base',evaluation_partition='dev_a',runtime_seconds=1,per_dataset={'a':dict(maa=.5,clusterness=.5,combined_score=.5,registration_coverage=.5,failure_count=0)},cpu_peak_memory_bytes=dict(measured=True,value=1,reason='fixture'),gpu_peak_memory_bytes=dict(measured=False,value=None,reason='CPU fixture'),seeds={'main':20261111},environment={'python':'synthetic'})
        for k in ['benchmark_hash','dataset_inventory_hash','split_hash','evaluator_hash','threshold_hash']:e[k]=b[k]='a'*64
        e.update(code_snapshot_sha256='b'*64,model_config_hash='c'*64,git_commit='synthetic-fixture',weights_hashes={'none_reason':'synthetic'},submission_format_checks={'passed':True})
        return e,b
    def test_missing_pins_and_confirmation_lock(self):
        e,b=self.experiment();e['model_config_hash']=None
        with self.assertRaises(ValueError):validate_experiment(e,b)
    def test_registration_is_exclusive(self):
        e,b=self.experiment()
        with tempfile.TemporaryDirectory() as d:
            register(e,b,d)
            with self.assertRaises(FileExistsError):register(e,b,d)
    def test_export_synthetic_hash_and_exclusivity(self):
        import contextlib,io,zipfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'input').mkdir();f=p/'input/train_labels.csv';f.write_text('synthetic metadata')
            pins={'train_labels.csv':digest(f)}
            with contextlib.redirect_stdout(io.StringIO()):export(p/'input',p/'out.zip',pins)
            with zipfile.ZipFile(p/'out.zip') as z:self.assertEqual(z.read('train_labels.csv'),f.read_bytes())
            with self.assertRaises(FileExistsError):export(p/'input',p/'out.zip',pins)
            with self.assertRaises(ValueError):export(p/'input',p/'bad.zip',{'train_labels.csv':'wrong'})
    def test_export_notebook_embeds_exact_source(self):
        root=CONFIG.parents[2]
        nb=json.loads((root/'notebooks/02d_export_benchmark_metadata.ipynb').read_text())
        self.assertEqual(''.join(nb['cells'][1]['source']),(root/'scripts/export_benchmark_metadata.py').read_text())
        e,b=self.experiment();e['evaluation_partition']='confirmation'
        with self.assertRaises(ValueError):validate_experiment(e,b)
    def test_experiment_and_pair(self):
        e,b=self.experiment();self.assertTrue(validate_experiment(e,b));i=copy.deepcopy(e);i.update(role='improvement',experiment_id='new',baseline_reference_id='base');self.assertEqual(paired_comparison(e,i,b),{'a':0})
    def test_draft_and_changed_benchmark(self):
        e,b=self.experiment();b['status']='DRAFT'
        with self.assertRaises(ValueError):validate_experiment(e,b)
        b['status']='FROZEN';e['split_hash']='changed'
        with self.assertRaises(ValueError):validate_experiment(e,b)
    def test_incomplete_metrics_and_reference(self):
        e,b=self.experiment();e['per_dataset']={}
        with self.assertRaises(ValueError):validate_experiment(e,b)
        e,b=self.experiment();e['role']='improvement'
        with self.assertRaises(ValueError):validate_experiment(e,b)

if __name__=='__main__':unittest.main()
