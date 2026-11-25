"""CPU contracts; fixtures are never registered as research results."""
import json,sys,tempfile,unittest,zipfile,csv,ast
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/baselines/development'));sys.path.insert(0,str(ROOT/'scripts'))
import common as dev
import runner
import merge_baseline_batches as merger
from evaluate import evaluate

class DevelopmentTests(unittest.TestCase):
    def test_source_identity(self):
        identity=dev.verify_implementation();self.assertEqual(identity['algorithm_snapshot_hash'],'f57789a8bbd105de5a054040beb5289345ebb47e495b7e4ecf6b0fb25a792b3b')
    def test_frozen_model_configuration(self):
        smoke=dev.read(dev.SNAPSHOT/'configs/baselines/organizer_v1/smoke.json');cfg=dev.read(dev.CONFIG/'full_development.json')['model_config']
        self.assertEqual(cfg,{k:v for k,v in smoke.items() if k not in ['dataset','budget_seconds']})
    def test_membership_and_exclusion(self):
        rows=dev.read(dev.CONFIG/'development_inputs.json');self.assertEqual(len(rows),1563)
        self.assertEqual(sum(r['partition']=='dev_a' for r in rows),728)
        for name in ['imc2024_lizard','imc2024_piazza_san_marco','unknown','../ETs']:
            with self.assertRaises(ValueError):dev.inputs(name)
        frozen=dev.read(dev.SNAPSHOT/'configs/benchmark/imc2025_local_v1/image_membership.json')['records']
        self.assertEqual({(r['dataset'],r['image']) for r in rows},{(r['dataset'],r['image']) for r in frozen if r['partition'] in ['dev_a','dev_b']})
    def test_dataset_level_selection(self):
        rows=dev.inputs('ETs');self.assertEqual(len(rows),22);self.assertEqual(rows,dev.inputs('ETs'));self.assertEqual(set(rows[0]),{'dataset','image','opaque'})
    def test_checkpoint_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'embeddings.pt').write_bytes(b'fixture');dev.write(p/'result.json',{'status':'PASS'})
            dev.checkpoint(p,'B','key');self.assertTrue(dev.checkpoint_valid(p,'key'));self.assertFalse(dev.checkpoint_valid(p,'other'))
            (p/'embeddings.pt').write_bytes(b'changed');self.assertFalse(dev.checkpoint_valid(p,'key'))
    def test_portable_resume_and_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'file.json';f.write_text('{}');archive=p/'cache.zip'
            dev.write_archive(archive,{'datasets/ETs/B/result.json':f},{'kind':'cache','implementation_hash':'x'})
            runner.restore([archive],p/'work','x');runner.restore([archive],p/'work','x')
            (p/'work/datasets/ETs/B/result.json').write_text('changed')
            with self.assertRaises(ValueError):runner.restore([archive],p/'work','x')
            with self.assertRaises(ValueError):runner.restore([archive],p/'other','y')
    def test_unsafe_archive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','x')
            with zipfile.ZipFile(p) as z:
                with self.assertRaises(ValueError):dev.safe_zip_files(z)
    def test_archive_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('x','changed');z.writestr('archive_manifest.json',json.dumps({'files':{'x':'0'*64}}))
            with zipfile.ZipFile(p) as z:
                with self.assertRaises(ValueError):dev.verify_zip(z)
    def test_equal_dataset_weighting_and_partial(self):
        plan=[{'dataset':'a','partition':'dev_a','images':1},{'dataset':'b','partition':'dev_b','images':1000}]
        row=lambda v:{'scores':{k:{'combined_score':v} for k in ['participant','prose_sensitivity']}}
        partial=dev.aggregate({'a':row(.2)},plan);self.assertIsNone(partial['combined']['participant']);self.assertEqual(partial['status'],'PARTIAL')
        full=dev.aggregate({'a':row(.2),'b':row(.8)},plan);self.assertEqual(full['combined']['participant'],.5);self.assertEqual(full['status'],'COMPLETE_UNREGISTERED')
    def test_unknown_aggregation_rejected(self):
        with self.assertRaises(ValueError):dev.aggregate({'unknown':{}},[])
    def test_exclusive_record(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'record.json';dev.exclusive(p,{'value':1})
            with self.assertRaises(FileExistsError):dev.exclusive(p,{'value':2})
            self.assertEqual(dev.read(p)['value'],1)
    def test_environment_lock(self):
        lock=dev.read(dev.CONFIG/'full_environment_lock.json');runtime=lock['observed_full_environment'];dev.validate_environment(runtime,lock)
        with self.assertRaises(ValueError):dev.validate_environment(dict(runtime,python='3.13.16'),lock)
    def test_all_unregistered_failure_scores_zero(self):
        with tempfile.TemporaryDirectory() as d:
            rows=dev.serialize(dev.inputs('ETs'),{});rows.sort(key=lambda r:(r['dataset'],r['image']));p=Path(d)/'pred.csv';dev.write_predictions(p,rows)
            result=evaluate('ETs',p);self.assertEqual(result['predicted_outliers'],22)
            for value in result['scores'].values():self.assertEqual(value['combined_score'],0)
    def test_missing_prediction_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows=dev.serialize(dev.inputs('ETs'),{})[:-1];rows.sort(key=lambda r:(r['dataset'],r['image']));p=Path(d)/'pred.csv';dev.write_predictions(p,rows)
            with self.assertRaises(ValueError):evaluate('ETs',p)
    def test_independent_smoke_recomputation_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as d:
            with (ROOT/'docs/evidence/baseline_smoke_received/predictions.csv').open() as f:rows=list(csv.DictReader(f))
            rows.sort(key=lambda r:(r['dataset'],r['image']));p=Path(d)/'pred.csv';dev.write_predictions(p,rows)
            result=evaluate('ETs',p);self.assertAlmostEqual(result['scores']['participant']['combined_score'],.35894206549118385)
    def test_duplicate_batch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.zip';identity=dev.verify_implementation();dev.write_archive(p,{},dict(kind='report',implementation_hash=identity['implementation_hash']))
            with self.assertRaisesRegex(ValueError,'Duplicate batch'):merger.inspect_batches([p,p],identity)
    def test_missing_blocks_registration(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Missing datasets'):merger.merge([],Path(d)/'output',True)
            self.assertFalse((Path(d)/'output').exists())
    def test_completed_batch_roundtrip_and_duplicate_dataset(self):
        # Synthetic all-unregistered ETs fixture, never persisted outside a temporary directory.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'work/datasets/ETs';folder.mkdir(parents=True)
            identity=dev.verify_implementation();cfg=dev.read(dev.CONFIG/'full_development.json');lock=dev.read(dev.CONFIG/'full_environment_lock.json')
            mapping=dev.inputs('ETs');inventory={r['opaque']:{'original_sha256':'a'*64,'input_sha256':'b'*64} for r in mapping}
            rows=dev.serialize(mapping,{});rows.sort(key=lambda r:(r['dataset'],r['image']));dev.write_predictions(folder/'predictions.csv',rows)
            ev=evaluate('ETs',folder/'predictions.csv');dev.write(folder/'evaluation.json',ev);dev.write(folder/'identity_mapping.json',mapping);dev.write(folder/'input_hashes.json',inventory)
            metrics={'C':{'selected_pairs':0},'E':{'retained_pairs':0},'F':{'verified_pairs':0},'G':{'models':0,'registered_unique':0}}
            for stage in 'BCDEFGH':dev.write(folder/stage/'result.json',{'stage':stage,'status':'PASS','runtime_seconds':1,'metrics':metrics.get(stage,{})})
            dev.write(folder/'H/assignments.json',{r['opaque']:None for r in mapping})
            result=dict(dataset='ETs',partition='dev_b',images=22,benchmark_hash=dev.BENCHMARK,split_hash=dev.SPLIT,implementation_hash=identity['implementation_hash'],model_config_hash=dev.fingerprint(cfg['model_config']),checkpoint_hash=dev.fingerprint(dev.read(dev.CONFIG/'full_checkpoint_lock.json')),environment_hash=dev.validate_environment(lock['observed_full_environment'],lock),membership_hash=dev.fingerprint(mapping),input_inventory_hash=dev.fingerprint(inventory),registered_images=0,registration_coverage=0,failure_count=1,status='SCORED_ALGORITHMIC_FAILURE',runtime_by_stage={s:1 for s in 'BCDEFGH'},successful_stage_seconds=7,models=0,selected_pairs=0,retained_match_pairs=0,verified_pairs=0,**ev)
            dev.write(folder/'result.json',result);dev.write(folder/'complete.json',{'implementation_hash':identity['implementation_hash'],'files':{n:dev.sha(folder/n) for n in ['result.json','predictions.csv','evaluation.json','identity_mapping.json','input_hashes.json']}})
            dev.write(root/'work/session_fixture.json',{'runtime':lock['observed_full_environment']})
            files={p.relative_to(root/'work').as_posix():p for p in (root/'work').rglob('*') if p.is_file()}
            a=root/'a.zip';b=root/'b.zip'
            for target,batch in [(a,'fixture_a'),(b,'fixture_b')]:dev.write_archive(target,files,dict(kind='report',implementation_hash=identity['implementation_hash'],batch_id=batch))
            report=merger.merge([a],root/'review');self.assertEqual(report['summary']['completed_datasets'],1);self.assertIsNone(report['summary']['combined']['participant']);self.assertTrue((root/'review/scores_and_coverage.svg').exists())
            with self.assertRaisesRegex(ValueError,'Duplicate/conflicting dataset'):merger.inspect_batches([a,b],identity)
    def test_notebook_python_syntax(self):
        for cell in dev.read(ROOT/'notebooks/step03b_full_development_baseline.ipynb')['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    def test_label_isolation(self):
        source=(ROOT/'src/baselines/development/runner.py').read_text()
        self.assertNotIn('train_labels.csv',source);self.assertNotIn('train_thresholds.csv',source)
        self.assertLess(source.index("write_predictions(folder/'predictions.csv'"),source.index("stage='EVALUATION'"))

if __name__=='__main__':unittest.main()
