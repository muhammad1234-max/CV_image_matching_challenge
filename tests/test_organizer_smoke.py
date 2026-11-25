"""Synthetic engineering tests, not a model run or performance evidence."""
import ast,base64,csv,io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/baselines/organizer'))
from contracts import *
import run as smoke
class OrganizerTests(unittest.TestCase):
    def test_missing_competition_mount_message(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError,'attach image-matching-challenge-2025-ongoing'):
                smoke.prepare_images([],Path(d)/'absent',Path(d))
    def test_incomplete_dataset_is_not_silently_reduced(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaisesRegex(FileNotFoundError,'1/1 expected files missing'):
                smoke.prepare_images([{'dataset':'ETs','image':'missing.png','opaque':'image_0000.png'}],p,p)
            self.assertFalse((p/'images').exists())
    def test_pybind_and_python_enum_reporting(self):
        import enum
        class ImageSelectionMethod:
            __module__='pycolmap'
            name='MIN_UNCERTAINTY'
            def __int__(self):return 2
        class Choice(enum.Enum):AUTO=0
        options={'mapper':{'image_selection_method':ImageSelectionMethod()},'images':set(),'choice':Choice.AUTO}
        saved=json.loads(json.dumps(options,default=json_default))
        self.assertEqual(saved['mapper']['image_selection_method'],{'enum_type':'pycolmap.ImageSelectionMethod','name':'MIN_UNCERTAINTY','value':2})
        self.assertEqual(fingerprint(options),fingerprint(saved))
    def test_nested_pycolmap_options_reporting(self):
        class Options:
            __module__='pycolmap'
            def todict(self):return {'image_names':{'a','b'}}
        self.assertEqual(json.loads(json.dumps(Options(),default=json_default)),{'image_names':['a','b']})
    def test_pycolmap_set_options_reporting(self):
        options={'image_names':set(),'nested':{'selected':{'image_0002.png','image_0001.png'}}}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'runtime.json';write(path,{'pycolmap_options':options})
            saved=json.loads(path.read_text())['pycolmap_options']
            self.assertEqual(saved['image_names'],[])
            self.assertEqual(saved['nested']['selected'],['image_0001.png','image_0002.png'])
            self.assertEqual(fingerprint(options),fingerprint(saved))
    def test_unknown_report_type_stays_strict(self):
        with self.assertRaises(TypeError):fingerprint({'invalid':object()})
    def test_frozen_hash(self):self.assertEqual(verify_benchmark(ROOT)['benchmark_hash'],BENCHMARK)
    def test_wrong_benchmark(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'configs/benchmark/imc2025_local_v1';p.mkdir(parents=True);b=json.loads((ROOT/'configs/benchmark/imc2025_local_v1/benchmark.json').read_text());b['seed']=0;write(p/'benchmark.json',b)
            with self.assertRaises(ValueError):verify_benchmark(d)
    def test_input_isolation(self):
        rows=selection(ROOT);self.assertEqual(len(rows),22)
        self.assertTrue(all(set(r)=={'dataset','image','opaque'} and r['dataset']=='ETs' for r in rows))
        self.assertEqual(len({r['opaque'] for r in rows}),22)
    def test_forbidden_datasets(self):
        for name in ['imc2024_lizard_pond','pt_piazzasanmarco_grandplace','test','amy_gardens']:
            with self.assertRaises(ValueError):selection(ROOT,name)
    def test_deterministic_aliases(self):self.assertEqual(selection(ROOT),selection(ROOT))
    def test_serialization_and_missing(self):
        mapping=[dict(dataset='ETs',image='original.png',opaque='image_0000.png'),dict(dataset='ETs',image='missing.png',opaque='image_0001.png')]
        rows=serialize(mapping,{'image_0000.png':{'cluster':2,'rotation':[1,0,0,0,1,0,0,0,1],'translation':[1,2,3]}})
        self.assertEqual(rows[0]['image'],'original.png');self.assertEqual(rows[0]['scene'],'cluster2');self.assertEqual(rows[0]['translation_vector'],'1.000000000;2.000000000;3.000000000')
        self.assertEqual(rows[1]['scene'],'outliers');self.assertEqual(rows[1]['rotation_matrix'],';'.join(['nan']*9));self.assertEqual(len(rows),2)
    def test_unexpected_prediction(self):
        with self.assertRaises(ValueError):serialize([],{'unknown':None})
    def test_malformed_pose(self):
        with self.assertRaises(ValueError):serialize([dict(dataset='ETs',image='i',opaque='a')],{'a':{'cluster':0,'rotation':[1],'translation':[0,0,0]}})
    def test_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'features').write_text('a');write(p/'checkpoint.json',{'status':'PASS','cache_key':'key','artifacts':{'features':sha(p/'features')}})
            self.assertTrue(cache_valid(p,'key'));self.assertFalse(cache_valid(p,'new-config'))
            (p/'features').write_text('changed');self.assertFalse(cache_valid(p,'key'))
    def test_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError,'mounted asset'):require_files([Path(d)/'absent.pth'])
    def test_failed_process(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):smoke.stage_process([sys.executable,'-c','raise RuntimeError("COLMAP synthetic failure")'],Path(d)/'log',10)
            self.assertIn('COLMAP synthetic failure',(Path(d)/'log').read_text())
    def test_budget_exhausted(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TimeoutError):smoke.stage_process([],Path(d)/'log',0)
    def test_active_stage_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TimeoutError):smoke.stage_process([sys.executable,'-c','import time; time.sleep(10)'],Path(d)/'log',0.1)
    def test_gpu_stage_failure_receipt(self):
        cfg=json.loads((ROOT/'configs/baselines/organizer_v1/smoke.json').read_text())
        runtime=dict(python='synthetic',torch='synthetic',cuda_build='synthetic',gpu='synthetic',packages={},checkpoint_hashes={})
        with tempfile.TemporaryDirectory() as d,patch('preflight.run',return_value=runtime),patch.object(smoke,'prepare_images',return_value={}):
            p=Path(d)
            with patch.object(smoke,'stage_process',side_effect=RuntimeError('synthetic CUDA OOM')):
                r=smoke.run(ROOT,{'competition':'unused','dinov2':'unused'},p/'work',cfg)
            self.assertEqual(r['status'],'FAILED');self.assertEqual(r['stages']['A']['status'],'PASS');self.assertEqual(r['stages']['B']['status'],'FAILED');self.assertEqual(r['stages']['C']['status'],'NOT_RUN')
            self.assertFalse((p/'work/predictions.csv').exists())
    def test_preflight_failure_receipt(self):
        cfg=json.loads((ROOT/'configs/baselines/organizer_v1/smoke.json').read_text())
        with tempfile.TemporaryDirectory() as d,patch('preflight.run',side_effect=RuntimeError('synthetic missing CUDA')):
            p=Path(d);r=smoke.run(ROOT,{},p/'work',cfg)
            self.assertEqual(r['status'],'BLOCKED');self.assertEqual(r['stages']['B']['status'],'NOT_RUN');self.assertTrue(r['not_registered_baseline'])
            with zipfile.ZipFile(p/'imc_baseline_smoke_report.zip') as z:self.assertIn('receipt.json',z.namelist());self.assertFalse(any(n.endswith('.png') for n in z.namelist()))
    def test_flat_input_strips_scene_metadata(self):
        from PIL import Image,PngImagePlugin
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'data/train/ETs';source.mkdir(parents=True);meta=PngImagePlugin.PngInfo();meta.add_text('scene','secret_ground_truth')
            Image.new('RGB',(8,8),(7,8,9)).save(source/'scene_hint.png',pnginfo=meta);(p/'work').mkdir()
            mapping=[{'dataset':'ETs','image':'scene_hint.png','opaque':'image_0000.png'}];result=smoke.prepare_images(mapping,p/'data',p/'work')
            with Image.open(p/'work/images/image_0000.png') as im:self.assertNotIn('scene',im.info);self.assertEqual(im.getpixel((0,0)),(7,8,9))
            self.assertIn('input_sha256',result['image_0000.png'])
    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'work').mkdir();(p/'data').mkdir()
            with self.assertRaises(ValueError):smoke.prepare_images([{'dataset':'ETs','image':'../../../secret','opaque':'image_0000.png'}],p/'data',p/'work')
    def test_scoring_only_ets(self):
        with (ROOT/'configs/baselines/organizer_v1/scoring/ETs.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),22);self.assertEqual({r['dataset'] for r in rows},{'ETs'})
    def test_source_snapshot(self):
        snapshot=json.loads((ROOT/'configs/baselines/organizer_v1/source_snapshot.json').read_text())
        for n,h in snapshot['files'].items():self.assertEqual(sha(ROOT/n),h,n)
    def test_notebook_structure_payload(self):
        nb=json.loads((ROOT/'notebooks/step03a_baseline_gpu_smoke.ipynb').read_text());self.assertEqual(nb['nbformat'],4)
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']));self.assertEqual(c['outputs'],[])
        tree=ast.parse(''.join(nb['cells'][1]['source']));payload=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
            self.assertIn('src/baselines/organizer/run.py',z.namelist());self.assertFalse(any(n.endswith(('train_labels.csv','.pth','.whl')) for n in z.namelist()))
            snapshot=json.loads(z.read('configs/baselines/organizer_v1/source_snapshot.json'))
            for n,h in snapshot['files'].items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h)

if __name__=='__main__':unittest.main()
