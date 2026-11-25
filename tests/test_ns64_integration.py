import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/baselines/top10_reproduction'))
import ns64_adapter as adapter
import verify_dependencies as verifier


class Integration(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root=Path(t.name)
        self.mapping={'image_0001.png':'b.png','image_0000.png':'a.png'}
        self.frame=pd.DataFrame([dict(dataset='ETs',image=n,scene='cluster0',rotation_matrix='1;0;0;0;1;0;0;0;1',translation_vector='1;2;3') for n in self.mapping])

    def serialize(self):
        p=adapter.serialize_predictions(self.frame,self.mapping,self.root)
        return pd.read_csv(p,keep_default_na=False)

    def test_sorted_world_to_camera_unchanged(self):
        df=self.serialize()
        self.assertEqual(list(df.image),['a.png','b.png'])
        self.assertEqual(list(df.translation_vector),['1;2;3']*2)

    def test_missing_member(self):
        self.frame=self.frame.iloc[:1]
        with self.assertRaises(ValueError):self.serialize()

    def test_duplicate_member(self):
        self.frame['image']='image_0000.png'
        with self.assertRaises(ValueError):self.serialize()

    def test_wrong_dataset(self):
        self.frame['dataset']='confirmation'
        with self.assertRaises(ValueError):self.serialize()

    def test_invalid_rotation(self):
        self.frame['rotation_matrix']='-1;0;0;0;1;0;0;0;1'
        with self.assertRaises(ValueError):self.serialize()

    def test_unregistered(self):
        self.frame['scene']='outliers'
        self.frame['rotation_matrix']=';'.join(['nan']*9)
        self.frame['translation_vector']='nan;nan;nan'
        self.assertEqual(set(self.serialize().scene),{'outliers'})

    def test_outlier_with_pose(self):
        self.frame['scene']='outliers'
        with self.assertRaises(ValueError):self.serialize()

    def test_confirmation_rejected_before_read(self):
        with self.assertRaises(ValueError):adapter.stage_ets({'images':{}},self.root/'confirmation',self.root/'w')

    def test_no_labels_or_cache_reuse(self):
        directory=self.root/'train/ETs'
        directory.mkdir(parents=True)
        images={}
        for i in range(22):
            p=directory/f'scene_name{i}.png'
            p.write_bytes(bytes([i]))
            images[p.name]=adapter.digest(p)
        (directory/'train_labels.csv').write_text('forbidden')
        work=self.root/'attempt'
        mapping=adapter.stage_ets({'images':images},directory,work)
        self.assertEqual(set(mapping.values()),set(images))
        self.assertEqual(set(mapping),{p.name for p in (work/'test/ETs').iterdir()})
        with self.assertRaises(FileExistsError):adapter.stage_ets({'images':images},directory,work)

    def test_wrong_hash_prevents_loading(self):
        (self.root/'weight').write_bytes(b'other')
        found,path=verifier.locate([self.root],dict(filename='weight',sha256='x'))
        self.assertEqual(len(found),1)
        self.assertIsNone(path)

    def test_independent_imports(self):
        self.assertEqual(verifier.probe('import nonexistent_ns64_abc',self.root)['status'],'BLOCKED')
        self.assertEqual(verifier.probe('import json',self.root)['status'],'PASS')

    def test_no_source_failure_archive(self):
        out=self.root/'receipt.zip'
        self.assertEqual(verifier.verify({},[],self.root/'missing',out)['status'],'BLOCKED')
        with zipfile.ZipFile(out) as z:self.assertEqual(json.loads(z.read('result.json'))['status'],'BLOCKED')

    def test_ets_actual_frozen_scorer(self):
        import glob
        archive=next(p for p in ROOT.glob('imc_baseline_dev_batch_01_*.zip') if '_cache' not in p.name)
        with zipfile.ZipFile(archive) as z:
            prediction=self.root/'predictions.csv'
            prediction.write_bytes(z.read('datasets/ETs/predictions.csv'))
        output=self.root/'score.json'
        adapter.score(ROOT,prediction,output)
        scores=json.loads(output.read_text())['scores']
        for score in scores.values():self.assertAlmostEqual(score['combined_score'],0.3458737864,places=9)

    def test_model_alias_integrity(self):
        w=self.root/'model.pth'
        w.write_bytes(b'model')
        spec={'m':dict(filename=w.name,alias='M',sha256=adapter.digest(w))}
        adapter.check_model_paths({'M':w},spec)
        w.write_bytes(b'changed')
        with self.assertRaises(ValueError):adapter.check_model_paths({'M':w},spec)

    def test_correspondence_counts_and_coordinates(self):
        import numpy as np
        from types import SimpleNamespace
        xy=np.array([[8.,1.],[2.,6.]])
        keys=SimpleNamespace(keypoints={'a':xy,'b':xy[::-1]})
        indices=np.array([[0,1],[1,0]])
        pairs=SimpleNamespace(matches={'a':{'b':indices}})
        self.assertEqual(adapter.match_counts([keys],[pairs]),{'keypoints':4,'matches':2})
        np.testing.assert_array_equal(xy,[[8,1],[2,6]])
        indices[1,0]=2
        with self.assertRaises(ValueError):adapter.match_counts([keys],[pairs])

    def test_stage_failure_and_order_receipts(self):
        from types import SimpleNamespace
        cuda=SimpleNamespace(max_memory_allocated=lambda:0)
        with patch.dict(sys.modules,torch=SimpleNamespace(cuda=cuda)):
            def pairs(): return [(3,2),(0,1)]
            self.assertEqual(adapter.measured(pairs,'shortlist_generator',self.root)(),pairs())
            def fail():raise RuntimeError('SfM failed')
            with self.assertRaises(RuntimeError):adapter.measured(fail,'incremental_mapping',self.root)()
            def geometry():return SimpleNamespace(count=1,inliers={'a':{'b':[0,1]}})
            adapter.measured(geometry,'verify_matches',self.root)()
        rows=[json.loads(l) for l in (self.root/'stages.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['ordered_pairs'],[[3,2],[0,1]])
        self.assertEqual(rows[1]['status'],'FAILED')
        self.assertEqual(rows[2]['inliers'],2)

    def test_unapproved_receipt_stops_before_images(self):
        import ets_worker
        p=self.root/'receipt.zip'
        with zipfile.ZipFile(p,'w') as z:
            z.writestr('manifest.json','{}')
            z.writestr('result.json','{"status":"BLOCKED"}')
        with self.assertRaises(ValueError):ets_worker.approved_paths(p,{})

    def test_input_scanner_skips_competition(self):
        for directory in ['train/ETs','competitions/test','models']:
            p=self.root/directory
            p.mkdir(parents=True)
            (p/'w').write_bytes(b'1')
        self.assertEqual(list(verifier.input_files([self.root],'w')),[self.root/'models/w'])

    def test_changed_dino_config_blocks(self):
        spec={'dinov2':dict(alias='D',filename='model.bin',sha256='',sidecars={'config.json':'invalid'})}
        w=self.root/'model.bin'
        w.write_bytes(b'1')
        spec['dinov2']['sha256']=adapter.digest(w)
        (self.root/'config.json').write_text('{}')
        with self.assertRaises(ValueError):adapter.check_model_paths({'D':self.root},spec)
