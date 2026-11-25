"""Trusted Phase 3C.5 evaluation of the two returned Colab archives."""
import ast
import hashlib
import io
import itertools
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/baselines/top10_reproduction'))
from isolated_staging import CFG, PLAN_HASH, SEEDS, digest, ordered
from restore_and_score import validate_identity
from ns64_adapter import serialize_predictions, score


def read(path):
    return json.loads(path.read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def centers(frame):
    values = {}
    for row in frame.itertuples():
        r = np.array(row.rotation_matrix.split(';'), float).reshape(3, 3)
        t = np.array(row.translation_vector.split(';'), float).reshape(3)
        if row.scene == 'outliers':
            assert np.isnan(r).all() and np.isnan(t).all()
        else:
            assert np.isfinite(t).all() and np.allclose(r.T @ r, np.eye(3), atol=1e-6)
            assert abs(np.linalg.det(r)-1) < 1e-6
            values[row.image] = -r.T @ t
    return values


def compare(a, b):
    left, right = a.scene.to_dict(), b.scene.to_dict()
    pairs = list(itertools.combinations(sorted(left), 2))
    def together(labels, pair):
        i, j = pair
        return labels[i] == labels[j] and labels[i] != 'outliers'
    x = np.array([together(left,p) for p in pairs])
    y = np.array([together(right,p) for p in pairs])
    expected = x.sum()*y.sum()/len(pairs)
    out = dict(agreeing_pairs=int((x==y).sum()), total_pairs=len(pairs),
               adjusted_rand_index=float(((x & y).sum()-expected)/((x.sum()+y.sum())/2-expected)),
               outlier_disagreements=sum((left[i]=='outliers')!=(right[i]=='outliers') for i in left))
    ca, cb = centers(a.reset_index()), centers(b.reset_index())
    aligned = []
    for _,group in a[a.scene!='outliers'].groupby('scene'):
        ids = sorted(group.index)
        if set(ids) != {i for i in right if right[i]==right[ids[0]]}:
            continue
        u,v = [np.stack([c[i] for i in ids]) for c in (ca,cb)]
        u -= u.mean(axis=0)
        v -= v.mean(axis=0)
        l,s,r = np.linalg.svd(u.T@v)
        sign=np.sign(np.linalg.det(l@r))
        d=np.diag([1.,1.,sign])
        fit=u@(l@d@r)*(s@np.diag(d))/(u*u).sum()
        aligned.append(dict(images=len(ids),relative_center_rms=float(np.linalg.norm(fit-v)/np.linalg.norm(v))))
    out['similarity_aligned_centers']=aligned
    return out


def main():
    root=ROOT/'reports/phase3c5'
    root.mkdir(exist_ok=True)
    clean=ROOT/'reports/phase3c4_namespace'
    stage=clean/'clean_vm_staging'
    plan=read(CFG/'pilot.json')
    assert digest(CFG/'pilot.json')==PLAN_HASH
    assert digest(CFG/'ns64_pilot_source.zip.bin')==plan['source_zip_sha256']
    lock=read(ROOT/'configs/baselines/ns64_isolated_v1/experiment.json')
    pre=read(clean/'clean_vm_precheck_pass_received.json')
    boundary=read(clean/'clean_vm_post_staging_received.json')
    nb=read(ROOT/'notebooks/step03c5_ns64_neutral_orders.ipynb')
    worker=ast.literal_eval(ast.parse(''.join(nb['cells'][1]['source'])).body[0].value)
    hist=ROOT/'reports/phase3c1_ets_pilot'
    frames={'historical':pd.read_csv(hist/'predictions.csv').set_index('image')}
    assert digest(stage/'opaque_inputs.zip')==boundary['archive_sha256']
    summary={'benchmark':plan['benchmark'],'runs':{},'comparisons':{}}
    for seed in SEEDS:
        key=str(seed)
        file=ROOT/f'imc_ns64_order_seed_{seed}.zip'
        dest=root/key
        dest.mkdir(exist_ok=True)
        mapping=read(stage/'private.json')[key]
        assert mapping=={f'image_{i:04d}.png':n for i,n in enumerate(ordered(plan['images'],seed))}
        with zipfile.ZipFile(stage/'opaque_inputs.zip') as inputs:
            hashes=[sha(inputs.read(f'{seed}/{n}')) for n in sorted(mapping)]
        assert hashes==[plan['images'][mapping[n]] for n in sorted(mapping)]
        permutation=sha('\n'.join(hashes).encode())
        with zipfile.ZipFile(file) as z:
            assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
            def load(n):
                return json.loads(z.read(n+'.json'))
            env,run,identity,pred=map(load,['environment','run','identity','prediction'])
            validate_identity(identity,seed)
            assert run['worker_sha256']==sha(worker.encode())==env['probe_sha256']
            assert run['experiment_id']==identity['experiment_id'] and run['seed']==identity['seed']==seed
            assert identity['config_sha256']==plan['config_sha256'] and identity['source_revision']==plan['revision']
            assert run['status']=='PASS' and 0<run['seconds']<=run['budget_seconds']==1800
            assert env['status']=='CLEAN_VM_PRECHECK_PASS' and env['unexpected_artifact_count']==0
            for k in ['source_inventory_hash','assets','environment_freeze','gpu','torch','pycolmap','python','administrative_paths']:
                assert env[k]==pre[k],k
            assert load('boundary')==boundary and boundary['status']=='POST_STAGING_CHECKS_PASS'
            assert permutation==identity['permutation_sha256']==lock['conditions'][identity['experiment_id']]['permutation_sha256']==boundary['order_hashes'][key]
            raw=z.read('opaque.csv')
            assert sha(raw)==pred['sha256']
            restored=serialize_predictions(pd.read_csv(io.BytesIO(raw)),mapping,dest)
            frame=pd.read_csv(restored)
            assert len(frame)==22 and set(frame.image)==set(plan['images'])
            assert len(centers(frame))==pred['registered'] and (frame.scene=='outliers').sum()==pred['outliers']
            evaluation=dest/'evaluation.json'
            score(ROOT,restored,evaluation)
            ev=read(evaluation)
            assert ev['predicted_clusters']==pred['clusters']
            frames[key]=frame.set_index('image')
            stages=[json.loads(line) for line in z.read('stages.jsonl').splitlines()]
            assert stages and all(s['status']=='PASS' for s in stages)
            summary['runs'][key]=dict(archive_sha256=digest(file),restored_sha256=digest(restored),run=run,identity=identity,prediction=pred,evaluation=ev,stages=stages,resources=load('resources'))
    for a,b in itertools.combinations(frames,2):
        summary['comparisons'][a+' vs '+b]=compare(frames[a],frames[b])
    (root/'verified.json').write_text(json.dumps(summary,indent=2))
    for k,v in summary['runs'].items():
        print(k,v['evaluation']['scores'],v['resources'],[(s['stage'],s.get('count',s.get('verified_pairs',s.get('retained_pairs')))) for s in v['stages']])
    print(summary['comparisons'])


if __name__=='__main__':
    main()
