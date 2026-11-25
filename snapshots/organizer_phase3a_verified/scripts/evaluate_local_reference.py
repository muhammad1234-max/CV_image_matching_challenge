"""Pinned local evaluator adapter with strict coverage; no hidden-Kaggle parity claim."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import types
import numpy as np
from benchmark_contract import digest,load_records,TRAIN

ROOT=Path(__file__).resolve().parents[1]
PIN='c7ae94306ba0ae4ce9c00e9abc7e15bd99f3091fa063b5826b0dd567a6c3f261'
ORIGINAL='best_ind = np.lexsort((-mAA_table[i], -cluster_table[i]))[0]'
SENSITIVITY='best_ind = prose_assignment(mAA_table[i], cluster_table[i], user_scene_sum_table[i])'

def prose_assignment(maa,overlap,cluster_size):
    purity=np.divide(overlap,cluster_size,out=np.full_like(maa,-1,dtype=float),where=cluster_size>0)
    # np.lexsort: mAA primary, purity secondary, original position final tie.
    return int(np.lexsort((np.arange(len(maa)),-purity,-maa))[0])

def load_metric(variant='participant'):
    if variant not in ('participant','prose_sensitivity'):raise ValueError('Unknown evaluator variant')
    source=ROOT/'docs/evidence/public_sources/first_metric.py.txt'
    if digest(source)!=PIN:raise ValueError('Pinned evaluator bytes changed')
    text=source.read_text(encoding='utf-8')
    if variant=='prose_sensitivity':
        if text.count(ORIGINAL)!=1:raise ValueError('Assignment patch target changed')
        text=text.replace(ORIGINAL,SENSITIVITY)
    module=types.ModuleType('local_reference_'+variant);module.prose_assignment=prose_assignment
    exec(compile(text,str(source),'exec'),module.__dict__)
    return module

def score_strict(gt,pred,thresholds,variant='participant'):
    truth=load_records(gt);prediction=load_records(pred)
    if {(r['dataset'],r['image']) for r in truth}!={(r['dataset'],r['image']) for r in prediction}:
        raise ValueError('Missing/unexpected prediction keys or datasets; evaluator cannot silently drop them')
    module=load_metric(variant);values,_=module.tth_from_csv(thresholds)
    for row in truth:
        if row['scene']=='outliers':continue
        try:ts=values[row['dataset']][row['scene']]
        except KeyError as exc:raise ValueError('Missing scene thresholds') from exc
        if len(ts)<=2 or not np.isfinite(ts).all() or (ts<0).any() or (ts[2:]<=0).any():raise ValueError('Unsupported threshold convention; resolve before scoring')
    # Separate calls expose components without altering original aggregation.
    results={}
    for name,mode in [('combined_score','harmonic'),('maa','mAA'),('clusterness','clusterness')]:
        with contextlib.redirect_stdout(io.StringIO()):
            totals,per_dataset=module.score(gt_csv=gt,user_csv=pred,thresholds_csv=thresholds,mask_csv=None,combo_mode=mode,inl_cf=0,strict_cf=-1,skip_top_thresholds=2,to_dec=3)
        results[name]={'overall':totals[0]/100,'per_dataset':{k:v/100 for k,v in per_dataset[0].items()}}
    return {'variant':variant,'exact_kaggle_parity':False,'score_units':'fraction','components':results}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gt',required=True);p.add_argument('--pred',required=True);p.add_argument('--thresholds',required=True);p.add_argument('--variant',default='participant',choices=['participant','prose_sensitivity'])
    a=p.parse_args();print(json.dumps(score_strict(a.gt,a.pred,a.thresholds,a.variant),indent=2))
