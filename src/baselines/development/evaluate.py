"""Isolated scoring/diagnostics process, invoked only after predictions are finalized."""
import csv,json,sys,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from common import *
if not hasattr(hashlib,'file_digest'):
    def file_digest(f,name):
        h=hashlib.new(name)
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return h
    hashlib.file_digest=file_digest
sys.path.insert(0,str(SNAPSHOT/'scripts'))
from benchmark_contract import load_records
from evaluate_local_reference import score_strict
def evaluate(dataset,prediction):
    authorized=inputs(dataset);folder=CONFIG/'development_scoring'/dataset
    gt=load_records(folder/'train_labels.csv');pred=load_records(prediction)
    expected={(r['dataset'],r['image']) for r in authorized}
    if {(r['dataset'],r['image']) for r in gt}!=expected or {(r['dataset'],r['image']) for r in pred}!=expected:raise ValueError('Prediction/GT coverage differs from frozen membership')
    # Ordering is identity order, independent of labels; all datasets use frozen sorted membership.
    if [(r['dataset'],r['image']) for r in pred]!=sorted(expected):raise ValueError('Prediction identity order changed')
    scores={}
    for variant in ['participant','prose_sensitivity']:
        result=score_strict(folder/'train_labels.csv',prediction,folder/'train_thresholds.csv',variant)
        scores[variant]={k:v['per_dataset'][dataset] for k,v in result['components'].items()}
    truth={(r['dataset'],r['image']):r for r in gt};cross=Counter();clusters=defaultdict(set);scenes=defaultdict(set);false_outliers=missed_outliers=0
    for p in pred:
        t=truth[p['dataset'],p['image']];cross[t['scene'],p['scene']]+=1
        false_outliers+=t['scene']!='outliers' and p['scene']=='outliers';missed_outliers+=t['scene']=='outliers' and p['scene']!='outliers'
        if t['scene']!='outliers' and p['scene']!='outliers':clusters[p['scene']].add(t['scene']);scenes[t['scene']].add(p['scene'])
    return {'scores':scores,'true_scene_count':len({r['scene'] for r in gt}-{'outliers'}),'true_outliers':sum(r['scene']=='outliers' for r in gt),'predicted_outliers':sum(r['scene']=='outliers' for r in pred),'predicted_clusters':len({r['scene'] for r in pred}-{'outliers'}),'diagnostics':{'false_outliers':false_outliers,'missed_outliers':missed_outliers,'merged_clusters':{k:sorted(v) for k,v in clusters.items() if len(v)>1},'split_scenes':{k:sorted(v) for k,v in scenes.items() if len(v)>1},'contingency':[{'true_scene':a,'predicted_scene':b,'images':n} for (a,b),n in sorted(cross.items())]}}
if __name__=='__main__':
    verify_implementation();write(sys.argv[3],evaluate(sys.argv[1],Path(sys.argv[2])))
