"""Register only a newly independently scored complete development collection."""
import argparse,json
from pathlib import Path
from merge_ns64_development import merge,ROOT,read

def register(index,review):
    target=ROOT/'experiments/baseline/ns64_mast3rhybrid_022c_dev_full.json'
    if target.exists():
        raise FileExistsError(target)
    merge(index,review)
    coverage=read(review/'coverage.json')
    if not coverage['full']['complete']:
        raise ValueError('Missing datasets; independent review saved without full registration')
    organizer=read(ROOT/'experiments/baseline/organizer_baseline_v1_dev_full.json')
    datasets={}
    for entry in read(index):
        name=entry['dataset']
        scores=read(review/name/'evaluation.json')
        ref=organizer['datasets'][name]
        datasets[name]=dict(evaluation=scores,archive=entry,deltas={v:{k:x-ref['scores'][v][k] for k,x in values.items()} for v,values in scores['scores'].items()})
    record=dict(baseline_id='ns64_mast3rhybrid_022c_dev_full',comparison_baseline_id='organizer_baseline_v1_dev_full',benchmark=read(ROOT/'reports/phase3c8/preparation.json')['benchmark'],seed=20261111,coverage=coverage,datasets=datasets)
    with target.open('x') as stream:json.dump(record,stream,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('index',type=Path);p.add_argument('review',type=Path)
    a=p.parse_args();register(a.index,a.review)
