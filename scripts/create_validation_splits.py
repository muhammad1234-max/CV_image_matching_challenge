"""Deterministic dataset-disjoint planning split; refuse silent replacement."""
import argparse,csv,hashlib,json
from pathlib import Path

def create(source,output,seed=20261111):
    source=Path(source);output=Path(output)
    if output.exists(): raise FileExistsError('Split exists: version changes require supervisor review')
    groups=set()
    with source.open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f): groups.add(row['dataset'])
    if len(groups)<3: raise ValueError('Need at least three independent datasets; no random image fallback')
    ordered=sorted(groups,key=lambda x:hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
    n=max(1,len(groups)//5)
    assignment={d:('confirmation' if i<n else 'development' if i<2*n else 'train') for i,d in enumerate(ordered)}
    with source.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    output.mkdir(parents=True)
    with source.open(newline='',encoding='utf-8-sig') as f,(output/'split.csv').open('w',newline='') as out:
        w=csv.writer(out);w.writerow(['dataset','scene','image','partition'])
        for r in csv.DictReader(f): w.writerow([r['dataset'],r['scene'],r['image'],assignment[r['dataset']]])
    metadata={'seed':seed,'input_sha256':digest,'assignment':assignment,'status':'DRAFT_REQUIRES_DUPLICATE_GROUP_AUDIT','split_sha256':hashlib.sha256((output/'split.csv').read_bytes()).hexdigest()}
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2));return metadata

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20261111)
    a=p.parse_args();create(a.csv,a.output,a.seed)
