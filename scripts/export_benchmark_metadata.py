"""Bounded Kaggle metadata-only export. CPU, no internet or image reads."""
import hashlib,json,zipfile
from pathlib import Path
PINS={'train_labels.csv':'f65b20be20cb2cbfd288871c20f9863fd6f052f484a97b236090989af945ea0c','sample_submission.csv':'cce6dddfdf979f0b33e821587d98d6ed0fa97042a0cc0ae4cca1a77a3cf65aab','train_thresholds.csv':'5b92db83d510ab19e66bd1ef4c23865245d75b4a981986e6d8eb1aee06fc9fe2'}
def export(root='/kaggle/input',output='/kaggle/working/imc_benchmark_metadata.zip',pins=PINS):
    root=Path(root).resolve();found={};report=[];total=0
    for name,expected in pins.items():
        matches=list(root.rglob(name))
        if len(matches)!=1:raise ValueError(f'Expected exactly one {name}, found {len(matches)}; attach only the audited competition input')
        p=matches[0].resolve()
        if not p.is_relative_to(root):raise ValueError('Input outside mounted root')
        total+=p.stat().st_size
        if total>10*1024*1024:raise ValueError('Metadata exceeds 10 MiB bound')
        raw=p.read_bytes();actual=hashlib.sha256(raw).hexdigest()
        if actual!=expected:raise ValueError(f'{name} changed from audited source: {actual}; do not freeze')
        found[name]=raw;report.append({'file':name,'sha256':actual,'bytes':len(raw),'source_path':str(p)})
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in found.items():z.writestr(name,raw)
        z.writestr('metadata_manifest.json',json.dumps({'files':report,'images_read':0},indent=2))
    print(json.dumps({'output':str(output),'metadata_bytes':total,'files':report},indent=2))
if __name__=='__main__':export()
