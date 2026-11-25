"""Streaming CSV analysis; SQLite bounds duplicate-index memory. No image decoding."""
import argparse
from contextlib import closing
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import numpy as np

FIELDS=['dataset','scene','image','rotation_matrix','translation_vector']

def inspect(csv_path, output, image_root=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    with Path(csv_path).open(newline='',encoding='utf-8-sig') as stream:
        observed=next(csv.reader(stream),[])
    if observed==['image_id']+FIELDS:
        from contract_overlap_audit import validate_submission
        # Sample placeholders receive encoding checks, never physical pose checks.
        return validate_submission(csv_path,output)
    counts=Counter(); issues=Counter(); rows=0
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        with closing(sqlite3.connect(str(Path(temporary)/'ids.sqlite'))) as db:
            db.execute('create table ids (dataset text,image text,scene text,primary key(dataset,image))')
            with Path(csv_path).open(newline='',encoding='utf-8-sig') as stream:
                reader=csv.DictReader(stream)
                if reader.fieldnames != FIELDS: raise ValueError('Expected exact five-column header')
                for row in reader:
                    rows+=1
                    if None in row or any(row.get(k) is None for k in FIELDS):
                        issues['malformed_row']+=1; continue
                    d,s,i=(row[k] for k in FIELDS[:3])
                    if not all((d,s,i)): issues['empty_identifier']+=1
                    counts[d,s]+=1
                    prior=db.execute('select scene from ids where dataset=? and image=?',(d,i)).fetchone()
                    if prior:
                        issues['duplicate_image']+=1
                        if prior[0]!=s: issues['inconsistent_scene']+=1
                    else: db.execute('insert into ids values (?,?,?)',(d,i,s))
                    try:
                        r=np.array([float(v) for v in row['rotation_matrix'].split(';')]).reshape(3,3)
                        t=np.array([float(v) for v in row['translation_vector'].split(';')]).reshape(3)
                        if np.isnan(r).all() and np.isnan(t).all(): issues['all_nan_pose']+=1
                        elif not (np.isfinite(r).all() and np.isfinite(t).all()): issues['nonfinite_pose']+=1
                        elif not (np.allclose(r.T@r,np.eye(3),atol=1e-3) and abs(np.linalg.det(r)-1)<1e-3): issues['invalid_rotation']+=1
                    except ValueError: issues['malformed_pose']+=1
                    if image_root:
                        root=Path(image_root).resolve()
                        candidates=[root/d/s/i,root/d/i]
                        safe=[p for p in candidates if p.resolve().is_relative_to(root)]
                        if len(safe)!=2: issues['unsafe_path']+=1
                        elif not any(p.is_file() for p in safe): issues['missing_image']+=1
            unique=db.execute('select count(*) from ids').fetchone()[0]; db.close()
    datasets=Counter()
    for (d,s),n in counts.items(): datasets[d]+=n
    summary={'rows':rows,'unique_image_records':unique,'datasets':dict(sorted(datasets.items())),
             'scenes':[{'dataset':d,'scene':s,'rows':n} for (d,s),n in sorted(counts.items())],
             'outlier_rows':sum(n for (d,s),n in counts.items() if s=='outliers'),
             'scene_size_distribution':dict(Counter(n for (d,s),n in counts.items() if s!='outliers')),
             'issues':dict(issues),'image_existence_checked':bool(image_root)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    with (output/'scene_counts.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','scene','rows']);w.writerows((d,s,n) for (d,s),n in sorted(counts.items()))
    sizes=summary['scene_size_distribution']; maximum=max(sizes.values(),default=1)
    bars=''.join(f'<rect x="80" y="{j*25+20}" width="{400*v/maximum}" height="18"/><text x="0" y="{j*25+34}">{k}: {v}</text>' for j,(k,v) in enumerate(sorted(sizes.items())))
    (output/'scene_size_distribution.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="520" height="{max(60,len(sizes)*25+30)}">{bars}</svg>')
    return summary

def inventory(root, target):
    with Path(target).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['relative_path','bytes','type'])
        for p in sorted(Path(root).iterdir()):
            if p.is_symlink(): continue
            if p.is_dir():
                for child in p.rglob('*'):
                    if child.is_file() and not child.is_symlink(): w.writerow([str(child.relative_to(root)),child.stat().st_size,child.suffix.lower()])
            elif p.is_file(): w.writerow([p.name,p.stat().st_size,p.suffix.lower()])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('csv');parser.add_argument('--output',required=True);parser.add_argument('--image-root')
    a=parser.parse_args();print(json.dumps(inspect(a.csv,a.output,a.image_root),indent=2))
