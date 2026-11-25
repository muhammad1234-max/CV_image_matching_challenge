"""Offline CPU metadata audit. Embedded verbatim in the portable notebook."""
import csv
import hashlib
import json
import os
import platform
import sqlite3
import tempfile
import zipfile
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

FIELDS = ['dataset', 'scene', 'image', 'rotation_matrix', 'translation_vector']
MAX_FILES = 1000000
MAX_ROWS = 1000000
MAX_METADATA_BYTES = 10 * 1024**2
MAX_INVENTORY_RECORDS = 20000

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def header(path):
    try:
        with path.open(encoding='utf-8-sig', newline='') as f:
            return next(csv.reader(f), [])
    except (OSError, UnicodeError, csv.Error):
        return []

def discover(input_root, override=None):
    """Bounded shallow metadata discovery, independent of mount names."""
    input_root = Path(input_root).resolve()
    if override:
        chosen = Path(override).resolve()
        if not chosen.is_relative_to(input_root) or not chosen.is_dir():
            raise ValueError('INPUT_PATH must be an attached directory beneath input_root')
        return chosen, [str(chosen)], 'manual'
    candidates = set()
    visited = 0
    for directory, dirs, files in os.walk(input_root, followlinks=False):
        visited += 1
        if visited > 10000:
            raise ValueError('Discovery limit reached. Set INPUT_PATH explicitly.')
        base = Path(directory)
        dirs[:] = [d for d in dirs if not (base/d).is_symlink() and len((base/d).relative_to(input_root).parts) <= 5]
        for name in files:
            p = base/name
            if p.suffix.lower() == '.csv' and not p.is_symlink() and p.stat().st_size <= MAX_METADATA_BYTES:
                h = header(p)
                if set(FIELDS).issubset(h) and ((base/'train').is_dir() or (base/'test').is_dir()):
                    candidates.add(base)
    candidates = sorted(candidates)
    if len(candidates) != 1:
        raise ValueError(f'Found {len(candidates)} candidate roots. Attach competition data or set INPUT_PATH explicitly. Candidates: {[str(p) for p in candidates]}')
    return candidates[0], [str(p) for p in candidates], 'automatic structure heuristic; provenance requires user confirmation'

def pose_state(row):
    try:
        r = np.array([float(x) for x in row['rotation_matrix'].split(';')]).reshape(3, 3)
        t = np.array([float(x) for x in row['translation_vector'].split(';')]).reshape(3)
        if np.isnan(r).all() and np.isnan(t).all(): return 'all_nan'
        if not np.isfinite(r).all() or not np.isfinite(t).all(): return 'nonfinite'
        if not np.allclose(r.T@r, np.eye(3), atol=1e-3) or abs(np.linalg.det(r)-1) > 1e-3: return 'invalid_rotation'
        return 'valid'
    except (ValueError, KeyError, AttributeError):
        return 'malformed'

def analyze_csv(path, db, role):
    counts, poses, issues = Counter(), Counter(), Counter()
    db.execute('drop table if exists records')
    db.execute('create table records(dataset text,image text,scene text,primary key(dataset,image))')
    rows = 0
    complete = True
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        h = reader.fieldnames
        if h != FIELDS:
            return {'status':'unsupported_header', 'columns':h, 'complete':False}, counts
        for row in reader:
            if rows >= MAX_ROWS:
                complete = False; break
            rows += 1
            if None in row or any(row.get(k) is None for k in FIELDS):
                issues['malformed_row'] += 1; continue
            d, s, i = (row[k] for k in FIELDS[:3])
            if not all((d,s,i)): issues['empty_identifier'] += 1
            old = db.execute('select scene from records where dataset=? and image=?',(d,i)).fetchone()
            if old:
                issues['duplicate_composite_key'] += 1
                if old[0] != s: issues['inconsistent_scene'] += 1
            else:
                db.execute('insert into records values(?,?,?)',(d,i,s))
                counts[d,s] += 1
            poses[pose_state(row)] += 1
            if role in ('train','test'):
                candidates = [f'{role}/{d}/{s}/{i}', f'{role}/{d}/{i}']
                if any('..' in Path(p).parts or Path(i).is_absolute() for p in candidates):
                    issues['unsafe_image_reference'] += 1
                else:
                    found = db.execute('select count(*) from files where path in (?,?)',candidates).fetchone()[0]
                    if not found: issues['unresolved_image_reference'] += 1
    unique = db.execute('select count(*) from records').fetchone()[0]
    missing_test = extra_test = None
    if role == 'test':
        missing_test = db.execute("select count(*) from images i where role='test' and not exists(select 1 from records r where r.dataset=i.dataset and r.image=i.image)").fetchone()[0]
        extra_test = db.execute("select count(*) from records r where not exists(select 1 from images i where i.role='test' and r.dataset=i.dataset and r.image=i.image)").fetchone()[0]
    return {'status':'parsed','columns':h,'rows':rows,'unique_dataset_image_keys':unique,'complete':complete,
            'pose_states':dict(poses),'issues':dict(issues),'sample_keys_missing_visible_test_images':missing_test,
            'sample_keys_without_visible_test_image':extra_test,
            'identifier_format':'opaque image string; uniqueness key is (dataset,image)'}, counts

def audit(input_root='/kaggle/input', output='/kaggle/working/imc_audit', input_path=None, train_csv=None, sample_csv=None):
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=True)
    if out.is_relative_to(Path(input_root).resolve()): raise ValueError('Outputs must be outside read-only inputs')
    probe = out/'.write_probe'; probe.write_text('ok'); probe.unlink()
    artifacts = []
    def save(name, value):
        dump(out/name,value); artifacts.append(name)
    save('execution_environment.json',{'utc_timestamp':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
         'numpy':np.__version__,'platform':platform.system(),'gpu_required':False,'network_used':False,'output_write_probe':'passed'})
    status = {'state':'STARTED','real_kaggle_execution':str(input_root)=='/kaggle/input' and Path('/kaggle/working').is_dir(),
              'statistics_complete':False,'split_frozen':False,'limitations':[]}
    try:
        root,candidates,method = discover(input_root,input_path)
        status.update(input_root=str(root),discovery=method,candidates=candidates)
        inventory, metadata, archives = [], [], []
        total_bytes = entries = 0
        scan_complete = True
        with tempfile.TemporaryDirectory(dir=out) as temp, closing(sqlite3.connect(str(Path(temp)/'index.sqlite'))) as db:
            db.execute('create table files(path text primary key)')
            db.execute('create table images(role text,dataset text,image text)')
            db.execute('create index image_lookup on images(role,dataset,image)')
            for directory, dirs, names in os.walk(root,followlinks=False):
                base = Path(directory)
                dirs[:] = [d for d in dirs if not (base/d).is_symlink()]
                if len(inventory)<MAX_INVENTORY_RECORDS:
                    inventory.append({'path':base.relative_to(root).as_posix(),'type':'directory','bytes':None})
                for name in names:
                    p=base/name
                    if p.is_symlink(): continue
                    if entries>=MAX_FILES: scan_complete=False;break
                    rel=p.relative_to(root).as_posix();size=p.stat().st_size;ext=p.suffix.lower()
                    entries+=1;total_bytes+=size
                    db.execute('insert into files values(?)',(rel,))
                    if len(inventory)<MAX_INVENTORY_RECORDS: inventory.append({'path':rel,'type':ext or 'file','bytes':size})
                    parts=Path(rel).parts
                    if ext in ('.jpg','.jpeg','.png','.webp','.tif','.tiff') and len(parts)>=3 and parts[0] in ('train','test'):
                        db.execute('insert into images values(?,?,?)',(parts[0],parts[1],'/'.join(parts[2:]) if parts[0]=='test' else parts[-1]))
                    if ext in ('.csv','.json','.txt') and len(metadata)<1000:
                        item={'path':rel,'bytes':size,'sha256':None,'columns':None}
                        if size<=MAX_METADATA_BYTES and len(metadata)<200:
                            if ext=='.csv': item['columns']=header(p)
                            with p.open('rb') as f:item['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
                        metadata.append(item)
                    if ext in ('.zip','.tar','.gz','.7z') and len(archives)<1000:
                        item={'path':rel,'bytes':size,'inspection':'not inspected; no extraction'}
                        if ext=='.zip' and size<=64*1024**2:
                            try:
                                with zipfile.ZipFile(p) as z:
                                    members=z.infolist();item.update(inspection='central directory only',members=len(members),expanded_bytes=sum(m.file_size for m in members),member_name_sample=[m.filename for m in members[:100]])
                            except zipfile.BadZipFile:item['inspection']='invalid ZIP'
                        archives.append(item)
                if not scan_complete:break
            db.commit()
            save('dataset_inventory.json',{'root':str(root),'scanned_files':entries,'reported_bytes':total_bytes,'scan_complete':scan_complete,
                 'inventory_records_complete':len(inventory)<MAX_INVENTORY_RECORDS,'records':inventory,'metadata':metadata,'archives':archives,
                 'metadata_catalog_may_be_capped':len(metadata)==1000,'archive_catalog_may_be_capped':len(archives)==1000,
                 'note':'Inventory includes path metadata only. Directory bytes null; total counts each file once. Large archives not opened.'})
            def choose(override, word):
                if override:
                    p=(root/override).resolve()
                    if not p.is_relative_to(root) or not p.is_file():raise ValueError('CSV override must name a file inside selected root')
                    return p
                options=[root/m['path'] for m in metadata if m['columns']==FIELDS and word in Path(m['path']).name.lower()]
                return options[0] if len(options)==1 else None
            labels=choose(train_csv,'train');sample=choose(sample_csv,'sample')
            train_result,counts=analyze_csv(labels,db,'train') if labels else ({'status':'unavailable_or_ambiguous','complete':False},Counter())
            sample_result,_=analyze_csv(sample,db,'test') if sample else ({'status':'unavailable_or_ambiguous','complete':False},Counter())
            train_result['source']=str(labels.relative_to(root)) if labels else None
            sample_result['source']=str(sample.relative_to(root)) if sample else None
            sample_result['pose_representation']='nine row-major rotation values and three translation values, separated by semicolons'
            sample_result['coverage_limit']='Visible test files only; archive-only or unrecognized layouts make unresolved references inconclusive. Sample poses are not ground truth.'
            save('pose_validation.json',train_result);save('submission_contract.json',sample_result)
            groups=Counter();scenes=Counter();outs=Counter()
            for (d,s),n in counts.items():
                groups[d]+=n
                if s=='outliers':outs[d]+=n
                else:scenes[d]+=1
            def table(name,columns,rows):
                with (out/name).open('w',newline='',encoding='utf-8') as f:
                    w=csv.writer(f);w.writerow(columns);w.writerows(rows)
                artifacts.append(name)
            table('dataset_groups.csv',['dataset','unique_images','scenes','outliers'],[(d,n,scenes[d],outs[d]) for d,n in sorted(groups.items())])
            table('scene_statistics.csv',['dataset','scene','unique_images'],[(d,s,n) for (d,s),n in sorted(counts.items())])
            table('outlier_statistics.csv',['dataset','outliers','unique_images','percent'],[(d,outs[d],n,100*outs[d]/n) for d,n in sorted(groups.items())])
            distribution=Counter(n for (d,s),n in counts.items() if s!='outliers')
            visible={role:dict(db.execute('select dataset,count(*) from images where role=? group by dataset',(role,))) for role in ('train','test')}
            save('dataset_summary.json',{'training_datasets':len(groups) if labels else None,'training_scenes':sum(scenes.values()) if labels else None,
                'unique_training_images':sum(groups.values()) if labels else None,'training_outliers':sum(outs.values()) if labels else None,
                'scene_size_distribution':dict(distribution),'visible_image_files_by_dataset':visible,'independent_groups':'UNKNOWN: dataset names are not proof of independence',
                'metadata_statistics_complete':train_result.get('complete',False),'file_scan_complete':scan_complete})
            thresholds=[]
            for m in metadata:
                if m['columns'] and {'dataset','scene','thresholds'}.issubset(m['columns']) and m['bytes']<=MAX_METADATA_BYTES:
                    values=[]
                    with (root/m['path']).open(newline='',encoding='utf-8-sig') as f:
                        for j,r in enumerate(csv.DictReader(f)):
                            if j>=10000:break
                            try:
                                ts=[float(x) for x in r['thresholds'].split(';')]
                                valid=bool(ts) and all(np.isfinite(x) and x>0 for x in ts)
                            except (ValueError,TypeError):ts=[];valid=False
                            values.append({'dataset':r.get('dataset'),'scene':r.get('scene'),'thresholds':ts if valid else None,'valid':valid})
                    thresholds.append({'source':m['path'],'sha256':m['sha256'],'rows':values,'complete':len(values)<10000,'units':'not inferred'})
            save('threshold_provenance.json',{'sources':thresholds,'synthetic_substitution':False,'official_parity':'not established'})
            bars=''.join(f'<text x="0" y="{j*22+18}">{n} images: {freq} scenes</text>' for j,(n,freq) in enumerate(sorted(distribution.items())[:100]))
            (out/'scene_sizes.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="{max(40,min(100,len(distribution))*22+20)}">{bars}</svg>')
            artifacts.append('scene_sizes.svg')
            review=f'''# Validation review\nObserved labeled dataset names: {len(groups) if labels else 'unavailable'}. Independent source groups: UNKNOWN.\nThe 60/20/20 allocation is NOT approved or frozen. With {len(groups)} named groups, each 20% partition would contain approximately {max(1,len(groups)//5) if groups else 0} groups; inspect dataset_groups.csv, outlier_statistics.csv and scene_statistics.csv for imbalance.\nBefore assignment, audit source provenance, exact duplicates and near-duplicates across groups. No image-content duplicate scan was run. Names and sizes cannot establish independence.\nIf few independent groups remain, consider leave-one-independent-group-out development CV, with a separately protected confirmation set if feasible. Its compute cost needs a later baseline pilot; feasibility is not established by this audit. Scene-disjoint diagnostics can complement dataset-level generalization but cannot replace it. Small scenes remain reported; do not exclude them post hoc.\nFreeze baseline, data, evaluator, thresholds and conditions once approved; every improvement uses that same version. Never tune on confirmation. Official prose mAA-first assignment remains distinct from participant overlap-first code; no parity claim.\n'''
            (out/'validation_recommendation.md').write_text(review);artifacts.append('validation_recommendation.md')
            status.update(state='AUDITED',statistics_complete=scan_complete and train_result.get('complete',False),
                limitations=['No source-independence or near-duplicate audit','No hidden test labels accessed','Image-reference resolution supports train/dataset/scene/image and test/dataset/image; unresolved is not proof of missing content','Archive extraction not performed'])
    except (OSError,ValueError,csv.Error,UnicodeError,sqlite3.Error) as exc:
        status.update(state='BLOCKED',error_type=type(exc).__name__,message=str(exc))
    save('audit_status.json',status)
    # Explicit allowlist: never archive input files, SQLite indices, or old outputs.
    with zipfile.ZipFile(out/'imc_dataset_audit.zip','w',zipfile.ZIP_DEFLATED) as archive:
        used=0
        included=[]; omitted=[]
        for name in artifacts:
            size=(out/name).stat().st_size
            if size>10*1024**2 or used+size>25*1024**2:
                omitted.append(name);continue
            archive.write(out/name,name);used+=size
            included.append({'file':name,'bytes':size,'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()})
        archive.writestr('archive_manifest.json',json.dumps({'included':included,'omitted_due_to_size_cap':omitted},indent=2))
    return status
