"""Offline CPU Phase 2C audit; no predictions or split generation."""
import csv
import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from kaggle_dataset_audit import discover

TRAIN_FIELDS=['dataset','scene','image','rotation_matrix','translation_vector']
SUBMISSION_FIELDS=['image_id']+TRAIN_FIELDS
EXPECTED_THRESHOLD_HASH='5b92db83d510ab19e66bd1ef4c23865245d75b4a981986e6d8eb1aee06fc9fe2'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def csv_rows(path, fields):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=fields:raise ValueError(f'Expected schema {fields}, observed {r.fieldnames}')
        for index,row in enumerate(r):
            if None in row or any(v is None for v in row.values()):raise ValueError(f'Malformed CSV row {index}')
            yield index,row

def write_table(path,fields,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(fields);w.writerows(rows)

def validate_submission(sample, output, visible=None):
    """Preserve strings and row order in a private-to-run mapping, excluded from ZIP."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    issues=Counter();counts=Counter();scenes=Counter();forms=Counter();types=Counter()
    ids=set();keys=set();filenames=defaultdict(set);missing_visible=0;rows=0
    with (out/'sample_row_identity.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['row_index']+SUBMISSION_FIELDS)
        for index,row in csv_rows(sample,SUBMISSION_FIELDS):
            rows+=1;writer.writerow([index]+[row[k] for k in SUBMISSION_FIELDS])
            ident=row['image_id'];key=(row['dataset'],row['image'])
            if not ident.strip():issues['empty_image_id']+=1
            if not all(v.strip() for v in key):issues['empty_dataset_or_image']+=1
            if ident in ids:issues['duplicate_image_id']+=1
            if key in keys:issues['duplicate_dataset_image']+=1
            ids.add(ident);keys.add(key);counts[key[0]]+=1;filenames[Path(key[1]).name].add(key[0])
            scenes[row['scene']]+=1
            for k,v in row.items():
                kind='empty' if v=='' else 'integer_like_string' if re.fullmatch(r'[+-]?\d+',v) else 'string'
                types[k,kind]+=1
            try:
                r=[float(x) for x in row['rotation_matrix'].split(';')];t=[float(x) for x in row['translation_vector'].split(';')]
                if len(r)!=9 or len(t)!=3:raise ValueError()
                forms['all_nan' if all(math.isnan(x) for x in r+t) else 'numeric_vectors']+=1
            except ValueError:forms['malformed_vector_encoding']+=1
            if visible is not None and key not in visible:missing_visible+=1
    result={'source_sha256':sha(sample),'columns':SUBMISSION_FIELDS,'rows':rows,'unique_image_id':len(ids),'unique_dataset_image':len(keys),
            'unique_image_strings':len({i for _,i in keys}),'issues':dict(issues),'per_dataset':dict(counts),
            'column_types':'CSV strings preserved without coercion','observed_lexical_types':{f'{k}:{v}':n for (k,v),n in types.items()},
            'scene_value_counts':dict(scenes),'pose_encoding_counts':dict(forms),'physical_pose_validation_applied':False,
            'placeholder_semantics':'Patterns reported only; numeric values or scene strings do not establish prediction meaning',
            'repeated_basenames_across_datasets':sum(len(v)>1 for v in filenames.values()),
            'sample_rows_without_visible_image':missing_visible if visible is not None else None,
            'visible_test_keys_absent_from_sample':len(visible-keys) if visible is not None else None,
            'visible_coverage_note':'Unmatched rows are not proof of invalid submission or hidden-test size. Visible inputs only.',
            'row_identity_mapping':'sample_row_identity.csv (internal; excluded from portable ZIP)','complete':True,'scorer_acceptance':'UNVERIFIED'}
    write_json(out/'sample_submission_contract.json',result)
    write_table(out/'sample_dataset_coverage.csv',['dataset','sample_rows','visible_test_keys'],[(d,n,sum(x==d for x,_ in visible) if visible is not None else '') for d,n in sorted(counts.items())])
    return result

def threshold_tokens(raw):
    result=[]
    for token in raw.split(';'):
        value=None
        try:
            number=float(token)
            kind='nonfinite' if not math.isfinite(number) else 'zero' if number==0 else 'negative' if number<0 else 'positive'
            value=number if math.isfinite(number) else None
        except ValueError:kind='missing' if not token.strip() else 'unparsed_encoding'
        result.append({'raw':token,'type':kind,'finite_value':value})
    return result

def thresholds(path,out):
    digest=sha(path);rows=[]
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f);columns=reader.fieldnames
        if not {'dataset','scene','thresholds'}.issubset(columns or []):raise ValueError('Unrecognized threshold schema')
        for idx,row in enumerate(reader):
            if row.get('thresholds') is None:raise ValueError('Missing threshold field')
            tokens=threshold_tokens(row['thresholds']);reasons=sorted({t['type'] for t in tokens if t['type']!='positive'})
            rows.append([idx,row.get('dataset'),row.get('scene'),row['thresholds'],json.dumps(tokens),';'.join(reasons),json.dumps(row)])
    write_table(Path(out)/'thresholds_raw_and_parsed.csv',['row_index','dataset','scene','raw_thresholds','parsed_tokens_json','previous_positive_validator_rejection','original_row_json'],rows)
    info={'sha256':digest,'expected_sha256':EXPECTED_THRESHOLD_HASH,'hash_matches_previous':digest==EXPECTED_THRESHOLD_HASH,'columns':columns,'rows':len(rows),
          'interpretation':'UNRESOLVED: preserve raw tokens; participant parser uses float(token) on semicolon-separated strings. A representable float is not proof of official intended semantics.',
          'configuration_created':False,'complete':True}
    write_json(Path(out)/'threshold_source.json',info)
    target=[r for r in rows if r[1]=='imc2023_haiper' and r[2] in ('bike','chairs','fountain')]
    (Path(out)/'threshold_reconciliation.md').write_text('# Threshold inspection\n\n'+json.dumps(info,indent=2)+'\n\nHaiper raw diagnostics (no substituted values):\n'+json.dumps(target,indent=2)+'\n\nA failed positivity check does not prove invalid metadata. Review raw tokens and actual evaluator behavior before creating an executable threshold configuration. The source was not modified.\n')
    return info

def pose_category(row):
    try:
        r=np.array([float(x) for x in row['rotation_matrix'].split(';')]).reshape(3,3);t=np.array([float(x) for x in row['translation_vector'].split(';')]).reshape(3)
        x=np.r_[r.ravel(),t]
        if np.isnan(x).all():return 'fully_nan',False
        if np.isfinite(x).all():
            ok=np.allclose(r.T@r,np.eye(3),atol=1e-3,rtol=0) and abs(np.linalg.det(r)-1)<=1e-3
            return 'fully_finite',bool(ok)
        if np.isnan(x).any() and not np.isinf(x).any():return 'mixed_nan_finite',False
        return 'other_invalid',False
    except ValueError:return 'other_invalid',False

def pose_audit(path,out):
    counts=Counter();bad_rot=Counter();scenes=defaultdict(set);n=0
    for _,row in csv_rows(path,TRAIN_FIELDS):
        n+=1;category,rotation_ok=pose_category(row);flag=row['scene']=='outliers'
        counts[row['dataset'],flag,category]+=1;scenes[row['scene']].add(row['dataset'])
        if category=='fully_finite' and not rotation_ok:bad_rot[row['dataset']]+=1
    write_table(Path(out)/'pose_outlier_crosstab.csv',['dataset','is_outlier','pose_state','count'],[(d,o,s,n) for (d,o,s),n in sorted(counts.items())])
    result={'rows':n,'source_sha256':sha(path),'rotation_atol':1e-3,'rotation_rtol':0,'determinant_abs_tolerance':1e-3,
        'finite_invalid_rotations_by_dataset':dict(bad_rot),'nonoutlier_all_nan':sum(n for (d,o,s),n in counts.items() if not o and s=='fully_nan'),
        'outlier_not_all_nan':sum(n for (d,o,s),n in counts.items() if o and s!='fully_nan'),'complete':True}
    write_json(Path(out)/'pose_integrity.json',result);return scenes

def constrained_groups(datasets, exact_edges):
    parent={d:d for d in datasets}
    def find(d):
        while parent[d]!=d:d=parent[d]
        return d
    for a,b in exact_edges:parent[find(b)]=find(a)
    groups=defaultdict(list)
    for d in sorted(datasets):groups[find(d)].append(d)
    return sorted(groups.values())

def overlap(root,out,seconds=120,max_images=10000,near_distance=5):
    """Train-only byte hashes and 64-bit thumbnail dHash. Resume via versioned cache."""
    start=time.monotonic();deadline=start+seconds;root=Path(root);out=Path(out)
    files=[];scan_complete=True;warnings=[]
    for directory,dirs,names in os.walk(root/'train',followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink()]
        for name in names:
            p=Path(directory)/name
            if p.is_symlink() or p.suffix.lower() not in ('.jpg','.jpeg','.png','.webp','.tif','.tiff'):continue
            if time.monotonic()>deadline or len(files)>=max_images:scan_complete=False;break
            parts=p.relative_to(root/'train').parts
            if len(parts)>=2:files.append((p,parts[0],p.stat().st_size,p.stat().st_mtime_ns))
        if not scan_complete:break
    files.sort(key=lambda x:x[0].as_posix())
    catalog_hash=hashlib.sha256(json.dumps([(str(p.relative_to(root)),d,s,m) for p,d,s,m in files]).encode()).hexdigest()
    cache_path=out/'overlap_cache.json';cache={}
    if cache_path.exists():
        old=json.loads(cache_path.read_text())
        if old.get('catalog_hash')==catalog_hash and old.get('algorithm')=='dhash64-exif-gray-v1':cache=old['entries']
    size_groups=defaultdict(set)
    for p,d,size,m in files:size_groups[size].add(d)
    exact_needed=sum(len(size_groups[s])>1 for _,_,s,_ in files)
    errors=[]
    for p,d,size,m in files:
        if time.monotonic()>deadline:break
        key=p.relative_to(root).as_posix();entry=cache.setdefault(key,{})
        try:
            if len(size_groups[size])>1 and 'sha256' not in entry:
                h=hashlib.sha256()
                with p.open('rb') as f:
                    while True:
                        if time.monotonic()>deadline:raise TimeoutError()
                        chunk=f.read(1024*1024)
                        if not chunk:break
                        h.update(chunk)
                entry['sha256']=h.hexdigest()
            if 'dhash' not in entry:
                with Image.open(p) as im:
                    im.draft('RGB',(128,128));im.thumbnail((128,128))
                    small=ImageOps.exif_transpose(im).convert('L').resize((9,8))
                    arr=np.asarray(small);bits=(arr[:,1:]>arr[:,:-1]).ravel()
                    entry['dhash']=int(''.join('1' if b else '0' for b in bits),2)
        except TimeoutError:break
        except (OSError,ValueError,Image.DecompressionBombError) as exc:errors.append({'path':key,'error_type':type(exc).__name__})
    write_json(cache_path,{'catalog_hash':catalog_hash,'algorithm':'dhash64-exif-gray-v1','entries':cache})
    candidates=[];edges=set();comparisons=0;pair_complete=True
    dataset_counts=Counter(d for _,d,_,_ in files)
    total_pairs=(len(files)**2-sum(n*n for n in dataset_counts.values()))//2
    normalized=lambda p:re.sub(r'[^a-z0-9]','',p.stem.lower())
    for i,(p,a,_,_) in enumerate(files):
        for q,b,_,_ in files[i+1:]:
            if a==b:continue
            if time.monotonic()>deadline or len(candidates)>=20000:pair_complete=False;break
            comparisons+=1;x=cache.get(p.relative_to(root).as_posix(),{});y=cache.get(q.relative_to(root).as_posix(),{})
            kind=None;distance=None
            if x.get('sha256') and x.get('sha256')==y.get('sha256'):kind='exact_bytes';edges.add(tuple(sorted((a,b))))
            elif 'dhash' in x and 'dhash' in y:
                distance=(x['dhash']^y['dhash']).bit_count()
                if distance<=near_distance:kind='perceptual_candidate_UNREVIEWED'
            if kind is None and normalized(p)==normalized(q):kind='normalized_filename_only'
            if kind:candidates.append((a,b,p.relative_to(root).as_posix(),q.relative_to(root).as_posix(),kind,distance))
        if not pair_complete:break
    write_table(out/'dataset_overlap_candidates.csv',['dataset_a','dataset_b','path_a','path_b','evidence','dhash_distance'],candidates)
    info={'inventory_images':len(files),'inventory_complete':scan_complete,'inventory_catalog_sha256':catalog_hash,'exact_hash_candidates':exact_needed,
          'exact_hashes_completed':sum('sha256' in x for x in cache.values()),'thumbnail_hashes_completed':sum('dhash' in x for x in cache.values()),
          'cross_dataset_pairs_compared':comparisons,'total_cross_dataset_pairs':total_pairs,'pair_scan_complete':pair_complete,
          'complete':scan_complete and pair_complete and sum('dhash' in x for x in cache.values())==len(files) and sum('sha256' in x for x in cache.values())==exact_needed,
          'decode_errors':errors,'elapsed_seconds':time.monotonic()-start,'budget_seconds':seconds,'budget_note':'Cooperative time limit; a single image decode/stat may overrun. JPEG draft/thumbnail; one image at a time. No test imagery read.',
          'near_distance':near_distance,'near_matches_manually_reviewed':False,'resume':'Rerun with same mounts/output; cache reused only for identical size/mtime/path catalog. Increase budget for pair stage; it restarts deterministically.',
          'exact_group_constraints':constrained_groups({d for _,d,_,_ in files},edges)}
    write_json(out/'overlap_status.json',info);return info

def run(input_root='/kaggle/input',output='/kaggle/working/imc_contract_audit',input_path=None,seconds=120,max_images=10000):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    if out.is_relative_to(Path(input_root).resolve()):raise ValueError('Output cannot be under input')
    status={'state':'STARTED','split_frozen':False,'warnings':[]}
    names=['environment.json','audit_status.json']
    write_json(out/'environment.json',{'utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'numpy':np.__version__,'network':False,'gpu':False})
    try:
        root,_,_=discover(input_root,input_path);status['root']=str(root)
        # Only visible test paths are enumerated; their image bytes are never read.
        visible=set()
        for p in (root/'test').rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.tif','.tiff'):
                parts=p.relative_to(root/'test').parts
                if len(parts)>=2:visible.add((parts[0],'/'.join(parts[1:])))
        validate_submission(root/'sample_submission.csv',out,visible)
        names+=['sample_submission_contract.json','sample_dataset_coverage.csv']
        thresholds(root/'train_thresholds.csv',out);names+=['thresholds_raw_and_parsed.csv','threshold_source.json','threshold_reconciliation.md']
        scenes=pose_audit(root/'train_labels.csv',out);names+=['pose_outlier_crosstab.csv','pose_integrity.json']
        info=overlap(root,out,seconds,max_images);names+=['dataset_overlap_candidates.csv','overlap_status.json']
        shared={s:sorted(ds) for s,ds in scenes.items() if s!='outliers' and len(ds)>1}
        licenses=[]
        for p in (root/'train').rglob('*LICENSE*'):
            if p.is_file() and len(licenses)<100:
                with p.open(encoding='utf-8',errors='replace') as f:prefix=f.read(65536)
                licenses.append({'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size,'urls_in_first_64KiB':sorted(set(re.findall(r'https?://[^\s<>"\)]+',prefix)))[:20]})
        (out/'source_provenance_review.md').write_text('# Provenance evidence\nShared scene names (not proof of duplication):\n'+json.dumps(shared,indent=2)+'\nLicense/source metadata (URL sample, not a complete attribution audit):\n'+json.dumps(licenses,indent=2)+'\nExact byte matches require grouping. Perceptual candidates require human review of training images in Kaggle; no thumbnails exported. No matches does not establish independent sources.\n')
        (out/'validation_grouping_proposal.md').write_text('# DRAFT grouping; no split frozen\nMust-link groups from observed exact duplicates:\n'+json.dumps(info['exact_group_constraints'],indent=2)+'\nUncertain shared-scene groups:\n'+json.dumps(shared,indent=2)+'\nConservatively keep the two dioscuri datasets together pending provenance review even without byte duplicates. Historical IMC and PhotoTourism naming suggests source families but does not establish independence or mandatory merging. Inspect near candidates before approval. Outlier-heavy domains (heritage, dioscuri/baalshamin, lizard/pond, ETs) must inform representative development/confirmation coverage without model scores. Prefer grouped development folds; assess leave-one-group-out cost later. Scene-disjoint diagnostics are supplementary. Protect confirmation from tuning. Freeze identical baseline, data, evaluator, thresholds and conditions for all improvements only after supervisor approval.\n')
        names+=['source_provenance_review.md','validation_grouping_proposal.md']
        status.update(state='AUDITED',overlap_complete=info['complete'],threshold_interpretation='UNRESOLVED',submission_acceptance='UNVERIFIED')
        if not info['complete']:status['warnings'].append('PARTIAL_OVERLAP_AUDIT')
        status['warnings']+=['NEAR_MATCHES_UNREVIEWED','SOURCE_INDEPENDENCE_NOT_ESTABLISHED','THRESHOLD_SEMANTICS_REQUIRE_REVIEW']
    except (OSError,ValueError,csv.Error) as exc:status.update(state='BLOCKED',error_type=type(exc).__name__,message=str(exc))
    write_json(out/'audit_status.json',status)
    with zipfile.ZipFile(out/'imc_contract_audit.zip','w',zipfile.ZIP_DEFLATED) as z:
        manifest=[];omitted=[];size=0
        for name in names:
            p=out/name
            if p.stat().st_size>10*1024**2 or size+p.stat().st_size>25*1024**2:omitted.append(name);continue
            size+=p.stat().st_size;z.write(p,name);manifest.append({'file':name,'sha256':sha(p),'bytes':p.stat().st_size})
        z.writestr('archive_manifest.json',json.dumps({'included':manifest,'omitted':omitted},indent=2))
    return status
