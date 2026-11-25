"""Frozen Linux worker adapter for the published Haiper-independent ns64 entry point."""
from contextlib import closing
import csv
import json
from pathlib import Path
import time
import numpy as np
from ..geometry_capture_v1.database import inventory, snapshot, save, file_sha, read_only, MAX_ID
from ..geometry_graph_v1.schemas import Edge, Graph
from ..reconstruction_feedback_v1 import make_copy, inspect
from . import candidate, decide, POLICY


def capture(mapper, work, dataset, **kwargs):
    import pycolmap
    root=work/"feedback"
    root.mkdir()
    names=[f"image_{i:04d}.png" for i in range(POLICY["authorized"][dataset])]
    parent=root/"parent.db"
    manifest=snapshot(Path(kwargs["database_path"]),parent)
    images,_,_,pairs,_=inventory(parent,names)
    ids={v["name"]:k for k,v in images.items()}
    edges=[]
    for (i,j),p in pairs.items():
        count=len(p.get("verified",()))
        if count:
            a,b=sorted((images[i]["name"],images[j]["name"]))
            edges.append(Edge(a,b,count,None))
    graph=Graph(tuple(names),tuple(sorted(edges,key=lambda e:e.pair)),{},"","verified_counts")
    proposal=candidate(graph)
    options=kwargs["options"]
    defaults=pycolmap.IncrementalPipelineOptions().todict()
    def differences(actual,base):
        diff={}
        for key,value in actual.items():
            if isinstance(value,dict):
                nested=differences(value,base[key])
                if nested:
                    diff[key]=nested
            elif value!=base[key]:
                diff[key]=value
        return diff
    forbidden=[]
    if proposal:
        for a in proposal["sides"][0]:
            for b in proposal["sides"][1]:
                i,j=sorted((ids[a],ids[b]))
                forbidden.append(i*MAX_ID+j)
    job=dict(dataset=dataset,manifest=manifest,names=names,proposal=proposal,forbidden=sorted(forbidden),
             image_path=str(kwargs["image_path"]),options=options.summary(),overrides=differences(options.todict(),defaults),
             image_hashes={n:file_sha(Path(kwargs["image_path"])/n) for n in names},pairs=[e.pair for e in edges])
    save(root/"job.json",job)
    target=root/"pass.db"
    audit=make_copy(parent,target,manifest["backup_sha256"])
    save(root/"pass_input.json",audit)
    dest=root/"pass_models"
    dest.mkdir()
    start=time.monotonic()
    maps=mapper(**{**kwargs,"database_path":str(target),"output_path":str(dest)})
    with closing(read_only(target)) as db:
        assert inspect(db)==audit["after"]
    for k,model in maps.items():
        model_dir=dest/str(k)
        model_dir.mkdir(exist_ok=True)
        model.write(str(model_dir))
    save(root/"pass_mapper.json",dict(seconds=time.monotonic()-start,models=list(maps)))
    return maps


def quality(models):
    errors={}
    groups=[]
    for model in models.values():
        names=[]
        for i in model.reg_image_ids():
            image=model.images[i]
            if image.name in errors:
                raise ValueError("overlapping reconstruction models")
            names.append(image.name)
            residual=np.full(len(image.points2D),np.nan)
            for k,obs in enumerate(image.points2D):
                if obs.has_point3D():
                    xy=image.project_point(model.points3D[obs.point3D_id].xyz)
                    if xy is None or not np.isfinite(xy).all():
                        raise ValueError("invalid projected feature")
                    residual[k]=np.linalg.norm(xy-obs.xy)
            errors[image.name]=residual
        groups.append(dict(images=sorted(names),points=model.num_points3D(),
                           mean_track_length=model.compute_mean_track_length(),mean_reprojection=model.compute_mean_reprojection_error()))
    return dict(errors=errors,models=groups)


def run_trial(work):
    import pycolmap
    import resource
    root=work/"feedback"
    job=json.loads((root/"job.json").read_bytes())
    proposal=job["proposal"]
    parent=root/"parent.db"
    assert file_sha(parent)==job["manifest"]["backup_sha256"]
    assert {n:file_sha(Path(job["image_path"])/n) for n in job["names"]}==job["image_hashes"]
    maps={int(p.name):pycolmap.Reconstruction(str(p)) for p in (root/"pass_models").iterdir() if p.is_dir()}
    baseline=quality(maps)
    save(root/"baseline_quality.json",baseline["models"])
    if not proposal or not any(set(m["images"])==set(sum((list(x) for x in proposal["sides"]),[])) for m in baseline["models"]):
        return dict(selected="pass_through",reason="no eligible fully registered component")
    target=root/"trial.db"
    audit=make_copy(parent,target,job["manifest"]["backup_sha256"],job["forbidden"])
    assert audit["before"]["records"]==job["manifest"]["original_logical"]
    save(root/"trial_input.json",audit)
    options=pycolmap.IncrementalPipelineOptions()
    options.mergedict(job["overrides"])
    assert options.summary()==job["options"]
    dest=root/"trial_models"
    dest.mkdir()
    start=time.monotonic()
    maps=pycolmap.incremental_mapping(database_path=str(target),image_path=job["image_path"],output_path=str(dest),options=options)
    with closing(read_only(target)) as db:
        assert inspect(db)==audit["after"]
    trial=quality(maps)
    save(root/"trial_quality.json",dict(models=trial["models"],seconds=time.monotonic()-start,
                                        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    poses={}
    for k,m in maps.items():
        for i in m.reg_image_ids():
            im=m.images[i]
            poses[im.name]=(f"cluster{k}",im.cam_from_world.rotation.matrix(),im.cam_from_world.translation)
    with (root/"trial.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["dataset","scene","image","rotation_matrix","translation_vector"])
        for n in job["names"]:
            scene,r,t=poses.get(n,("outliers",np.full(9,np.nan),np.full(3,np.nan)))
            w.writerow([job["dataset"],scene,n,";".join(map(str,r.ravel())),";".join(map(str,t.ravel()))])
    return decide(proposal["sides"],baseline,trial)

