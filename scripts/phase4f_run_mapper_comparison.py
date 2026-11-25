"""Mapper-only worker. Run from the clean Colab bundle as a Python module."""
import csv,json,sys,time,resource
from contextlib import closing
from pathlib import Path
import numpy as np
from src.novel_model.reconstruction_feedback_v1 import make_copy,inspect,read_only,file_sha
from src.novel_model.geometry_capture_v1.database import save

def run(branch,parent,images,out):
    import pycolmap, torch
    planpath=Path("plan.json")
    plan=json.loads(planpath.read_bytes())
    assert sys.version_info[:3]==(3,11,11) and pycolmap.__version__=="3.11.1" and torch.cuda.get_device_name(0)=="Tesla T4"
    assert file_sha(Path("/content/ns64_clean/source/conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-022-c.yaml"))==plan["config"]
    if branch not in ("pass_through","active"):
        raise ValueError(branch)
    if branch=="active":
        gate=json.loads(Path("parity_gate.json").read_bytes())
        assert gate["pass"] and gate["plan"]==file_sha(planpath) and gate["prediction"]==file_sha(Path("pass_through/prediction.csv"))
    assert {p.name:file_sha(p) for p in images.iterdir()}==plan["image_hashes"]
    out.mkdir(exist_ok=False)
    target=out/"input.db"
    audit=make_copy(parent,target,plan["parent"],plan["forbidden_ids"] if branch=="active" else [])
    assert audit["before"]["records"]==plan["records"]
    save(out/"input.json",audit)
    options=pycolmap.IncrementalPipelineOptions()
    options.max_num_models=25
    options.min_model_size=3
    options.num_threads=1
    assert options.summary()==plan["options"],"Mapper options drift"
    dest=out/"models"
    dest.mkdir()
    start=time.monotonic()
    maps=pycolmap.incremental_mapping(database_path=str(target), image_path=str(images),output_path=str(dest),options=options)
    elapsed=time.monotonic()-start
    poses={}
    stats=[]
    for k,model in sorted(maps.items()):
        stats.append(dict(model=k,cameras=model.num_reg_images(),points=model.num_points3D(),
                          reprojection=model.compute_mean_reprojection_error(),track_length=model.compute_mean_track_length()))
        for i in model.reg_image_ids():
            im=model.images[i]
            if im.name in poses:
                raise ValueError("ambiguous model membership")
            pose=im.cam_from_world
            if callable(pose):
                pose=pose()
            poses[im.name]=(f"cluster{k}",pose.rotation.matrix(),pose.translation)
    assert set(poses)<=set(plan["image_hashes"])
    with (out/"prediction.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["dataset","scene","image","rotation_matrix","translation_vector"])
        for n in sorted(plan["image_hashes"]):
            scene,r,t=poses.get(n,("outliers",np.full(9,np.nan),np.full(3,np.nan)))
            w.writerow(["imc2023_haiper",scene,n,";".join(map(str,r.ravel())),";".join(map(str,t.ravel()))])
    with closing(read_only(target)) as db:
        assert inspect(db)==audit["after"],"database changed during mapping"
    save(out/"result.json",dict(branch=branch,plan=file_sha(planpath),prediction=file_sha(out/"prediction.csv"),
         models=stats,registered=len(poses),outliers=sorted(set(plan["image_hashes"])-poses.keys()),seconds=elapsed,
         peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,options=options.summary()))
if __name__=="__main__":
    branch,parent,images=sys.argv[1:]
    run(branch,Path(parent),Path(images),Path(branch))

