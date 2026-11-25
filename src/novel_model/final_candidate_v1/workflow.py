"""Linux mapper stages. Only opaque images, pinned source and earlier receipts enter here."""
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from contextlib import closing
from types import SimpleNamespace
import numpy as np

from . import POLICY, decide, export_observations, file_sha, partition
from ..geometry_capture_v1.database import read_only, save, snapshot
from ..reconstruction_feedback_v1 import inspect, make_copy

STAGES = ("capture","original","trial","decision")


def read(path):
    return json.loads(path.read_bytes())


def files_in(folder):
    return {p.relative_to(folder).as_posix():file_sha(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p != folder/"seal.json"}


def seal(folder, stage, identity, parents):
    save(folder/"seal.json", dict(stage=stage,identity=identity,parents=parents,files=files_in(folder)))


def verify(folder, stage, identity, parents):
    receipt = read(folder/"seal.json")
    if receipt != dict(stage=stage,identity=identity,parents=parents,files=files_in(folder)):
        raise ValueError(f"invalid {stage} checkpoint")
    return file_sha(folder/"seal.json")


def capture(kwargs, folder):
    """Hook at the exact first call to the published mapper, following verification and pruning."""
    import pycolmap
    target = folder/"parent.db"
    records = snapshot(Path(kwargs["database_path"]),target)
    options = kwargs["options"]
    reference = mapper_options(pycolmap)
    if options.summary() != reference.summary():
        raise ValueError("published mapper parameters differ")
    ids = sorted(Path(kwargs["image_path"]).glob("*.png"))
    save(folder/"job.json",dict(parent=records,options=options.summary(),
         image_hashes={p.name:file_sha(p) for p in ids}))


def mapper_options(module):
    options = module.IncrementalPipelineOptions()
    options.num_threads = 1
    options.min_model_size = 3
    options.max_num_models = 25
    return options


def write_prediction(maps, ds, names, images, dest):
    """Use the pinned published reconstruction and clustering/pose serializer."""
    import pandas as pd
    from data import IMC2025TestData
    from colmap import get_reconstructions
    from pipelines.common import results_to_submission_df
    frame = pd.DataFrame([dict(image_id=str(i),dataset=ds,scene="cluster0",image=n,
                               rotation_matrix="",translation_vector="") for i,n in enumerate(names)])
    scene = SimpleNamespace(dataset=ds,scene="cluster0",image_paths=[images/n for n in names],
                            data_schema=IMC2025TestData(frame,data_root_dir=images.parent.parent))
    out,_ = get_reconstructions(maps,scene,None,None)
    table = results_to_submission_df({ds:{"cluster0":out}},schema="imc2025")
    if sorted(table.image) != names or set(table.dataset) != {ds}:
        raise ValueError("published prediction coverage")
    from .submission import validate
    validate(table.to_dict("records"))
    table.to_csv(dest,index=False)


def map_stage(root, stage, dest, ds, images):
    import pycolmap
    if (root/"capture/no_mapper.json").exists():
        if stage == "original":
            from .submission import COLUMNS,validate
            with (root/"capture/opaque.csv").open(newline="") as stream:
                rows=list(csv.DictReader(stream))
            for row in rows:
                row["image_id"]=str(int(row["image"][6:10]))
            validate(rows)
            with (dest/"prediction.csv").open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        save(dest/"branch.json",dict(available=stage=="original",valid_export=False,groups=[],proposal=None,
                                    reason="published pipeline returned before mapping"))
        return
    job = read(root/"capture/job.json")
    names = sorted(job["image_hashes"])
    if {p.name:file_sha(p) for p in images.iterdir()} != job["image_hashes"]:
        raise ValueError("mapper image bytes differ")
    parent = root/"capture/parent.db"
    digest = job["parent"]["backup_sha256"]
    mask = []
    if stage == "trial":
        original = read(root/"original/branch.json")
        proposal = original["proposal"]
        groups = original["groups"]
        if (not original["valid_export"] or not proposal or
            sorted(sum(proposal["sides"],[])) not in [sorted(g) for g in groups]):
            save(dest/"branch.json",dict(available=False,reason="no eligible fully registered component or original export"))
            return
        repeated,mask = partition(parent,digest,names)
        if json.loads(json.dumps(repeated)) != proposal:
            raise ValueError("proposal replay disagreement")
    with tempfile.TemporaryDirectory(dir=root) as scratch:
        scratch = Path(scratch)
        # The legacy backup/audit functions can create SQLite WAL sidecars on read.
        # Work only on a verified scratch copy of the checkpoint.
        scratch_parent = scratch/"sealed.db"
        shutil.copyfile(parent,scratch_parent)
        parent = scratch_parent
        target = scratch/"mapping.db"
        audit = make_copy(parent,target,digest,mask)
        if audit["before"]["records"] != job["parent"]["original_logical"]:
            raise ValueError("database logical records")
        save(dest/"input.json",audit)
        options = mapper_options(pycolmap)
        if options.summary() != job["options"]:
            raise ValueError("mapper options drift")
        models = scratch/"models"
        models.mkdir()
        maps = pycolmap.incremental_mapping(database_path=str(target),image_path=str(images),
                                             output_path=str(models),options=options)
        with closing(read_only(target)) as db:
            if inspect(db) != audit["after"]:
                raise ValueError("mapper database changed")
        write_prediction(maps,ds,names,images,dest/"prediction.csv")
        proposal = None
        try:
            if stage == "original":
                proposal,_ = partition(parent,digest,names)
            groups,errors = export_observations(maps,parent)
            np.savez_compressed(dest/"errors.npz",**errors)
            counts = [dict(images=sorted(m.images[i].name for i in m.reg_image_ids()),points=m.num_points3D())
                      for m in maps.values()]
            summary = dict(valid_export=True,groups=groups,models=counts)
        except (ValueError,IndexError,KeyError,RuntimeError) as error:
            summary = dict(valid_export=False,groups=[],export_error=str(error))
        save(dest/"branch.json",dict(available=True,proposal=proposal,**summary))


def quality(folder):
    summary = read(folder/"branch.json")
    with np.load(folder/"errors.npz",allow_pickle=False) as data:
        return summary["groups"], {n:data[n] for n in data}


def select(root, dest):
    base, variant = (root/p for p in ("original","trial"))
    original, trial = read(base/"branch.json"), read(variant/"branch.json")
    action = dict(action="ABSTAIN",selected="original",per_image={})
    if not trial["available"] or not original["valid_export"] or not trial.get("valid_export"):
        action["reason"] = trial.get("reason") or "indexed export unavailable"
    else:
        try:
            action = decide(original["proposal"]["sides"],quality(base),quality(variant))
        except ValueError as error:
            action["reason"] = "invalid indexed evidence: "+str(error)
    selected = root/action["selected"]/"prediction.csv"
    shutil.copyfile(selected,dest/"final.csv")
    save(dest/"selection.json",dict(**action,final_sha256=file_sha(selected),
        original_sha256=file_sha(base/"prediction.csv")))
    return action


def main():
    stage, ds, root, dest, images = sys.argv[1:]
    if stage not in ("original","trial","decision") or ds not in POLICY["development"]:
        raise ValueError("stage or development dataset")
    root,dest,images = map(Path,(root,dest,images))
    sys.path.insert(0,"/content/ns64_clean/source")
    if stage == "decision":
        select(root,dest)
    else:
        map_stage(root,stage,dest,ds,images)


if __name__ == "__main__":
    main()
