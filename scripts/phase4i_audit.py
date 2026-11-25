"""Verify reusable development assets and run the sole historical diagnostic (no score access)."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile
from src.novel_model.final_candidate_v1 import file_sha,partition
from src.novel_model.final_candidate_v1.workflow import save,read
from contextlib import closing
from src.novel_model.geometry_capture_v1.database import logical,read_only

def run():
    out=Path("reports/phase4i")
    plan=read(Path("reports/phase4f/plan.json"))
    db=Path(plan["parent"]+".db")
    with closing(read_only(db)) as conn:
        assert logical(conn)==plan["records"]
    t=time.perf_counter()
    proposal,ids=partition(db,plan["parent"],sorted(plan["image_hashes"]))
    assert list(map(list,proposal["sides"]))==plan["sides"] and ids==sorted(plan["forbidden_ids"])
    diagnostic=dict(status="PASS",task="proposal compatibility",parent=file_sha(db),seconds=time.perf_counter()-t,
                    pairs=len(plan["pairs"]),removed_verified=len(proposal["cross"]),forbidden_ids=len(ids),
                    side_sizes=list(map(len,proposal["sides"])),indexed_original=False,indexed_trial=False,
                    real_final_candidate_score=None)
    save(out/"haiper_diagnostic.json",diagnostic)
    history=[]
    for entry in read(Path("reports/ns64_full/completed_archive_index.json")):
        ds=entry["dataset"]
        with zipfile.ZipFile(entry["archive"]) as z:
            run=json.loads(z.read("run.json"))
            usage=json.loads(z.read("resources.json"))
            stats=json.loads(z.read("prediction.json"))
        score=read(Path("reports/phase3c8/four_pilot_merge")/ds/"evaluation.json")["scores"]
        org=read(Path("experiments/baseline/organizer_baseline_v1_dev_full.json"))["datasets"][ds]
        history.append(dict(dataset=ds,organizer=org["scores"]["participant"],ns64=score["participant"],
            archive=entry["sha256"],seconds=run["seconds"],registration=stats,
            resources={k:v for k,v in usage.items() if "peak" in k}))
    branches={}
    for key in ("pass_through","active"):
        scores=read(Path("reports/phase4f")/(key+"_evaluation.json"))["scores"]
        stats=read(Path("reports/phase4f")/(key+"_return")/"result.json")
        branches[key]=dict(scores=scores,registered=stats["registered"],seconds=stats["seconds"],rss_kib=stats["peak_rss_kib"])
    save(out/"observed.json",dict(historical_design_sets=history,haiper_fixed_comparison=branches,
        v2=read(Path("reports/phase4h/decision.json")),prospective_groups_completed=0,final_scores={}))
    files=[Path(x["archive"]) for x in read(Path("reports/ns64_full/completed_archive_index.json"))]
    files+=list(Path(".").glob("phase4f_*_return.zip"))+list(Path(".").glob("phase4g1_*_return.zip"))
    files+=[p for d in Path(".").glob("phase4g_*outputs") for p in d.rglob("*.zip")]
    rows=[]
    expected={x["archive"]:x["sha256"] for x in read(Path("reports/ns64_full/completed_archive_index.json"))}
    for p in files:
        with zipfile.ZipFile(p) as z:
            hits=[n for n in z.namelist() if Path(n).suffix.lower() in (".db",".sqlite",".npz",".h5") or
                  Path(n).stem.lower() in ("points3d","cameras","images") and Path(n).suffix in (".txt",".bin")]
        digest=file_sha(p)
        if p.as_posix() in expected: assert expected[p.as_posix()]==digest
        rows.append(dict(file=p.as_posix(),sha256=digest,numerical_members=hits))
    freeze=read(out/"freeze.json")
    availability={}
    for ds in freeze["order"]:
        path=Path(ds+"_opaque.zip")
        entry=dict(present=path.exists(),parent_available=False)
        if path.is_file():
            with zipfile.ZipFile(path) as z:
                names=sorted(z.namelist())
                assert names==[f"image_{i:04d}.png" for i in range(freeze["datasets"][ds]["images"])]
                perm=hashlib.sha256("\n".join(hashlib.sha256(z.read(n)).hexdigest() for n in names).encode()).hexdigest()
                assert perm==freeze["datasets"][ds]["permutation_sha256"]
            entry.update(sha256=file_sha(path),permutation=perm,bytes=path.stat().st_size)
        availability[ds]=entry
    save(out/"inventory.json",dict(checked=rows,historical_workspace_search=read(Path("reports/phase4h/search.json")),
        numerical_parent=dict(sha256=file_sha(db),bytes=db.stat().st_size,dataset="imc2023_haiper"),
        independent_inputs=availability,local_nvidia_smi=shutil.which("nvidia-smi"),external_session_verified=False))

if __name__=="__main__":run()
