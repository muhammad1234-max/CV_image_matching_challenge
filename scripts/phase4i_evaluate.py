"""Administrative scoring only after all four stage archives and download digests are returned."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from scripts.build_ns64_76 import inventory,ordered
from src.novel_model.final_candidate_v1 import POLICY,file_sha,partition
from src.novel_model.final_candidate_v1.workflow import STAGES,quality,read,save,select,verify
from src.novel_model.final_candidate_v1.session import unpack
from src.novel_model.final_candidate_v1.submission import COLUMNS,validate
from src.novel_model.reconstruction_feedback_v2 import decide as legacy_decide

FREEZE=Path("reports/phase4i/freeze.json")


def score(ds,path,out):
    subprocess.run([sys.executable,"src/baselines/development/evaluate.py",ds,str(path),str(out)],check=True)
    return read(out)


def receive(ds, downloads, pinned, root, freeze):
    spec=freeze["datasets"][ds]
    ident=dict(model=freeze["version"],seed=POLICY["seed"],dataset=ds,protocol=freeze["protocol"],
               worker=spec["worker_sha256"])
    parents={}
    for stage in STAGES:
        dest=root/stage
        dest.mkdir()
        unpack(downloads/(ds+"_"+stage+".zip"),dest,pinned[stage])
        parents[stage]=verify(dest,stage,ident,dict(parents))
    replay=root/"replay"
    replay.mkdir()
    selection=select(root,replay)
    if (file_sha(replay/"final.csv")!=file_sha(root/"decision/final.csv")
            or read(replay/"selection.json")!=read(root/"decision/selection.json")):
        raise ValueError("independent decision replay")
    identity=read(root/"capture/identity.json")
    experiment=read(Path("configs/baselines/ns64_isolated_v1/experiment.json"))
    if (identity["benchmark"]!=experiment["benchmark"] or identity["config_sha256"]!=experiment["configuration_sha256"]
        or identity["permutation_sha256"]!=spec["permutation_sha256"] or identity["seed"]!=POLICY["seed"]):
        raise ValueError("reference pipeline identity")
    if (root/"capture/job.json").exists():
        job=read(root/"capture/job.json")
        names=sorted(job["image_hashes"])
        sequence="\n".join(job["image_hashes"][n] for n in names)
        if hashlib.sha256(sequence.encode()).hexdigest()!=spec["permutation_sha256"]: raise ValueError("images")
        proposal,mask=partition(root/"capture/parent.db",job["parent"]["backup_sha256"],names)
        base,trial=(read(root/k/"branch.json") for k in ("original","trial"))
        if base["valid_export"] and json.loads(json.dumps(proposal))!=base["proposal"]:
            raise ValueError("proposal mismatch")
        for stage,excluded in (("original",[]),("trial",mask)):
            if stage=="trial" and not trial["available"]: continue
            audit=read(root/stage/"input.json")
            if audit["forbidden"]!=excluded or audit["before"]["records"]!=job["parent"]["original_logical"]:
                raise ValueError("branch input mismatch")
    return selection


def alternatives(root):
    original=root/"original/prediction.csv"
    trial=root/"trial/prediction.csv"
    choices={"ns64_pass":original,"automatic":root/"decision/final.csv"}
    b,t=(read(root/k/"branch.json") for k in ("original","trial"))
    if not t["available"]:
        return choices
    choices["ungated_proposal"]=trial
    if b["valid_export"] and t["valid_export"]:
        sides=b["proposal"]["sides"]
        pairs=[dict(models=v["models"],errors=quality(root/k)[1]) for v,k in ((b,"original"),(t,"trial"))]
        choice=legacy_decide(sides,*pairs)
        choices["legacy_feedback"]=trial if choice["selected"]=="trial" else original
        # Matched-index cost alone, without coverage and difficult-feature safeguards.
        import numpy as np
        gs,es=quality(root/"original")
        gt,et=quality(root/"trial")
        same=set(es)==set(et) and all(es[n].shape==et[n].shape for n in es)
        cost={}
        if same:
            for n,a in es.items():
                m=np.isfinite(a)
                shared=m & np.isfinite(et[n])
                cap=POLICY["pixel_cap"]
                if m.any():
                    cost[n]=float(((np.minimum(et[n][shared],cap)-np.minimum(a[shared],cap)).sum()
                                  +cap*(m.sum()-shared.sum()))/m.sum())
        gain=all(all(n in cost for n in side) and np.median([cost[n] for n in side]) < -POLICY["pixel_margin"]
                 for side in sides)
        choices["paired_cost_alone"]=trial if gain else original
    return choices


def evaluate(ds,downloads,pins,out):
    freeze=read(FREEZE)
    for path,digest in freeze["files"].items():
        if file_sha(path)!=digest: raise ValueError("frozen dependency drift "+path)
    local=read(Path("reports/phase4i/inference_pins.json"))
    if file_sha(Path("reports/phase4i/inference_pins.json"))!=freeze["protocol"]:
        raise ValueError("inference manifest drift")
    for name,pin in local.items():
        if name.startswith(("src/","configs/")) and file_sha(name)!=pin:
            raise ValueError("candidate drift: "+name)
    spec=freeze["datasets"][ds]
    registry=read(Path("experiments/baseline/organizer_baseline_v1_dev_full.json"))["datasets"][ds]
    org=Path(spec["organizer_prediction"])
    if file_sha(org)!=spec["organizer_sha256"]: raise ValueError("organizer bytes")
    if out.exists() and any(p.name!="failures" for p in out.iterdir()): raise FileExistsError(out)
    with tempfile.TemporaryDirectory(dir=downloads) as directory:
        root=Path(directory)
        sel=receive(ds,downloads,read(pins),root,freeze)
        choices=alternatives(root)
        mapping={f"image_{i:04d}.png":n for i,n in enumerate(ordered(inventory(ds),POLICY["seed"]))}
        out.mkdir(parents=True,exist_ok=True)
        results={"organizer":score(ds,org,out/"organizer.json")}
        if results["organizer"]["scores"]!=registry["scores"]: raise ValueError("organizer score drift")
        for method,path in choices.items():
            with path.open(newline="") as stream: rows=list(csv.DictReader(stream))
            validate(rows)
            if sorted(r["image"] for r in rows)!=sorted(mapping) or any(r["dataset"]!=ds for r in rows):
                raise ValueError("coverage")
            for row in rows:
                if row["image_id"]!=str(int(row["image"][6:10])): raise ValueError("opaque image ID mismatch")
                row["image"]=mapping[row["image"]]
                del row["image_id"]
            pred=out/(method+".csv")
            with pred.open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=COLUMNS[1:])
                writer.writeheader()
                writer.writerows(sorted(rows,key=lambda r:r["image"]))
            results[method]=score(ds,pred,out/(method+".json"))
        timings={k:read(root/k/"time.json")["seconds"] for k in STAGES}
        usage={}
        for k in STAGES:
            p=root/k/"heartbeat.jsonl"
            if p.is_file():
                rows=[json.loads(line) for line in p.read_text().splitlines()]
                usage[k]={key:max((r["totals"][key] for r in rows),default=None) for key in ("rss","pss","uss")}
                usage[k]["device_peak_used_mib"]=max((u for r in rows for _,_,u in r["gpu"]),default=None)
        deltas={ref:{k:v-results[ref]["scores"]["participant"][k] for k,v in
                     results["automatic"]["scores"]["participant"].items()} for ref in ("ns64_pass","organizer")}
        def endpoints(v):
            diag=v["diagnostics"]
            return dict(registered=len(mapping)-v["predicted_outliers"],outliers=v["predicted_outliers"],
                        false_outliers=diag["false_outliers"],missed_outliers=diag["missed_outliers"],
                        merged_clusters=len(diag["merged_clusters"]),split_scenes=len(diag["split_scenes"]))
        metrics={k:endpoints(v) for k,v in results.items()}
        for ref in deltas:
            deltas[ref].update({k:x-metrics[ref][k] for k,x in metrics["automatic"].items()})
        save(out/"comparison.json",dict(dataset=ds,group=spec["group"],freeze=file_sha(FREEZE),action=sel["action"],
            reason=sel["reason"],paired_deltas=deltas,metrics=results,counts=metrics,images=len(mapping),
            staged_original_seconds=timings["capture"]+timings["original"],automatic_seconds=sum(timings.values()),
            registered={k:len(mapping)-v["predicted_outliers"] for k,v in results.items()},
            unavailable=sorted({"ungated_proposal","legacy_feedback","paired_cost_alone"}-choices.keys()),
            stage_seconds=timings,usage=usage,trial=read(root/"trial/branch.json"),
            organizer_usage={k:registry[k] for k in ("runtime_by_stage","cpu_peak_rss_bytes","gpu_peak_allocated_bytes","failure_count")}))


def record_failure(ds, archive, pin, output):
    freeze=read(FREEZE)
    if file_sha(archive)!=pin: raise ValueError("download hash mismatch")
    with zipfile.ZipFile(archive) as z:
        failure=json.loads(z.read("failure.json"))
    expected=dict(model=freeze["version"],dataset=ds,seed=POLICY["seed"],protocol=freeze["protocol"],
                  worker=freeze["datasets"][ds]["worker_sha256"])
    if failure["identity"]!=expected or failure["stage"] not in STAGES or not failure["error"]:
        raise ValueError("foreign or invalid failure")
    folder=output/ds/"failures"
    folder.mkdir(exist_ok=True,parents=True)
    save(folder/(pin+".json"),dict(dataset=ds,freeze=file_sha(FREEZE),archive_sha256=pin,**failure))


def aggregate(folder):
    rows=[read(folder/ds/"comparison.json") for ds in POLICY["development"] if (folder/ds/"comparison.json").is_file()]
    if any(r["freeze"]!=file_sha(FREEZE) or r["group"]!=POLICY["development"][r["dataset"]] for r in rows):
        raise ValueError("mixed protocol")
    failures=[read(p) for ds in POLICY["development"] for p in sorted((folder/ds/"failures").glob("*.json"))]
    if any(f["freeze"]!=file_sha(FREEZE) for f in failures): raise ValueError("foreign failure record")
    return dict(completed=[r["dataset"] for r in rows],planned=list(POLICY["development"]),failures=failures,
        missing=sorted(set(POLICY["development"])-{r["dataset"] for r in rows}),
        equal_group_mean={ref:{key:sum(r["paired_deltas"][ref][key] for r in rows)/len(rows)
                              for key in rows[0]["paired_deltas"][ref]} for ref in ("ns64_pass","organizer")} if rows else {})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("dataset",choices=POLICY["development"])
    p.add_argument("downloads",type=Path)
    p.add_argument("pins",type=Path)
    p.add_argument("results",type=Path)
    p.add_argument("--failure",help="SHA-256 for failure ZIP supplied as downloads; the pins argument is unused")
    args=p.parse_args()
    if args.failure:
        record_failure(args.dataset,args.downloads,args.failure,args.results)
    else:
        evaluate(args.dataset,args.downloads,args.pins,args.results/args.dataset)
    save(args.results/"aggregate.json",aggregate(args.results))
