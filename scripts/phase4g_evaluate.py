"""Administrator only. Score after the branch choice and prediction bytes are sealed."""
import argparse,csv,hashlib,io,json,subprocess,sys,zipfile
from pathlib import Path
from scripts.build_ns64_76 import inventory,ordered

def run(dataset,archive,output):
    pins=json.loads(Path("reports/phase4g/protocol_manifest.json").read_bytes())
    pin=pins["datasets"][dataset]
    sha=lambda b:hashlib.sha256(b).hexdigest()
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(set(z.namelist())):
            raise ValueError("duplicate receipt members")
        read=lambda n:json.loads(z.read(n))
        selection=read("selection.json")
        if selection["status"]!="COMPLETE" or selection["protocol"]!=pins["model_bundle"] or selection["worker"]!=pin["worker"]:
            raise ValueError("incomplete or incompatible run")
        selected=selection["selected"]
        assert selected in ("pass_through","trial")
        base=z.read("opaque.csv")
        final=z.read("final.csv")
        assert sha(base)==selection["pass_sha256"]==pin["opaque_baseline"]
        assert sha(final)==selection["final_sha256"]
        assert final==(base if selected=="pass_through" else z.read("feedback/trial.csv"))
        job=read("feedback/job.json")
        assert job["dataset"]==dataset and read("feedback/pass_input.json")["before"]["records"]==job["manifest"]["original_logical"]
        if selected=="trial":
            assert read("decision.json")=={k:selection[k] for k in read("decision.json")}
            assert read("feedback/trial_input.json")["forbidden"]==job["forbidden"]
    original=ordered(inventory(dataset),20261111)
    mapping={f"image_{i:04}.png":n for i,n in enumerate(original)}
    output.mkdir(parents=True,exist_ok=False)
    results={}
    for branch,data in (("pass_through",base),("automatic",final)):
        rows=list(csv.DictReader(io.StringIO(data.decode())))
        assert sorted(r["image"] for r in rows)==sorted(mapping) and all(r["dataset"]==dataset for r in rows)
        for row in rows:row["image"]=mapping[row["image"]]
        pred=output/(branch+"_restored.csv")
        with pred.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows,key=lambda r:r["image"]))
        score=output/(branch+"_metrics.json")
        subprocess.run([sys.executable,"src/baselines/development/evaluate.py",dataset,str(pred),str(score)],check=True)
        results[branch]=json.loads(score.read_bytes())
    a,b=[results[k]["scores"] for k in ("automatic","pass_through")]
    results["delta"]={v:{k:x-b[v][k] for k,x in scores.items()} for v,scores in a.items()}
    (output/"comparison.json").write_text(json.dumps(results,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",choices=["stairs","imc2023_theather_imc2024_church"])
    parser.add_argument("archive",type=Path)
    parser.add_argument("output",type=Path)
    run(**vars(parser.parse_args()))
