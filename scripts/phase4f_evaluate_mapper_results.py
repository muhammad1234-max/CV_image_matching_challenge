"""Administrative evaluation after opaque branch export."""
import json,sys,zipfile,csv,io
from pathlib import Path
from scripts.compare_ns64_mapper_parity import evaluate
from src.novel_model.geometry_capture_v1.parity import compare,parse
from src.novel_model.geometry_capture_v1.database import save,file_sha

def evaluate_return(folder):
    root=Path("reports/phase4f")
    p=root/"plan.json"
    plan=json.loads(p.read_bytes())
    record=json.loads((folder/"result.json").read_bytes())
    audit=json.loads((folder/"input.json").read_bytes())
    pred=folder/"prediction.csv"
    assert record["prediction"]==file_sha(pred) and record["plan"]==file_sha(p) and record["options"]==plan["options"]
    assert audit["before"]["records"]==plan["records"]
    name=record["branch"]
    assert name in ("active","pass_through")
    if name=="pass_through":
        assert audit["after"]==audit["before"] and not audit["forbidden"]
    else:
        assert json.loads((root/"parity_gate.json").read_bytes())["pass"] and audit["forbidden"]==plan["forbidden_ids"]
    rows=parse(pred.read_bytes())
    ref=(root/"reference.csv").read_bytes()
    assert file_sha(root/"reference.csv")==plan["reference_sha256"]
    mapping=json.loads(Path("reports/phase3c7/private.json").read_bytes())
    hist=Path("reports/phase3c7/restored_predictions.csv")
    assert file_sha(hist)=="6901de90ed8cd8cd8b1ab3dd2aba75a986fa64aeb621724b3857f5e91fa4c1c9"
    restored={r["image"]:r for r in csv.DictReader(io.StringIO(hist.read_text()))}
    with zipfile.ZipFile("reports/phase3c7/worker.zip") as z:
        for row in csv.DictReader(io.StringIO(z.read("opaque.csv").decode())):
            n=mapping[row["image"]]
            assert restored[n]=={**row,"image":n}
    scores=evaluate(pred.read_bytes(),mapping,root,name)
    parity=compare(ref,pred.read_bytes())
    historical=json.loads(Path("reports/phase4c/preflight.json").read_bytes())["reference_scores"]
    delta={v:{k:x-historical[v][k] for k,x in vals.items()} for v,vals in scores.items()}
    save(root/(name+"_analysis.json"),dict(scores=scores,delta_from_historical=delta,poses=parity))
    if name=="pass_through":
        ok=parity["pose_equivalent"] and all(abs(x)<=1e-12 for vals in delta.values() for x in vals.values())
        save(root/"parity_gate.json",{"pass":ok,"plan":file_sha(p),"prediction":file_sha(pred)})
if __name__=="__main__":
    evaluate_return(Path(sys.argv[1]))
