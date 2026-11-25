"""Administrator-only development evaluation after sealing the label-free features."""
import csv, json, hashlib, zipfile, io
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

def main():
    out=Path("reports/phase4e")
    data=(out/"features.json").read_bytes()
    assert hashlib.sha256(data).hexdigest()==json.loads((out/"freeze.json").read_bytes())["features"]
    rows=json.loads(data)["edges"]
    admin=Path("reports/phase3c7")
    mapping=json.loads((admin/"private.json").read_bytes())
    restored=(admin/"restored_predictions.csv").read_bytes()
    assert hashlib.sha256(restored).hexdigest()=="6901de90ed8cd8cd8b1ab3dd2aba75a986fa64aeb621724b3857f5e91fa4c1c9"
    historical={r["image"]:r for r in csv.DictReader(io.StringIO(restored.decode()))}
    with zipfile.ZipFile(admin/"worker.zip") as z:
        for r in csv.DictReader(io.StringIO(z.read("opaque.csv").decode())):
            name=mapping[r["image"]]
            assert historical[name]=={**r,"image":name}
    folder=Path("configs/baselines/organizer_v1/development_scoring/imc2023_haiper")
    with (folder/"train_labels.csv").open() as f:
        labels={r["image"]:r["scene"] for r in csv.DictReader(f)}
    assert set(mapping.values())==set(labels)
    groups=defaultdict(list)
    boundaries=defaultdict(list)
    for r in rows:
        a,b=[labels[mapping[n]] for n in r["pair"]]
        kind="outlier_incident" if "outliers" in (a,b) else ("within" if a==b else "cross")
        groups[kind].append(r)
        boundaries[str(r["component_size"])+("_boundary" if r["boundary"] else "_internal")].append(r)
    def stats(rs):
        vals={"inliers":[r["n"] for r in rs],"overlap":[r["overlap"] for r in rs]}
        for k in ("cycle","multiple","clean_multiview","conflict"):
            vals[k]=[r[k]/r["n"] for r in rs]
        return dict(edges=len(rs),links=sum(r["n"] for r in rs),zero_cycle=sum(r["cycle"]==0 for r in rs),
                    quantiles_25_50_75={k:np.quantile(v,[.25,.5,.75]).tolist() for k,v in vals.items()})
    bins={}
    for kind,rs in groups.items():
        for k in sorted({r["n"].bit_length()-1 for r in rs}):
            bins.setdefault(str(k),{})[kind]=stats([r for r in rs if r["n"].bit_length()-1==k])
    result=dict(classes={k:stats(v) for k,v in groups.items()},boundaries={k:stats(v) for k,v in boundaries.items()},log2_bins=bins)
    # Pair each cross edge once with the nearest remaining within edge in log-count
    # order, deterministic with opaque endpoint ties. No threshold selection.
    pool=sorted(groups["within"],key=lambda r:r["pair"])
    matched=[]
    for cross in sorted(groups["cross"],key=lambda r:(r["n"],r["pair"])):
        idx=min(range(len(pool)),key=lambda i:abs(np.log(pool[i]["n"]/cross["n"])))
        within=pool.pop(idx)
        matched.append((within,cross))
    result["count_matched"]={
        "pairs":len(matched),
        "median_abs_log_count_ratio":float(np.median([abs(np.log(w["n"]/c["n"])) for w,c in matched])),
        "median_paired_cycle_delta_within_minus_cross":float(np.median([w["cycle"]/w["n"]-c["cycle"]/c["n"] for w,c in matched])),
        "within_cycle_higher":sum(w["cycle"]/w["n"]>c["cycle"]/c["n"] for w,c in matched)}
    result["class_boundary_counts"]={k:dict(Counter((str(r["component_size"])+("_boundary" if r["boundary"] else "_internal")) for r in rs)) for k,rs in groups.items()}
    (out/"admin_aggregates.json").write_text(json.dumps(result,indent=2))
    print([(k,v["edges"],v["zero_cycle"]) for k,v in result["classes"].items()])
    for k,v in result["boundaries"].items():
        print(k,v["edges"],v["quantiles_25_50_75"]["cycle"][1],v["quantiles_25_50_75"]["conflict"][1])
if __name__=="__main__":
    main()

