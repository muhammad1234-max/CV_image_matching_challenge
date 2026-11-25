"""Run as python -m scripts.phase4e_capture on the frozen Phase 4C parent."""
import json, zipfile, hashlib, time
from pathlib import Path
import psutil,yaml,numpy as np
from src.novel_model.geometry_capture_v1.database import inventory,file_sha
from src.novel_model.track_consistency_v1 import analyze

def main():
    start=time.perf_counter()
    cpu=time.process_time()
    cfg=yaml.safe_load(Path("configs/novel_model/track_consistency_v1.yaml").read_text())
    db=Path(cfg["parent"]+".db")
    baseline=Path("reports/phase4d/replay")
    validation=json.loads((baseline/"validation.json").read_bytes())
    archive=Path("phase4c_haiper_receipts_20261111.zip")
    assert file_sha(db)==validation["parent_sha256"]==cfg["parent"]
    assert file_sha(archive)==validation["receipt_sha256"] and validation["parent_verified"]
    with zipfile.ZipFile(archive) as z:
        m=json.loads(z.read("capture/manifest.json"))
    graph=json.loads((baseline/"graph.json").read_bytes())
    assert file_sha(baseline/"graph.json")==json.loads((baseline/"graph_trust.json").read_bytes())["payload_sha256"]
    ims,cams,keys,pairs,arrays=inventory(db,graph["nodes"],{int(k):v for k,v in m["offset_sha256"].items()})
    assert arrays==m["arrays"] and {str(k):v for k,v in ims.items()}==m["images"]
    name=lambda i:ims[i]["name"]
    links={(name(i),name(j)):p["verified"] for (i,j),p in pairs.items() if len(p.get("verified",[]))}
    assert {tuple(sorted(pair)):len(v) for pair,v in links.items()}=={(e["i"],e["j"]):e["inliers"] for e in graph["edges"]}
    sizes={name(i):[cams[v["camera_id"]][k] for k in ("width","height")] for i,v in ims.items()}
    imported=time.perf_counter()-start
    result=analyze({name(i):v for i,v in keys.items()},sizes,links,cfg["grid"],cfg["chunk"])
    assert (len(ims),len(links),result["links"],result["keypoints"])==(54,454,607136,944534)
    result["seconds"].update(import_and_hash=imported,cpu=time.process_time()-cpu)
    result["peak_bytes"]=psutil.Process().memory_info().peak_wset
    frozen=json.loads(Path("reports/phase4d/haiper.json").read_bytes())
    for row in result["edges"]:
        a,b=row["pair"]
        comp=next(c for c in frozen["components"] if a in c["nodes"])
        row["component_size"]=len(comp["nodes"])
        row["boundary"]=(a in comp["sides"][0])!=(b in comp["sides"][0])
    dest=Path("reports/phase4e/features.json")
    dest.write_text(json.dumps(result,indent=2))
    Path("reports/phase4e/freeze.json").write_text(json.dumps({"features":file_sha(dest),"parent":file_sha(db),"proposal":file_sha(Path("reports/phase4d/haiper.json")),"source_revision":m["source_revision"],"config":m["config_sha256"],"checkpoints":m["checkpoints"],"permutation":m["permutation_sha256"]},indent=2))
if __name__=="__main__":
    main()
