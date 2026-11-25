"""Freeze a label-free boundary and verify a clean pass-through snapshot."""
import json, zipfile, statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from src.novel_model.geometry_capture_v1.database import file_sha, save, MAX_ID
from src.novel_model.geometry_capture_v1.parity import compare
from src.novel_model.reconstruction_feedback_v1 import make_copy

def main():
    out=Path("reports/phase4f")
    old=Path("reports/phase4d/haiper.json")
    frozen=json.loads(Path("reports/phase4e/freeze.json").read_bytes())
    assert file_sha(old)==frozen["proposal"]
    r=json.loads(old.read_bytes())
    nbr={n:set() for n in r["nodes"]}
    for a,b in r["pairs"]:
        nbr[a].add(b)
        nbr[b].add(a)
    candidates=[]
    for c in r["components"]:
        left,right=map(set,c["sides"])
        crossing=[(a,b) for a,b in r["pairs"] if (a in left and b in right) or (b in left and a in right)]
        overlap=statistics.median(len(nbr[a]&nbr[b])/len(nbr[a]|nbr[b]) for a,b in crossing)
        candidates.append((overlap,c,crossing))
    overlap,c,cross=min(candidates,key=lambda x:x[0])
    assert len(c["nodes"])==31 and len(cross)==68
    archive=Path("phase4c_haiper_receipts_20261111.zip")
    validated=json.loads(Path("reports/phase4d/replay/validation.json").read_bytes())
    assert validated["parent_verified"] and file_sha(archive)==validated["receipt_sha256"]
    with zipfile.ZipFile(archive) as z:
        m=json.loads(z.read("capture/manifest.json"))
        with zipfile.ZipFile("reports/phase3c7/worker.zip") as h:
            save(out/"original_parity.json",compare(h.read("opaque.csv"),z.read("capture/original.csv")))
        (out/"reference.csv").write_bytes(z.read("capture/original.csv"))
    names={v["name"]:int(i) for i,v in m["images"].items()}
    ids=[]
    for a in c["sides"][0]:
        for b in c["sides"][1]:
            i,j=sorted([names[a],names[b]])
            ids.append(i*MAX_ID+j)
    parent=Path(frozen["parent"]+".db")
    assert m["backup_sha256"]==frozen["parent"]==validated["parent_sha256"]
    with TemporaryDirectory(dir=out) as temp:
        audit=make_copy(parent,Path(temp)/"pass.db",frozen["parent"])
    assert audit["before"]==audit["after"] and audit["before"]["records"]==m["original_logical"]
    save(out/"copy_audit.json",audit)
    save(out/"plan.json",dict(parent=frozen["parent"],source=m["source_revision"],config=m["config_sha256"],seed=m["seed"],
        permutation=m["permutation_sha256"],image_hashes=m["image_sha256"],proposal_sha256=file_sha(old),receipt_sha256=file_sha(archive),
        options=m["mapper_options"],sides=c["sides"],verified_crossing=cross,forbidden_ids=sorted(ids),
        keep=[tuple(p) not in set(cross) for p in r["pairs"]],pairs=r["pairs"],
        medians={str(len(k["nodes"])):o for o,k,_ in candidates},rule="lowest boundary median endpoint-neighborhood Jaccard",
        reference_sha256=file_sha(out/"reference.csv"),records=m["original_logical"]))
if __name__=="__main__":
    main()
