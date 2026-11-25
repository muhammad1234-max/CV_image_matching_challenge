"""Verify the built compatibility kit and package evidence, excluding operator inputs."""
import json
from pathlib import Path
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from scripts.phase5a5_build import build,read,save,R,sha

def main():
    release=read(R/"release.json")
    build()
    assert release==read(R/"release.json")
    with tempfile.TemporaryDirectory(dir="tmp") as tmp:
        kit=Path(tmp).resolve()
        with zipfile.ZipFile(R/"validation_kit.zip") as z:z.extractall(kit)
        sys.path[:0]=[str(kit),str(kit/"tests")]
        from scripts.phase5a5_persistence import create,restore
        from src.runner_5a5.kaggle import transition,JOURNAL
        from src.runner_5a3.provenance import Contract
        from src.runner_5a1.runner import Runner
        rev=release["revision"]
        dest=kit/"preview"
        dest.mkdir()
        contract=Contract(read(kit/"manifest.json")["pins"],rev)
        pin=transition(kit/("journal-"+JOURNAL+".json"),dest/"attempts.json",contract.identity,rev)
        runner=Runner(dest,contract,lambda *args:None,pin)
        out=kit/"custody.bin"
        directory=kit/"disposable"
        directory.mkdir()
        digest=create(directory,out,rev)
        recovered=restore(out,digest,rev)
        save(R/"local_rehearsal.json",dict(platform=sys.platform,revision=rev,synthetic_checkpoint=digest,
             restored=recovered,custody_preview_sha256=pin,events=runner.events,committed=False))
    counts={}
    for name in ("full","targeted"):
        suite=ET.parse(R/(name+".xml")).getroot().find("testsuite")
        assert all(int(suite.get(k))==0 for k in ("errors","failures","skipped"))
        counts[name]=int(suite.get("tests"))
    assert counts=={"full":406,"targeted":12}
    save(R/"status.json",dict(classification="BLOCKED — DEPENDENCY COMPATIBILITY",windows_tests=counts,
         latest_live_kaggle="phase5a4_kaggle_preflight.json",live_phase5a5_receipts_received=False,
         api_attempt="Authentication required",capture_started=False))
    files=[*Path("docs").glob("phase5a5_*.md"),*Path("notebooks").glob("step05a5*.ipynb"),
           *Path("src/runner_5a5").glob("*.py"),*Path("tests").glob("test_phase5a5*.py"),
           *Path("scripts").glob("phase5a5*.py"),Path("phase5a4_kaggle_preflight.json"),
           *[p for p in R.iterdir() if p.is_file() and p.name!="SHA256.json"]]
    hashes={p.as_posix():sha(p) for p in sorted(files)}
    save(R/"SHA256.json",hashes)
    target=Path("phase5a5_kaggle_migration_review.zip")
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as z:
        for p in [*files,R/"SHA256.json"]:z.write(p,p.as_posix())
    import hashlib
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in hashes.items())
    print(target,sha(target))
if __name__=="__main__":main()
