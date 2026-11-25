"""Publish the CPU review archive only after the embedded and project suites pass."""
import json
import hashlib
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from scripts.phase5a3_build import build,read,save,sha

def package():
    report=Path("reports/phase5a3")
    expected=read(report/"revision.json")
    build()
    assert read(report/"revision.json")==expected
    counts={}
    for name in ("generated","suite","regression"):
        suite=ET.parse(report/(name+".xml")).getroot().find("testsuite")
        assert all(suite.get(key)=="0" for key in ("errors","failures","skipped"))
        counts[name]=int(suite.get("tests"))
    assert counts==dict(generated=24,suite=394,regression=68)
    original=read(Path("reports/phase5a2/preflight.json"))
    archive=Path(original["dataset"]+"_opaque.zip")
    with archive.open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest()==original["image_archive"]
    with zipfile.ZipFile(archive) as z:
        names=sorted(z.namelist())
        assert names==[f"image_{i:04d}.png" for i in range(138)]
        assert sha("\n".join(sha(z.read(n)) for n in names).encode())==original["permutation"]
    fixture=Path("tmp/phase5a3_generated/test_rehearsal0/experiment")
    before=original["generated_child_emits_identity"]
    save(report/"preflight.json",dict(scope="Historical pre-repair CPU reproduction",generated_identity=before,
         previous_notebook=original["notebook"],previous_runner=original["runner"],evidence="test_pre_repair_children"))
    snapshots={s:read(fixture/s/"identity.json") for s in ("capture","original","trial")}
    save(report/"examples.json",dict(scope="Synthetic child execution only",previous_generated_identity=before,
        after=snapshots,decision=read(fixture/"decision/selection.json"),
        seals={s:sha((fixture/s/"seal.json").read_bytes()) for s in ("capture","original","trial","decision")}))
    save(report/"release.json",dict(tests=counts,old_runner_tests=48,frozen_candidate_tests=20,
         receipt_schema="phase5a3.branch.1",checkpoint_schema="phase5a3.checkpoint.1",
         image_archive=original["image_archive"],permutation=original["permutation"],images=138,
         inference_seconds=0,scope="CPU release rehearsal only",**expected))
    members=[*Path("src/runner_5a3").glob("*.py"),*Path("scripts").glob("phase5a3*.py"),
             Path("tests/test_phase5a3.py"),Path("tests/phase5a3_cpu.py"),Path("tests/test_phase5a1.py"),
             Path("notebooks/step05a3_dioscuri_colab.ipynb"),*Path("docs").glob("phase5a3_*.md")]
    members+=sorted(report/n for n in ("revision.json","release.json","scientific_comparison.json","examples.json",
         "preflight.json","runner.zip","generated.xml","generated.log","suite.xml","suite.log","regression.xml","regression.log"))
    hashes={p.as_posix():sha(p.read_bytes()) for p in members}
    save(report/"SHA256.json",hashes)
    target=Path("phase5a3_production_child_repair_review.zip")
    with zipfile.ZipFile(target,"w") as z:
        for p in sorted([*members,report/"SHA256.json"]):
            z.writestr(zipfile.ZipInfo(p.as_posix()),p.read_bytes(),zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and all(sha(z.read(k))==v for k,v in hashes.items())
    print(target,sha(target.read_bytes()))

if __name__=="__main__":package()
