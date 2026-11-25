"""Build the small supervisor ZIP from a positive list. Reopen and verify every member."""
from pathlib import Path
import json
import hashlib
import zipfile
from xml.etree.ElementTree import parse

from src.novel_model.final_candidate_v1.workflow import read,save
from src.novel_model.geometry_capture_v1.database import file_sha

OUT=Path("reports/phase4i")
ARCHIVE=Path("phase4i_final_candidate_review.zip")

def main():
    freeze=read(OUT/"freeze.json")
    for p,h in freeze["files"].items():
        if file_sha(p)!=h: raise ValueError("frozen compatibility file drift "+p)
    inputs=read(OUT/"inference_pins.json")
    paths=[Path(p) for p in inputs if p.startswith(("src/","configs/"))]
    for p in paths:
        if file_sha(p)!=inputs[p.as_posix()]: raise ValueError("inference dependency drift")
    paths.extend([Path("AGENTS.md"),Path("notebooks/step04i_final_candidate_colab.ipynb"),
                  Path("tests/test_final_candidate_v1.py"),*Path("scripts").glob("phase4i_*.py"),
                  *Path("docs").glob("phase4i_*.md")])
    paths.extend(Path(p) for p in [
        "reports/phase4f/active_return/result.json","reports/phase4f/pass_through_return/result.json",
        "reports/phase4f/parity_gate.json","reports/phase4h/decision.json","reports/phase4h/decision.sha256",
        "reports/phase4h/search.json","docs/phase4f_score_comparison.md","docs/phase4h_offline_replay.md",
        "docs/phase4g1_supervisor_handoff.md","reports/ns64_full/completed_archive_index.json"])
    paths.extend(OUT/n for n in ("freeze.json","inference_pins.json","haiper_diagnostic.json","inventory.json",
                  "targeted.xml","targeted.log","suite.xml","suite.log","observed.json","native_export.json"))
    for p in (OUT/"suite.xml",OUT/"targeted.xml"):
        suite=parse(p).getroot().find("testsuite")
        if any(int(suite.get(k)) for k in ("failures","errors","skipped")): raise ValueError(p)
    partial=ARCHIVE.with_suffix(".partial")
    manifest={}
    with zipfile.ZipFile(partial,"w") as z:
        for p in sorted(set(paths)):
            if p.suffix not in (".py",".yaml",".md",".json",".xml",".log",".sha256",".ipynb"):
                raise ValueError(p)
            data=p.read_bytes()
            manifest[p.as_posix()]=hashlib.sha256(data).hexdigest()
            z.writestr(zipfile.ZipInfo(p.as_posix()),data,compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("SHA256.json",json.dumps(manifest,sort_keys=True,indent=2))
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{"SHA256.json"}
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in manifest.items())
    partial.replace(ARCHIVE)
    save(OUT/"archive.json",dict(file=ARCHIVE.name,sha256=file_sha(ARCHIVE),
                                 bytes=ARCHIVE.stat().st_size,members=len(manifest)+1))
    print((OUT/"archive.json").read_text())

if __name__=="__main__":main()
