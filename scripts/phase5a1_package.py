"""Package reviewed operational code and evidence from an explicit file selection."""
import difflib
import hashlib
import json
from pathlib import Path
import zipfile
from scripts.phase5a1_build import build,read,save,sha

def package():
    build()
    out=Path("reports/phase5a1")
    code=sorted([*Path("src/runner_5a1").glob("*.py"),*Path("tests").glob("test_phase5a1*.py"),
                 *Path("scripts").glob("phase5a1*.py")])
    (out/"runner_only.diff").write_text("".join("".join(difflib.unified_diff([],p.read_text().splitlines(True),
                        fromfile="/dev/null",tofile=p.as_posix()))+"\n" for p in code),encoding="utf8")
    schema=dict(version="phase5a1.checkpoint.1",authority="src/runner_5a1/contract.py",
                common=["time.json","contract.json","usage.json","heartbeat.jsonl","seal.json"],
                capture=["parent.db","job.json","identity.json","environment.json","boundary.json","input_receipt.json","checkpoint_verify_matches.json"],
                original_complete=["prediction.csv","input.json","branch.json","errors.npz"],
                trial_complete=["prediction.csv","input.json","branch.json","errors.npz"],
                decision=["final.csv","selection.json"],
                unavailable={"NO_ELIGIBLE_GEOMETRY":["branch.json","failure.json"],
                             "TRIAL_TIMEOUT":["branch.json","failure.json","input.json"]},
                observation_failure=["branch.json","prediction.csv","input.json","failure.json"],
                branch_provenance=["identity.json","environment.json","boundary.json","input_receipt.json"])
    save(out/"checkpoint_schema.json",schema)
    files=code+sorted(Path("docs").glob("phase5a1_*.md"))+[Path("notebooks/step05a1_dioscuri_colab.ipynb")]
    files+=sorted(f for f in out.iterdir() if f.is_file() and f.name not in {"changed_files.json","SHA256.json"})
    changed=sorted([p.as_posix() for p in files]+[(out/n).as_posix() for n in ("changed_files.json","SHA256.json")])
    save(out/"changed_files.json",dict(added=changed,modified_scientific_files=[],export="phase5a1_runner_repair_review.zip"))
    files += [out/"changed_files.json",Path("reports/phase5a/runner_probe.json")]
    hashes={p.as_posix():sha(p.read_bytes()) for p in files}
    save(out/"SHA256.json",hashes)
    target=Path("phase5a1_runner_repair_review.zip")
    with zipfile.ZipFile(target,"w") as z:
        for p in sorted([*files,out/"SHA256.json"]):
            z.writestr(zipfile.ZipInfo(p.as_posix()),p.read_bytes(),zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and set(z.namelist())==set(hashes)|{(out/"SHA256.json").as_posix()}
        assert all(sha(z.read(k))==v for k,v in hashes.items())
    print(target,sha(target.read_bytes()),len(files)+1)
if __name__=="__main__":
    package()
