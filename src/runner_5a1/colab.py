"""Thin Colab transport around the same Runner exercised by CPU regression tests."""
import json
from pathlib import Path
import shutil
import tempfile
from .contract import DS, Contract, Fatal, Execution, equal, check, read, save, file_sha
from .runner import Runner
from src.novel_model.final_candidate_v1 import session

def execute(runner,stage,d,end,content=Path("/content")):
    kit=session.KIT
    p=runner.contract.p
    equal(session.identity(DS),runner.contract.identity,"actual kit identity")
    for name,pin in read(kit/"inference_pins.json").items():
        equal(file_sha(kit/name),pin,name)
    if stage!="decision":
        equal(file_sha(content/(DS+"_opaque.zip")),p["image_archive"],"opaque input archive digest")
    with tempfile.TemporaryDirectory(dir=runner.root) as t:
        work=Path(t)
        command=session.stage_command(stage,work,DS,d,runner.root)
        # Preserve the child exception across the process boundary (the monitor reports only exit code).
        wrapper=work/"entry.py"
        wrapper.write_text("import sys, runpy, json, os\nfrom pathlib import Path\nsys.argv=sys.argv[1:]\ntry:\n runpy.run_path(sys.argv[0],run_name='__main__')\nexcept Exception as e:\n Path(os.environ['PHASE4I_DEST'],'child_error.json').write_text(json.dumps(dict(type=type(e).__name__,reason=str(e))))\n raise\n")
        try:
            session.monitored(command[:3]+[str(wrapper)]+command[3:],work,d,end,stage)
        except TimeoutError:
            if stage!="trial" or not (d/"input.json").is_file():
                raise
            reason="trial deadline exceeded"
            save(d/"branch.json",dict(available=False,reason=reason))
            save(d/"failure.json",dict(stage=stage,kind="EXECUTION_RECOVERABLE",code="TRIAL_TIMEOUT",reason=reason))
        except RuntimeError as exc:
            if (d/"child_error.json").is_file():
                child=read(d/"child_error.json")
                if child["type"] in ("AssertionError","ValueError","KeyError") or any(
                      k in child["reason"].lower() for k in ("digest","mismatch","inventory","hash","differ")):
                    raise Fatal(f"{stage} child {child}") from exc
                raise Execution(f"{stage} child {child}") from exc
            raise
        finally:
            for name in ("identity.json","input_receipt.json","environment.json","boundary.json",
                         "checkpoint_verify_matches.json","preflight.json","current_stage.json","resources.json"):
                if (work/name).is_file():shutil.copyfile(work/name,d/name)
        heart=[json.loads(line) for line in (d/"heartbeat.jsonl").read_text().splitlines()]
        check(bool(heart),"missing process samples")
        save(d/"usage.json",dict(stage=stage,samples=len(heart),**{
            "peak_"+k:max(row["totals"][k] for row in heart) for k in ("rss","pss","uss")}))
        if stage in ("original","trial") and (d/"branch.json").exists():
            b=read(d/"branch.json")
            if b["available"] is False and not (d/"failure.json").exists():
                save(d/"failure.json",dict(stage=stage,kind="EXECUTION_RECOVERABLE",code="NO_ELIGIBLE_GEOMETRY",reason=b["reason"]))
            if b.get("valid_export") is False:
                save(d/"failure.json",dict(stage=stage,kind=Execution.kind,code="OBSERVATION_EXPORT",reason=b.get("export_error")))

def interface(root,deadline,pins,revision):
    from google.colab import files
    def custody(journal,pin):
        copy=journal.with_name("journal-"+pin+".json")
        shutil.copyfile(journal,copy)
        files.download(str(copy))
        equal(input("After saving the journal externally, enter its printed SHA256 "+pin+" : ").strip(),pin,"journal custody")
    pin=input("First authorized attempt only: NEW. On restart enter the latest externally retained journal SHA256: ").strip()
    if pin!="NEW" and not (root/"attempts.json").exists():
        check(len(pin)==64 and all(c in "0123456789abcdef" for c in pin),"journal SHA256")
        uploaded=files.upload()
        data=uploaded["journal-"+pin+".json"]
        import hashlib
        equal(hashlib.sha256(data).hexdigest(),pin,"journal upload")
        root.mkdir(parents=True,exist_ok=True)
        (root/"attempts.json").write_bytes(data)
    runner=Runner(root,Contract(pins,revision),custody,pin)
    def run(stage):
        archive=Path("/content")/(DS+"_"+stage+".zip")
        external=input("Saved archive SHA256: ").strip() if archive.exists() else None
        if external:
            runner.restore(stage,archive,external)
        elif not (root/stage).exists():
            retry=input("Supervisor-authorized retry journal hash: ").strip() if runner.events[-1]["status"]=="failed" else None
            runner.run(stage,lambda s,d,end:execute(runner,s,d,end),deadline,retry)
        pin=runner.export(stage,archive,external)
        print(archive.name,pin,flush=True)
        files.download(str(archive))
    return run
