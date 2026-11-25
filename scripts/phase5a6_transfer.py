"""Thin Colab transport around the same Runner exercised by CPU regression tests."""
import json
from pathlib import Path
import shutil
import tempfile
from src.runner_5a3.provenance import Contract, command as generate_command, RECEIPTS
from src.runner_5a1.contract import DS, Fatal, Execution, equal, check, read, save, file_sha
from src.runner_5a1.runner import Runner
from src.novel_model.final_candidate_v1 import session

def execute(runner,stage,d,end,opaque,generate_command,launch):
    kit=session.KIT
    p=runner.contract.p
    equal(session.identity(DS),runner.contract.identity,"actual kit identity")
    for name,pin in read(kit/"inference_pins.json").items():
        equal(file_sha(kit/name),pin,name)
    if stage!="decision":
        equal(file_sha(opaque),p["image_archive"],"opaque input archive digest")
    with tempfile.TemporaryDirectory(dir=runner.root) as t:
        work=Path(t)
        command=generate_command(session,stage,work,d,runner.root,runner.contract)
        # Preserve the child exception across the process boundary (the monitor reports only exit code).
        wrapper=work/"entry.py"
        wrapper.write_text("import sys, runpy, json, os\nfrom pathlib import Path\nsys.argv=sys.argv[1:]\ntry:\n runpy.run_path(sys.argv[0],run_name='__main__')\nexcept Exception as e:\n Path(os.environ['PHASE4I_DEST'],'child_error.json').write_text(json.dumps(dict(type=type(e).__name__,reason=str(e))))\n raise\n")
        try:
            launch(command[:3]+[str(wrapper)]+command[3:],work,d,end,stage)
        except TimeoutError:
            if stage!="trial" or not (d/"input.json").is_file():
                raise
            reason="trial deadline exceeded"
            save(d/"branch.json",dict(available=False,reason=reason))
            save(d/"failure.json",dict(stage=stage,kind="EXECUTION_RECOVERABLE",code="TRIAL_TIMEOUT",reason=reason))
        except RuntimeError as exc:
            if (d/"child_error.json").is_file():
                child=read(d/"child_error.json")
                if child["type"] in ("Fatal","AssertionError","ValueError","KeyError","FileNotFoundError") or any(
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
        if stage in ("original","trial"):
            for name in RECEIPTS:
                check((work/name).is_file() and (d/name).is_file(),"missing child receipt "+name)
                equal(file_sha(d/name),file_sha(work/name),"transferred "+name)
            try:
                runner.contract.branch(stage,d,runner.root)
            except Fatal:
                raise
            except (OSError,KeyError,ValueError,TypeError) as exc:
                raise Fatal(f"{stage} invalid transferred evidence: {exc}") from exc

