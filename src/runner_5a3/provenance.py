"""Child-produced branch evidence. Restore recomputes the same bindings from sealed parents."""
import json
import os
from pathlib import Path
import tempfile
from src.runner_5a1.contract import Contract as Base, DS, STAGES, check, equal, read, save, file_sha, digest, outcome, partition

SCHEMA="phase5a3.branch.1"
RECEIPTS=("environment.json","boundary.json","input_receipt.json","identity.json")
OUTPUTS=("input.json","branch.json","prediction.csv","errors.npz","failure.json")

def canonical(value):
    return digest(json.dumps(value,sort_keys=True,separators=(",",":")).encode())

class Contract(Base):
    def header(self,*args):
        return dict(super().header(*args),version="phase5a3.checkpoint.1")

    def binding(self,stage,root):
        parent=root/"capture/parent.db"
        pin=file_sha(parent)
        equal(pin,read(root/"capture/job.json")["parent"]["backup_sha256"],"numerical parent")
        mask=partition(parent,pin,self.names)[1] if stage=="trial" else []
        return dict(schema=SCHEMA,stage=stage,dataset=DS,runner=self.runner,protocol=self.identity["protocol"],
                    worker=self.identity["worker"],candidate_config=self.p["config_sha256"],parent=pin,
                    predecessors={s:file_sha(root/s/"seal.json") for s in STAGES[:STAGES.index(stage)]},
                    proposed_mask=canonical(mask))

    def branch(self,stage,d,root):
        super().branch(stage,d,root)
        actual=read(d/"identity.json")["branch_execution"]
        expected=self.binding(stage,root)
        timeout=(d/"failure.json").exists() and read(d/"failure.json").get("code")=="TRIAL_TIMEOUT"
        equal(actual,dict(expected,status="STARTED") if timeout else terminal(expected,stage,d),"branch execution receipt")

def terminal(binding,stage,d):
    audit=read(d/"input.json") if (d/"input.json").exists() else None
    return dict(binding,status=outcome(stage,d),mapper_logical_sha256=canonical(audit["after"]) if audit else None,
                outputs={n:file_sha(d/n) for n in OUTPUTS if (d/n).is_file()})

def run_child(main,stage,root,d,work,source,pins,revision,seed):
    from src.novel_model.final_candidate_v1 import session
    contract=Contract(pins,revision)
    equal(session.identity(DS),contract.identity,"child worker and protocol")
    equal(file_sha(session.KIT/"configs/novel_model/final_candidate_v1.yaml"),pins["config_sha256"],"child candidate configuration")
    equal(seed,20261111,"actual seed")
    equal(file_sha(source/"conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-022-c.yaml"),pins["published_config"],"live config")
    binding=contract.binding(stage,root)
    identity=dict(source_revision=pins["source_revision"],config_sha256=pins["published_config"],
        benchmark=pins["benchmark"],seed=seed,permutation_sha256=read(work/"input_receipt.json")["permutation_sha256"],
        experiment_id=f"ns64_isolated_v1_{DS}_{seed}",method_id="ns64_mast3rhybrid_022c",
        comparison_baseline_id="organizer_baseline_v1_dev_full",branch_execution=dict(binding,status="STARTED"))
    path=work/"identity.json"
    check(not path.exists(),"stale branch identity")
    save(path,identity)
    contract.provenance(work)
    main()
    b=read(d/"branch.json")
    if not b["available"]:
        save(d/"failure.json",dict(kind="EXECUTION_RECOVERABLE",stage=stage,code="NO_ELIGIBLE_GEOMETRY",reason=b["reason"]))
    elif b.get("valid_export") is False:
        save(d/"failure.json",dict(kind="EXECUTION_FATAL",stage=stage,code="OBSERVATION_EXPORT",reason=b["export_error"]))
    # Validate the same branch evidence before atomically publishing terminal status.
    with tempfile.TemporaryDirectory(dir=work) as t:
        view=Path(t)
        for name in RECEIPTS:
            os.link(work/name,view/name)
        for name in OUTPUTS:
            if (d/name).exists():
                os.link(d/name,view/name)
        Base.branch(contract,stage,view,root)
        identity["branch_execution"]=terminal(binding,stage,view)
    save(path,identity)

def command(session,stage,work,d,root,contract):
    argv=session.stage_command(stage,work,DS,d,root)
    if stage in ("original","trial"):
        script=Path(argv[3])
        code=script.read_text()
        check(code.endswith("\nmain()\n"),"unexpected frozen child entry")
        code=code.removesuffix("main()\n")
        code+="from src.runner_5a3.provenance import run_child\n"
        script.write_text(code+f"run_child(main,{stage!r},Path({str(root)!r}),Path({str(d)!r}),work,source,{contract.p!r},{contract.runner!r},seed)\n")
    return argv
