"""Phase 5A.1 run body with the attempt lookup scoped to the replacement generation."""
import time
import uuid
from src.runner_5a1.runner import equal,check,read,save,file_sha,seal,Execution,Fatal

def run_stage(self,stage,execute,deadline,retry_pin=None):
    equal(file_sha(self.journal),self.pin,"attempt history")
    parents=self.parents(stage)
    prior=self.events[-1]
    check(prior["status"]!="started","unresolved interrupted attempt; reserved time encumbered")
    if prior["status"]=="failed":
        check(retry_pin==self.pin and prior["stage"]==stage,"explicit retry authorization required")
    check(not any(e["stage"]==stage and (e["status"]=="complete" or
              (stage=="trial" and e["status"]=="started")) for e in self.current()),"stage already attempted or complete")
    check(not (self.root/stage).exists(),"existing stage requires restore verification")
    if stage=="trial":
        check(read(self.root/"original/branch.json")["valid_export"] is True,"original observations required")
    if stage=="decision":
        check(read(self.root/"trial/branch.json").get("valid_export") is not False,"trial observation export failed")
    spent=sum(e.get("seconds",0) for e in self.events)
    now=time.monotonic()
    remaining=min(14400-spent,deadline-now)
    check(remaining>1200 and (stage!="capture" or remaining>=8400),"insufficient session budget")
    end=now+min(remaining-1200,5400 if stage=="trial" else remaining)
    self.record("started",stage,reserved=end-now,parents=parents)
    started=time.monotonic()
    d=self.root/("attempt-"+uuid.uuid4().hex)
    d.mkdir()
    try:
        execute(stage,d,end)
        elapsed=time.monotonic()-started
        save(d/"time.json",dict(seconds=elapsed))
        try:
            save(d/"contract.json",self.contract.header(stage,d,parents))
        except (KeyError,FileNotFoundError) as exc:
            raise Fatal(f"{stage} required output: {exc}") from exc
        equal(self.parents(stage),parents,"predecessors changed during execution")
        self.contract.validate(stage,d,self.root,parents,sealed=False)
        seal(d,stage,self.contract.identity,parents)
        self.contract.validate(stage,d,self.root,parents)
        d.rename(self.root/stage)
    except BaseException as exc:
        kind=getattr(exc,"kind",Execution.kind)
        if "digest mismatch" in str(exc).lower():
            kind=Fatal.kind
        self.record("failed",stage,seconds=time.monotonic()-started,kind=kind,reason=str(exc),diagnostics=d.name)
        if kind==Fatal.kind and not isinstance(exc,Fatal):
            raise Fatal(str(exc)) from exc
        raise
    self.record("complete",stage,seconds=time.monotonic()-started,seal=file_sha(self.root/stage/"seal.json"))
