"""Operational wrapper for a single operator, stage and frozen experiment at a time."""
from pathlib import Path
import json
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from .contract import DS, STAGES, Fatal, Execution, check, equal, read, save, file_sha
from src.novel_model.final_candidate_v1.workflow import seal

class Runner:
    def __init__(self, root, contract, custody, journal_pin):
        self.root=Path(root)
        self.root.mkdir(exist_ok=True,parents=True)
        self.contract=contract
        self.custody=custody
        self.journal=self.root/"attempts.json"
        if journal_pin=="NEW":
            check(not any(self.root.iterdir()),"new experiment requires empty directory")
            self.events=[]
            self.record("created","",identity=contract.identity,runner=contract.runner)
        else:
            equal(file_sha(self.journal),journal_pin,"latest externally retained journal")
            self.events=read(self.journal)
            equal(self.events[0],dict(status="created",stage="",identity=contract.identity,runner=contract.runner),"journal identity")
        self.pin=file_sha(self.journal)

    def record(self,status,stage,**details):
        if self.journal.exists():
            equal(file_sha(self.journal),self.pin,"journal concurrency or tampering")
        events=[*self.events,dict(status=status,stage=stage,**details)]
        partial=self.journal.with_suffix(".partial")
        save(partial,events)
        os.replace(partial,self.journal)
        self.events=events
        self.pin=file_sha(self.journal)
        self.custody(self.journal,self.pin)

    def parents(self,stage):
        check(stage in STAGES,"unknown stage")
        parents={}
        for s in STAGES[:STAGES.index(stage)]:
            d=self.root/s
            self.contract.validate(s,d,self.root,dict(parents))
            parents[s]=file_sha(d/"seal.json")
            self.registered(s,parents[s])
        return parents

    def registered(self,stage,sha):
        check(any(e.get("seal")==sha and e["status"]=="complete" and e["stage"]==stage for e in self.events),
              "seal missing from externally retained attempt history")

    def verify(self,stage,d):
        self.contract.validate(stage,d,self.root,self.parents(stage))
        sha=file_sha(d/"seal.json")
        self.registered(stage,sha)
        return sha

    def unpack(self,stage,archive,pin,d,registered=True):
        equal(file_sha(archive),pin,"externally recorded archive SHA256")
        with zipfile.ZipFile(archive) as z:
            members=z.infolist()
            names=[i.filename for i in members]
            check(len(set(n.casefold() for n in names))==len(names) and len(names)<=64 and
                  all(re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_-]*[.][a-zA-Z0-9]+",n) for n in names)
                  and all((i.external_attr>>28) in (0,8) for i in members)
                  and sum(i.file_size for i in members)<=16*1024**3,"unsafe archive inventory")
            z.extractall(d)
        self.contract.validate(stage,d,self.root,self.parents(stage))
        sha=file_sha(d/"seal.json")
        if registered:
            self.registered(stage,sha)
        return sha

    def restore(self,stage,archive,pin):
        with tempfile.TemporaryDirectory(dir=self.root) as t:
            d=Path(t)/"stage"
            sha=self.unpack(stage,archive,pin,d)
            target=self.root/stage
            if target.exists():
                equal(self.verify(stage,target),sha,"local checkpoint conflict")
            else:
                d.rename(target)
        return sha

    def export(self,stage,archive,existing_pin=None):
        archive=Path(archive)
        d=self.root/stage
        sha=self.verify(stage,d)
        with tempfile.TemporaryDirectory(dir=archive.parent) as t:
            tmp=Path(t)
            if archive.exists():
                check(existing_pin is not None,"existing ZIP needs external digest")
                try:
                    existing=self.unpack(stage,archive,existing_pin,tmp/"verify",registered=False)
                except Exception:
                    # Preserve bytes; a wrong external pin also requires explicit review.
                    archive.rename(archive.with_name(archive.name+".quarantine-"+uuid.uuid4().hex))
                    raise
                equal(existing,sha,"existing valid checkpoint conflict")
            else:
                candidate=tmp/"checkpoint.zip"
                with zipfile.ZipFile(candidate,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                    for f in sorted(d.iterdir()):
                        z.write(f,f.name)
                equal(self.unpack(stage,candidate,file_sha(candidate),tmp/"verify"),sha,"export round trip")
                os.link(candidate,archive)
        return file_sha(archive)

    def run(self,stage,execute,deadline,retry_pin=None):
        equal(file_sha(self.journal),self.pin,"attempt history")
        parents=self.parents(stage)
        prior=self.events[-1]
        check(prior["status"]!="started","unresolved interrupted attempt; reserved time encumbered")
        if prior["status"]=="failed":
            check(retry_pin==self.pin and prior["stage"]==stage,"explicit retry authorization required")
        check(not any(e["stage"]==stage and (e["status"]=="complete" or
                  (stage=="trial" and e["status"]=="started")) for e in self.events),"stage already attempted or complete")
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
