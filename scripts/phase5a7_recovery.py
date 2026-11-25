"""One-time recovery journal extension. This module has no GPU dispatch entry point."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import json
from src.runner_5a1.contract import equal, check, read, save, file_sha
from src.runner_5a1.runner import Runner
from src.runner_5a7_primitives import run_stage

LATEST="f19165adfbf7fc12d1d17f1b4817c4d3d27a94560048655219d09e93e813f487"
LOST="e82d74536d7d4124deeffa56abf1cb74f262f12e9973537b8514449fd0165966"
ARCHIVE="5a3e45ecdae559e4860a15859a820dff10980e84c692098d4d09050bfe6a8406"
DEBIT=1648.8952883439997
ATTEMPT="dioscuri-custody-recovery-01"

def transition(previous,contract,authorization,proof_pin,loss_checked):
    equal(file_sha(previous),LATEST,"prior completed capture")
    equal(authorization,"Phase 5A.7 single recovery CAPTURE","supervisor authorization")
    check(loss_checked,"prior artifact availability unresolved")
    check(len(proof_pin)==64 and all(c in "0123456789abcdef" for c in proof_pin),"persistence proof required")
    history=read(previous)
    equal(history[0]["identity"],contract.identity,"experiment")
    equal(history[-1],dict(stage="capture",status="complete",seconds=DEBIT,seal=LOST),"historical completion")
    check(contract.runner!=history[0]["runner"],"new operational release required")
    return dict(stage="capture",status="recovery_authorized",reason="checkpoint custody loss before ORIGINAL or TRIAL",
                authorization=authorization,attempt=ATTEMPT,prior_journal=LATEST,lost_seal=LOST,
                lost_archive=ARCHIVE,prior_time=DEBIT,remaining_seconds=14400-DEBIT,allowance=1,
                previous_release=history[0]["runner"],release=contract.runner,persistence_proof=proof_pin)

class Custodian:
    """Independent administrator's persistent, append-only single-successor registry."""
    def __init__(self,folder):
        self.folder=Path(folder)
        self.folder.mkdir(exist_ok=True,parents=True)

    def latest(self,pin):
        tip=LATEST
        seen=set()
        while (self.folder/(tip+".next")).exists():
            check(tip not in seen,"cyclic custody registry")
            seen.add(tip)
            tip=file_sha(self.folder/(tip+".next"))
        equal(pin,tip,"stale custody head")
        return pin

    def append(self,prior,path,pin):
        self.latest(prior)
        equal(file_sha(path),pin,"journal to retain")
        with TemporaryDirectory(dir=self.folder) as tmp:
            copy=Path(tmp)/"retained"
            with copy.open("xb") as f:
                f.write(path.read_bytes())
                f.flush()
                os.fsync(f.fileno())
            equal(file_sha(copy),pin,"retained bytes")
            snapshot=self.folder/("journal-"+pin+".json")
            if not snapshot.exists():os.link(copy,snapshot)
            equal(file_sha(snapshot),pin,"stored snapshot")
            try:
                os.link(copy,self.folder/(prior+".next"))
            except FileExistsError:
                raise ValueError("custody predecessor already consumed")

class RecoveryRunner(Runner):
    def __init__(self,root,contract,registry,pin,previous,event):
        self.root=Path(root)
        self.contract=contract
        self.journal=self.root/"attempts.json"
        self.registry=registry
        self.pin=registry.latest(pin)
        equal(file_sha(self.journal),pin,"operator journal")
        equal(file_sha(registry.folder/("journal-"+pin+".json")),pin,"independent custody")
        equal(file_sha(previous),LATEST,"historical bytes")
        self.events=read(self.journal)
        past=read(previous)
        equal(self.events[:len(past)+1],past+[event],"unaltered history and transition")
        equal(event,transition(previous,contract,event["authorization"],event["persistence_proof"],True),"transition")
        self.offset=len(past)+1
        self.anchor=pin
        def custody(path,new):
            registry.append(self.anchor,path,new)
            self.anchor=new
        self.custody=custody

    def current(self):
        return self.events[self.offset:]

    def registered(self,stage,seal):
        equal([e["seal"] for e in self.current() if e["stage"]==stage and e["status"]=="complete"],
              [seal],"current registered seal")
        check(seal!=LOST,"historical seal unavailable")

    def parents(self,stage):
        parents=super().parents(stage)
        for s,seal in parents.items():
            check(any(e["status"]=="externally_restored" and e["stage"]==s and e["seal"]==seal
                      for e in self.current()),"predecessor awaiting external restore")
        return parents

    def run(self,stage,execute,deadline):
        self.registry.latest(self.pin)
        check(not any(e["stage"]==stage and e["status"]=="started" for e in self.current()),"attempt allowance consumed")
        return run_stage(self,stage,execute,deadline)

    def record(self,status,stage,**details):
        super().record(status,stage,attempt=ATTEMPT,**details)

    def activate(self,stage,mounted_archive,archive_pin,mounted_journal,journal_pin,version):
        # Called by a fresh reader, following independent retrieval from the approved private version.
        private_saved_output(mounted_archive,mounted_journal,version)
        equal(file_sha(mounted_journal),journal_pin,"externally retrieved journal")
        equal(journal_pin,self.pin,"completion journal")
        check(self.events[-1]["status"]=="complete" and self.events[-1]["stage"]==stage,"await completed stage")
        check(file_sha(mounted_archive)!=ARCHIVE,"lost archive cannot masquerade as replacement")
        seal=self.restore(stage,mounted_archive,archive_pin)
        self.record("externally_restored",stage,archive=archive_pin,seal=seal,version=version,
                    parent=file_sha(self.root/"capture/parent.db"),completion_journal=journal_pin)

def private_saved_output(archive,journal,version):
    import re
    check(re.fullmatch(r"(?:[\w-]+/[\w-]+/versions/|https://www\.kaggle\.com/code/[\w-]+/[\w-]+\?scriptVersionId=)[1-9][0-9]*",version),"immutable private notebook version required")
    root=Path("/kaggle/input")
    paths=[Path(p).resolve(strict=True) for p in (archive,journal)]
    for path in paths:
        check(path.is_relative_to(root),"artifact outside independent input mount")
        matches=[]
        for row in Path("/proc/self/mountinfo").read_text().splitlines():
            fields=row.split()
            mount=Path(fields[4])
            if mount.is_relative_to(root) and path.is_relative_to(mount):
                matches.append((len(mount.parts),fields[5]))
        check(matches and "ro" in max(matches)[1].split(","),"read-only source mount required")
    equal(paths[0].parent,paths[1].parent,"checkpoint and journal version directory")

def validate_persistence(source,producer_pin,reader_receipt,revision,private_version,owner_checked_private):
    equal(revision,"a4139e05464bb7052bac562a311131667d0f92cb275e134ff79447cd12111309","released persistence notebook")
    check(owner_checked_private is True,"owner verification of private version required")
    private_saved_output(Path(source)/"phase5a7_fixture_capture.bin",Path(source)/"phase5a7_fixture_journal.json",private_version)
    from scripts.phase5a7_persistence import verify
    external=read(reader_receipt)
    check(str(external.get("retrieved_from","")).startswith("/kaggle/input/"),"missing external version mount")
    restored=verify(source,producer_pin,revision)
    equal(external,dict(restored,retrieved_from=external["retrieved_from"]),"independent restore receipt")
    return file_sha(reader_receipt)

def initiate(root,previous,contract,registry,event,proof,inventory):
    equal(set(inventory),{"local","kaggle_outputs","versions","private_datasets","other_destinations","runtime"},"custody search scope")
    check(all(v=="checked_absent" for v in inventory.values()),"original archive available or unresolved")
    equal(validate_persistence(**proof),event["persistence_proof"],"live private-version proof")
    # The control plane must first verify the remote producer, proof and original-artifact inventory.
    equal(event,transition(previous,contract,event["authorization"],event["persistence_proof"],True),"authorized transition")
    registry.latest(LATEST)
    check(not (root/"capture").exists(),"existing capture requires verification")
    root.mkdir(exist_ok=True,parents=True)
    path=root/"attempts.json"
    check(not path.exists(),"refuse journal replacement")
    save(path,[*read(previous),event])
    pin=file_sha(path)
    registry.append(LATEST,path,pin)
    return RecoveryRunner(root,contract,registry,pin,previous,event)

def restore_historical(root,previous,contract,archive):
    equal(file_sha(previous),LATEST,"completed original journal")
    check(archive.is_file(),"original archive missing")
    equal(file_sha(archive),ARCHIVE,"historical archive bytes")
    root.mkdir(parents=True,exist_ok=False)
    (root/"attempts.json").write_bytes(previous.read_bytes())
    prior=Runner(root,contract,lambda *args: None,LATEST)
    seal=prior.restore("capture",archive,ARCHIVE)
    equal(seal,LOST,"historical registered seal")
    return seal
