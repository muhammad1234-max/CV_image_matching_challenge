"""Operational checkpoint contract. All seals retain the frozen evaluator's identity format."""
from contextlib import closing
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import numpy as np

from src.novel_model.final_candidate_v1 import partition, immutable_db
from src.novel_model.final_candidate_v1.submission import validate as validate_rows, COLUMNS
from src.novel_model.final_candidate_v1.workflow import STAGES, read, save, file_sha, files_in, select
from src.novel_model.reconstruction_feedback_v1 import inspect, make_copy

VERSION = "phase5a1.checkpoint.1"
DS = "imc2024_dioscuri_baalshamin"
class Fatal(ValueError):
    kind = "INTEGRITY_FATAL"
class Execution(RuntimeError):
    kind = "EXECUTION_FATAL"
class Recoverable(RuntimeError):
    kind = "EXECUTION_RECOVERABLE"

def check(value, reason):
    if not value:
        raise Fatal(reason)

def equal(actual, expected, field):
    if actual != expected:
        raise Fatal(f"{field}: expected {expected!r}, observed {actual!r}")

def digest(b):
    return hashlib.sha256(b).hexdigest()

def outcome(stage, folder):
    if stage in ("capture","decision"):
        return "COMPLETE"
    branch=read(folder/"branch.json")
    if branch["available"] is False:
        return "UNAVAILABLE"
    check(branch["available"] is True,"invalid availability")
    if branch["valid_export"] is False:
        return "OBSERVATIONS_UNAVAILABLE"
    check(branch["valid_export"] is True,"invalid export flag")
    return "COMPLETE"

class Contract:
    def __init__(self,pins,runner):
        self.p=pins
        self.runner=runner
        self.names=[f"image_{i:04d}.png" for i in range(pins["images"])]
        self.identity=dict(model="final_candidate_v1",seed=20261111,dataset=DS,
                           protocol=pins["protocol_sha256"],worker=pins["worker"])

    def header(self,stage,folder,parents):
        return dict(version=VERSION,runner=self.runner,stage=stage,outcome=outcome(stage,folder),
                    identity=self.identity,config=self.p["config_sha256"],source=self.p["source_revision"],
                    permutation=self.p["image_permutation"],parents=parents)

    def predictions(self,path):
        with path.open(newline="") as f:
            reader=csv.DictReader(f)
            equal(reader.fieldnames,COLUMNS,"prediction header")
            rows=list(reader)
        validate_rows(rows)
        equal(sorted(r["image"] for r in rows),self.names,"prediction coverage")
        for r in rows:
            check(r["dataset"]==DS and r["image_id"]==str(int(r["image"][6:10])),"prediction identity")
        return rows

    def capture(self,d):
        check(not (d/"no_mapper.json").exists(),"capture requires numerical parent")
        job=read(d/"job.json")
        equal(file_sha(d/"parent.db"),job["parent"]["backup_sha256"],"parent.db SHA256")
        hashes=job["image_hashes"]
        equal(sorted(hashes),self.names,"capture membership")
        equal(digest("\n".join(hashes[n] for n in self.names).encode()),self.p["image_permutation"],"capture order")
        equal(digest(job["options"].encode()),self.p["options_sha256"],"mapper options")
        self.provenance(d)
        return self.database(d,job)

    def provenance(self,d):
        ident=read(d/"identity.json")
        for k,v in dict(source_revision=self.p["source_revision"],config_sha256=self.p["published_config"],
                        benchmark=self.p["benchmark"],seed=20261111,permutation_sha256=self.p["image_permutation"],
                        experiment_id=f"ns64_isolated_v1_{DS}_20261111",method_id="ns64_mast3rhybrid_022c",
                        comparison_baseline_id="organizer_baseline_v1_dev_full").items():
            equal(ident[k],v,k)
        receipt=read(d/"input_receipt.json")
        equal(receipt["archive_sha256"],self.p["image_archive"],"opaque archive")
        equal(receipt["permutation_sha256"],self.p["image_permutation"],"opaque permutation")
        env=read(d/"environment.json")
        check(env["status"]=="CLEAN_VM_PRECHECK_PASS" and read(d/"boundary.json")["status"]=="PASS","clean VM receipts")
        equal(env["source_inventory_hash"],self.p["source_inventory"],"published inventory")
        equal(env["assets"],self.p["weight_pins"],"weight inventory")
        equal(digest(env["environment_freeze"].encode()),self.p["environment"],"package freeze")

    def database(self,d,job):
        match=read(d/"checkpoint_verify_matches.json")
        check(match["status"]=="PASS" and match["stage"]=="verify_matches" and type(match["verified_pairs"]) is int and match["verified_pairs"]>=0,"verification receipt")
        with closing(immutable_db(d/"parent.db")) as db:
            equal(db.execute("pragma integrity_check").fetchall(),[("ok",)],"SQLite integrity")
            equal(db.execute("pragma foreign_key_check").fetchall(),[],"SQLite references")
            equal(sorted(x[0] for x in db.execute("select name from images")),self.names,"SQLite names")
            tables=inspect(db)["records"]
            check({"images","cameras","keypoints","matches","two_view_geometries"}<=tables["tables"].keys(),"SQLite tables")
            equal(tables,job["parent"]["original_logical"],"logical records")
            counts=dict(db.execute("select i.name,k.rows from images i join keypoints k using(image_id)"))
            check(set(counts)==set(self.names) and all(n>=0 for n in counts.values()),"keypoint coverage")
        return counts

    def branch(self,stage,d,root):
        self.provenance(d)
        b=read(d/"branch.json")
        mode=outcome(stage,d)
        cap=root/"capture"
        parent=cap/"parent.db"
        job=read(cap/"job.json")
        pin=job["parent"]["backup_sha256"]
        prop,mask=partition(parent,pin,self.names)
        prop=json.loads(json.dumps(prop))
        if stage=="trial":
            orig=read(root/"original/branch.json")
            check(orig["valid_export"] is True,"indexed original required")
            eligible=prop is not None and sorted(sum(prop["sides"],[])) in [sorted(g) for g in orig["groups"]]
        if mode=="UNAVAILABLE":
            check(set(b)=={"available","reason"},"contradictory unavailable branch")
            check(stage=="trial","missing original reconstruction")
            failure=read(d/"failure.json")
            check(failure["kind"]==Recoverable.kind and failure["reason"]==b["reason"] and failure["stage"]=="trial","fallback receipt")
            code=failure["code"]
            check((code=="NO_ELIGIBLE_GEOMETRY" and not eligible and b["reason"]=="no eligible fully registered component or original export")
                  or (code=="TRIAL_TIMEOUT" and eligible and b["reason"]=="trial deadline exceeded"),"unrecognized fallback")
            check(not any((d/k).exists() for k in ("prediction.csv","errors.npz")),"unavailable trial has prediction or residuals")
            if code=="NO_ELIGIBLE_GEOMETRY":
                check(not (d/"input.json").exists(),"unexpected masked input")
                return
        if stage=="trial":
            check(eligible,"ineligible trial")
        audit=read(d/"input.json")
        with closing(immutable_db(parent)) as db:
            equal(audit["before"],inspect(db),"parent audit")
        equal(audit["forbidden"],mask if stage=="trial" else [],"pair mask")
        if stage=="original":
            equal(audit["before"],audit["after"],"unmasked database")
        else:
            with tempfile.TemporaryDirectory() as t:
                source=Path(t)/"parent.db"
                shutil.copyfile(parent,source)
                replay=make_copy(source,Path(t)/"trial.db",pin,mask)
                equal(audit,replay,"masked logical records")
        if mode=="UNAVAILABLE":
            return
        rows=self.predictions(d/"prediction.csv")
        if mode=="OBSERVATIONS_UNAVAILABLE":
            fail=read(d/"failure.json")
            check(b["export_error"] in ("invalid depth","invalid projection") and fail==dict(
                stage=stage,kind=Execution.kind,code="OBSERVATION_EXPORT",reason=b["export_error"])
                and not (d/"errors.npz").exists(),"invalid observation failure")
            return
        check(not (d/"failure.json").exists(),"contradictory failure receipt")
        equal(b["proposal"],prop if stage=="original" else None,"frozen proposal")
        clusters={}
        for row in rows:
            if row["scene"]!="outliers":
                clusters.setdefault(row["scene"],[]).append(row["image"])
        groups=sorted(sorted(g) for g in clusters.values())
        equal(sorted(sorted(g) for g in b["groups"]),groups,"prediction / indexed membership")
        equal(sorted(sorted(m["images"]) for m in b["models"]),groups,"model membership")
        check(all(type(m["points"]) is int and m["points"]>=0 for m in b["models"]),"model point count")
        with closing(immutable_db(parent)) as db:
            counts=dict(db.execute("select i.name,k.rows from images i join keypoints k using(image_id)"))
        names=sorted(n for g in groups for n in g)
        with np.load(d/"errors.npz",allow_pickle=False) as arrays:
            equal(sorted(arrays.files),names,"indexed names")
            for n in names:
                a=arrays[n]
                check(a.dtype.kind=="f" and a.shape==(counts[n],) and not np.isinf(a).any() and not (a<0).any(),"indexed residual "+n)

    def validate(self,stage,d,root,parents,sealed=True):
        """Caller supplies parents only after recursively validating the actual predecessor chain."""
        try:
            check(stage in STAGES and list(parents)==list(STAGES[:STAGES.index(stage)]),"predecessor stages")
            common={"contract.json","time.json","usage.json","heartbeat.jsonl","worker.log","seal.json"}
            provenance={"environment.json","boundary.json","identity.json","input_receipt.json","current_stage.json","preflight.json","resources.json","checkpoint_verify_matches.json"}
            branch={"branch.json","input.json","prediction.csv","failure.json","errors.npz"}
            allowed=common | ({"selection.json","final.csv"} if stage=="decision" else provenance | (
                {"parent.db","job.json"} if stage=="capture" else branch))
            check(all(f.is_file() and not f.is_symlink() and f.name in allowed for f in d.iterdir()),"unexpected stage inventory")
            expected=self.header(stage,d,parents)
            equal(read(d/"contract.json"),expected,"contract identity/outcome")
            sec=read(d/"time.json")["seconds"]
            check(type(sec) in (float,int) and math.isfinite(sec) and sec>=0,"elapsed seconds")
            usage=read(d/"usage.json")
            check(usage["stage"]==stage and all(type(usage[k]) is int and usage[k]>=0 for k in ("peak_pss","peak_rss","peak_uss"))
                  and usage["samples"]>0,"resource accounting")
            samples=[json.loads(line) for line in (d/"heartbeat.jsonl").read_text().splitlines()]
            equal(len(samples),usage["samples"],"resource sample count")
            check(all(s["stage"]==stage for s in samples),"resource stage")
            for k in ("rss","pss","uss"):
                equal(usage["peak_"+k],max(s["totals"][k] for s in samples),"sample peak "+k)
            if stage=="capture":
                self.capture(d)
            elif stage in ("original","trial"):
                self.branch(stage,d,root)
            else:
                check(read(root/"trial/branch.json").get("valid_export") is not False,"trial observation export failed")
                with tempfile.TemporaryDirectory() as tmp:
                    replay=Path(tmp)
                    action=select(root,replay)
                    check(not action["reason"].startswith("invalid indexed evidence"),"invalid decision evidence")
                    equal(read(d/"selection.json"),read(replay/"selection.json"),"decision replay")
                    equal(file_sha(d/"final.csv"),file_sha(replay/"final.csv"),"selected bytes")
                self.predictions(d/"final.csv")
            if sealed:
                equal(read(d/"seal.json"),dict(stage=stage,identity=self.identity,parents=parents,files=files_in(d)),"payload seal")
        except Fatal:
            raise
        except Exception as exc:
            raise Fatal(f"{stage} invalid checkpoint: {exc}") from exc
