"""Trusted workstation entry point after all four real stage archives are retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
from scripts.phase4i_evaluate import evaluate
from scripts.phase5a6_production import Contract
from src.runner_5a1.contract import DS,STAGES,read,equal,check
from src.runner_5a1.runner import Runner
from src.runner_5a5.kaggle import sha,save

def receive(folder, journal_pin, archive_pins, output):
    release=read(Path("reports/phase5a6/release.json"))
    live=read(folder/"phase5a6_preflight.json")
    equal(live["release"],release["production_release"],"live release")
    equal(live["checks"]["preflight"],"PASS","live preflight")
    pins=read(archive_pins)
    equal(set(pins),set(STAGES),"independent archive digests")
    journal=folder/("journal-"+journal_pin+".json")
    equal(sha(journal),journal_pin,"external final journal")
    events=read(journal)
    migrated=next(Path("kaggle_upload/phase5a6_initial_custody").glob("journal-*.json"))
    equal(events[:2],read(migrated),"committed custody")
    equal([(e["stage"],e["status"]) for e in events[2:]],
          [(s,k) for s in STAGES for k in ("started","complete")],"authorized attempt sequence")
    check(sum(e.get("seconds",0) for e in events)<=14400,"budget exceeded")
    with tempfile.TemporaryDirectory(dir=folder) as directory:
        root=Path(directory)
        runroot=root/"validated"
        runroot.mkdir()
        shutil.copyfile(journal,runroot/"attempts.json")
        with zipfile.ZipFile("reports/phase5a5/validation_kit.zip") as z:
            cfg=json.loads(z.read("manifest.json"))
            for n,pin in read(Path("reports/phase5a6/scientific.json")).items():
                if Path(n).is_file():
                    equal(sha(n),pin,n)
                else:
                    equal(hashlib.sha256(z.read(n)).hexdigest(),pin,n)
        equal(live["checks"]["freeze"],cfg["normalized_freeze"],"installed lock")
        contract=Contract(dict(release["pins"],environment=live["checks"]["raw_freeze_sha256"]),live["release"])
        runner=Runner(runroot,contract,lambda *args:None,journal_pin)
        for stage in STAGES:
            archive=folder/(DS+"_"+stage+".bin")
            runner.restore(stage,archive,pins[stage])
            os.link(archive,root/(DS+"_"+stage+".zip"))
        evaluate(DS,root,archive_pins,output)
        comparison=read(output/"comparison.json")
        scores=comparison["metrics"]
        pairs={}
        for method,reference in [("automatic","ns64_pass"),("automatic","organizer"),("ungated_proposal","ns64_pass")]:
            if method in scores:
                pairs[method+" minus "+reference]={variant:{k:x-scores[reference]["scores"][variant][k]
                    for k,x in vals.items()} for variant,vals in scores[method]["scores"].items()}
        save(output/"paired_variants.json",pairs)
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    parser.add_argument("journal_pin")
    parser.add_argument("archive_pins",type=Path)
    parser.add_argument("output",type=Path)
    receive(**vars(parser.parse_args()))
