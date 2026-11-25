"""Generate and restore a synthetic checkpoint with the approved validator."""
import json
from pathlib import Path
import tempfile
import time
import zipfile

from src.runner_5a3.provenance import Contract
from src.runner_5a1.runner import Runner
from src.runner_5a5.kaggle import sha

def create(directory,bundle,revision):
    from test_phase5a1 import Fixture
    fixture=Fixture(directory)
    fixture.r=Runner(directory/"synthetic",Contract(fixture.p,revision),lambda *args:None,"NEW")
    fixture.r.run("capture",fixture.child,time.monotonic()+14400)
    archive=directory/"capture.zip"
    pin=fixture.r.export("capture",archive)
    with zipfile.ZipFile(bundle,"x",zipfile.ZIP_DEFLATED) as z:
        z.write(archive,"capture.zip")
        z.write(fixture.r.journal,"attempts.json")
        z.writestr("smoke.json",json.dumps(dict(scope="synthetic",pins=fixture.p,runner=revision,archive=pin,journal=fixture.r.pin)))
    return sha(bundle)

def restore(bundle,expected,revision):
    if sha(bundle)!=expected:raise ValueError("External synthetic custody digest differs")
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(bundle) as z:
        root=Path(tmp)
        if sorted(z.namelist())!=["attempts.json","capture.zip","smoke.json"] or sum(i.file_size for i in z.infolist())>16*1024**2:
            raise ValueError("Unexpected synthetic package")
        z.extractall(root)
        manifest=json.loads((root/"smoke.json").read_bytes())
        if manifest["scope"]!="synthetic" or manifest["runner"]!=revision:raise ValueError("Wrong synthetic custody")
        runner=Runner(root,Contract(manifest["pins"],revision),lambda *args:None,manifest["journal"])
        seal=runner.restore("capture",root/"capture.zip",manifest["archive"])
        if runner.parents("original")!={"capture":seal} or runner.pin!=manifest["journal"]:
            raise ValueError("Restore changed synthetic history")
        return dict(capture_seal=seal,restored_journal=runner.pin,next_stage="original",repeated_capture=False)
