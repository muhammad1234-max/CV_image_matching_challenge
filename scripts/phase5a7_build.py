"""Verify the released bytes and assemble isolated CPU persistence notebooks."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
from scripts.phase5a5_build import archive,read,save,sha
from src.runner_5a5.kaggle import verify_opaque

R=Path("reports/phase5a7")
OLD="fcc175e98945bb0e2bf5bd1328d07cd88f7e523c4bf89cb8a848e28af00fb9ac"
ORIGINAL="387775c3c7bd6e05d68cf9eeaf19f823860f8bd45423d6d302d7e7503f141537"
JOURNALS=["35ea6c250304d1fa81e490ea82219dfb62737e4076b5fb437658485326e99a4a",
          "e92ca62c66e762ecb76f2a5e713007e1dee75694ab7a80a1e609d30cd871f475",
          "f19165adfbf7fc12d1d17f1b4817c4d3d27a94560048655219d09e93e813f487"]
def digest(b):return hashlib.sha256(b).hexdigest()
def build():
    R.mkdir(exist_ok=True)
    with zipfile.ZipFile("phase5a6_capture_recovery_review.zip") as z:
        pins=json.loads(z.read("SHA256.json"))
        assert len(pins)==6 and z.testzip() is None
        assert all(digest(z.read(n))==v for n,v in pins.items())
        snapshots=[]
        for i,pin in enumerate(JOURNALS):
            name="journal-"+pin+".json"
            data=z.read(name)
            assert digest(data)==pin
            path=(Path("kaggle_upload/phase5a6_initial_custody") if i==0 else Path("."))/name
            assert path.read_bytes()==data
            snapshots.append(json.loads(data))
        assert snapshots[0]==snapshots[1][:-1] and snapshots[1]==snapshots[2][:-1]
        status=json.loads(z.read("recovery_status.json"))
        assert snapshots[-1][-1]==status["completed_capture"] and snapshots[-1][-1]["seconds"]==1648.8952883439997
        assert json.loads(z.read("phase5a6_preflight.json"))["checks"]["preflight"]=="PASS"
        assert z.read("phase5a6_preflight.json")==Path("phase5a6_preflight.json").read_bytes()
        assert z.read("release.json")==Path("reports/phase5a6/release.json").read_bytes()
    nb=Path("notebooks/step05a6_dioscuri_kaggle.ipynb")
    assert sha(nb)==ORIGINAL
    tree=ast.parse("".join(read(nb)["cells"][1]["source"]))
    payload=base64.b64decode(next(ast.literal_eval(n.value.args[0]) for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=="payload"))
    assert digest(payload)==OLD
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        science=read("reports/phase5a6/scientific.json")
        for n,pin in science.items():
            if n in z.namelist():assert digest(z.read(n))==pin
            if Path(n).is_file():assert sha(n)==pin
            else:assert n in z.namelist()
        cfg=json.loads(z.read("manifest.json"))
        for n,pin in cfg["mounts"].items():assert sha(Path("kaggle_upload/phase5a5_private_inputs")/n)==pin
        verify_opaque(Path("kaggle_upload/phase5a5_private_inputs/dioscuri_opaque.bin"),cfg["pins"]["image_permutation"])
        files={n:z.read(n) for n in z.namelist() if n.endswith(".py") and (n.startswith("src/") or n in ("tests/test_phase5a1.py","scripts/phase5a6_production.py"))}
        files.update({n:z.read(n) for n in z.namelist() if n.startswith("configs/novel_model/")})
    p=Path("scripts/phase5a7_persistence.py")
    files[p.as_posix()]=p.read_bytes()
    blob=archive(files)
    revision=digest(blob)
    setup=f"""import base64, hashlib, io, sys, tempfile, zipfile
from pathlib import Path
revision={revision!r}
payload=base64.b64decode({base64.b64encode(blob).decode()!r})
assert hashlib.sha256(payload).hexdigest()==revision
scratch=tempfile.TemporaryDirectory()
kit=Path(scratch.name)
with zipfile.ZipFile(io.BytesIO(payload)) as z: z.extractall(kit)
sys.path[:0]=[str(kit),str(kit/'tests')]
from scripts.phase5a7_persistence import produce, verify, RECEIPT, save
output=Path('/kaggle/working')
"""
    producer="""assert not any(p.is_file() for p in Path('/kaggle/input').rglob('*')), 'Detach all inputs'
print('Producer receipt SHA256:',produce(output,revision),flush=True)
"""
    reader="""sources=list(Path('/kaggle/input').rglob(RECEIPT))
assert len(sources)==1, 'Attach the completed private producer version as notebook output'
pin=input('Paste the receipt SHA256 printed in the completed producer version log: ').strip()
result=verify(sources[0].parent,pin,revision)
result['retrieved_from']=str(sources[0].parent)
save(output/'phase5a7_external_restore.json',result)
print(result,flush=True)
"""
    names={}
    for role,code in (("producer",producer),("reader",reader)):
        note=("Private CPU saved-version producer. Internet off, no inputs. Use Save Version > Save & Run All." if role=="producer" else "Fresh private CPU reader. Internet off. Attach only the completed producer version output and run interactively.")
        cells=[dict(cell_type="markdown",metadata={},id="instructions",source=[note]),
               dict(cell_type="code",metadata={},id=role,source=(setup+code).splitlines(True),outputs=[],execution_count=None)]
        p=Path("notebooks/step05a7_custody_"+role+".ipynb")
        save(p,dict(nbformat=4,nbformat_minor=5,metadata={"kernelspec":dict(name="python3",display_name="Python 3",language="python")},cells=cells))
        names[p.name]=sha(p)
    save(R/"verified.json",dict(production=OLD,notebook=ORIGINAL,fixture_release=revision,fixture_notebooks=names,
         manifest=pins,science=science,journals=JOURNALS,inputs=cfg["mounts"],worker=cfg["pins"]["worker"],
         permutation=cfg["pins"]["image_permutation"],recovery_status=status))
    return revision
if __name__=="__main__":print(build())
