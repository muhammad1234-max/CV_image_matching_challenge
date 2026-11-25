import ast,base64,hashlib,io,json,zipfile
from pathlib import Path
from scripts.phase5a5_build import archive,save
from scripts.phase5a7_recovery import LATEST,DEBIT
from scripts.phase5a7_build import OLD,ORIGINAL

def digest(b):return hashlib.sha256(b).hexdigest()
def build():
    parent=Path("notebooks/step05a6_dioscuri_kaggle.ipynb")
    assert digest(parent.read_bytes())==ORIGINAL
    nb=json.loads(parent.read_bytes())
    src="".join(nb["cells"][1]["source"])
    expr=next(n.value for n in ast.parse(src).body if isinstance(n,ast.Assign) and n.targets[0].id=="payload")
    payload=base64.b64decode(ast.literal_eval(expr.args[0]))
    assert digest(payload)==OLD
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        files={n:z.read(n) for n in z.namelist()}
    name=f"journal-{LATEST}.json"
    files[name]=Path(name).read_bytes()
    assert digest(files[name])==LATEST
    receipt=Path("phase5a7_private_external_restore.json").read_bytes()
    proof=json.loads(receipt)
    producer=Path("reports/phase5a7/verified_kaggle_fixture/phase5a7_producer.json")
    assert proof["status"]=="FIXTURE_RESTORE_PASS" and digest(producer.read_bytes())==proof["producer_sha256"]
    assert all(proof[k]==json.loads(producer.read_bytes())[k] for k in ("archive","bytes","journal","release","parent","seal"))
    code=files["scripts/phase5a6_production.py"].decode().split('    runroot=repo/"actual"')[0]
    a=code.index('    latest=digest_input')
    b=code.index('    kaggle.verify_opaque',a)
    code=code[:a]+f'''    latest={LATEST!r}
    journal=repo/("journal-"+latest+".json")
    equal(kaggle.sha(journal),latest,"completed capture journal")
    history=read(journal)
    spent=sum(e.get("seconds",0) for e in history)
    equal(spent,{DEBIT!r},"historical capture debit")
    budget=14400-spent
    check(min(remaining,budget)>=8400,"insufficient capture window before setup")
    mounts=dict(cfg["mounts"])
    mounted=kaggle.inputs(Path("/kaggle/input"),mounts)
'''+code[b:]
    code=code.replace('phase5a6_preflight.json','phase5a7_gpu_preflight.json')
    code+='''    space(repo,output,plan)
    live=kaggle.sample(os.getpid())
    kaggle.guard(live,idle,policy)
    record("resources_after_setup",live)
    record("gpu_after_setup",check_gpu(env))
    record("historical_journal",latest)
    record("stage_seconds_remaining",budget)
    record("recovery_captures_started",0)
    record("session_seconds_remaining",deadline-time.monotonic())
    check(min(deadline-time.monotonic(),budget)>=8400,"insufficient window after setup")
    record("preflight","PASS")
'''
    code+= '\n'
    files["scripts/phase5a7_gpu_gate.py"]=code.encode()
    files["phase5a7_persistence_evidence.json"]=receipt
    blob=archive(files)
    rev=digest(blob)
    cell=src.replace(base64.b64encode(payload).decode(),base64.b64encode(blob).decode()).replace(OLD,rev).replace('from scripts.phase5a6_production import preflight','from scripts.phase5a7_gpu_gate import preflight').replace('run=preflight','preflight').replace("PREFLIGHT PASS. Select the next authorized stage cell.","PREFLIGHT PASS. Download phase5a7_gpu_preflight.json.").replace("phase5a6_blocker.json","phase5a7_gpu_blocker.json")
    compile(code,"gate","exec")
    compile(cell,"loader","exec")
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="run" and isinstance(n.func.value,ast.Name) and n.func.value.id=="runner" for n in ast.walk(ast.parse(code)))
    nb["cells"]=nb["cells"][:2]
    nb["cells"][0]["source"]=["Phase 5A.7 recovery GPU preflight. Fresh private T4 session. Attach only ns64_runtime.bin and dioscuri_opaque.bin. Internet on for setup. Enter actual remaining session seconds. This notebook ends at preflight."]
    nb["cells"][1]["source"]=cell.splitlines(True)
    target=Path("notebooks/step05a7_recovery_gpu_preflight.ipynb")
    save(target,nb)
    save(Path("reports/phase5a7/gpu_preflight_release.json"),dict(preflight_release=rev,parent=OLD,notebook=digest(target.read_bytes()),journal=LATEST,restore_receipt=digest(receipt),visibility="owner-confirmed private",private_reader_run="operator-reported; receipt has no run timestamp",actual_gpu_result=None))
    print(target,rev)
if __name__=="__main__":build()
