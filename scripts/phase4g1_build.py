import ast,hashlib,json
from pathlib import Path
from scripts.phase4g_build_notebooks import RUN

def build():
    monitor=Path("src/novel_model/resource_monitor_v2/__init__.py")
    policy=Path("configs/novel_model/phase4g1_resource_policy.yaml")
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    pins=json.loads(Path("reports/phase4g/protocol_manifest.json").read_bytes())
    receipt={"original_policy":pins["files"]["configs/novel_model/reconstruction_feedback_v2.yaml"],
             "model":pins["model_bundle"],"monitor":sha(monitor),"policy":sha(policy),"notebooks":{}}
    for ds,pin in pins["datasets"].items():
        old=Path(f"notebooks/step04g_{ds}_colab.ipynb")
        assert sha(old)==pin["notebook"]
        nb=json.loads(old.read_bytes())
        cell=nb["cells"][-1]
        source="".join(cell["source"])
        assert source.endswith(RUN)
        code=RUN.replace("def launch(args,label):",monitor.read_text()+"\nPOLICY="+repr(json.loads(policy.read_bytes()))+"\ndef launch(args,label):")
        i=code.index("                    try:\n                        process=")
        j=code.index("                if proc.returncode",i)
        code=code[:i]+"""                    row=sample(proc.pid)
                    row.update(stage=label,remaining=deadline-time.monotonic(),elapsed=time.monotonic()-start)
                    with (work/'heartbeat.jsonl').open('a') as f:print(json.dumps(row),file=f)
                    guard(row,initial,POLICY)
                    peak=max(peak,row['totals']['rss'])
                    if time.monotonic()>=deadline:raise TimeoutError(label)
                    time.sleep(POLICY['sample_seconds'])
"""+code[j:]
        code=code.replace("    launch([str(work/'worker.py')", """    initial=sample(os.getpid())
    initial.update(remaining_seconds=remaining,processes=[p.info for p in psutil.process_iter(['pid','name'])])
    (work/'resource_preflight.json').write_text(json.dumps(initial))
    guard(initial,initial,POLICY)
    if remaining<MIN_TIME:raise RuntimeError('Insufficient session time for historical capture and bounded trial')
    launch([str(work/'worker.py')""".replace("MIN_TIME","1800" if ds=="stairs" else "3000"))
        code=code.replace("record.update(protocol=",f"record.update(experiment_id='phase4g1_{ds}_20261111',resource_policy='{sha(policy)}',monitor='{sha(monitor)}',protocol=")
        code=code.replace("archive=root/('phase4g_'","archive=root/('phase4g1_'").replace("    files.download(str(archive))","    with zipfile.ZipFile(archive) as check: assert check.testzip() is None\n    files.download(str(archive))")
        cell["source"]=(source[:-len(RUN)]+code).splitlines(True)
        ast.parse("".join(cell["source"]))
        new=Path(f"notebooks/step04g1_{ds}_colab.ipynb")
        new.write_text(json.dumps(nb,indent=1))
        receipt["notebooks"][ds]=sha(new)
    Path("reports/phase4g1/manifest.json").write_text(json.dumps(receipt,indent=2))
if __name__=="__main__":build()
