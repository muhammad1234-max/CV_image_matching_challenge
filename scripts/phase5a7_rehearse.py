"""Exercise the shipped notebook cells in separate interpreters with workstation fixture mounts."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
from src.runner_5a5.kaggle import sha,save

def main():
    root=Path.cwd()
    reports=root/"reports/phase5a7"
    with tempfile.TemporaryDirectory(dir=root/"tmp") as tmp:
        temp=Path(tmp)
        producer=temp/"saved_output"
        reader=temp/"reader_output"
        producer.mkdir()
        reader.mkdir()
        dest=None
        for role in ("producer","reader"):
            nb=root/f"notebooks/step05a7_custody_{role}.ipynb"
            code="".join(json.loads(nb.read_bytes())["cells"][1]["source"])
            dest=producer if role=="producer" else reader
            code=code.replace("Path('/kaggle/working')",f"Path({str(dest)!r})").replace("Path('/kaggle/input')",f"Path({str(temp/'empty' if role=='producer' else producer)!r})")
            script=temp/(role+".py")
            script.write_text(code,encoding="utf8")
            pin=sha(producer/"phase5a7_producer.json") if role=="reader" else ""
            result=subprocess.run([sys.executable,str(script)],cwd=temp,text=True,input=pin+"\n",capture_output=True,
                                  env=dict(os.environ,TEMP=str(temp),TMP=str(temp)))
            if result.returncode:raise RuntimeError(result.stderr)
        receipt=json.loads((reader/"phase5a7_external_restore.json").read_bytes())
        receipt.pop("retrieved_from")
        save(reports/"workstation_transfer.json",dict(location="separate workstation processes; local storage",**receipt))
        print(receipt["bytes"],receipt["status"])
if __name__=="__main__":main()
