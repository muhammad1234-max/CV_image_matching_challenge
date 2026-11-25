"""Execute CPU integration tests with import precedence given to verified notebook bytes."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    root=Path.cwd()
    report=root/"reports/phase5a3"
    nb=root/"notebooks/step05a3_dioscuri_colab.ipynb"
    revision=json.loads((report/"revision.json").read_bytes())
    assert sha(nb.read_bytes())==revision["notebook"]
    tree=ast.parse("".join(json.loads(nb.read_bytes())["cells"][3]["source"]))
    assignments={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    with tempfile.TemporaryDirectory(dir=root/"tmp") as t:
        kit=Path(t)
        for node,pin in ((assignments["EMBED"],ast.literal_eval(assignments["KIT_PIN"])),(assignments["ops"].args[0],revision["runner"])):
            raw=base64.b64decode(ast.literal_eval(node))
            assert sha(raw)==pin
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for name in z.namelist():
                    if name.startswith(("src/","configs/")):
                        assert z.read(name)==(root/name).read_bytes(),name
                z.extractall(kit)
        code="""import sys,pathlib,pytest
kit=pathlib.Path(sys.argv[1])
sys.path.insert(0,str(kit))
from src.runner_5a3 import colab,provenance
from src.runner_5a1 import runner
from src.novel_model.final_candidate_v1 import session,workflow
assert all(pathlib.Path(m.__file__).is_relative_to(kit) for m in (colab,provenance,runner,session,workflow))
raise SystemExit(pytest.main(sys.argv[2:]))
"""
        with (report/"generated.log").open("w") as log:
            subprocess.run([sys.executable,"-c",code,str(kit),"tests/test_phase5a3.py","-q","--tb=short",
                         "--basetemp=tmp/phase5a3_generated","--junitxml=reports/phase5a3/generated.xml"],stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=="__main__":main()
