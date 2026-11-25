"""CPU-only check with the real pinned PyCOLMAP options, without model execution."""
import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/baselines/organizer'))
from contracts import json_default,fingerprint,write
import pycolmap
options=pycolmap.IncrementalPipelineOptions().todict()
normalized=json.loads(json.dumps(options,default=json_default,allow_nan=False))
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'runtime.json';write(p,{'pycolmap_options':options})
    assert json.loads(p.read_text())['pycolmap_options']==normalized
assert fingerprint(options)==fingerprint(normalized)
print(json.dumps({'pycolmap_version':pycolmap.__version__,'status':'PASS','option_keys':list(normalized),'normalized_options':normalized},indent=2))
