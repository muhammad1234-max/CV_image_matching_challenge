"""Runs inside the isolated container. Contains no inference entry point."""
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

root=Path('/input')
out=Path('/output')
receipt={'status':'BLOCKED'}
try:
    inventory={}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            with path.open('rb') as handle:
                inventory[path.relative_to(root).as_posix()]=hashlib.file_digest(handle,'sha256').hexdigest()
    receipt['package_hash']=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if receipt['package_hash']!=sys.argv[2]:
        raise ValueError('Mounted inventory mismatch')
    worker=runpy.run_path('/input/worker.py')
    worker['probe']()
    canary=Path(sys.argv[1])
    for path in (canary,Path('/proc/1/root')/canary.relative_to('/')):
        try:
            with path.open('rb'): pass
        except (FileNotFoundError,PermissionError):
            continue
        raise ValueError('Administrative canary readable')
    receipt['network_interfaces']=sorted(p.name for p in Path('/sys/class/net').iterdir())
    receipt['process_ids']=sorted(int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit())
    if receipt['network_interfaces']!=['lo'] or os.getpid()!=1 or os.getgid()!=65534:
        raise ValueError('Unexpected namespace or identity')
    import torch
    import asmk.hamming, curope, pycolmap, mast3r.model, dust3r.inference
    import yaml
    if torch.__version__!='2.5.1+cu121' or not torch.cuda.is_available() or pycolmap.__version__!='3.11.1':
        raise RuntimeError('Incompatible runtime')
    source=root/'source'
    sys.path.insert(0,str(source))
    definitions=yaml.safe_load((source/'conf/models.yaml').read_text())
    aliases=json.loads((root/'aliases.json').read_text())
    for alias,name in aliases.items():
        path=Path(name).resolve(strict=True)
        if not path.is_relative_to(root/'models'):
            raise ValueError('Model alias escapes mount')
        definitions[alias]=dict(local=name,kernel=name)
    aliasfile=out/'models.json'
    aliasfile.write_text(json.dumps(definitions))
    os.environ.update(DEFAULT_MODEL_LIST_PATH=str(aliasfile),DEFAULT_EXTRA_DIR=str(out/'extra'))
    from pipeline import create_pipeline
    from config import PipelineConfig
    from data import resolve_model_path
    for key,name in aliases.items():
        if Path(resolve_model_path(key)).resolve()!=Path(name):
            raise ValueError('Published alias resolver mismatch')
    configuration=PipelineConfig.load_config(source/worker['CONFIG'])
    receipt.update(status='PASS',config_type=configuration.type,device=torch.cuda.get_device_name(0),
                   canary='denied',torch=torch.__version__,pycolmap=pycolmap.__version__)
except Exception as error:
    receipt.update(error_type=type(error).__name__,module=getattr(error,'name',None))
    raise
finally:
    (out/'readiness.json').write_text(json.dumps(receipt,indent=2))
