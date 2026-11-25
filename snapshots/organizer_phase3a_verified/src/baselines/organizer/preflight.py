"""Original package ABI gate and offline model-asset pinning; no GPU installs locally."""
import importlib.metadata as md,json,os,platform,shutil,subprocess,sys
from pathlib import Path
from contracts import sha,require_files,write,json_default
def run(root,mounts,work,remaining):
    report={'python':platform.python_version(),'platform':platform.platform(),'gpu':None,'packages_before':{d.metadata['Name']:d.version for d in md.distributions()}}
    try:
        import torch
        report.update(torch=torch.__version__,cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0)
    except Exception as e:report['gpu_inspection_error']=str(e)
    write(work/'runtime.json',report)
    if sys.platform!='linux' or sys.version_info[:2]!=(3,10):raise RuntimeError('Original wheels require Linux CPython 3.10; current runtime '+platform.python_version()+'. No package substitution performed. Use a verified Python 3.10 GPU runtime or request an explicit adaptation review.')
    source=root/'docs/evidence/phase3a_sources';manifest=json.loads((root/'configs/baselines/organizer_v1/assets.json').read_text())
    wheels=[Path(mounts['packages'])/n for n in manifest['wheels']]
    for p in wheels:
        require_files([p])
        if sha(p)!=manifest['wheels'][p.name]:raise ValueError('Wheel hash mismatch '+p.name)
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*[str(p) for p in wheels]],check=True,timeout=max(1,remaining()))
    for name,expected in manifest['package_versions'].items():
        if md.version(name)!=expected:raise ValueError('Installed package version mismatch '+name)
    import torch,pycolmap,kornia,h5py,transformers,lightglue
    import numpy as np
    if int(np.__version__.split('.')[0])>=2:raise RuntimeError('Historical compiled package ABI under NumPy 2 is not verified. Use a reviewed NumPy 1.x environment; no silent downgrade.')
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is unavailable')
    report.update(torch=torch.__version__,cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(0),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,packages={d.metadata['Name']:d.version for d in md.distributions()})
    report['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=20).stdout
    report['pycolmap_options']=json.loads(json.dumps(pycolmap.IncrementalPipelineOptions().todict(),default=json_default,allow_nan=False))
    write(work/'runtime.json',report)
    dino=Path(mounts['dinov2']);aliked=Path(mounts['aliked']);lg=Path(mounts['lightglue'])
    require_files([dino/'config.json',dino/'preprocessor_config.json',aliked,lg])
    weights=list(dino.glob('*.bin'))+list(dino.glob('*.safetensors'))
    if not weights:raise FileNotFoundError('Missing local DINOv2 weights; runtime downloads disabled')
    model_cfg=json.loads((dino/'config.json').read_text())
    write(work/'dinov2_configuration.json',{'model':model_cfg,'preprocessing':json.loads((dino/'preprocessor_config.json').read_text())})
    if model_cfg.get('model_type')!='dinov2' or model_cfg.get('hidden_size')!=768 or model_cfg.get('patch_size')!=14:raise ValueError('Expected DINOv2 base ViT-B/14 config')
    hashes=require_files([p for p in dino.iterdir() if p.is_file()]+[aliked,lg])
    lock=work/'checkpoint_hashes.json'
    if lock.exists() and json.loads(lock.read_text())!=hashes:raise ValueError('Checkpoint bytes changed from first preflight; use a new run directory')
    write(lock,hashes)
    cache=work/'torch/hub/checkpoints';cache.mkdir(parents=True,exist_ok=True)
    for src,name in [(aliked,'aliked-n16.pth'),(lg,'aliked_lightglue.pth'),(lg,'aliked_lightglue_v0-1_arxiv-pth')]:shutil.copyfile(src,cache/name)
    report['checkpoint_hashes']=hashes;return report
