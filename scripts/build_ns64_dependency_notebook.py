import hashlib
import json
from pathlib import Path

root=Path('configs/baselines/top10_reproduction_v1')
src=Path('snapshots/top10_ns')
plan=json.loads((root/'pilot.json').read_text())
locks=json.loads(Path('configs/baselines/organizer_v1/full_checkpoint_lock.json').read_text())
sources={'superpoint':'https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/weights/superpoint_v1.pth','aliked':'https://raw.githubusercontent.com/Shiaoming/ALIKED/main/models/aliked-n16.pth','dinov2':'https://huggingface.co/facebook/dinov2-base/blob/main/pytorch_model.bin','isc':'https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_ft_v107.pth.tar'}
aliases=['MAGICLEAP_SUPERPOINT','ALIKED_LIGHTGLUE_N16','MAST3R','MAST3R_RETRIEVAL','MAST3R_RETRIEVAL_CODEBOOK','ISC','DINOV2_BASE']
sizes={'aliked':2738091,'dinov2':346392713,'superpoint':5206086,'mast3r':2754910614,'retrieval':8399111,'codebook':268435691,'isc':420418996}
assets={}
for (k,f),alias in zip(plan['weight_files'].items(),aliases):
    remote=root/('upstream_'+k+'.json')
    meta=json.loads(remote.read_text()) if remote.exists() else {}
    if meta.get('size')!=sizes.get(k):meta={}
    assets[k]=dict(filename=f,alias=alias,url=sources.get(k,'https://download.europe.naverlabs.com/ComputerVision/MASt3R/'+f),sha256=meta.get('sha256',locks.get(f)),size=sizes.get(k))
assets['superpoint']['sha256']='52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e'
assets['dinov2']['sidecars']={k:locks[k] for k in ['config.json','preprocessor_config.json']}
license_notes={'superpoint':'Magic Leap academic or nonprofit noncommercial research license',
    'dinov2':'Meta Apache-2.0 for code and weights','aliked':'ALIKED BSD-3-Clause; retain upstream notices',
    'isc':'ISC repository MIT; training-data rights and checkpoint redistribution permission require separate review'}
formats={'superpoint':'SuperPoint CNN, state dict','aliked':'ALIKED N16 state dict','dinov2':'ViT-B/14 Transformers directory',
    'mast3r':'AsymmetricMASt3R ViT-Large / base decoder 512 PyTorch checkpoint',
    'retrieval':'training-free retrieval head checkpoint','codebook':'64k ASMK codebook pickle','isc':'ISCNet EfficientNetV2-M fine-tuned v107 PyTorch tar checkpoint'}
modules={'superpoint':'models._research_only.magicleap.superpoint','aliked':'features.lightglue.aliked',
    'dinov2':'global_descriptors.dinov2','mast3r':'models.mast3r.model','retrieval':'mast3r.retrieval.processor',
    'codebook':'asmk.asmk_method','isc':'models.isc.model'}
for k,a in assets.items():
    a.update(format=formats[k],module=modules[k],license=license_notes.get(k,'CC BY-NC-SA 4.0 and upstream training dataset conditions, including MapFree. Review CHECKPOINTS_NOTICE.'),
             published_sha256=a['sha256'] if k=='dinov2' else None,observed_in_previous_kaggle=k in ('aliked','dinov2'),load_status='UNTESTED')
probes={'python':"import platform,sys; assert platform.system()=='Linux' and sys.version_info[:2]==(3,11);print(sys.version)",
'gpu':"import subprocess;subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],check=True)",
'torch':"import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_capability());assert torch.__version__.split('+')[0]=='2.5.1';print(torch.ones(1,device='cuda').item())",
'distributed':"import torch.distributed as d;from distributed import DistConfig;print(DistConfig.single());assert d.is_nccl_available()",
'pycolmap':"import pycolmap as p;print(p.__version__);assert p.__version__=='3.11.1';p.IncrementalPipelineOptions();p.SiftMatchingOptions();assert callable(p.incremental_mapping) and callable(p.match_exhaustive)",
'driver_toolkit':"import shutil,subprocess,torch;print(torch.version.cuda);assert torch.version.cuda=='12.1';print(subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True));print(shutil.which('nvcc') or 'runtime only')",
'pipeline':"from kernel import run;from config import PipelineConfig;print(PipelineConfig.load_config("+repr(plan['config'])+").type)"}
for mod in ['asmk.hamming','asmk.asmk_method','curope','mast3r.model','dust3r.inference','lightglue','models._research_only.magicleap.superpoint','models.isc.model','faiss']:probes[mod]='import torch;import '+mod
for pkg,ver in {'torchvision':'0.20.1','numpy':'1.26.4','kornia':'0.8.0','transformers':'4.47.0'}.items():probes[pkg]=f"import {pkg};assert {pkg}.__version__.split('+')[0]=={ver!r}"
prefix="import sys,torch;from pathlib import Path;p=Path(sys.argv[1]);"
retrieval="from models.mast3r.model import get_mast3r_model;from mast3r.retrieval.processor import Retriever; m=get_mast3r_model(p,torch.device('cuda'));r=Retriever(sys.argv[2],backbone=m,device='cuda');assert r.asmk is not None"
loaders={k:prefix+v for k,v in {'superpoint':"from models._research_only.magicleap.superpoint import SuperPoint;SuperPoint(dict(model_path=str(p),fix_sampling=True,max_keypoints=4096,keypoint_threshold=.0005)).eval().cuda()", 'aliked':"from features.lightglue.aliked import ALIKED;ALIKED(str(p),model_name='aliked-n16',max_num_keypoints=4096).eval().cuda()", 'dinov2':"from transformers import AutoModel,AutoImageProcessor;AutoModel.from_pretrained(p.parent,local_files_only=True).eval().cuda();AutoImageProcessor.from_pretrained(p.parent,local_files_only=True)",'mast3r':"from models.mast3r.model import get_mast3r_model;m=get_mast3r_model(p,torch.device('cuda'));v={'instance':['left'],'img':torch.zeros(1,3,224,224,device='cuda'),'true_shape':torch.tensor([[224,224]])};\nwith torch.inference_mode(): a,b=m(v,dict(v,instance=['right']));assert torch.isfinite(a['pts3d']).all();torch.cuda.synchronize()", 'retrieval':retrieval,'codebook':retrieval,'isc':"from models.isc.model import create_model;create_model(p,torch.device('cuda'))"}.items()}
manifest={'revision':plan['revision'],'archive_sha256':plan['source_zip_sha256'],'config':plan['config'],'config_sha256':plan['config_sha256'],'assets':assets,'probes':probes,'loaders':loaders}

env={'status':'proposed and untested','python':'3.11.11','platform':'linux-x86_64','torch':'2.5.1+cu121','torchvision':'0.20.1+cu121','cuda_toolkit':'12.1','numpy':'1.26.4','kornia':'0.8.0','transformers':'4.47.0','pycolmap':'3.11.1','pycolmap_cp311_linux_sha256':'9bb98585e540a403df718b26ffdf475eba74891d16bf59ba94eef0f3acb827a5','asmk_revision':'2a96d9c03a841dffdfddabc699a20512dcd09363','croco_revision':'d7de0705845239092414480bd829228723bf20de','compiler_target':'GCC/G++ 11, T4 sm_75','upstream_wheels':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (src/'bundle/oss').rglob('*.whl')}}
manifest.update(bundled_wheels=env['upstream_wheels'],pycolmap_filename='pycolmap-3.11.1-cp311-cp311-manylinux_2_28_x86_64.whl',pycolmap_sha256=env['pycolmap_cp311_linux_sha256'])
(root/'assets_phase3c1.json').write_text(json.dumps(manifest,indent=2))
(root/'linux_target.json').write_text(json.dumps(env,indent=2))
worker=Path('src/baselines/top10_reproduction/verify_dependencies.py').read_text()
launch="""from pathlib import Path
import sys, subprocess, tempfile, os, signal
from IPython.display import FileLink, display
working=Path('/kaggle/working')
receipt=working/'imc_ns64_dependency_receipt.zip'
if receipt.exists(): raise FileExistsError('Use a fresh session to preserve the previous receipt')
script=working/'phase3c1_verify.py'
script.write_text(WORKER)
try:
    child_env=os.environ.copy()
    child_env['MPLBACKEND']='Agg'
    child=subprocess.Popen(['/kaggle/working/ns64_env/bin/python',str(script)],env=child_env,start_new_session=True)
    try:
        if child.wait(timeout=1800): raise RuntimeError('Dependency worker failed')
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        print('Time budget reached. Download partial receipt.')
finally:
    if receipt.exists(): display(FileLink(str(receipt)))
"""
worker+="\nroot=Path('/kaggle/input')\nbundle=next(input_files([root],'ns64_pilot_source.zip.bin'),root/'missing-source')\nr=verify("+repr(manifest)+",[root,Path('/kaggle/working/ns64_weights')],bundle,Path('/kaggle/working/imc_ns64_dependency_receipt.zip'))\nprint(r['status'],{k:v['status'] for k,v in r['checks'].items()})\n"
launch=launch.replace('write_text(WORKER)','write_text('+repr(worker)+')')
for code in (launch,worker):compile(code,'dependency-notebook','exec')
nb=json.loads(Path('notebooks/step03c_top10_reproduction_pilot.ipynb').read_text())
nb['cells'][0]['source']=launch.splitlines(True)
Path('notebooks/step03c1_ns64_dependencies.ipynb').write_text(json.dumps(nb,indent=2))

lines=['# Active checkpoint inventory','',
       'Prior Kaggle PASS for source and images is independent of model loading. All seven GPU loader checks are pending. Meta publishes the DINOv2 SHA256 as the LFS object identity, matching the Kaggle mount. Other rows have no confirmed author-published SHA256. Hashes below are project measurements, from the original download or the previous pinned Kaggle mount. An unavailable value is explicitly marked pending. Sizes are bytes.','',
       '| Model | Expected file and authoritative source | Size | SHA256 | Previous mount |',
       '|---|---|---:|---|---|']
for k,a in assets.items():
    lines.append(f"| {k} | [{a['filename']}]({a['url']}) | {a['size'] or 'pending'} | {a['sha256'] or 'pending'} | {'present' if a['observed_in_previous_kaggle'] else 'missing'} |")
lines+=['','| Model | Architecture and serialization | Required module | License context |','|---|---|---|---|']
for k,a in assets.items():
    lines.append(f"| {k} | {a['format']} | {a['module']} | {a['license']} |")
lines+=['','The retrieval loader derives its codebook filename from the head filename, so those two must be colocated. DINOv2 also needs both hashed JSON sidecars. The alternative Hugging Face MASt3R safetensors file has a different serialization. DAD and VGGT have no active weight references in the selected YAML.','',
'[ns64 model paths](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/07b5695b99a8ae4725d5d952f0065c2157fadd7d/conf/models.yaml) | [NAVER checkpoint notice](https://github.com/naver/mast3r/blob/main/CHECKPOINTS_NOTICE) | [Magic Leap terms](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE) | [ALIKED](https://github.com/Shiaoming/ALIKED) | [ISC](https://github.com/lyakaap/ISC21-Descriptor-Track-1st)']
Path('docs/step03c1_asset_manifest.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
