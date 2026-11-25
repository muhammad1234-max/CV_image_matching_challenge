"""Load original organizer function definitions, never its top-level dataset loop."""
import ast,json,os
from pathlib import Path
def load(root,dino_dir):
    import torch,numpy as np,h5py,kornia as K,kornia.feature as KF
    from lightglue import ALIKED
    from transformers import AutoImageProcessor,AutoModel
    from tqdm import tqdm
    from database import COLMAPDatabase
    from h5_to_db import add_keypoints,add_matches
    nb=json.loads((Path(root)/'docs/evidence/phase3a_sources/organizer.ipynb').read_text());defs=[]
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        try:tree=ast.parse(''.join(cell['source']))
        except SyntaxError:continue
        defs.extend(n for n in tree.body if isinstance(n,ast.FunctionDef))
    ns=dict(torch=torch,np=np,h5py=h5py,K=K,KF=KF,ALIKED=ALIKED,AutoImageProcessor=AutoImageProcessor,AutoModel=AutoModel,tqdm=tqdm,os=os,F=torch.nn.functional,COLMAPDatabase=COLMAPDatabase,add_keypoints=add_keypoints,add_matches=add_matches)
    tree=ast.Module(body=defs,type_ignores=[])
    for n in ast.walk(tree):
        if isinstance(n,ast.Constant) and n.value=='/kaggle/input/dinov2/pytorch/base/1':n.value=str(dino_dir)
    exec(compile(ast.fix_missing_locations(tree),'organizer_v1_functions','exec'),ns);return ns
