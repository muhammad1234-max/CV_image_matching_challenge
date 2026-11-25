"""CPU boundary for executing the notebook children. Only fixture paths, deployment pins
and external platform/native services are substituted. The generated control flow runs."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
if __name__=="__main__":
    fixture_spec=json.loads(Path(sys.argv[1]).read_bytes())
    sys.path.insert(0,fixture_spec["kit"])
from src.novel_model.final_candidate_v1 import workflow
from src.runner_5a3.provenance import DS,read,save,file_sha

class CPU:
    def __init__(self,fx,base):
        self.fx=fx
        self.base=base
        self.failure=""
        self.original=base/"dbfixture"
        self.original.mkdir()
        fx.child("capture",self.original,0)
        self.config=base/"ns64_clean/source/conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-022-c.yaml"
        self.config.parent.mkdir(parents=True)
        self.config.write_text("synthetic native configuration")
        (base/"ns64_clean/source/conf/models.yaml").write_text("{}")
        (base/"ns64_clean/assets.json").write_text("{}")
        self.mem=base/"meminfo"
        self.mem.write_text("MemAvailable: 20000000 kB")
        self.p=fx.p

    def monitor(self,cmd,work,d,end,stage):
        from contextlib import ExitStack
        root=self.base
        fx=self.fx
        published=read(self.original/"identity.json")
        published.update(benchmark=self.p["benchmark"],source_revision=self.p["source_revision"],config_sha256=self.p["published_config"])
        env=dict(status="CLEAN_VM_PRECHECK_PASS",source_inventory_hash=self.p["source_inventory"],assets={},
                 environment_freeze="freeze",gpu="Tesla T4, 580.82.07, 15360 MiB",administrative_paths={})
        def probe():
            save(work/"environment.json",env)
        def sha(s):return hashlib.sha256(s.encode()).hexdigest()
        substitutions={"/content/ns64_clean":str(root/"ns64_clean"),"/proc/meminfo":str(self.mem),
            "/content/"+DS+"_opaque.zip":str(root/(DS+"_opaque.zip")),
            "1b7bc97f6f54a2f5daa967a06e67211cd565f84d2d9403cd60f1faa50af543b0":self.p["image_permutation"],
            "696df0470cebc09a532d45466819b43bd363656402b761a23bd6a9ee403b0ca0":file_sha(root/"ns64_clean/assets.json"),
            "8bb71679bf5735f6e3c8225885be88c1f530639094df310a5a29e6691498455b":sha("freeze")}
        class Platform(ast.NodeTransformer):
            def visit_Constant(self,node):
                return ast.copy_location(ast.Constant(substitutions[node.value]),node) if node.value in substitutions else node
            def visit_Attribute(self,node):
                if ast.unparse(node)=="sys.version_info":
                    return ast.copy_location(ast.parse("(3,11,11)",mode="eval").body,node)
                return self.generic_visit(node)
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name) and node.func.id=="exec" and isinstance(node.args[0],ast.Constant) and node.args[0].value.startswith("SOURCE_HASHES ="):
                    return ast.copy_location(ast.Call(ast.Name("fixture_probe",ast.Load()),[],[]),node)
                return self.generic_visit(node)
        class Options:
            def summary(self):return "options"
        class Map:
            def __init__(self,group):
                self.images={fx.names.index(n)+1:NS(name=n) for n in group}
            def reg_image_ids(self):return list(self.images)
            def num_points3D(self):return 40
        def map_native(**kw):
            if self.failure=="timeout":raise TimeoutError(stage)
            if self.failure=="unknown":raise RuntimeError("unknown native failure")
            if self.failure=="digest":raise ValueError("trial input digest mismatch")
            groups=[fx.names[:12]] if stage=="original" else [fx.names[:6],fx.names[6:12]]
            return {i:Map(g) for i,g in enumerate(groups)}
        native=NS(IncrementalPipelineOptions=Options,incremental_mapping=map_native,__version__="3.11.1")
        impl=NS(pycolmap=native)
        for name in ("run_local_feature_extraction_and_matching_stage","run_detector_free_matching_stage",
                     "run_point_tracking_matching_stage","fuse_matching_sets_late"):
            setattr(impl,name,lambda:None)
        impl.verify_matches=lambda:NS(count=31,inliers={})
        class Pipeline:
            shortlist_generator=lambda *a:None
            def run(self,frame,data):
                impl.verify_matches()
                native.incremental_mapping(database_path=str(fx.tmp/"dbfixture/parent.db"),
                                           image_path=str(work/"test"/DS),options=Options())
        cuda=NS(max_memory_allocated=lambda:0,max_memory_reserved=lambda:0,get_device_name=lambda i=0:"Tesla T4",
                mem_get_info=lambda:(12*1024**3,15*1024**3))
        modules=dict(resource=NS(getrusage=lambda _:NS(ru_maxrss=100),RUSAGE_SELF=0),
                     torch=NS(cuda=cuda,version=NS(cuda="12.1"),device=lambda x:x),pycolmap=native,
                     config=NS(PipelineConfig=NS(load_config=lambda x:{})),pipeline=NS(create_pipeline=lambda *a,**k:Pipeline()),
                     data=NS(IMC2025TestData=lambda *a,**k:None),pipelines=NS(imc2025_pipeline=impl))
        modules["pipelines.imc2025_pipeline"]=impl
        def observations(maps,parent):
            groups=[[row.name for row in model.images.values()] for model in maps.values()]
            return groups,{n:np.ones(40) for g in groups for n in g}
        def predictions(maps,ds,names,images,target):
            groups,_=observations(maps,None)
            with target.open("w",newline="") as f:
                w=csv.writer(f)
                w.writerow(["image_id","dataset","scene","image","rotation_matrix","translation_vector"])
                for i,n in enumerate(names):
                    scene=next((str(k) for k,g in enumerate(groups) if n in g),"outliers")
                    w.writerow([i,ds,scene,n,"1;0;0;0;1;0;0;0;1" if i<12 else ";".join(["nan"]*9),
                                "0;0;0" if i<12 else "nan;nan;nan"])
        script=Path(cmd[-2])
        tree=ast.fix_missing_locations(Platform().visit(ast.parse(script.read_text())))
        def child(*args,**kwargs):
            exec(compile(tree,str(script),"exec"),dict(__name__="__main__",__file__=str(script),fixture_probe=probe))
        previous=Path.cwd()
        oldpath=list(sys.path)
        try:
            with ExitStack() as stack:
                for ctx in (patch.dict(sys.modules,modules),patch.object(workflow,"export_observations",observations),
                            patch.object(workflow,"write_prediction",predictions),patch("runpy.run_path",child),
                            patch("subprocess.check_output",return_value=b"freeze"),patch.object(sys,"argv",cmd[3:]),
                            patch.dict(os.environ,PHASE4I_DEST=str(d),STAIRS_DEADLINE=str(time.monotonic()+20000))):
                    stack.enter_context(ctx)
                os.chdir(work)
                try:exec(compile(Path(cmd[3]).read_text(),cmd[3],"exec"),{"__name__":"__main__"})
                except SystemExit as e:
                    if e.code:raise
                except TimeoutError:
                    (d/"child_error.json").unlink(missing_ok=True)
                    raise
                except Exception as e:raise RuntimeError("child exited 1") from e
        finally:
            sys.path[:]=oldpath
            os.chdir(previous)
            (d/"heartbeat.jsonl").write_text(json.dumps(dict(stage=stage,totals=dict(rss=2,uss=1,pss=1))))

def separate_process_monitor(cpu,kit,cmd,work,d,end,stage):
    """Execute the generated entrypoint in a fresh interpreter for the notebook rehearsal."""
    import subprocess
    state=dict(base=str(cpu.base),p=cpu.p,names=cpu.fx.names,cmd=cmd,work=str(work),
               dest=str(d),end=end,stage=stage,kit=str(kit))
    spec=work/"cpu-platform.json"
    spec.write_text(json.dumps(state))
    subprocess.run([sys.executable,str(Path(__file__).resolve()),str(spec)],check=True)

if __name__=="__main__":
    spec=fixture_spec
    assert Path(workflow.__file__).is_relative_to(spec["kit"])
    fixture=CPU.__new__(CPU)
    fixture.base=Path(spec["base"])
    fixture.fx=NS(tmp=fixture.base,names=spec["names"])
    fixture.original=fixture.base/"dbfixture"
    fixture.mem=fixture.base/"meminfo"
    fixture.p=spec["p"]
    fixture.failure=""
    fixture.monitor(spec["cmd"],Path(spec["work"]),Path(spec["dest"]),spec["end"],spec["stage"])
