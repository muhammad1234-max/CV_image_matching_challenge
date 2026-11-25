"""Prepare Haiper inputs from authenticated returns, then replay in a clean directory."""
import ast,base64,csv,hashlib,io,json,re,sys,subprocess,tempfile,zipfile
from pathlib import Path
from src.novel_model.geometry_capture_v1.database import file_sha,read_only,pair_ids,logical
from contextlib import closing

MODEL="af0eaa71875e61f265284e45fb13b2770b1bd711e4ec7941873c53832be02c01"
def sha(b):return hashlib.sha256(b).hexdigest()
def encode(x):return json.dumps(x,sort_keys=True,indent=2,allow_nan=False).encode()
def verify(b,h):
    if sha(b)!=h:raise ValueError("digest mismatch")
def branch(z,name,plan,plan_hash):
    r=json.loads(z.read("result.json"))
    audit=json.loads(z.read("input.json"))
    assert r["branch"]==name and r["plan"]==plan_hash and r["options"]==plan["options"]
    assert audit["before"]["records"]==plan["records"]
    assert audit["forbidden"]==(plan["forbidden_ids"] if name=="active" else [])
    pred=z.read("prediction.csv")
    verify(pred,r["prediction"])
    rows=list(csv.DictReader(io.StringIO(pred.decode())))
    assert sorted(v["image"] for v in rows)==sorted(plan["image_hashes"])
    models=[]
    for m in r["models"]:
        names=sorted(v["image"] for v in rows if v["scene"]==f"cluster{m['model']}")
        assert len(names)==m["cameras"] and type(m["points"]) is int and m["points"]>=0
        models.append(dict(images=names,points=m["points"]))
    return dict(models=models),pred

def main():
    out=Path("reports/phase4h")
    pins=json.loads(Path("reports/phase4g/protocol_manifest.json").read_bytes())
    nb=Path("notebooks/step04g_stairs_colab.ipynb")
    verify(nb.read_bytes(),pins["datasets"]["stairs"]["notebook"])
    tree=ast.parse("".join(json.loads(nb.read_bytes())["cells"][-1]["source"]))
    kit=base64.b64decode(next(ast.literal_eval(s.value) for s in tree.body if isinstance(s,ast.Assign) and getattr(s.targets[0],"id","")=="KIT"))
    verify(kit,MODEL)
    with zipfile.ZipFile("phase4f_reconstruction_review.zip") as ref:
        pins4=json.loads(ref.read("SHA256.json"))
    pp=Path("reports/phase4f/plan.json")
    verify(pp.read_bytes(),pins4[pp.as_posix()])
    plan=json.loads(pp.read_bytes())
    db=Path(plan["parent"]+".db")
    assert file_sha(db)==plan["parent"]
    with closing(read_only(db)) as con:
        assert logical(con)==plan["records"]
        ids=dict(con.execute("select image_id,name from images"))
        assert all(re.fullmatch(r"image_\d{4}\.png",n) for n in ids.values())
        edges=[[*sorted(ids[i] for i in pair_ids(k)),n] for k,n in con.execute("select pair_id,rows from two_view_geometries where rows>0")]
    assert sorted([a,b] for a,b,_ in edges)==plan["pairs"]
    data=dict(nodes=sorted(ids.values()),edges=edges,sides=plan["sides"])
    preds={}
    for name in ("pass_through","active"):
        f=Path(f"phase4f_{name}_return.zip")
        verify(f.read_bytes(),pins4[f.name])
        with zipfile.ZipFile(f) as z:data[name],preds[name]=branch(z,name,plan,file_sha(pp))
    pkg=out/"inference_only.zip"
    with zipfile.ZipFile(pkg,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.zip",kit)
        z.writestr("inputs.json",encode(data))
    code="""import sys,json,site
sys.path.append(site.getusersitepackages())
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from src.novel_model.reconstruction_feedback_v2 import candidate,decide
from src.novel_model.geometry_graph_v1.schemas import Graph,Edge
d=json.loads(Path('inputs.json').read_bytes())
g=Graph(tuple(d['nodes']),tuple(Edge(a,b,n,None) for a,b,n in sorted(d['edges'])),{},'', 'verified_counts')
c=candidate(g)
assert list(map(list,c['sides']))==d['sides']
print(json.dumps(decide(c['sides'],d['pass_through'],d['active'])))
"""
    with tempfile.TemporaryDirectory(dir=out) as t:
        with zipfile.ZipFile(pkg) as z:z.extractall(t)
        with zipfile.ZipFile(Path(t)/"model.zip") as z:z.extractall(t)
        result=json.loads(subprocess.check_output([sys.executable,"-I","-c",code],cwd=t))
    assert result["selected"]=="pass_through" and result["reason"]=="affected point loss"
    (out/"selected.csv").write_bytes(preds["pass_through"])
    receipt=dict(decision=result,selected_sha256=sha(preds["pass_through"]),model=MODEL,parent=plan["parent"],package=file_sha(pkg),plan=file_sha(pp))
    (out/"decision.json").write_bytes(encode(receipt))
    (out/"decision.sha256").write_text(file_sha(out/"decision.json"))
if __name__=="__main__":main()

