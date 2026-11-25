"""CPU-only sparse-file roundtrip on PyCOLMAP 3.11.1. No matching, GPU or research labels."""
import json
from pathlib import Path
import sqlite3
from contextlib import closing
import tempfile
import numpy as np
import pycolmap

# Load the production exporter directly so the diagnostic needs no spectral or training stack.
import runpy
export_observations=runpy.run_path("src/novel_model/final_candidate_v1/observations.py")["export_observations"]
from src.novel_model.geometry_capture_v1.database import save

def run():
    assert pycolmap.__version__=="3.11.1"
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        db=root/"parent.db"
        with closing(sqlite3.connect(db)) as con, con:
            con.execute("create table images(image_id integer,camera_id integer,name text)")
            con.execute("create table keypoints(image_id integer,rows integer,cols integer,data blob)")
            for i,x in ((1,50),(2,40)):
                con.execute("insert into images values(?,1,?)",(i,f"image_{i-1:04d}.png"))
                con.execute("insert into keypoints values(?,2,2,?)",(i,np.array([[x,50],[0,0]],dtype="<f4").tobytes()))
        models={}
        # An equivalent gauge with double scale and a different landmark ID.
        for label,scale,pid in (("original",1,7),("gauge",2,900)):
            folder=root/label
            folder.mkdir()
            (folder/"cameras.txt").write_text("1 PINHOLE 100 100 100 100 50 50\n")
            (folder/"images.txt").write_text(f"1 1 0 0 0 0 0 0 1 image_0000.png\n50 50 {pid} 0 0 -1\n"
                f"2 1 0 0 0 {-scale} 0 0 1 image_0001.png\n40 50 {pid} 0 0 -1\n")
            (folder/"points3D.txt").write_text(f"{pid} 0 0 {10*scale} 1 2 3 0 1 0 2 0\n")
            model=pycolmap.Reconstruction(str(folder))
            groups,errors=export_observations({0:model},db)
            assert groups==[["image_0000.png","image_0001.png"]]
            assert all(v.shape==(2,) and abs(v[0])<1e-8 and np.isnan(v[1]) for v in errors.values())
            models[label]=model
        with closing(sqlite3.connect(db)) as con, con:
            con.execute("update keypoints set data=? where image_id=1",
                        (np.array([[0,0],[50,50]],dtype="<f4").tobytes(),))
        try: export_observations({0:models["gauge"]},db)
        except ValueError as error: assert "row or pixel" in str(error)
        else: raise AssertionError("feature permutation went undetected")
    return dict(status="PASS",pycolmap=pycolmap.__version__,fixture="two native sparse text models",
                different_point_ids=[7,900],scale_ratio=2,common=2,untriangulated=2,
                residual_px=0,permuted_database_rows="rejected",device="Windows CPU")

if __name__=="__main__":
    save(Path("reports/phase4i/native_export.json"),run())
