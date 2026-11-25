import numpy as np
import pytest
from src.novel_model.track_consistency_v1 import analyze

def run(pairs, n=3):
    names=sorted({x for p in pairs for x in p})
    keys={x:np.zeros((n,2)) for x in names}
    data={p:np.asarray(v,int).reshape(-1,2) for p,v in pairs.items()}
    return analyze(keys,{x:(20,20) for x in keys},data)

triangle={("a","b"):[[0,1]],("b","c"):[[1,2]],("c","a"):[[2,0]]}

def test_exact_and_degenerate_coordinates():
    r=run(triangle)
    assert (r["exact_cycles"],r["supported_observations"],r["clean_multiview_tracks"])==(1,3,1)
    assert all(list(e["grid_cells"].values())==[[1,0],[1,0]] for e in r["edges"])

def test_coherent_false():
    # Identical associations can refer to repeated but distinct physical surfaces.
    assert run(triangle)["exact_cycles"]==1

def test_broken_conflict():
    r=run({**triangle,("c","a"):[[2,1]]})
    assert r["image_triangles"]==1 and r["exact_cycles"]==0 and r["conflicting_tracks"]==1

def test_disjoint_triangle():
    r=run({p:[[0,1]] for p in triangle})
    assert r["image_triangles"]==1 and r["clean_multiview_tracks"]==0

def test_partial_sparse():
    r=run({("a","b"):[[0,0],[1,1]],("a","c"):[[0,0]],("b","c"):[]})
    assert r["links"]==3 and r["exact_cycles"]==0 and r["clean_multiview_tracks"]==1

@pytest.mark.parametrize("bad",[[[3,0]],[[-1,0]],[[1,0],[1,0]]])
def test_invalid(bad):
    with pytest.raises(ValueError): run({("a","b"):bad})

def test_reverse_duplicate():
    with pytest.raises(ValueError): run({("a","b"):[[0,0]],("b","a"):[[0,0]]})

def test_reindex():
    r=run({(b.upper(),a.upper()):[[2-j,2-i] for i,j in v] for (a,b),v in triangle.items()})
    assert r["exact_cycles"]==1 and r["conflicting_tracks"]==0

def test_hub():
    r=run({("hub",str(i)):[[0,0]] for i in range(128)})
    assert r["track_hypotheses"]==1 and r["exact_cycles"]==0
