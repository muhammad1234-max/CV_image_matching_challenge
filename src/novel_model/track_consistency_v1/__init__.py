"""Sparse exact-observation diagnostics. No pose, label or camera calibration inputs."""
from time import perf_counter
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def analyze(keys, sizes, pairs, grid=4, chunk=2048):
    start = perf_counter()
    names = sorted(keys)
    offsets = {}
    owners = []
    total = 0
    for k, name in enumerate(names):
        xy = keys[name]
        if xy.ndim != 2 or xy.shape[1] < 2 or not np.isfinite(xy[:, :2]).all():
            raise ValueError("invalid coordinates")
        offsets[name] = total
        total += len(xy)
        owners.append(np.full(len(xy), k, dtype=np.int32))
    owner = np.concatenate(owners)
    links = {}
    neighbors = {n:set() for n in names}
    for (a,b), ij in sorted(pairs.items()):
        if a == b or a not in keys or b not in keys or ij.ndim != 2 or ij.shape[1] != 2:
            raise ValueError("invalid pair")
        if a > b:
            a,b = b,a
            ij = ij[:, ::-1]
        if (a,b) in links or ij.dtype.kind not in "iu":
            raise ValueError("duplicate pair or invalid dtype")
        if not len(ij):
            continue
        if ij.min()<0 or ij[:,0].max()>=len(keys[a]) or ij[:,1].max()>=len(keys[b]):
            raise ValueError("index bounds")
        uv = ij.astype(np.int64) + [offsets[a],offsets[b]]
        if len(np.unique(uv,axis=0)) != len(uv):
            raise ValueError("duplicate correspondence")
        links[a,b] = uv
        neighbors[a].add(b)
        neighbors[b].add(a)
    all_links = np.concatenate(list(links.values())) if links else np.empty((0,2),int)
    u,v = all_links.T
    adjacency = csr_matrix((np.ones(2*len(u),bool),(np.r_[u,v],np.r_[v,u])),shape=(total,total))
    built = perf_counter()
    _, labels = connected_components(adjacency,directed=False)
    used = np.unique(all_links)
    components, distinct_images_count = np.unique(labels[used]*np.int64(len(names))+owner[used],return_counts=True)
    bad = np.unique(components[distinct_images_count>1]//len(names))
    conflicting = np.isin(labels,bad)
    multiview = (np.bincount(labels)[labels]>=3) & ~conflicting
    tracked = perf_counter()
    supported = np.zeros(total,bool)
    rows = []
    for (a,b), uv in links.items():
        exact = np.zeros(len(uv),int)
        paths = np.zeros_like(exact)
        for pos in range(0,len(uv),chunk):
            x,y = uv[pos:pos+chunk].T
            hits = adjacency[x].multiply(adjacency[y]).tocoo()
            exact[pos:pos+len(x)] = np.bincount(hits.row,minlength=len(x))
            witness = np.unique(hits.row.astype(np.int64)*len(names)+owner[hits.col])
            paths[pos:pos+len(x)] = np.bincount(witness//len(names),minlength=len(x))
        ok = exact>0
        supported[uv[ok].ravel()] = True
        cover = {}
        for side,name in enumerate((a,b)):
            xy = keys[name][uv[:,side]-offsets[name],:2]
            cells = np.clip(np.floor(xy*grid/np.asarray(sizes[name])).astype(int),0,grid-1)
            cover[name] = [len(np.unique(cells[mask],axis=0)) for mask in (ok,~ok)]
        rows.append(dict(pair=[a,b],n=len(uv),cycle=int(ok.sum()),witness_images=int(paths.sum()),
                         multiple=int((paths>=2).sum()),cycle_incidences=int(exact.sum()),
                         clean_multiview=int(multiview[uv[:,0]].sum()), conflict=int(conflicting[uv[:,0]].sum()),
                         duplicate_endpoint_uses=[len(uv)-len(np.unique(col)) for col in uv.T],
                         common_images=len(neighbors[a]&neighbors[b]), overlap=len(neighbors[a]&neighbors[b])/len(neighbors[a]|neighbors[b]),
                         grid_cells=cover))
    return dict(edges=rows, linked_observations=len(used),keypoints=total,links=len(u),
                track_hypotheses=len(np.unique(labels[used])), conflicting_tracks=len(bad),
                clean_multiview_tracks=len(np.unique(labels[multiview])),supported_observations=int(supported.sum()),
                exact_cycles=sum(r["cycle_incidences"] for r in rows)//3,
                image_triangles=sum(r["common_images"] for r in rows)//3,
                seconds=dict(graph=built-start,tracks=tracked-built,diagnostics=perf_counter()-tracked))
