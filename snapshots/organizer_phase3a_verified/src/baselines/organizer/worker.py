"""One killable inference stage; only opaque filenames enter this process."""
import argparse,json,sys,time,traceback,shutil,sqlite3
from pathlib import Path
from contracts import write
def execute(stage,state,out):
    import torch,numpy as np
    # Checkpoints must be cached by preflight; never let a missing cache trigger a download.
    def offline_download(*args,**kwargs):raise RuntimeError('Runtime checkpoint download prohibited; verify mounted assets and cache names')
    torch.hub.download_url_to_file=offline_download
    root=Path(state['source_root']);work=Path(state['work']);cfg=state['config'];images=[str(work/'images'/n) for n in state['images']]
    sys.path.insert(0,str(root/'docs/evidence/phase3a_sources'))
    from reference import load
    import pycolmap,h5py
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(min(1,cfg['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory))
    ref=load(root,state['dino_dir']);device=torch.device('cuda:0')
    if stage=='B':
        emb=ref['get_global_desc'](images,device=device);torch.save(emb,out/'embeddings.pt')
        return {'images':len(images),'embedding_shape':list(emb.shape)}
    if stage=='C':
        emb=torch.load(work/'B/embeddings.pt',map_location='cpu');ref['get_global_desc']=lambda *a,**k:emb
        pairs=ref['get_image_pairs_shortlist'](images,sim_th=cfg['sim_th'],min_pairs=cfg['min_pairs'],exhaustive_if_less=cfg['exhaustive_if_less'],device=device)
        write(out/'pairs.json',pairs);return {'possible_pairs':len(images)*(len(images)-1)//2,'selected_pairs':len(pairs)}
    if stage=='D':
        ref['detect_aliked'](images,str(out),cfg['max_keypoints'],cfg['resize'],device)
        with h5py.File(out/'keypoints.h5') as f:counts={k:len(f[k]) for k in f}
        return {'feature_counts':counts,'failed_images':len(images)-len(counts)}
    if stage=='E':
        for n in ['keypoints.h5','descriptors.h5']:shutil.copyfile(work/'D'/n,out/n)
        pairs=json.loads((work/'C/pairs.json').read_text());ref['match_with_lightglue'](images,pairs,str(out),device,min_matches=cfg['min_matches'],verbose=True)
        with h5py.File(out/'matches.h5') as f:counts={a+'/'+b:len(f[a][b]) for a in f for b in f[a]}
        return {'retained_pairs':len(counts),'retained_matches':counts,'raw_counts_location':'E.log original per-pair counts; omitted zero-match pairs have zero'}
    if stage=='F':
        for n in ['keypoints.h5','descriptors.h5','matches.h5']:shutil.copyfile(work/'E'/n,out/n)
        ref['import_into_colmap'](str(work/'images'),str(out),str(out/'colmap.db'))
        pycolmap.match_exhaustive(str(out/'colmap.db'))
        with sqlite3.connect(out/'colmap.db') as db:rows=db.execute('SELECT pair_id,rows FROM two_view_geometries').fetchall()
        return {'verified_pairs':sum(n>0 for _,n in rows),'inliers':dict(rows),'method':'original pycolmap.match_exhaustive defaults'}
    if stage=='G':
        opts=pycolmap.IncrementalPipelineOptions();opts.min_model_size=cfg['min_model_size'];opts.max_num_models=cfg['max_num_models'];(out/'models').mkdir(exist_ok=True)
        maps=pycolmap.incremental_mapping(database_path=str(work/'F/colmap.db'),image_path=str(work/'images'),output_path=str(out/'models'),options=opts)
        poses={};counts={}
        for index,model in maps.items():
            counts[str(index)]=len(model.images)
            for image in model.images.values():poses[image.name]={'cluster':index,'rotation':image.cam_from_world.rotation.matrix().reshape(-1).tolist(),'translation':image.cam_from_world.translation.tolist()}
        write(out/'poses.json',poses);return {'models':len(maps),'model_sizes':counts,'registered_unique':len(poses),'total':len(images)}
    if stage=='H':
        poses=json.loads((work/'G/poses.json').read_text());write(out/'assignments.json',{n:poses.get(n) for n in state['images']})
        return {'outliers':sum(n not in poses for n in state['images']),'cluster_assignment':'model index; unregistered -> outliers'}
    raise ValueError('Unknown inference stage')
def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('state');p.add_argument('out');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);t=time.monotonic();record={'stage':a.stage,'status':'FAILED'}
    try:record.update(metrics=execute(a.stage,json.loads(Path(a.state).read_text()),out),status='PASS')
    except Exception as e:record.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
    finally:
        record['runtime_seconds']=time.monotonic()-t
        try:
            import torch,resource
            record['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated();record['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved();record['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        except Exception:pass
        write(out/'result.json',record)
if __name__=='__main__':main()
