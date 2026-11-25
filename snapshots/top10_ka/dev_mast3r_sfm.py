import sys

sys.path.append("/kaggle/input/smly-mast3r-pkgs/mast3r/dust3r")
sys.path.append("/kaggle/input/smly-mast3r-pkgs/mast3r")
import os
import tempfile

from dust3r.utils.image import load_images  # type: ignore
from mast3r.image_pairs import make_pairs  # type: ignore
from mast3r.retrieval.processor import Retriever  # type: ignore
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment  # type: ignore
from mast3r.model import AsymmetricMASt3R  # type: ignore

import torch


def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])


def main(filelist: list[str]) -> list[dict[str, str]]:
    optim_level = "refine+depth"
    retrieval_model = "/kaggle/input/imc-models/pytorch/bundle/1/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
    model_weight_path = "/kaggle/input/imc-models/pytorch/bundle/1/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    device = "cuda:0"
    image_size = 512
    lr1 = 0.07
    niter1 = 300
    lr2 = 0.01
    niter2 = 300
    shared_intrinsics = False
    matching_conf_thr = 0.0

    model = AsymmetricMASt3R.from_pretrained(model_weight_path).to(device)

    # Ignore FutureWarning
    imgs = load_images(filelist, size=image_size, verbose=False)
    assert len(imgs) > 1, "Not enough images loaded"
    n_images = len(imgs)

    # scenegraph_type = "complete"
    # scene_graph_params = [scenegraph_type]
    scenegraph_type = "retrieval"
    winsize = min(20, n_images)  # Na
    refid = min(10, n_images - 1)  # k
    scene_graph_params = [scenegraph_type, str(winsize), str(refid)]

    scene_graph = "-".join(scene_graph_params)

    sim_matrix = None
    if "retrieval" in scenegraph_type:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)

        # Cleanup
        del retriever
        torch.cuda.empty_cache()

    pairs = make_pairs(
        imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix
    )

    with tempfile.TemporaryDirectory() as outdir:
        # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
        cache_dir = tempfile.mkdtemp(suffix="_cache", dir=outdir)

        os.makedirs(cache_dir, exist_ok=True)
        scene = sparse_global_alignment(
            filelist,
            pairs,
            cache_dir,
            model,
            lr1=lr1,
            niter1=niter1,
            lr2=lr2,
            niter2=niter2,
            device=device,
            opt_depth="depth" in optim_level,
            shared_intrinsics=shared_intrinsics,
            matching_conf_thr=matching_conf_thr,
        )

    im_poses = scene.get_im_poses().cpu().numpy()
    pose_list = []
    for idx in range(3):
        rotation_matrix = im_poses[idx][:3, :3]
        translation_vector = im_poses[idx][:3, 3]
        pose_list.append(
            {
                "filepath": filelist[idx],
                "rotation_matrix": arr_to_str(rotation_matrix),
                "translation_vector": arr_to_str(translation_vector),
            }
        )

    return pose_list


if __name__ == "__main__":
    filelist = [
        "/kaggle/input/image-matching-challenge-2025/train/ETs/et_et000.png",
        "/kaggle/input/image-matching-challenge-2025/train/ETs/et_et001.png",
        "/kaggle/input/image-matching-challenge-2025/train/ETs/et_et002.png",
    ]
    print(main(filelist))
