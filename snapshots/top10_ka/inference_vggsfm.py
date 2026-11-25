import multiprocessing
import argparse
import subprocess
import os
import pandas as pd
import shutil
import json
import gc
import torch
from pathlib import Path
from config import Config
from pipeline import Pipeline
from reconstruction import reconstruction

from loguuuuuuu import logger


def parse_sample_submission(
    base_path: str,
    input_csv: str,
    target_datasets: list,
    prefix: str,
    verbose: bool = False,
) -> dict[str, dict[str, list[Path]]]:
    """Construct a dict describing the test data as

    {"dataset": {"scene": [<image paths>]}}
    """
    data_dict = {}
    with open(input_csv, "r") as f:
        for i, l in enumerate(f):
            # Skip header
            # if i == 0:
            #     print("header:", l)

            if l and i > 0:
                image_id, dataset, scene, image_path, _, _ = l.strip().split(",")
                image_path = f"{prefix}/{dataset}/{image_path}"
                if target_datasets is not None and dataset not in target_datasets:
                    continue

                ### Temp for bug?
                if not os.path.isfile(os.path.join(base_path, image_path)):
                    continue

                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                data_dict[dataset][scene].append(Path(os.path.join(base_path, image_path)))

    if verbose:
        for dataset in data_dict:
            for scene in data_dict[dataset]:
                logger.info(
                    f"dataset: {dataset}, scene: {scene}, num_images: {len(data_dict[dataset][scene])}"
                )

    return data_dict


def worker_reconstruction(input_queue, submission_path_list, config_name, keep_flag):
    while True:
        reconstruction_inputs = input_queue.get()
        if reconstruction_inputs is None:
            break
        data_dict, dataset, scene, base_path, work_dir, cluster_work_dir, colmap_mapper_options = (
            reconstruction_inputs
        )
        submission_path = reconstruction(
            data_dict, dataset, scene, base_path, work_dir, cluster_work_dir, colmap_mapper_options
        )
        submission_path_list.append(submission_path)

        # Keep intermediate files
        if keep_flag:
            keep_dir = "/kaggle/working/output/{config_name:s}/{dataset:s}".format(
                config_name=config_name,
                dataset=str(dataset),
            )
            # Remove if exists
            if not Path(keep_dir).exists():
                shutil.rmtree(keep_dir, ignore_errors=True)

            Path(keep_dir).mkdir(parents=True, exist_ok=True)
            shutil.copytree(work_dir, keep_dir, dirs_exist_ok=True)


def run(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--target_datasets", nargs="*", required=False)
    parser.add_argument("--output_dir", required=False)
    parser.add_argument(
        "--prefix", type=str, default="/kaggle/input/image-matching-challenge-2025/test"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/kaggle/input/imc25-codes/configs/rotate_640_840_1024_1280_crop.json",
        required=True,
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        default=False,
        help="Keep the intermediate files on the output directory",
    )
    args = parser.parse_args()

    if args.output_dir:
        config.output_dir = args.output_dir

    if args.target_datasets:
        config.target_datasets = args.target_datasets

    # Check output_dir
    if config.check_exist_dir:
        if os.path.isdir(config.output_dir):
            raise Exception(f"{config.output_dir} is already exists.")

    logger.info(f"Config: {args.config}")

    os.makedirs(config.output_dir, exist_ok=True)
    base_path = os.path.join(config.input_dir_root, "image-matching-challenge-2025")
    feature_dir = os.path.join(config.output_dir, "feature_outputs")
    shutil.copy(args.config, config.output_dir)

    data_dict = parse_sample_submission(
        base_path, config.input_csv, config.target_datasets, args.prefix
    )
    datasets = list(data_dict.keys())

    manager = multiprocessing.Manager()
    submission_path_list = manager.list()

    input_queue = multiprocessing.Queue()
    worker_reconstruction_process = multiprocessing.Process(
        target=worker_reconstruction,
        args=(input_queue, submission_path_list, Path(args.config).stem, args.keep),
    )
    worker_reconstruction_process.start()
    for dataset in datasets:
        for scene in data_dict[dataset]:
            logger.info(f'Processing pipeline for dataset "{dataset}"')
            work_dir = Path(os.path.join(feature_dir, f"{dataset}_{scene}"))
            work_dir.mkdir(parents=True, exist_ok=True)

            # Exec Pipeline
            with open(args.config, "r") as f:
                pipeline_config = json.load(f)

            pipeline = Pipeline(
                data_dict[dataset][scene],
                work_dir,
                config.input_dir_root,
                pipeline_config,
                args.device_id,
            )
            pipeline.exec()
            logger.info(f'Processing pipeline for dataset "{dataset}" ... DONE')

            # Reconstruction & Save CSV
            print("Start Clustering & Reconstruction")
            gc.collect()
            torch.cuda.empty_cache()

            data_dict_clusters = pipeline.clustering(dataset, scene, data_dict, work_dir)
            for scene_name in data_dict_clusters[dataset].keys():
                print(
                    "Cluster {}={}".format(
                        scene_name, len(data_dict_clusters[dataset][scene_name])
                    )
                )
                # TODO: DUST3R など対策を考える
                if len(data_dict_clusters[dataset][scene_name]) < 3:
                    continue

                cluster_work_dir = work_dir / scene_name
                cluster_work_dir.mkdir(parents=True, exist_ok=True)
                input_queue.put(
                    (
                        data_dict_clusters,
                        dataset,
                        scene_name,
                        base_path,
                        work_dir,
                        cluster_work_dir,
                        config.colmap_mapper_options,
                    )
                )

            # input_queue.put((data_dict, dataset, scene, base_path, work_dir, config.colmap_mapper_options))

    input_queue.put(None)
    worker_reconstruction_process.join()

    # TODO: Refine cluster SfM results...
    # * recon_refine_vggsfm.py を dataset, scene, model_name, base_path 指定で実行可能にする
    # * vggsfm_v2_0_0.bin と vggsfm_conf.yml を指定して実行可能にする
    # * cluster_data_dict.pkl が存在しない場合は実行しない（起こり得るっぽい）
    logger.info("Start Refining SfM results with VGG-SfM")
    print(dataset, args.device_id, Path(args.config).stem, args.prefix, base_path)
    cmd = [
        "python",
        "/kaggle/input/imc25-codes/tasks/debug/recon_refine_vggsfm.py",
        "--dataset",
        dataset,
        "--device",
        f"cuda:{args.device_id}",
        "--model_name",
        Path(args.config).stem,
        "--images_dir",
        f"{args.prefix}/{dataset}",
        "--base_path",
        base_path,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc.wait()
    logger.info("Refining SfM results with VGG-SfM ... DONE")

    # Concat Submission
    if len(submission_path_list) > 0:
        submission_df_list = [pd.read_csv(p) for p in submission_path_list]
        submission_df = pd.concat(submission_df_list).reset_index(drop=True)
    else:
        submission_df = pd.DataFrame.from_dict(
            {
                "image_path": [],
                "dataset": [],
                "scene": [],
                "rotation_matrix": [],
                "translation_vector": [],
            }
        )

    out_path = os.path.join(config.output_dir, "submission.csv")
    print(f"Save summary to {out_path}")
    submission_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    cfg = Config
    run(cfg)
