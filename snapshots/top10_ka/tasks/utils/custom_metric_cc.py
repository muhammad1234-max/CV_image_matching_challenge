import re
import sys

sys.path.append("/kaggle/input/imc25-codes")
from pathlib import Path

from clustering import clustering
import numpy as np
import polars as pl

SAMPLE_SUBMISSION = "/kaggle/input/image-matching-challenge-2025/sample_submission.csv"
GT_CSV = "/kaggle/input/image-matching-challenge-2025/train_labels.csv"
THRESH_CSV = "/kaggle/input/image-matching-challenge-2025/train_thresholds.csv"


def set_outliers(df: pl.DataFrame, df_sub: pl.DataFrame) -> pl.DataFrame:
    """
    Set outliers in the DataFrame.
    """
    df_out = df_sub.filter(~pl.col("image").is_in(df["image"]))
    df_out = df_out.with_columns(
        pl.lit("outliers").alias("scene"),
    )
    df = pl.concat([df, df_out])
    return df


def custom_score_clustering(
    data_dict_clusters: dict[str, dict[str, list[Path]]],
) -> None:
    for dataset in data_dict_clusters.keys():
        rows = []
        for scene_name in data_dict_clusters[dataset].keys():
            for filepath in data_dict_clusters[dataset][scene_name]:
                rows.append(
                    {
                        "dataset": dataset,
                        "scene": scene_name,
                        "image": Path(filepath).name,
                        "rotation_matrix": "",
                        "translation_vector": "",
                    }
                )
        df = pl.DataFrame(rows)
        df_gt = pl.read_csv(GT_CSV).filter(pl.col("dataset") == dataset)
        df = set_outliers(df, df_gt)

        scene_list = (
            df_gt.filter(pl.col("scene") != "outliers")
            .group_by("scene")
            .agg(pl.len())
            .sort("len", descending=True)["scene"]
            .to_list()
        )
        overlap_mapping = {}
        selected_user_clusters = []
        for scene_name in scene_list:
            user_clusters = df.filter(pl.col("scene") != "outliers")["scene"].unique().to_list()
            best_size = 0
            best_user_cluster = None
            for user_cluster in user_clusters:
                if user_cluster in selected_user_clusters:
                    continue
                df_scene = df.filter(pl.col("scene") == user_cluster)
                df_scene_gt = df_gt.filter(pl.col("scene") == scene_name)
                sz = len(set(df_scene["image"].to_list()) & set(df_scene_gt["image"].to_list()))

                if best_size < sz:
                    best_size = sz
                    best_user_cluster = user_cluster
            if best_user_cluster is not None:
                selected_user_clusters.append(best_user_cluster)
                overlap_mapping[scene_name] = best_user_cluster
            else:
                overlap_mapping[scene_name] = None

        scores = []
        for scene_name in overlap_mapping.keys():
            user_cluster_name = overlap_mapping[scene_name]
            df_scene_gt = df_gt.filter(pl.col("scene") == scene_name)
            if user_cluster_name is None:
                score = 0.0
                # print(scene_name, None, len(df_scene_gt), 0, 0, score)
                scores.append(score)
                continue

            df_scene = df.filter(pl.col("scene") == user_cluster_name)
            sz = len(set(df_scene["image"].to_list()) & set(df_scene_gt["image"].to_list()))
            sz2 = len(set(df_scene["image"].to_list()) | set(df_scene_gt["image"].to_list()))
            score = sz / sz2 if sz > 0 else 0.0
            # print(scene_name, user_cluster_name, len(df_scene_gt), len(df_scene), sz, score)
            scores.append(score)

        print(dataset, np.mean(scores) * 100)


def clusteringV2(
    dataset: str,
    image_paths: list[Path],
    work_dir: str | Path,
) -> dict[str, dict[str, list[str | Path]]]:
    """
    {prefix}{\d+}.png の prefix ごとに group を作成する
    """
    pat_count = {}
    valid_count = 0
    pat = re.compile(r"^(.*?)(\d+).png$")
    for image_path in image_paths:
        m = pat.match(Path(image_path).name)
        if m:
            prefix = m.groups()[0]
            if prefix not in pat_count:
                pat_count[prefix] = 0
            pat_count[prefix] += 1
            valid_count += 1
    num_patterns = len(pat_count)

    if num_patterns >= 10:
        pat = re.compile(r"^(.*?)([\d_\-]+).png$")
        pat_count = {}
        valid_count = 0
        for image_path in image_paths:
            m = pat.match(Path(image_path).name)
            if m:
                prefix = m.groups()[0]
                if prefix not in pat_count:
                    pat_count[prefix] = 0
                pat_count[prefix] += 1
                valid_count += 1
        num_patterns = len(pat_count)

    if valid_count > len(image_paths) * 0.5 and num_patterns < 10:
        data_cluster_dict = {dataset: {}}
        for cluster_idx, prefix in enumerate(pat_count.keys(), start=1):
            cluster_name = f"cluster{cluster_idx:d}"
            if cluster_name not in data_cluster_dict[dataset]:
                data_cluster_dict[dataset][cluster_name] = []
            for image_path in image_paths:
                m = pat.match(Path(image_path).name)
                if m and m.groups()[0] == prefix:
                    data_cluster_dict[dataset][cluster_name].append(image_path)

        return data_cluster_dict

    return clustering(
        dataset,
        image_paths,
        work_dir,
        quantile=0.1,
        eps=None,
    )


def main():
    feature_out = "/kaggle/working/output/v7/"
    # dataset = "ETs"
    # dataset = "stairs"
    # dataset = "amy_gardens"
    # dataset = "fbk_vineyard"
    # dataset = "pt_brandenburg_british_buckingham"
    # dataset = "pt_piazzasanmarco_grandplace"
    # dataset = "pt_sacrecoeur_trevi_tajmahal"
    # dataset = "pt_stpeters_stpauls"
    dataset = "imc2024_lizard_pond"

    image_paths = list(
        Path(f"/kaggle/input/image-matching-challenge-2025/train/{dataset:s}/").glob("*.png")
    )
    work_dir = f"/kaggle/working/output/v7/{dataset:s}"
    data_dict_clusters = clustering(
        dataset,
        image_paths,
        work_dir,
    )
    custom_score_clustering(data_dict_clusters)  # type: ignore


if __name__ == "__main__":
    main()
