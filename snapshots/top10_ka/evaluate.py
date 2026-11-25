import argparse
import tempfile
from pathlib import Path
import warnings
import json

import polars as pl

import imc25metric
from loguuuuuuu import logger


SAMPLE_SUBMISSION = "/kaggle/input/image-matching-challenge-2025/sample_submission.csv"
GT_CSV = "/kaggle/input/image-matching-challenge-2025/train_labels.csv"
THRESH_CSV = "/kaggle/input/image-matching-challenge-2025/train_thresholds.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working",
        type=str,
        default="/kaggle/working/output/v16/fbk_vineyard/",
        help="Path to the working directory",
    )
    return parser.parse_args()


def _load_partial_results(path: Path) -> pl.DataFrame:
    return pl.read_csv(path / "submission.csv")


def main(args: argparse.Namespace) -> None:
    feature_out = Path(args.working)
    eval_json = feature_out / "eval.json"
    if eval_json.exists():
        logger.warning("Already evaluated, skipping...")
        return

    df = pl.concat([_load_partial_results(path) for path in feature_out.glob("cluster*")])
    df = df.with_columns(pl.col("image_path").str.split("/").list.last().alias("image")).drop(
        "image_path"
    )

    assert len(df["dataset"].unique()) == 1
    dataset_name = df["dataset"].unique()[0]
    df_sub = pl.read_csv(SAMPLE_SUBMISSION).filter(pl.col("dataset") == dataset_name)
    dataset_image_names = set(df_sub["image"].to_list())
    outlier_file_names = list(dataset_image_names - set(df["image"].to_list()))
    logger.info(
        "Cluster size: \033[92m{}\033[0m | Cluster images: \033[93m{}\033[0m, Dataset images: \033[93m{}\033[0m, Missing Images: \033[93m{}\033[0m".format(
            len(df["scene"].unique()), len(df), len(dataset_image_names), len(outlier_file_names)
        )
    )
    df = pl.concat(
        [
            df,
            pl.DataFrame(
                {
                    "dataset": [dataset_name] * len(outlier_file_names),
                    "scene": ["outliers"] * len(outlier_file_names),
                    "rotation_matrix": ["nan;nan;nan;nan;nan;nan;nan;nan;nan"]
                    * len(outlier_file_names),
                    "translation_vector": ["nan;nan;nan"] * len(outlier_file_names),
                    "image": outlier_file_names,
                }
            ),
        ]
    )
    df = df_sub.select(["image_id", "dataset", "image"]).join(
        df, how="left", on=["dataset", "image"]
    )
    assert len(df) == len(df_sub)

    with (
        tempfile.NamedTemporaryFile(delete=True, suffix=".csv") as f,
        tempfile.NamedTemporaryFile(delete=True, suffix=".csv") as f_gt,
        tempfile.NamedTemporaryFile(delete=True, suffix=".csv") as f_thresh,
    ):
        df.write_csv(f.name)
        f.flush()

        pl.read_csv(GT_CSV).filter(pl.col("dataset") == dataset_name).write_csv(f_gt.name)
        f_gt.flush()

        pl.read_csv(THRESH_CSV).filter(pl.col("dataset") == dataset_name).write_csv(f_thresh.name)
        f_thresh.flush()

        # Ignore RuntimeWarning warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Calculate the score
            logger.info("Calculating score...")
            dataset_scores = imc25metric.score(
                gt_csv=f_gt.name,
                user_csv=f.name,
                thresholds_csv=f_thresh.name,
                mask_csv=None,
                inl_cf=0,
                strict_cf=-1,
                verbose=True,
            )
            print(">>", dataset_scores, type(dataset_scores))
            json.dump(
                dataset_scores,
                open(eval_json, "w"),
                indent=4,
            )


if __name__ == "__main__":
    main(parse_args())
