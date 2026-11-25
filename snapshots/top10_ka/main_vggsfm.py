import os
import argparse
import subprocess
import shutil
from pathlib import Path

import pandas as pd
from glob import glob

from loguuuuuuu import logger


INPUT_ROOT = "/kaggle/input/image-matching-challenge-2025"
OUTPUT_ROOT = "/kaggle/working"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMC25 Inference")
    parser.add_argument(
        "--skip",
        action="store_true",
        default=False,
        help="Skip the main process and only copy the submission file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/kaggle/input/imc25-codes/configs/v1.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Default: all (or dataset name like 'ETs')",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="/kaggle/input/image-matching-challenge-2025/test",
        help="Prefix for the dataset directory",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        default=False,
        help="Keep the intermediate files on the output directory",
    )
    parser.add_argument("--output_dir", type=str, default="/tmp", help="Output directory")
    return parser.parse_args()


def partition_with_index(lst, lst2):
    indexed_lst = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    groupA = []
    groupB = []
    groupA2 = []
    groupB2 = []

    for index, item in indexed_lst:
        if sum([lst[i] for i in groupA]) <= sum([lst[i] for i in groupB]):
            groupA.append(index)
            groupA2.append(lst2[index])
        else:
            groupB.append(index)
            groupB2.append(lst2[index])

    return groupA2, groupB2


def make_command(
    device_id: int, target_datasets: list[str], args: argparse.Namespace
) -> list[str]:
    prefix = args.prefix
    config_path = args.config
    keep_flag = args.keep
    shutil.rmtree(f"/tmp/tmp{device_id}", ignore_errors=True)

    cmd = [
        "python",
        "/kaggle/input/imc25-codes/inference_vggsfm.py",
        "--device_id",
        f"{device_id}",
        "--output_dir",
        f"/tmp/tmp{device_id}",
        "--prefix",
        prefix,
        "--config",
        config_path,
    ]
    if keep_flag:
        cmd.append("--keep")

    if len(target_datasets) > 0:
        cmd.append("--target_datasets")
        for scene in target_datasets:
            cmd.append(scene)
    return cmd


def main(args: argparse.Namespace) -> None:
    config_name = Path(args.config).stem
    if (
        Path("/kaggle/working/output") / config_name / args.dataset / "processing_time.csv"
    ).exists():
        logger.warning("Already evaluated, skipping...")
        return

    datasets = [x for x in os.listdir(f"{args.prefix}/") if os.path.isdir(f"{args.prefix}/{x}")]
    if args.dataset != "all":
        datasets = [x for x in datasets if x == args.dataset]

    data_num_list = [len(list(glob(f"{args.prefix}/{dataset}/*.png"))) for dataset in datasets]
    datasets0, datasets1 = partition_with_index(data_num_list, datasets)

    df_sub = pd.read_csv("/kaggle/input/image-matching-challenge-2025/sample_submission.csv")
    df_sub = df_sub[["image_id", "dataset", "image"]]

    if args.skip and len(df_sub["image_id"].unique()) == 1945:
        # `--skip` かつ interactive session の場合はスキップ
        # * submission のときは `--skip` をつければ OK
        # * local は `--skip` をつけなければ validation できる

        # quick sub
        shutil.copyfile(
            "/kaggle/input/image-matching-challenge-2025/sample_submission.csv",
            "/kaggle/working/submission.csv",
        )
        return

    if len(datasets0) == 1 and len(datasets1) == 0:
        # DEBUG mode
        proc0 = subprocess.Popen(
            make_command(0, datasets0, args),
        )
        proc0.wait()
        return

    proc0 = subprocess.Popen(
        make_command(0, datasets0, args),
    )
    proc1 = subprocess.Popen(
        make_command(1, datasets1, args),
    )
    proc0.wait()
    proc1.wait()

    df1 = pd.read_csv("/tmp/tmp0/submission.csv")
    df2 = pd.read_csv("/tmp/tmp1/submission.csv")
    df = pd.concat([df1, df2]).reset_index(drop=True)
    df.to_csv("/kaggle/working/solved_result.csv", index=False)

    # filter outlier cluster
    df = df[df["translation_vector"] != "0.0;0.0;0.0"]

    df.loc[:, "image"] = df["image_path"].map(lambda x: Path(x).name)
    df = df.drop_duplicates(subset=["dataset", "image"])[
        ["dataset", "image", "scene", "rotation_matrix", "translation_vector"]
    ]
    df_sub = df_sub.merge(df, how="left", on=["dataset", "image"])

    df_sub.loc[:, "scene"] = df_sub["scene"].fillna("outliers")

    df_sub.loc[:, "rotation_matrix"] = df_sub["rotation_matrix"].fillna(
        "nan;nan;nan;nan;nan;nan;nan;nan;nan"
    )
    df_sub.loc[:, "translation_vector"] = df_sub["translation_vector"].fillna("nan;nan;nan")
    df_sub[
        ["image_id", "dataset", "scene", "image", "rotation_matrix", "translation_vector"]
    ].to_csv("/kaggle/working/submission.csv", index=False)


if __name__ == "__main__":
    main(parse_args())
