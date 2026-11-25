import sys

if "/kaggle/input/imc25-codes" not in sys.path:
    sys.path.append("/kaggle/input/imc25-codes")

import subprocess
import os
import time
import pickle

from loguuuuuuu import logger


class Mast3rSFM:
    def __init__(
        self,
        data_dict_clusters,
        dataset,
        scene_name,
        base_path,
        work_dir,
        cluster_work_dir,
        device_id,
    ) -> None:
        self.data_dict_clusters = data_dict_clusters
        self.dataset = dataset
        self.scene_name = scene_name
        self.base_path = base_path
        self.work_dir = work_dir
        self.cluster_work_dir = cluster_work_dir
        self.device_id = device_id

    def exec(self) -> str:
        cluster_data_dict_path = os.path.join(self.cluster_work_dir, "cluster_data_dict.pkl")
        with open(cluster_data_dict_path, "wb") as f:
            pickle.dump(self.data_dict_clusters, f)

        submission_path = os.path.join(self.cluster_work_dir, "submission.csv")

        # print("======================")
        # print("data_dict_clusters", self.data_dict_clusters)
        # print("dataset", self.dataset)
        # print("scene_name", self.scene_name)
        # print("base_path", self.base_path)
        # print("work_dir", self.work_dir)
        # print("cluster_work_dir", self.cluster_work_dir)
        # print("device_id", self.device_id)

        start_ts = time.time()
        logger.info(f"\033[95m[mast3r]\033[0m {self.dataset}, {self.scene_name}")
        cmd = [
            "python",
            "/kaggle/input/imc25-codes/mast3r_proc.py",
            "--dataset",
            self.dataset,
            "--scene_name",
            self.scene_name,
            "--base_path",
            self.base_path,
            "--submission_path",
            submission_path,
            "--cluster_data_dict_path",
            cluster_data_dict_path,
            "--device_id",
            str(self.device_id),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        end_ts = time.time()
        logger.info(
            f"\033[95m[mast3r]\033[0m {self.dataset}, {self.scene_name} ... DONE ({end_ts - start_ts:.2f}s)"
        )

        return submission_path
