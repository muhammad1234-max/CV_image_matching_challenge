import re
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from loguuuuuuu import logger


def clustering(
    dataset: str,
    image_paths: list[Path],
    work_dir: str | Path,
    quantile: float = 0.1,
    eps: float | None = None,
) -> dict[str, dict[str, list[str | Path]]]:
    # data_dict_clusters = clustering(data_dict[dataset][scene])
    # data_dict ... scene が cluster0 だけ
    # data_dict_clusters .. scene が cluster1 ~ clusterN
    df = pd.read_csv(Path(work_dir) / "image_pair.csv")
    thresh = df["match_num"].quantile(quantile)
    nodes = list(set(df["key1"].unique().tolist() + df["key2"].unique().tolist()))
    g = nx.Graph()
    for i, row in df.iterrows():
        g.add_edge(row["key1"], row["key2"], weight=row["match_num"])
    mst = nx.maximum_spanning_tree(g)
    for e in mst.edges():
        if eps is not None:
            if mst[e[0]][e[1]]["weight"] < eps:
                mst.remove_edge(e[0], e[1])
        else:
            if mst[e[0]][e[1]]["weight"] < thresh:
                mst.remove_edge(e[0], e[1])

    data_dict_clusters = {dataset: {}}
    for i, cluster in enumerate(nx.connected_components(mst), start=1):
        data_dict_clusters[dataset][f"cluster{i}"] = []
        cc_filenames = list(set([key_filename for key_filename in cluster]))
        for p in image_paths:
            if Path(p).name in cc_filenames:
                data_dict_clusters[dataset][f"cluster{i}"].append(p)

    return data_dict_clusters


class MSTThresh:
    def __init__(self, args):
        self.args: dict[str, Any] = args
        self.quantile = self.args.get("quantile", 0.1)
        self.eps: float | None = self.args.get("eps", None)

    def __call__(
        self, dataset: str, image_paths: list[Path], work_dir: Path
    ) -> dict[str, dict[str, list[str | Path]]]:
        return clustering(dataset, image_paths, work_dir, quantile=self.quantile, eps=self.eps)


class KNNGraph:
    def __init__(self, args):
        self.args: dict[str, Any] = args

    def __call__(
        self, dataset: str, image_paths: list[Path], work_dir: Path
    ) -> dict[str, dict[str, list[str | Path]]]:
        return clustering(dataset, image_paths, work_dir)


def clusteringV2(
    dataset: str,
    image_paths: list[Path],
    work_dir: str | Path,
    quantile: float = 0.1,
    eps: float | None = None,
) -> dict[str, dict[str, list[str | Path]]]:
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

    logger.warning(
        f"Clustering failed. Using default clustering with quantile={quantile}, eps={eps}"
    )

    return clustering(
        dataset,
        image_paths,
        work_dir,
        quantile=quantile,
        eps=eps,
    )


class MSTThreshV2:
    def __init__(self, args):
        self.args: dict[str, Any] = args
        self.quantile = self.args.get("quantile", 0.1)
        self.eps: float | None = self.args.get("eps", None)

    def __call__(
        self, dataset: str, image_paths: list[Path], work_dir: Path
    ) -> dict[str, dict[str, list[str | Path]]]:
        return clusteringV2(dataset, image_paths, work_dir, quantile=self.quantile, eps=self.eps)


def clusteringV3(
    dataset: str,
    image_paths: list[Path],
    work_dir: str | Path,
    quantile: float = 0.1,
    eps: float | None = None,
    max_cluster_size: int = 5,
) -> dict[str, dict[str, list[str | Path]]]:
    # data_dict_clusters = clustering(data_dict[dataset][scene])
    # data_dict ... scene が cluster0 だけ
    # data_dict_clusters .. scene が cluster1 ~ clusterN
    df = pd.read_csv(Path(work_dir) / "image_pair.csv")
    thresh = df["match_num"].quantile(quantile)
    nodes = list(set(df["key1"].unique().tolist() + df["key2"].unique().tolist()))
    g = nx.Graph()
    for i, row in df.iterrows():
        g.add_edge(row["key1"], row["key2"], weight=row["match_num"])

    # NOTE: cluster は高々5とする
    # threshold 以下を全て削除するのではなく、小さな weight から削除していく

    mst = nx.maximum_spanning_tree(g)
    for e in sorted([(e_, mst[e_[0]][e_[1]]["weight"]) for e_ in mst.edges()], key=lambda x: x[1]):
        if len(list(nx.connected_components(mst))) >= max_cluster_size:
            break

        # edge check
        if e not in mst.edges():
            continue

        if eps is not None:
            if mst[e[0]][e[1]]["weight"] < eps:
                mst.remove_edge(e[0], e[1])
        else:
            if mst[e[0]][e[1]]["weight"] < thresh:
                mst.remove_edge(e[0], e[1])

    data_dict_clusters = {dataset: {}}
    for i, cluster in enumerate(nx.connected_components(mst), start=1):
        data_dict_clusters[dataset][f"cluster{i}"] = []
        cc_filenames = list(set([key_filename for key_filename in cluster]))
        for p in image_paths:
            if Path(p).name in cc_filenames:
                data_dict_clusters[dataset][f"cluster{i}"].append(p)

    return data_dict_clusters


class MSTThreshV3:
    def __init__(self, args):
        self.args: dict[str, Any] = args
        self.quantile = self.args.get("quantile", 0.1)
        self.eps: float | None = self.args.get("eps", None)
        self.max_cluster_size = self.args.get("max_cluster_size", 5)

    def __call__(
        self, dataset: str, image_paths: list[Path], work_dir: Path
    ) -> dict[str, dict[str, list[str | Path]]]:
        return clusteringV3(
            dataset,
            image_paths,
            work_dir,
            quantile=self.quantile,
            eps=self.eps,
            max_cluster_size=self.max_cluster_size,
        )
