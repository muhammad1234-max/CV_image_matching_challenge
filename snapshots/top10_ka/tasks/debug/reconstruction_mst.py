import math
import struct
import collections
import os
import json
import pickle
import shutil
import tempfile
from pathlib import Path
import networkx as nx
from copy import deepcopy
import pycolmap
import pandas as pd
import h5py
from tqdm import tqdm
import sqlite3
import numpy as np

from reconstruction import import_into_colmap



CONFIG_PRIOR_UNKNOWN = 0  
CONFIG_PRIOR_CONSISTENT = 10
CONFIG_PRIOR_INCONSISTENT = 11

ROTATION_ERROR_THRESHOLD_DEG = 5.0  
TRANSLATION_ERROR_THRESHOLD_DEG = 10.0  
MIN_INLIERS_FOR_COMPARISON = 100  


def image_ids_to_pair_id(image_id1, image_id2):
    """COLMAPのpair_id計算方法"""
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return 2147483647 * image_id1 + image_id2


def qvec2rotmat(qvec):
    """Quaternion (w, x, y, z) to Rotation Matrix"""
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def calculate_relative_pose(pose1_abs, pose2_abs):

    R1 = qvec2rotmat(pose1_abs[0])
    t1 = pose1_abs[1]
    R2 = qvec2rotmat(pose2_abs[0])
    t2 = pose2_abs[1]

    # T_cam_world = T_world_cam.inverse()
    # R_world_cam1 = R1.T
    # C1 = -R1.T @ t1 # Camera center in world
    # R_world_cam2 = R2.T
    # C2 = -R2.T @ t2

    # T_cam2_cam1 = T_world_cam2 * T_cam1_world
    R_cam2_cam1 = R2 @ R1.T
    t_cam2_cam1 = t2 - R_cam2_cam1 @ t1

    return R_cam2_cam1, t_cam2_cam1


def rotation_angular_error(R1, R2):
    dot = np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def translation_angular_error(t1, t2):
    t1_norm = t1 / (np.linalg.norm(t1) + 1e-8)
    t2_norm = t2 / (np.linalg.norm(t2) + 1e-8)
    dot = np.clip(t1_norm.dot(t2_norm), -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def get_keypoints(db_cursor, image_id):
    keypoints = {}
    data = db_cursor.execute(
        f"SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)
    ).fetchone()
    if data is None:
        return keypoints

    row, col, kpts_data = data
    assert col == 2
    kpts = np.frombuffer(kpts_data, dtype=np.float32)  # type: ignore
    kpts = kpts.reshape((row, col))
    for idx in range(kpts.shape[0]):
        x, y = kpts[idx][0], kpts[idx][1]
        keypoints[idx] = np.array([x, y])
    return keypoints


def get_matches(db_cursor, pair_id):
    matches = []
    rows = db_cursor.execute(
        f"SELECT pair_id, data FROM matches WHERE pair_id=?", (pair_id,)
    ).fetchone()
    if rows and rows[1] is not None:
        match_data = np.frombuffer(rows[1], dtype=np.uint32).reshape(-1, 2)
        matches = match_data.tolist()  # (point2D_idx1, point2D_idx2)のリスト
    return matches


CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    CAMERA_MODELS = {
        CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
        CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
        CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
        CameraModel(model_id=3, model_name="RADIAL", num_params=5),
        CameraModel(model_id=4, model_name="OPENCV", num_params=8),
        CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
        CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
        CameraModel(model_id=7, model_name="FOV", num_params=5),
        CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
        CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
        CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
    }
    CAMERA_MODEL_IDS = dict(
        [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
    )

    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


class Reconstruction:
    def __init__(self, images, cameras):
        self.images = images
        self.cameras = cameras


def build_database_with_camera_prior_consistency_check(
    db_path: str,
    prior_model_dir: str,
    verified_db_path: str,
) -> None:
    """
    consistency check によって matches を filtering した colmap_verify.db を作成
    """
    # Reconstruction をロードする
    update_pairs = {}
    maps = {}
    for prior_model_path in Path(prior_model_dir).glob("*"):
        images = read_images_binary(prior_model_path / "images.bin")
        cameras = read_cameras_binary(prior_model_path / "cameras.bin")
        map_id = int(prior_model_path.stem)
        recon = Reconstruction(images, cameras)
        maps[map_id] = recon
        # NOTE: pycolmap.Reconstruction だと qvec がアクセスできない？
        # recon = pycolmap.Reconstruction(str(prior_model_path))
        # maps[map_id] = recon

        rec_prior = recon
        print(
            f"Prior reconstruction loaded: {len(rec_prior.images)} images, {len(rec_prior.cameras)} cameras."
        )

        prior_image_ids = set(rec_prior.images.keys())

        print("Connecting to database...")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        pairs_to_check = []
        all_pairs = cursor.execute("SELECT pair_id FROM two_view_geometries").fetchall()
        print(
            f"Found {len(all_pairs)} pairs in two_view_geometries. Filtering based on prior model..."
        )

        db_images = cursor.execute("SELECT image_id, name FROM images").fetchall()
        imageid_to_name = {img_id: name for img_id, name in db_images}
        name_to_imageid = {name: img_id for img_id, name in db_images}

        prior_image_ids_in_db = set()
        prior_imageid_map_rec_to_db = {}
        for rec_img_id, rec_img in rec_prior.images.items():
            if rec_img.name in name_to_imageid:
                db_img_id = name_to_imageid[rec_img.name]
                prior_image_ids_in_db.add(db_img_id)
                prior_imageid_map_rec_to_db[rec_img_id] = db_img_id
            else:
                print(
                    f"Warning: Image {rec_img.name} (ID {rec_img_id} in prior) not found in database."
                )

        print(f"Found {len(prior_image_ids_in_db)} prior images present in the database.")

        # Priorモデルに存在する画像のペアを生成
        prior_db_ids_list = sorted(list(prior_image_ids_in_db))
        pairs_to_check_db_ids = []
        for i in range(len(prior_db_ids_list)):
            for j in range(i + 1, len(prior_db_ids_list)):
                id1_db = prior_db_ids_list[i]
                id2_db = prior_db_ids_list[j]
                pairs_to_check_db_ids.append((id1_db, id2_db))

        print(f"Generated {len(pairs_to_check_db_ids)} pairs to check based on prior images.")

        updates = []  # (prior_config_value, pair_id) のリスト

        print("Starting relative pose consistency check...")
        for id1_db, id2_db in tqdm(pairs_to_check_db_ids):
            pair_id = image_ids_to_pair_id(id1_db, id2_db)

            # データベースに two_view_geometry が存在するか確認 (念のため)
            geom_exists = cursor.execute(
                "SELECT 1 FROM two_view_geometries WHERE pair_id=?", (pair_id,)
            ).fetchone()
            if not geom_exists:
                continue

            # --- 1. Priorから相対姿勢を取得 ---
            rec_id1 = next(
                (
                    rec_id
                    for rec_id, db_id in prior_imageid_map_rec_to_db.items()
                    if db_id == id1_db
                ),
                None,
            )
            rec_id2 = next(
                (
                    rec_id
                    for rec_id, db_id in prior_imageid_map_rec_to_db.items()
                    if db_id == id2_db
                ),
                None,
            )

            if rec_id1 is None or rec_id2 is None:
                print(f"Warning: Could not map db ids ({id1_db}, {id2_db}) back to prior rec ids.")
                continue

            img1_prior = rec_prior.images[rec_id1]
            img2_prior = rec_prior.images[rec_id2]
            cam1_prior = rec_prior.cameras[img1_prior.camera_id]
            cam2_prior = rec_prior.cameras[img2_prior.camera_id]

            pose1_abs_prior = (img1_prior.qvec, img1_prior.tvec)
            pose2_abs_prior = (img2_prior.qvec, img2_prior.tvec)
            R_prior, t_prior = calculate_relative_pose(pose1_abs_prior, pose2_abs_prior)
            # print(R_prior, t_prior)


            kp1 = get_keypoints(cursor, id1_db)
            kp2 = get_keypoints(cursor, id2_db)
            matches_indices = get_matches(cursor, pair_id)  # (idx_in_kp1, idx_in_kp2)

            if len(matches_indices) < MIN_INLIERS_FOR_COMPARISON:
                updates.append((CONFIG_PRIOR_UNKNOWN, pair_id))  
                continue

            points1 = np.array(
                [kp1[idx1] for idx1, idx2 in matches_indices if idx1 in kp1 and idx2 in kp2]
            )
            points2 = np.array(
                [kp2[idx2] for idx1, idx2 in matches_indices if idx1 in kp1 and idx2 in kp2]
            )

            if len(points1) < MIN_INLIERS_FOR_COMPARISON:
                updates.append((CONFIG_PRIOR_UNKNOWN, pair_id))  
                continue


            # print(cam1_prior)
            pycolmap_cam1 = pycolmap.Camera(
                model=0,
                width=cam1_prior.width,
                height=cam1_prior.height,
                params=cam1_prior.params,
                # id=cam1_prior.id,
            )
            pycolmap_cam2 = pycolmap.Camera(
                model=0,
                width=cam2_prior.width,
                height=cam2_prior.height,
                params=cam2_prior.params,
                # id=cam2_prior.id,
            )

            # Essential Matrix 推定オプション
            estimation_options = pycolmap.RANSACOptions()
            estimation_options.min_num_trials = 32
            estimation_options.max_error = 2.0  # ransac error threshold
            # estimation_options = pycolmap.EssentialMatrixEstimationOptions()
            # estimation_options.min_num_trials = 32
            # estimation_options.max_error_px = 2.0  # RANSACのエラー閾値

            ret = pycolmap.estimate_essential_matrix(
                points1, points2, pycolmap_cam1, pycolmap_cam2, estimation_options
            )

            # --- 3. 比較と結果の記録 ---
            prior_config_result = CONFIG_PRIOR_UNKNOWN
            if ret is not None and ret["num_inliers"] >= MIN_INLIERS_FOR_COMPARISON:
                relative_pose_est = ret["cam2_from_cam1"]
                # Rigid3d(rotation_xyzw=[0.0130674, -0.00788319, -0.000383833, 0.999883], translation=[0.123667, 0.496256, -0.859323])

                # Rotation3d オブジェクトから回転行列を取得
                R_est = relative_pose_est.rotation.matrix()  # 3x3 numpy array

                # translation 属性から並進ベクトルを取得
                t_est = relative_pose_est.translation  # 3x1 numpy array (スケール不定の場合が多い)

                # R_est = ret["R"]  # Cam1 -> Cam2 の回転
                # t_est = ret["t"]  # Cam1 -> Cam2 の並進 (スケール不定)

                rot_err = rotation_angular_error(R_prior, R_est)
                trans_err = translation_angular_error(t_prior, t_est)

                # print(
                #     ">num_inliers, rot_err, trans_err",
                #     ret["num_inliers"],
                #     rot_err,
                #     trans_err,
                #     imageid_to_name[id1_db],
                #     imageid_to_name[id2_db],
                # )

                if (
                    rot_err <= ROTATION_ERROR_THRESHOLD_DEG
                    and trans_err <= TRANSLATION_ERROR_THRESHOLD_DEG
                ):
                    prior_config_result = CONFIG_PRIOR_CONSISTENT
                else:
                    prior_config_result = CONFIG_PRIOR_INCONSISTENT
            else:
                # 推定失敗またはインライア不足
                prior_config_result = CONFIG_PRIOR_INCONSISTENT  # Inconsistent扱いにする方が安全か

            updates.append((prior_config_result, pair_id))

        print(f"\nFinished checking pairs. Updating {len(updates)} entries in the database...")
        update_pairs[map_id] = updates

    conn.close()

    # map_id => [(PRIOR_CONSISTENCY_CONFIG, pair_id)]
    print(update_pairs)
    shutil.copy(db_path, verified_db_path)

    conn = sqlite3.connect(str(verified_db_path))
    cursor = conn.cursor()
    # cursor.execute("ALTER TABLE two_view_geometries ADD COLUMN prior_config INTEGER DEFAULT 0")

    # remove pair_id on matches and two_view_geometries
    for map_id, updates in update_pairs.items():
        # cursor.executemany(
        #     "UPDATE two_view_geometries SET prior_config = ? WHERE pair_id = ?", updates
        # )
        update_pair_ids = [(row[1],) for row in updates if row[0] != CONFIG_PRIOR_CONSISTENT]
        cursor.executemany("DELETE FROM two_view_geometries WHERE pair_id = ?", update_pair_ids)

    conn.commit()
    conn.close()


def mst_aided_reconstruction(
    dataset: str,
    cluster_name: str,
    images_dir: str,
    scene_work_dir: str,
    cluster_work_dir: str,
) -> None:
    image_pair_path = f"{scene_work_dir}/image_pair.csv"
    h_w_exif_path = f"{scene_work_dir}/h_w_exif.json"
    fms_path = f"{scene_work_dir}/fms.pkl"
    cluster_data_dict_path = os.path.join(cluster_work_dir, "cluster_data_dict.pkl")

    assert Path(image_pair_path).exists(), f"{image_pair_path} does not exist"
    assert Path(h_w_exif_path).exists(), f"{h_w_exif_path} does not exist"
    assert Path(image_pair_path).exists(), f"{image_pair_path} does not exist"
    assert Path(cluster_data_dict_path).exists(), f"{cluster_data_dict_path} does not exist"

    # * image_pair.csv から cluster に含まれる画像を取り出して MST を計算する
    with open(h_w_exif_path, "r") as f:
        h_w_exif = json.load(f)

    with open(fms_path, "rb") as f:
        fms = pickle.load(f)

    with open(cluster_data_dict_path, "rb") as f:
        data_dict = pickle.load(f)

    images_filepaths = [p.name for p in data_dict[dataset][cluster_name]]

    # * MST を構築する
    nodes = set(images_filepaths)
    g = nx.Graph()
    with open(image_pair_path, "r") as f:
        for i, row in enumerate(f):
            if i == 0:
                continue
            elems = row.strip().split(",")
            key1, key2 = elems[:2]
            if key1 not in nodes or key2 not in nodes:
                continue
            match_num = int(elems[-1])
            g.add_edge(key1, key2, weight=int(match_num))

    mst = nx.maximum_spanning_tree(g)
    mst_edges = list(mst.edges())
    fms_mst = {
        k: v for k, v in fms.items() if (k[0], k[1]) in mst_edges or (k[1], k[0]) in mst_edges
    }

    # * cluster_work_dir に MST の {kpts,matches,2vg} を取り出した colmap_mst.db を作成
    mst_database_path = Path(cluster_work_dir) / "colmap_mst.db"
    with (
        tempfile.TemporaryDirectory() as feature_dir_tmpdir,
        tempfile.TemporaryDirectory() as h5_store_feature_dir_tmpdir,
    ):
        keypoint_path = Path(h5_store_feature_dir_tmpdir) / "keypoints.h5"
        matches_path = Path(h5_store_feature_dir_tmpdir) / "matches.h5"
        orig_keypoint_path = Path(scene_work_dir) / "keypoints.h5"
        orig_matches_path = Path(scene_work_dir) / "matches.h5"

        # Copy MST keypints and matches to the h5_store_feature_dir_tmpdir
        with h5py.File(keypoint_path, "w") as f_out, h5py.File(orig_keypoint_path, "r") as f_in:
            for k in f_in.keys():
                f_out[k] = f_in[k][()]  # type: ignore

        with h5py.File(matches_path, "w") as f_out, h5py.File(orig_matches_path, "r") as f_in:
            for lhs in f_in.keys():
                group = f_in[lhs]
                for rhs in group.keys():  # type: ignore
                    if (lhs, rhs) in fms_mst or (rhs, lhs) in fms_mst:
                        out = f_in[lhs][rhs][()]  # type: ignore
                        out_group = f_out.require_group(lhs)
                        out_group.create_dataset(rhs, data=out)

        import_into_colmap(
            target_filenames=images_filepaths,
            orig_feature_dir=h5_store_feature_dir_tmpdir,
            feature_dir=feature_dir_tmpdir,
            h_w_exif=h_w_exif,
            fms=fms_mst,
        )
        tmp_mst_database_path = Path(feature_dir_tmpdir) / "colmap.db"
        shutil.move(tmp_mst_database_path, mst_database_path)

    # * colmap_mst.db で incremental mapping を実行して colmap_mst_rec direcotry に出力
    colmap_mst_rec_dir = Path(cluster_work_dir) / "colmap_mst_rec"
    colmap_mst_rec_dir.mkdir(parents=True, exist_ok=True)
    mst_colmap_mapper_options = {
        "min_model_size": 3,  # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
        "max_num_models": 5,
        # "num_threads": 1,
    }
    mapper_options = pycolmap.IncrementalPipelineOptions(**mst_colmap_mapper_options)
    maps = pycolmap.incremental_mapping(
        database_path=str(mst_database_path),
        image_path=images_dir,
        output_path=str(colmap_mst_rec_dir),
        options=mapper_options,
    )


# --- 定数・ヘルパー関数 (前回のコードから流用) ---
CONFIG_PRIOR_INCONSISTENT = 11


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return 2147483647 * image_id1 + image_id2


def filter_reconstruction_observations(rec, db_path):
    """
    データベースの prior_config に基づいて Reconstruction オブジェクトの観測をフィルタリングする
    (Reconstruction オブジェクトを直接変更する)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # prior_config が INCONSISTENT なペアIDのセットを取得
    inconsistent_pair_ids = set()
    rows = cursor.execute(
        "SELECT pair_id FROM two_view_geometries WHERE prior_config = ?",
        (CONFIG_PRIOR_INCONSISTENT,),
    ).fetchall()
    for row in rows:
        inconsistent_pair_ids.add(row[0])

    conn.close()
    print(f"Found {len(inconsistent_pair_ids)} inconsistent pairs based on prior_config.")

    observations_to_delete = []  # (point3D_id, image_id, point2D_idx) のタプルリスト

    print("Scanning observations for filtering...")
    # Reconstruction 内の3D点をループ
    for point3D_id, point3D in tqdm(rec.points3D.items()):
        track_elements = list(point3D.track.elements)  # イテレーション用にコピー

        # トラック内の観測ペアをチェック
        for i in range(len(track_elements)):
            for j in range(i + 1, len(track_elements)):
                el1 = track_elements[i]
                el2 = track_elements[j]

                pair_id = image_ids_to_pair_id(el1.image_id, el2.image_id)

                if pair_id in inconsistent_pair_ids:
                    # このペアは不整合 -> 両方の観測を削除対象とする
                    # 重複を避けるため、後で set にするが、ここでは一旦リストに追加
                    observations_to_delete.append((point3D_id, el1.image_id, el1.point2D_idx))
                    observations_to_delete.append((point3D_id, el2.image_id, el2.point2D_idx))

    # 重複削除
    unique_observations_to_delete = set(observations_to_delete)
    print(
        f"Identified {len(unique_observations_to_delete)} observations potentially linked to inconsistent pairs."
    )

    # 実際に観測を削除
    num_deleted = 0
    print("Deleting observations from reconstruction...")
    for point3D_id, image_id, point2D_idx in tqdm(unique_observations_to_delete):
        # オブジェクトが存在するか確認
        if point3D_id in rec.points3D and image_id in rec.images:
            point3D = rec.points3D[point3D_id]
            image = rec.images[image_id]

            # 配列インデックスの範囲チェック
            if point2D_idx < len(image.points2D):
                # Point3Dのトラックから要素を削除
                try:
                    point3D.track.delete_element(image_id, point2D_idx)
                    # ImageのPoint2Dから3D点への参照をリセット
                    image.reset_point3D_for_point2D(point2D_idx)
                    num_deleted += 1
                except RuntimeError as e:
                    # すでに削除されている場合など
                    # print(f"Warning: Could not delete obs ({point3D_id}, {image_id}, {point2D_idx}): {e}")
                    pass
            else:
                print(
                    f"Warning: point2D_idx {point2D_idx} out of range for image {image_id} (size {len(image.points2D)}). Skipping deletion."
                )

    print(f"Deleted {num_deleted} observations from tracks and reset point3D IDs.")

    # トラック長が短くなった3D点や、観測がなくなった3D点をフィルタリング
    print("Filtering 3D points based on track length and reprojection error...")
    initial_num_points = len(rec.points3D)
    num_filtered_points = 0  # フィルタリングされた点の数をカウント
    print(">>>", initial_num_points)

    try:
 
        num_filtered = rec.filter_points3D(
            min_track_len=2, max_reproj_error=4.0, min_tri_angle=0.0
        )
        print(f"Filtered {num_filtered} 3D points using rec.filter_points3D.")
        num_filtered_points = num_filtered
    except AttributeError:
        print("Warning: rec.filter_points3D method not found.")
        print("Attempting manual filtering (track length and point3D.error)...")
        points_to_delete = []
        min_track_len = 2
        max_point_error = 4.0 

        for point3D_id, point3D in tqdm(rec.points3D.items()):
            if len(point3D.track.elements) < min_track_len:
                points_to_delete.append(point3D_id)
            elif point3D.error > max_point_error:  
                points_to_delete.append(point3D_id)

        if points_to_delete:
            print(f"Deleting {len(points_to_delete)} points manually...")
            deleted_count = 0
            for point3D_id in tqdm(points_to_delete):
                try:
                    rec.delete_point3D(point3D_id)
                    deleted_count += 1
                except Exception as e:
                    # print(f"Error deleting point {point3D_id}: {e}")
                    pass 
            print(f"Deleted {deleted_count} points manually.")
            num_filtered_points = deleted_count
        else:
            print("No points to filter manually based on current criteria.")

    final_num_points = len(rec.points3D)
    # print(f"Filtered {initial_num_points - final_num_points} 3D points.") # カウント方法変更
    print(
        f"Total points filtered in this step: {num_filtered_points}. Remaining points: {final_num_points}"
    )


    print("Filtering observations with negative depth (manual implementation)...")
    observations_to_delete_neg_depth = []
    num_checked_obs = 0

    # Reconstruction内の全3D点をループ
    for point3D_id, point3D in tqdm(rec.points3D.items()):
        p_world = point3D.xyz  # 3D点のワールド座標


        track_elements_copy = list(point3D.track.elements)
        for track_el in track_elements_copy:
            image_id = track_el.image_id
            point2D_idx = track_el.point2D_idx
            num_checked_obs += 1

            if image_id not in rec.images:
                continue

            image = rec.images[image_id]

            pose = image.cam_from_world
            R = pose.rotation.matrix()  # Rotation3dオブジェクトから回転行列を取得
            t = pose.translation  
            p_cam = R @ p_world + t

            if p_cam[2] <= 1e-9:  
                observations_to_delete_neg_depth.append((point3D_id, image_id, point2D_idx))

    # 重複を削除（リストに追加する方式だと重複する可能性があるため）
    unique_observations_to_delete_neg_depth = set(observations_to_delete_neg_depth)
    print(
        f"Checked {num_checked_obs} observations. Found {len(unique_observations_to_delete_neg_depth)} observations with non-positive depth."
    )

    # 実際に観測を削除
    num_deleted_neg_depth = 0
    if unique_observations_to_delete_neg_depth:
        print("Deleting observations with non-positive depth...")
        for point3D_id, image_id, point2D_idx in tqdm(unique_observations_to_delete_neg_depth):
            # オブジェクトがまだ存在するか確認
            if point3D_id in rec.points3D and image_id in rec.images:
                point3D = rec.points3D[point3D_id]
                image = rec.images[image_id]
                # Point2Dインデックスの範囲チェック
                if point2D_idx < len(image.points2D):
                    try:
                        # Point3Dのトラックから要素を削除
                        point3D.track.delete_element(image_id, point2D_idx)
                        # ImageのPoint2Dから3D点への参照をリセット
                        image.reset_point3D_for_point2D(point2D_idx)
                        num_deleted_neg_depth += 1
                    except RuntimeError as e:
                        # 他の理由ですでに削除されている可能性など
                        pass  # エラーを無視して続行
                # else: point2D_idx out of range warning は必要なら出す

        print(f"Deleted {num_deleted_neg_depth} observations with non-positive depth.")
    else:
        print("No observations with non-positive depth found to delete.")


def run_colmap_with_camera_prior(
    prior_model_dir: str,
    filtered_database_path: str,
    final_model_dir: str,
    images_dir: str,
) -> None:
    Path(final_model_dir).mkdir(parents=True, exist_ok=True)

    for model_dir in Path(prior_model_dir).glob("./*"):
        if not model_dir.is_dir():
            continue

        # ステップ2のモデルを読み込む
        print("Loading initial reconstruction...")
        rec_to_adjust = pycolmap.Reconstruction(str(model_dir))
        initial_num_images = len(rec_to_adjust.images)
        initial_num_points = len(rec_to_adjust.points3D)
        print(f"Initial reconstruction: {initial_num_images} images, {initial_num_points} points.")

        # Reconstructionオブジェクトをフィルタリング（上記関数を呼び出す）
        print("Filtering reconstruction based on prior config...")
        filter_reconstruction_observations(rec_to_adjust, filtered_database_path)
        filtered_num_images = len(rec_to_adjust.images)  # 画像数は変わらないはず
        filtered_num_points = len(rec_to_adjust.points3D)
        print(
            f"Filtered reconstruction: {filtered_num_images} images, {filtered_num_points} points."
        )

        # バンドル調整オプション
        ba_options = pycolmap.BundleAdjustmentOptions()
        ba_options.solver_options.max_num_iterations = 50  # 繰り返し回数
        ba_options.solver_options.function_tolerance = 1e-6
        # refine_focal_length などはデフォルトで True のことが多い（必要なら False に設定）
        # ba_options.refine_focal_length = False
        # ba_options.refine_principal_point = False
        # ba_options.refine_extra_params = False

        # フィルタリングされたReconstructionを使ってバンドル調整 (rec_to_adjust が直接変更される)
        print("Running final bundle adjustment...")
        summary = pycolmap.bundle_adjustment(reconstruction=rec_to_adjust, options=ba_options)
        print("Bundle adjustment finished.")
        print(summary)  # 最適化結果のサマリを表示

        target_dir = Path(final_model_dir) / model_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        rec_to_adjust.write(str(target_dir))


def main() -> None:
    # dataset = "amy_gardens"
    # dataset = "stairs"
    # dataset = "ETs"
    # dataset = "imc2024_lizard_pond"
    dataset = "fbk_vineyard"
    model_name = "v20"
    split_name = "train"
    base_path = "/kaggle/input/image-matching-challenge-2025"
    images_dir = f"{base_path}/{split_name}/{dataset}"
    work_dir = f"/kaggle/working/output/{model_name}/{dataset}"
    cluster_names = [p.name for p in Path(work_dir).glob("cluster*")]

    for cluster_name in cluster_names:
        cluster_work_dir = f"{work_dir}/{cluster_name}"
        cluster_data_dict_path = os.path.join(cluster_work_dir, "cluster_data_dict.pkl")
        with open(cluster_data_dict_path, "rb") as f:
            data_dict = pickle.load(f)

        mst_aided_reconstruction(
            dataset,
            cluster_name,
            images_dir,
            work_dir,
            cluster_work_dir,
        )
        build_database_with_camera_prior_consistency_check(
            f"{cluster_work_dir}/colmap.db",
            f"{cluster_work_dir}/colmap_mst_rec",
            f"{cluster_work_dir}/colmap_filtered.db",
        )

        if False:
            # TODO: BA の API は reconstruction しかないので、
            # TODO: mst reconstruction に 3d points を追加することになるが、できているか不明
            run_colmap_with_camera_prior(
                f"{cluster_work_dir}/colmap_mst_rec",
                f"{cluster_work_dir}/colmap_filtered.db",
                f"{cluster_work_dir}/colmap_final_rec",
                images_dir,
            )

        # TODO: なので colmap_filtered.db で incremental mapping を実行することも検討する
        # TODO: camera pose prior を想定する方法は不明
        output_path = Path(f"{cluster_work_dir}/colmap_final_rec")
        output_path.mkdir(parents=True, exist_ok=True)
        colmap_mapper_options = {
            "min_model_size": 3,  # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
            "max_num_models": 5,
            # "num_threads": 1,
        }
        mapper_options = pycolmap.IncrementalPipelineOptions(**colmap_mapper_options)
        maps = pycolmap.incremental_mapping(
            database_path=f"{cluster_work_dir}/colmap_filtered.db",
            image_path=str(images_dir),
            output_path=str(output_path),
            options=mapper_options,
        )

        map_to_submission_csv(
            f"{cluster_work_dir}/colmap_final_rec",
            f"{cluster_work_dir}/submission.csv",
            images_dir,
            dataset,
            cluster_name,
            data_dict,
            base_path,
            maps=maps,
        )


_EPS = np.finfo(float).eps * 4.0


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion."""
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        # print("special case")
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [
                1.0 - q[2, 2] - q[3, 3],
                q[1, 2] - q[3, 0],
                q[1, 3] + q[2, 0],
                0.0,
            ],
            [
                q[1, 2] + q[3, 0],
                1.0 - q[1, 1] - q[3, 3],
                q[2, 3] - q[1, 0],
                0.0,
            ],
            [
                q[1, 3] - q[2, 0],
                q[2, 3] + q[1, 0],
                1.0 - q[1, 1] - q[2, 2],
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis."""
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    data *= data
    np.sum(data, axis=axis, out=out)
    np.sqrt(out, out)
    return None


def affine_matrix_from_points(v0, v1, shear=False, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.
    v0 and v1 are shape (ndims, -1) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.
    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean traffansformation matrix
    is returned.
    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.
    The returned matrix performs rotation, translation and uniform scaling
    (if specified)."""

    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims : 2 * ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [
            [xx + yy + zz, 0.0, 0.0, 0.0],
            [yz - zy, xx - yy - zz, 0.0, 0.0],
            [zx - xz, xy + yx, yy - xx - zz, 0.0],
            [xy - yx, zx + xz, yz + zy, zz - xx - yy],
        ]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        # print (vector_norm(q), np.linalg.norm(q))
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M


def register_by_Horn(ev_coord, gt_coord, ransac_threshold, inl_cf, strict_cf):
    """Return the best similarity transforms T that registers 3D points pt_ev in <ev_coord> to
    the corresponding ones pt_gt in <gt_coord> according to a RANSAC-like approach for each
    threshold value th in <ransac_threshold>.

    Given th, each triplet of 3D correspondences is examined if not already present as strict inlier,
    a correspondence is a strict inlier if <strict_cf> * err_best < th, where err_best is the registration
    error for the best model so far.
    The minimal model given by the triplet is then refined using also its inliers if their total is greater
    than <inl_cf> * ninl_best, where ninl_best is th number of inliers for the best model so far. Inliers
    are 3D correspondences (pt_ev, pt_gt) for which the Euclidean distance |pt_gt-T*pt_ev| is less than th.
    """

    # remove invalid cameras, the index is returned
    idx_cams = np.all(np.isfinite(ev_coord), axis=0)
    ev_coord = ev_coord[:, idx_cams]
    gt_coord = gt_coord[:, idx_cams]

    # initialization
    n = ev_coord.shape[1]
    r = ransac_threshold.shape[0]
    ransac_threshold = np.expand_dims(ransac_threshold, axis=0)
    ransac_threshold2 = ransac_threshold**2
    ev_coord_1 = np.vstack((ev_coord, np.ones(n)))
    max_no_inl = np.zeros((1, r))
    best_inl_err = np.full(r, np.Inf)
    best_transf_matrix = np.zeros((r, 4, 4))
    best_err = np.full((n, r), np.Inf)
    strict_inl = np.full((n, r), False)
    triplets_used = np.zeros((3, r))

    # run on camera triplets
    for ii in range(n - 2):
        for jj in range(ii + 1, n - 1):
            for kk in range(jj + 1, n):
                i = [ii, jj, kk]
                triplets_used_now = np.full((n), False)
                triplets_used_now[i] = True
                # if both ii, jj, kk are strict inliers for the best current model just skip
                if np.all(strict_inl[i]):
                    continue
                # get transformation T by Horn on the triplet camera center correspondences
                transf_matrix = affine_matrix_from_points(
                    ev_coord[:, i], gt_coord[:, i], usesvd=False
                )
                # apply transformation T to test camera centres
                rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                # compute error and inliers
                err = np.sum((rotranslated - gt_coord) ** 2, axis=0)
                inl = np.expand_dims(err, axis=1) < ransac_threshold2
                no_inl = np.sum(inl, axis=0)
                # if the number of inliers is close to that of the best model so far, go for refinement
                to_ref = np.squeeze(((no_inl > 2) & (no_inl > max_no_inl * inl_cf)), axis=0)
                for q in np.argwhere(to_ref):
                    qq = q[0]
                    if np.any(np.all((np.expand_dims(inl[:, qq], axis=1) == inl[:, :qq]), axis=0)):
                        # already done for this set of inliers
                        continue
                    # get transformation T by Horn on the inlier camera center correspondences
                    transf_matrix = affine_matrix_from_points(
                        ev_coord[:, inl[:, qq]], gt_coord[:, inl[:, qq]]
                    )
                    # apply transformation T to test camera centres
                    rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                    # compute error and inliers
                    err_ref = np.sum((rotranslated - gt_coord) ** 2, axis=0)
                    err_ref_sum = np.sum(err_ref, axis=0)
                    err_ref = np.expand_dims(err_ref, axis=1)
                    inl_ref = err_ref < ransac_threshold2
                    no_inl_ref = np.sum(inl_ref, axis=0)
                    # update the model if better for each threshold
                    to_update = np.squeeze(
                        (no_inl_ref > max_no_inl)
                        | ((no_inl_ref == max_no_inl) & (err_ref_sum < best_inl_err)),
                        axis=0,
                    )
                    if np.any(to_update):
                        triplets_used[0, to_update] = ii
                        triplets_used[1, to_update] = jj
                        triplets_used[2, to_update] = kk
                        max_no_inl[:, to_update] = no_inl_ref[to_update]
                        best_err[:, to_update] = np.sqrt(err_ref)
                        best_inl_err[to_update] = err_ref_sum
                        strict_inl[:, to_update] = (
                            best_err[:, to_update] < strict_cf * ransac_threshold[:, to_update]
                        )
                        best_transf_matrix[to_update] = transf_matrix

    for i in range(r):
        print(
            f"Registered cameras {int(max_no_inl[0, i])}/{n} for threshold {ransac_threshold[0, i]}"
        )

    best_model = {
        "valid_cams": idx_cams,
        "no_inl": max_no_inl,
        "err": best_err,
        "triplets_used": triplets_used,
        "transf_matrix": best_transf_matrix,
    }
    return best_model


def map_to_submission_csv(
    reconstruction_dir: str,
    submission_path: str,
    images_dir: str,
    dataset: str,
    scene: str,
    data_dict: dict,
    base_path: str,
    maps: dict = {},
) -> None:
    # 1. Load the reconstruction
    for recon_dir in Path(reconstruction_dir).glob("*"):
        map_id = int(recon_dir.name)
        maps[map_id] = pycolmap.Reconstruction(str(recon_dir))

        # 2. Look for the best reconstruction: The incremental mapping offered by
        # pycolmap attempts to reconstruct multiple models, we must pick the best one
        images_registered = 0
        best_idx = None

        print("Looking for the best reconstruction")

        if isinstance(maps, dict):
            for idx1, rec in maps.items():
                print(idx1, rec.summary())
                try:
                    if len(rec.images) > images_registered:
                        images_registered = len(rec.images)
                        best_idx = idx1
                except Exception:
                    continue

        # Parse the reconstruction object to get the rotation matrix and translation vector
        # obtained for each image in the reconstruction
        results = {}
        camid_im_map = {}
        if best_idx is not None:
            for k, im in maps[best_idx].images.items():
                key = os.path.join(images_dir, im.name)
                results[key] = {}
                results[key]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
                results[key]["t"] = deepcopy(np.array(im.cam_from_world.translation))

                camid_im_map[im.camera_id] = im.name

        # TODO: 以下がどのような役割を果たしているかを理解する
        # best map からの R, t は上のコードで完了しているはず。以下はなぜ必要？：
        try:
            if best_idx is not None:
                for idx1, rec in maps.items():
                    u_cameras = []
                    g_cameras = []
                    if idx1 != best_idx:
                        # best ではないものから取り出し

                        for k, im in rec.images.items():
                            key = os.path.join(images_dir, im.name)
                            if key in results:
                                g_R = deepcopy(results[key]["R"])
                                g_t = deepcopy(results[key]["t"])
                                g_C = -g_R.T @ g_t

                                u_R = deepcopy(im.cam_from_world.rotation.matrix())
                                u_t = deepcopy(np.array(im.cam_from_world.translation))
                                u_C = -u_R.T @ u_t
                                g_cameras.append(g_C.reshape(3, 1))
                                u_cameras.append(u_C.reshape(3, 1))
                        if len(g_cameras) < 3:
                            continue

                        # NOTE: best にはない位置を取得して追加してるのかも
                        g_cameras = np.array(g_cameras).reshape(3, -1)
                        u_cameras = np.array(u_cameras).reshape(3, -1)
                        inl_cf = 0
                        strict_cf = -1
                        thresholds = np.array([0.025, 0.05, 0.1, 0.2, 0.5, 1.0])
                        model = register_by_Horn(
                            u_cameras, g_cameras, np.asarray(thresholds), inl_cf, strict_cf
                        )
                        T = np.squeeze(model["transf_matrix"][-1])
                        # print(T)
                        # print(T[:3].shape)
                        for k, im in rec.images.items():
                            key = os.path.join(images_dir, im.name)
                            if key not in results:
                                Tcw2 = np.eye(4)
                                Tcw2[:3, :3] = deepcopy(im.cam_from_world.rotation.matrix())
                                Tcw2[:3, 3] = deepcopy(np.array(im.cam_from_world.translation))
                                Tw2c = np.linalg.inv(Tcw2)
                                Tw1c = np.matmul(T, Tw2c)
                                Tcw1 = np.linalg.inv(Tw1c)
                                results[key]["R"] = deepcopy(Tcw1[:3, :3])
                                results[key]["t"] = deepcopy(Tcw1[:3, 3])
        except:
            pass
        # with open(os.path.join(work_dir, "camid_im_map.json"), "w") as f:
        #     json.dump(camid_im_map, f, indent=2)

        print(f"Registered: {dataset} / {scene} -> {len(results)} images")
        print(f"Total: {dataset} / {scene} -> {len(data_dict[dataset][scene])} images")

        # Create Submission
        submission = {
            "image_path": [],
            "dataset": [],
            "scene": [],
            "rotation_matrix": [],
            "translation_vector": [],
        }
        for image in data_dict[dataset][scene]:
            image_path = str(image.relative_to(base_path))
            submission["image_path"].append(image_path)
            submission["dataset"].append(dataset)

            # NOTE: outliers or scene
            submission["scene"].append(scene)

            if str(image) in results:
                R = results[str(image)]["R"].reshape(-1)
                T = results[str(image)]["t"].reshape(-1)
                # submission["scene"].append(scene)
            else:
                R = np.eye(3).reshape(-1)
                T = np.zeros((3))
                # submission["scene"].append("outliers")

            submission["rotation_matrix"].append(arr_to_str(R))
            submission["translation_vector"].append(arr_to_str(T))

        submission_df = pd.DataFrame.from_dict(submission)
        print(f"Save to {submission_path}")
        submission_df.to_csv(submission_path, index=False)


if __name__ == "__main__":
    main()
