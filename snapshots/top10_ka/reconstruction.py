import contextlib
import subprocess
import io
import time
import sys
import os
import numpy as np

import sqlite3
from PIL import ExifTags
import h5py
from tqdm import tqdm
import warnings
import pickle
import json

from loguuuuuuu import logger


def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])


MAX_IMAGE_ID = 2**31 - 1


CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""


CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""


CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(
    MAX_IMAGE_ID
)


CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""


CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""


CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""


CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"


CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def array_to_blob(array):
    return array.tostring()


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = lambda: self.executescript(
            CREATE_TWO_VIEW_GEOMETRIES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params, prior_focal_length=0, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params), prior_focal_length),
        )
        return cursor.lastrowid

    def add_image(self, name, camera_id, prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,
                name,
                camera_id,
                prior_q[0],
                prior_q[1],
                prior_q[2],
                prior_q[3],
                prior_t[0],
                prior_t[1],
                prior_t[2],
            ),
        )
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),),
        )

    def add_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2
        if image_id1 > image_id2:
            matches = matches[:, ::-1]
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),),
        )

    def add_two_view_geometry(
        self, image_id1, image_id2, matches, F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2
    ):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2
        if image_id1 > image_id2:
            matches = matches[:, ::-1]
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,)
            + matches.shape
            + (
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H),
            ),
        )


def get_focal(height, width, exif):
    max_size = max(height, width)
    focal_found, focal = False, None
    if exif is not None:
        focal_35mm = None
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == "FocalLengthIn35mmFilm":
                focal_35mm = float(value)
                break
        if focal_35mm is not None:
            focal_found = True
            focal = focal_35mm / 35.0 * max_size
            print(f"Focal found: {focal}")
    if focal is None:
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size
    return focal_found, focal


def create_camera(db, height, width, exif, camera_model):
    focal_found, focal = get_focal(height, width, exif)
    if camera_model == "simple-pinhole":
        model = 0  # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    if camera_model == "pinhole":
        model = 1  # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == "simple-radial":
        model = 2  # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == "radial":
        model = 3  # radial
        param_arr = np.array([focal, width / 2, height / 2, 0.0, 0.0])
    elif camera_model == "opencv":
        model = 4  # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0.0, 0.0, 0.0, 0.0])
    return db.add_camera(model, width, height, param_arr, prior_focal_length=int(focal_found))


def add_keypoints(db, target_filenames, feature_dir, h_w_exif, camera_model, single_camera=False):
    keypoint_f = h5py.File(os.path.join(feature_dir, "keypoints.h5"), "r")
    camera_id = None
    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys())):
        if filename not in target_filenames:
            continue
        keypoints = keypoint_f[filename][()]  # type: ignore
        if camera_id is None or not single_camera:
            height = h_w_exif[filename]["h"]
            width = h_w_exif[filename]["w"]
            exif = h_w_exif[filename]["exif"]
            camera_id = create_camera(db, height, width, exif, camera_model)

        image_id = db.add_image(filename, camera_id)
        fname_to_id[filename] = image_id
        db.add_keypoints(image_id, keypoints)
    return fname_to_id


def add_matches_and_fms(db, target_filenames, feature_dir, fname_to_id, fms):
    match_file = h5py.File(os.path.join(feature_dir, "matches.h5"), "r")
    added = set()
    for key_1 in match_file.keys():
        if key_1 not in target_filenames:
            continue
        group = match_file[key_1]
        for key_2 in group.keys():  # type: ignore
            if key_2 not in target_filenames:
                continue
            id_1 = fname_to_id[key_1]
            id_2 = fname_to_id[key_2]
            pair_id = (id_1, id_2)
            if pair_id in added:
                warnings.warn(f"Pair {pair_id} ({id_1}, {id_2}) already added!")
                continue
            added.add(pair_id)
            matches = group[key_2][()]  # type: ignore
            db.add_matches(id_1, id_2, matches)
            db.add_two_view_geometry(id_1, id_2, matches, fms[(key_1, key_2)])


def import_into_colmap(target_filenames, orig_feature_dir, feature_dir, h_w_exif, fms):
    db = COLMAPDatabase.connect(f"{feature_dir}/colmap.db")
    db.create_tables()
    fname_to_id = add_keypoints(
        db,
        target_filenames,
        orig_feature_dir,
        h_w_exif,
        camera_model="simple-radial",
        single_camera=False,
    )
    add_matches_and_fms(db, target_filenames, orig_feature_dir, fname_to_id, fms)
    db.commit()
    db.close()


def reconstruction(
    data_dict, dataset, scene, base_path, work_dir, cluster_work_dir, colmap_mapper_options
):
    # Import keypoint distances of matches into colmap for RANSAC
    images_dir = data_dict[dataset][scene][0].parent
    images_filepaths = [p.name for p in data_dict[dataset][scene]]
    with open(os.path.join(work_dir, "h_w_exif.json"), "r") as f:
        h_w_exif = json.load(f)

    with open(os.path.join(work_dir, "fms.pkl"), "rb") as f:
        fms = pickle.load(f)
    import_into_colmap(
        target_filenames=images_filepaths,
        orig_feature_dir=work_dir,
        feature_dir=cluster_work_dir,
        h_w_exif=h_w_exif,
        fms=fms,
    )

    database_path = f"{cluster_work_dir}/colmap.db"
    output_path = f"{cluster_work_dir}/colmap_rec_aliked"
    image_pair_path = f"{work_dir}/image_pair.csv"
    submission_path = os.path.join(cluster_work_dir, "submission.csv")
    cluster_data_dict_path = os.path.join(cluster_work_dir, "cluster_data_dict.pkl")
    with open(cluster_data_dict_path, "wb") as f:
        pickle.dump(data_dict, f)

    start_ts = time.time()
    logger.info(f"\033[93m[colmap]\033[0m {dataset}, {scene}")
    cmd = [
        "python",
        "/kaggle/input/imc25-codes/runcolmap.py",
        "--database_path",
        database_path,
        "--image_path",
        str(images_dir),
        "--output_path",
        output_path,
        "--image_pair_path",
        image_pair_path,
        "--submission_path",
        submission_path,
        "--dataset",
        dataset,
        "--scene",
        scene,
        "--base_path",
        base_path,
        "--cluster_data_dict_path",
        cluster_data_dict_path,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc.wait()
    end_ts = time.time()
    logger.info(f"\033[93m[colmap]\033[0m {dataset}, {scene} ... DONE ({end_ts - start_ts:.2f}s)")

    return submission_path
