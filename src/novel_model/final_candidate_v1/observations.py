"""Observation export depends only on NumPy, SQLite and the native reconstruction interface."""
from contextlib import closing
from pathlib import Path
import sqlite3
import numpy as np
from src.novel_model.geometry_capture_v1.database import array


def immutable_db(path):
    """Independent reads of a closed, checkpointed SQLite backup create no WAL sidecars."""
    path = Path(path)
    wal = Path(str(path)+"-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("expected a standalone database snapshot")
    return sqlite3.connect(path.resolve().as_uri()+"?mode=ro&immutable=1",uri=True)


def export_observations(models, parent):
    """Project optimized landmarks and audit the native track and database row both ways."""
    errors, groups = {}, []
    with closing(immutable_db(parent)) as db:
        for model in models.values():
            registered = set(model.reg_image_ids())
            names = []
            linked = 0
            for i in sorted(registered):
                im = model.images[i]
                row = db.execute("SELECT name,camera_id,rows,cols,data FROM images JOIN keypoints "
                                 "USING(image_id) WHERE image_id=?", (int(i),)).fetchone()
                if row is None or im.name != row[0] or im.camera_id != row[1] or im.name in errors:
                    raise ValueError("image or camera identity")
                keypoints = array(row[4], "<f4", (row[2],row[3]))
                if len(im.points2D) != len(keypoints):
                    raise ValueError("feature cardinality")
                residual = np.full(len(keypoints), np.nan)
                cam = model.cameras[im.camera_id]
                for k, obs in enumerate(im.points2D):
                    if not np.array_equal(obs.xy, keypoints[k,:2]):
                        raise ValueError("feature row or pixel coordinates changed")
                    if obs.has_point3D():
                        xyz = im.cam_from_world * model.points3D[obs.point3D_id].xyz
                        if not np.isfinite(xyz).all() or xyz[2] <= 0:
                            raise ValueError("invalid depth")
                        projected = cam.img_from_cam(xyz[:2]/xyz[2])
                        if projected is None or np.shape(projected) != (2,) or not np.isfinite(projected).all():
                            raise ValueError("invalid projection")
                        residual[k] = np.linalg.norm(projected - obs.xy)
                        linked += 1
                errors[im.name] = residual
                names.append(im.name)
            seen = set()
            for point_id, point in model.points3D.items():
                for el in point.track.elements:
                    key = (el.image_id, el.point2D_idx)
                    if key in seen or key[0] not in registered or not 0 <= key[1] < len(model.images[key[0]].points2D):
                        raise ValueError("invalid track element")
                    if model.images[key[0]].points2D[key[1]].point3D_id != point_id:
                        raise ValueError("track backlink mismatch")
                    seen.add(key)
            if len(seen) != linked:
                raise ValueError("missing track backlinks")
            if names:
                groups.append(sorted(names))
    return sorted(groups), errors


