"""Camera 1 -> camera 2. Standalone CPU solver; never receives reconstructed cameras."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


class Unavailable(ValueError):
    pass


def normalized(xy, cam):
    if cam["model"] not in (
        "SIMPLE_PINHOLE",
        "PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "OPENCV",
    ):
        raise Unavailable("unsupported_camera_model")
    p = cam["params"]
    if cam["model"] in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = p[:4]
        coeff = p[4:]
    else:
        fx, cx, cy = p[:3]
        fy = fx
        coeff = p[3:]
    if min(fx, fy) <= 0 or not (0 <= cx <= cam["width"] and 0 <= cy <= cam["height"]):
        raise Unavailable("invalid_intrinsics")
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    dist = np.zeros(5)
    dist[: len(coeff)] = coeff
    points = cv2.undistortPoints(
        np.ascontiguousarray(xy[:, :2], dtype=float).reshape(-1, 1, 2), k, dist
    ).reshape(-1, 2)
    xyz = np.column_stack((points, np.ones(len(points))))
    reprojected = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, dist)[0].reshape(
        -1, 2
    )
    if (
        not np.isfinite(points).all()
        or np.max(np.linalg.norm(reprojected - xy[:, :2], axis=1)) > 0.1
    ):
        raise Unavailable("distortion_inversion_unreliable")
    return points, float(min(fx, fy))


def corroborate(a, b):
    """Two disjoint estimates test the identifiability of the supplied focal scales."""
    if len(a) < 16:
        return False
    estimates = []
    order = np.random.default_rng(20261111).permutation(len(a))
    for ids in (order[::2], order[1::2]):
        F, _ = cv2.findFundamentalMat(a[ids], b[ids], cv2.FM_8POINT)
        if F is None or F.shape != (3, 3):
            return False

        def residual(logscale):
            u, v = np.exp(logscale)
            E = np.diag([v, v, 1]) @ F @ np.diag([u, u, 1])
            E = E / max(np.linalg.norm(E), 1e-15)
            S = E @ E.T
            return ((2 * S - np.eye(3) * np.trace(S)) @ E).ravel()

        fit = least_squares(residual, [0.0, 0.0], bounds=(-0.7, 0.7), max_nfev=60)
        sensitivity = np.linalg.svd(fit.jac, compute_uv=False)
        if (
            not fit.success
            or sensitivity[-1] < 0.005
            or sensitivity[-1] < 0.02 * sensitivity[0]
            or np.max(np.abs(fit.x)) > np.log(1.15)
            or np.linalg.norm(fit.fun) > 0.03
        ):
            return False
        estimates.append(fit.x)
    return np.max(np.abs(estimates[0] - estimates[1])) <= np.log(1.10)


def test_pose(a, b, r, t, focal):
    t = np.asarray(t, dtype=float).reshape(3)
    if np.linalg.norm(t) < 1e-12:
        return None
    t = t / np.linalg.norm(t)
    h = cv2.triangulatePoints(
        np.eye(3, 4), np.column_stack((r, t)), a.T.copy(), b.T.copy()
    )
    good = np.abs(h[3]) > 1e-10
    xyz = np.zeros((len(a), 3))
    xyz[good] = (h[:3, good] / h[3, good]).T
    other = xyz @ r.T + t
    ok = good & (xyz[:, 2] > 1e-9) & (other[:, 2] > 1e-9)
    if not np.any(ok):
        return None
    x, y = xyz[ok], other[ok]
    errors = (
        np.maximum(
            np.linalg.norm(x[:, :2] / x[:, 2, None] - a[ok], axis=1),
            np.linalg.norm(y[:, :2] / y[:, 2, None] - b[ok], axis=1),
        )
        * focal
    )
    v = y @ r
    angles = np.rad2deg(
        np.arctan2(np.linalg.norm(np.cross(x, v), axis=1), (x * v).sum(axis=1))
    )
    return dict(
        positive_fraction=float(ok.mean()),
        median_angle_deg=float(np.median(angles)),
        p75_error_px=float(np.percentile(errors, 75)),
    )


def recover(pair, xy1, xy2, cam1, cam2, quality1, quality2, gates):
    """A masked rotation carries a specific rejection reason. Unknown fields never vote."""
    try:
        if pair["config"] in (4, 5, 6):
            raise Unavailable("planar_or_panoramic_configuration")
        if pair["config"] not in (2, 3):
            raise Unavailable("unsupported_or_degenerate_configuration")
        if len(xy1) < gates["recovery_min_inliers"]:
            raise Unavailable("too_few_correspondences")
        a, fa = normalized(xy1, cam1)
        b, fb = normalized(xy2, cam2)
        f = max(fa, fb)
        for x in (a, b):
            s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
            if s[1] < 1e-5 or s[1] < 1e-3 * s[0]:
                raise Unavailable("collinear_points")
        hom, _ = cv2.findHomography(a, b, method=0)
        if hom is not None:
            predicted = cv2.perspectiveTransform(a.reshape(-1, 1, 2), hom).reshape(
                -1, 2
            )
            if np.percentile(np.linalg.norm(predicted - b, axis=1) * f, 75) <= 1:
                raise Unavailable("homography_explains_measurements")
        if not (quality1 and quality2) and not corroborate(a, b):
            raise Unavailable("uncorroborated_focal_prior")
        calibration = (
            "exif_consistent" if quality1 and quality2 else "two_disjoint_focal_checks"
        )

        def usable(stats):
            return (
                stats
                and stats["positive_fraction"]
                >= gates["recovery_min_positive_depth_fraction"]
                and stats["median_angle_deg"]
                >= gates["recovery_min_median_triangulation_deg"]
                and stats["p75_error_px"] <= 2
            )

        if pair["config"] == 2 and "qvec" in pair and "tvec" in pair:
            q = pair["qvec"]
            if abs(np.linalg.norm(q) - 1) < 1e-6:
                r = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
                stats = test_pose(a, b, r, pair["tvec"], f)
                if usable(stats):
                    return r, dict(
                        method="measured_qvec_tvec_validated",
                        calibration=calibration,
                        **stats,
                    )
        # Refit on normalized rays. Stored F is in distorted pixel coordinates.
        essential, _ = cv2.findFundamentalMat(a, b, cv2.FM_8POINT)
        if essential is None or essential.shape != (3, 3):
            raise Unavailable("essential_fit_failed")
        singular = np.linalg.svd(essential, compute_uv=False)
        if singular[1] <= 1e-12 or singular[1] / singular[0] < 0.85:
            raise Unavailable("essential_spectrum_inconsistent")
        r1, r2, t = cv2.decomposeEssentialMat(essential)
        choices = []
        for r in (r1, r2):
            for sign in (-1, 1):
                stats = test_pose(a, b, r, sign * t, f)
                if usable(stats):
                    choices.append((r, stats))
        if len(choices) != 1:
            raise Unavailable("no_unique_valid_decomposition")
        r, stats = choices[0]
        return r, dict(
            method="normalized_eight_point_v1", calibration=calibration, **stats
        )
    except (Unavailable, cv2.error, np.linalg.LinAlgError) as error:
        return None, {
            "reason": str(error)
            if isinstance(error, Unavailable)
            else type(error).__name__
        }
