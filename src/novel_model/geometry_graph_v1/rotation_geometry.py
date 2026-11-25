"""R(j<-i) maps vectors in camera i coordinates into camera j coordinates."""

import numpy as np


def checked_rotation(value, atol):
    raw = np.asarray(value)
    if raw.shape != (3, 3) or raw.dtype.kind not in "fiu":
        raise ValueError("rotation must be a numeric 3x3 array")
    r = raw.astype(np.float64)
    if (
        not np.isfinite(r).all()
        or abs(np.linalg.det(r) - 1) > atol
        or np.max(np.abs(r.T @ r - np.eye(3))) > atol
    ):
        raise ValueError("invalid SO(3) rotation")
    return np.frombuffer(r.tobytes(), dtype=np.float64).reshape(3, 3)


def degrees(r):
    """atan2 avoids loss of precision near both zero and pi; accepts stacks."""
    skew = r - np.swapaxes(r, -2, -1)
    sine = np.linalg.norm(skew, axis=(-2, -1)) / np.sqrt(8)
    cosine = (np.trace(r, axis1=-2, axis2=-1) - 1) / 2
    return np.rad2deg(np.arctan2(sine, cosine))


def alignment(source_frame, target_frame, target_from_source):
    return target_frame.T @ target_from_source @ source_frame
