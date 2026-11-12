"""Reading UniVTAC isaac51 HDF5 episodes.

Episode layout (60 Hz, one row per saved frame):
  embodiment/joint                         (T, 9)  7 arm joints + 2 finger joints
  embodiment/ee                            (T, 7)  position + quaternion
  observation/{head,wrist}/rgb             (T,)    JPEG bytes, 480x270
  tactile/{left,right}_tactile/rgb         (T,)    JPEG bytes, 320x240 (markerless)
  tactile/{left,right}_tactile/press_depth (T, 240, 320) float32, 0 where there is no contact
  tactile/{left,right}_tactile/marker      (T, 2, 63, 2) marker positions (reference, current)
Action convention (as in UniVTAC's policies): action[t] = joint[t + 1].

UniVTAC writes RGB arrays with cv2.imencode, which treats them as BGR, so decoding the JPEGs with cv2 gives back the
original RGB array (decoding with PIL would swap red and blue).
"""
import cv2
import h5py
import numpy as np
from PIL import Image

from . import paths

FINGERS = ("left", "right")


def decode_jpeg(buf):
    """UniVTAC JPEG bytes -> (H, W, 3) uint8 RGB."""
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def load_background(path=None):
    """Non-contact GelSight Mini image of the simulator, (240, 320, 3) uint8 RGB."""
    return np.asarray(Image.open(path or paths.GEL_BACKGROUND).convert("RGB"))


def open_episode(path):
    return h5py.File(path, "r")


def read_tactile_rgb(f, finger, idx=None):
    ds = f[f"tactile/{finger}_tactile/rgb"]
    idx = range(len(ds)) if idx is None else idx
    return np.stack([decode_jpeg(ds[i]) for i in idx])


def press_depth_max(f, finger):
    """Per-frame maximum indentation of one gel pad, shape (T,)."""
    return f[f"tactile/{finger}_tactile/press_depth"][:].max(axis=(1, 2))


def marker_displacement(f, finger):
    """Per-frame mean marker displacement (pixels) from the reference position, shape (T,)."""
    m = f[f"tactile/{finger}_tactile/marker"][:]
    return np.linalg.norm(m[:, 1] - m[:, 0], axis=-1).mean(axis=1)


def clip_indices(t, num_frames=4, stride=4, length=None):
    """Frame indices of the clip ending at t: (t - (n-1)*stride, ..., t), clamped to [0, length - 1]."""
    idx = np.arange(t - (num_frames - 1) * stride, t + 1, stride)
    hi = (length - 1) if length is not None else idx.max()
    return np.clip(idx, 0, hi)
