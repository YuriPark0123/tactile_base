"""Wire format between the simulator adapter and the policy server.

Transport: multiprocessing.connection on localhost (authkey b"worldtactile"); every message is an npz blob (stable
across numpy versions of the two environments).
Request : head, wrist (H, W, 3) uint8 RGB; tac_left, tac_right (4, 240, 320, 3) uint8 clips (t-12, t-8, t-4, t);
          joint (9,) float32; instruction, task (str); seed, step (action index); episode_start (1 on the first call)
          optional, for the tactile prediction log: tac_left_target / tac_right_target (the clip that ends at an
          earlier request's step + target_offset), target_pred_step (that request's step), target_step
Response: actions (H, 9) float32 absolute joint targets; target_offset
"""
import io

import cv2
import numpy as np

AUTHKEY = b"worldtactile"


def pack(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def unpack(blob):
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    return {k: z[k] for k in z.files}


def jpeg_roundtrip(img):
    """Apply the JPEG compression the training images went through (UniVTAC stores JPEG; decoded to true RGB)."""
    ok, buf = cv2.imencode(".jpg", img[..., ::-1])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
