"""Readers for one WildWorld sample dir (data_part<x>/<id>/): rgb.mp4, depth.mkv, state.csv, skeleton.npz.

rgb.mp4:      HEVC 1280x720, 30 fps CFR, no HUD; frame i <-> state.csv row i <-> skeleton.npz [:, i] (counts are equal).
state.csv:    one row per frame; tuple cells "(a, b, c)" (all-nan tuple = invalid camera on that frame).
              camera.K = (fx, fy, cx, cy) px at 1280x720 (fx changes per frame: zoom);
              camera.pos = world xyz; camera.rot = quaternion (x, y, z, w), camera-to-world in the OpenGL camera frame
              (x right, y up, looking along -z).  World is y-up (characters' rotations are pure yaw about y), metres.
skeleton.npz: '<kind><index>_<type_id>' -> (N, J, 3) float32 world joint positions; kind m = monster, n = player
              character, w = the player's weapon.  Bones in skeleton_edges.json ('em_<type>', 'npc', 'wp_<type>').
depth.mkv:    8-bit depth, lossless HEVC (see depth_code_to_m for the decoding we determined).
Projection (as the dataset's scripts/render_skeleton_video.py): p_gl = R^T (X - t); p_cv = p_gl * (1, -1, -1);
u = fx x / z + cx, v = fy y / z + cy.
"""
import ast, csv, json, os
import numpy as np
import av
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
# bone lists of the dataset (skeleton_edges.json from the WildWorld release); WW_EDGES overrides the location
EDGES = json.load(open(os.environ.get("WW_EDGES", f"{HERE}/skeleton_edges.json")))
EDGE_KEY = {"m": "em_{t}", "n": "npc", "w": "wp_{t}"}
SRC_W, SRC_H, SRC_FPS = 1280, 720, 30
csv.field_size_limit(1 << 30)


def _cell(s):
    s = s.strip()
    if s == "" or s.lower() == "nan": return float("nan")
    if s[0] == "(":
        try: return tuple(float(x) for x in s[1:-1].split(","))
        except ValueError: return ast.literal_eval(s)
    try: return float(s)
    except ValueError: return s


class Sample:
    def __init__(self, d):
        self.d = d.rstrip("/")
        with open(f"{self.d}/state.csv", newline="") as f:
            rd = csv.reader(f)
            self.header = next(rd)
            self.rows = [r for r in rd if r]
        self.N = len(self.rows)
        ci = {c: k for k, c in enumerate(self.header)}
        self.col = ci

        def tup(name, n):
            k = ci[name]; out = np.full((self.N, n), np.nan)
            for i, r in enumerate(self.rows):
                v = _cell(r[k]) if k < len(r) else float("nan")
                if isinstance(v, tuple) and len(v) == n: out[i] = v
            return out
        self.K = tup("camera.K", 4)
        self.cam_pos = tup("camera.pos", 3)
        self.cam_quat = tup("camera.rot", 4)
        self.cam_ok = ~(np.isnan(self.K).any(1) | np.isnan(self.cam_pos).any(1) | np.isnan(self.cam_quat).any(1))
        q = np.where(self.cam_ok[:, None], self.cam_quat, [0.0, 0.0, 0.0, 1.0])
        self.R = Rotation.from_quat(q).as_matrix()                        # (N,3,3) c2w, OpenGL camera frame
        z = np.load(f"{self.d}/skeleton.npz")
        self.skel = {k: z[k] for k in sorted(z.files)}

    def value(self, i, name):
        k = self.col.get(name)
        if k is None or k >= len(self.rows[i]): return None
        return _cell(self.rows[i][k])

    def row_dict(self, i):
        return {c: self.rows[i][k] for c, k in self.col.items() if k < len(self.rows[i])}

    @staticmethod
    def edges(key):
        kind, t = key[0], key.split("_", 1)[1]
        return np.array(EDGES.get(EDGE_KEY[kind].format(t=t), []), int).reshape(-1, 2)

    def to_cam(self, i, X):
        """X (n,3) world -> (n,3) OpenCV camera coords (x right, y down, z forward)."""
        return ((X - self.cam_pos[i]) @ self.R[i]) * np.array([1.0, -1.0, -1.0])

    def project(self, i, X):
        """X (n,3) world -> u, v (source px at 1280x720), z (m, forward)."""
        p = self.to_cam(i, X)
        fx, fy, cx, cy = self.K[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            return fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy, p[:, 2]

    def c2w_cv(self, i):
        """4x4 camera-to-world with an OpenCV camera (x right, y down, z forward)."""
        M = np.eye(4)
        M[:3, :3] = self.R[i] @ np.diag([1.0, -1.0, -1.0]); M[:3, 3] = self.cam_pos[i]
        return M


def iter_frames(path, start=0, fmt="rgb24", threads=4):
    """Yield (index, HxWxC uint8) from frame `start` on (decodes from the stream start; index by decode order)."""
    con = av.open(path)
    s = con.streams.video[0]
    s.thread_type = "AUTO"; s.codec_context.thread_count = threads
    try:
        for i, fr in enumerate(con.decode(s)):
            if i < start: continue
            yield i, fr.to_ndarray(format=fmt)
    finally:
        con.close()


def video_info(path):
    con = av.open(path)
    s = con.streams.video[0]
    info = dict(codec=s.codec_context.name, w=s.width, h=s.height, fps=float(s.average_rate or 0), frames=s.frames,
                pix_fmt=s.codec_context.pix_fmt,
                duration_s=float(s.duration * s.time_base) if s.duration else None)
    con.close()
    return info
