"""ww_clip.py <part>/<sample_id>: convert one WildWorld sample ($WW_SRC/<part>/<sample_id>/) into SolarRenderer
clips: non-overlapping 8 s windows, 24 fps, 1344x768, {render, white_box, skeleton}.mp4 + json per window.

Timing: the source is 30 fps CFR, one state.csv / skeleton.npz row per video frame.  Windows are 240 source frames,
back to back, the group centred in the sample; output frame j of a window starting at source frame s0 shows source
frame s0 + floor(1.25 j + 0.5) (src_frame in the json).  Samples shorter than 8 s give nothing.
Geometry: 10 px cropped left and right (1280 -> 1260), then scaled by 16/15 to 1344x768; K is scaled/shifted with it.
Streams: render; white_box (clay shading from engine depth, normals from depth, the same look as the CARLA white box);
skeleton (engine world-space joints of the player, the monsters and the weapon projected through the per-frame
camera, all bones drawn (no occlusion test: monster joints sit metres inside the body), coloured by entity and by
side: left / right / centre in the entity's own frame).  The data has no segmentation, so no class / mask stream.
Depth: depth.mkv is 8-bit grey in limited-range Y; g = (Y - 16) * 255 / 219 and planar depth z = C g / (B - g) metres
(fitted on projected joints, see DEPTH_* below).  It is dithered 1 code, so it is smoothed before the normals.
Json (uploaded last = clip complete) to clips/, or a json with the fail reasons to rejected/; once every window of a
sample is done, .../samples/<sample_id>.json (the resume marker).
Exit 0 = something uploaded, 3 = nothing to do / all rejected, other = failure.
"""
import json, os, subprocess, sys, time, shutil
import numpy as np, cv2, av
cv2.setNumThreads(int(os.environ.get("WW_CV_THREADS", "1")))   # 16 shards share one machine: 1 is fastest per frame

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ww_io import Sample, SRC_W, SRC_H, SRC_FPS, _cell
from scipy.spatial.transform import Rotation

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
SRC = os.environ.get("WW_SRC", "")               # storage prefix of the WildWorld copy (<part>/<sample_id>/...)
DST = os.environ["WW_DST"]                       # storage prefix for accepted clips, ending in /clips
REJ = DST.rsplit("/", 1)[0] + "/rejected"
SAMPLES = DST.rsplit("/", 1)[0] + "/samples"
WORK = os.environ.get("WW_WORK", "wildworld_work")
W, H, FPS, WIN_S = 1344, 768, 24, 8
SRC_WIN = SRC_FPS * WIN_S                      # 240 source frames per window
CROP_X = 10                                    # source columns cropped left and right (1280 -> 1260)
SC = H / SRC_H                                 # 16/15
STREAMS = ("render", "white_box", "skeleton")
LICENSE = "WildWorld (AlayaLab), non-commercial research only, no redistribution"
GEN = "wildworld convert v1"
# depth: g = (Y - 16) * 255 / 219, z = C g / (B - g)  (g = B z / (z + C); fit on projected joints of two samples,
# rms 0.7 code over 5-40 m; one code ~0.26 m at 7 m, ~0.4 m at 20 m, ~0.75 m at 50 m)
DEPTH_B, DEPTH_C = 270.0, 55.0
G_FAR = 250.0                                  # g at or above this: far / sky (z > ~640 m) = invalid for the white box
# rejects
MAX_FROZEN_RUN = 10                            # camera pose bit-identical for this many frames in a row: not the live camera
MAX_CAM_JUMP_M = 3.0                           # camera moves more than this between two source frames: a cut
SKEL_TOL_REL, SKEL_TOL_ABS = 0.15, 1.0         # joint "floats": nearest surface around it is farther than z*(1+rel)+abs
MIN_SKEL_ON = 0.8                              # median per-frame fraction of in-image joints not floating, per entity kind
MAX_BAD_SKEL_FRAMES = 0.1                      # ... and at most this fraction of measured frames below 0.5
MIN_MEASURED = 10                              # frames with >= 5 in-image joints of that kind needed to judge it
MAX_FLAT_DEPTH_FRAMES = 0                      # depth frames with almost no structure (std < 1 code) or mostly far

ENT_COL = {"n": {"c": (255, 220, 0), "l": (0, 255, 120), "r": (0, 160, 255)},
           "m": {"c": (255, 90, 40), "l": (255, 0, 200), "r": (150, 60, 255)},
           "w": {"c": (255, 255, 255), "l": (255, 255, 255), "r": (255, 255, 255)}}
ENT_COL_NOTE = ("RGB; player n: centre (255,220,0) left (0,255,120) right (0,160,255); monster m: centre (255,90,40) "
                "left (255,0,200) right (150,60,255); weapon w: white. A bone takes the side of its two joints "
                "(centre if they differ); white dots on the player's joints.")


def log(*a): print(time.strftime("%H:%M:%S"), NAME[:40], *a, flush=True)


def store(*args, check=True):
    r = subprocess.run([STORE, *args], capture_output=True, text=True)
    if check and r.returncode != 0: raise RuntimeError(f"storage {args[:2]} failed: {r.stderr[-400:]}")
    return r.stdout


def exists_remote(path):
    r = subprocess.run([STORE, "ls", path], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def ffmpeg_writer(path):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
                             "-preset", os.environ.get("WW_PRESET", "faster"), "-threads", "2", "-movflags", "+faststart",
                             path], stdin=subprocess.PIPE)


# ---------------------------------------------------------------- geometry
def fit(a, interp):
    """source (720, 1280[, c]) -> output (768, 1344[, c])"""
    return cv2.resize(a[:, CROP_X:SRC_W - CROP_X], (W, H), interpolation=interp)


def to_out(u, v): return (u - CROP_X) * SC, v * SC


def K_out(K4):
    fx, fy, cx, cy = K4
    return [[round(fx * SC, 4), 0, round((cx - CROP_X) * SC, 4)], [0, round(fy * SC, 4), round(cy * SC, 4)], [0, 0, 1]]


def depth_g(y):
    """raw limited-range Y plane (uint8) -> full-range grey code g (float32), unsmoothed"""
    return (y.astype(np.float32) - 16.0) * (255.0 / 219.0)


def g_to_m(g):
    return DEPTH_C * g / np.maximum(DEPTH_B - g, 1e-3)


SMOOTH = ((0.75, 1), (2.0, 1), (5.0, 2), (12.5, 4))  # (Gaussian sigma, pyramid step) at the half-res working scale
MEDIAN = True                                         # = sigma 1.5 / 4 / 10 / 25 source px
SMOOTH_LO, SMOOTH_HI = 1.5, 3.0                        # codes: full smoothing below, none above (a depth step)


def _up(a, ds, shape):
    while ds > 1:
        a = cv2.pyrUp(a); ds //= 2
    return a[:shape[0], :shape[1]]


def smooth_depth(g):
    """De-quantise the 8-bit depth code, coarse scales over fine: at each Gaussian scale (computed on a Gaussian pyramid
    level 1/ds and brought back with pyrUp, which is smooth: bilinear / bicubic upsampling leaves a grid in the normals)
    the blurred code replaces the running estimate with weight 1 where no depth step is in reach (max |g - blur| over a
    window of +-2 sigma < SMOOTH_LO codes: planes and one-code terraces), 0 where one is (> SMOOTH_HI); the weight map
    is blurred (and only ever lowered) so scale changes leave no seams.  Linear ramps are kept exactly."""
    out = g.copy()
    pyr = [g]
    while len(pyr) < 3: pyr.append(cv2.pyrDown(pyr[-1]))
    for sig, ds in SMOOTH:
        sm = pyr[{1: 0, 2: 1, 4: 2}[ds]]
        sg = sig / ds
        b = cv2.GaussianBlur(sm, (0, 0), sg)
        k = 2 * int(np.ceil(2.0 * sg)) + 1
        dev = cv2.dilate(cv2.absdiff(sm, b), np.ones((k, k), np.uint8))
        a = cv2.min(cv2.max((SMOOTH_HI - dev) * (1.0 / (SMOOTH_HI - SMOOTH_LO)), 0.0), 1.0)
        a = cv2.min(a, cv2.GaussianBlur(a, (0, 0), max(sg / 2, 0.5)))
        if ds > 1:
            b = _up(b, ds, g.shape); a = cv2.min(cv2.max(_up(a, ds, g.shape), 0.0), 1.0)
        out += a * (b - out)
    return out


L1 = np.array([-0.45, -0.35, 0.82], np.float32); L1 /= np.linalg.norm(L1)
L2 = np.array([-0.2, 0.6, 0.3], np.float32); L2 /= np.linalg.norm(L2)


def white_box(g, fx, fy, cx, cy):
    """Clay shading from the 8-bit engine depth (grey code g, source res); same look as the CARLA
    white box (same lights, edge and fog terms).  5x5 median on the raw codes (removes the one-code dither, keeps steps), a 64 px point-reflected border
    (the slope runs on to the frame edge), then everything at half resolution (640x360: the 8-bit depth carries no
    finer geometry once de-quantised, and it is 4x cheaper): smooth_depth, normals from depth, shading, edges, fog;
    resized to 1344x768 at the end.  Camera frame for the lighting: x forward, y right, z up."""
    if MEDIAN:                                  # median on the exact raw Y codes, then back to g
        y = np.clip(np.round(g * (219.0 / 255.0) + 16.0), 0, 255).astype(np.uint8)
        g = depth_g(cv2.medianBlur(y, 5))
    P = 64                                      # odd (point) reflection keeps the slope at the frame border
    gp = cv2.copyMakeBorder(g, P, P, P, P, cv2.BORDER_REFLECT_101)
    gp[:P] = 2 * gp[P:P + 1] - gp[:P]; gp[-P:] = 2 * gp[-P - 1:-P] - gp[-P:]
    gp[:, :P] = 2 * gp[:, P:P + 1] - gp[:, :P]; gp[:, -P:] = 2 * gp[:, -P - 1:-P] - gp[:, -P:]
    gh = cv2.pyrDown(np.clip(gp, 0, 254))
    h, w = SRC_H // 2, SRC_W // 2; Q = P // 2
    gs = smooth_depth(gh)[Q:Q + h, Q:Q + w]
    fx, fy, cx, cy = fx / 2, fy / 2, cx / 2, cy / 2
    inv = gs >= G_FAR
    dd = np.where(inv, np.float32(1000.0), g_to_m(gs)).astype(np.float32)
    ds = cv2.GaussianBlur(dd, (3, 3), 0)
    u = ((np.arange(w, dtype=np.float32) + 0.5 - cx) / fx)[None, :]
    v = ((np.arange(h, dtype=np.float32) + 0.5 - cy) / fy)[:, None]
    X, Y, Z = ds, ds * u, -ds * v
    kx = np.array([[-1, 0, 1]], np.float32); ky = kx.T
    f2 = lambda a, k: cv2.filter2D(a, cv2.CV_32F, k, borderType=cv2.BORDER_REPLICATE)
    uX, uY, uZ = f2(X, kx), f2(Y, kx), f2(Z, kx)
    vX, vY, vZ = f2(X, ky), f2(Y, ky), f2(Z, ky)
    nx = vY * uZ - vZ * uY; ny = vZ * uX - vX * uZ; nz = vX * uY - vY * uX
    r = cv2.magnitude(cv2.magnitude(nx, ny), nz) + 1e-8
    sg = np.where(nx * X + ny * Y + nz * Z > 0, -1.0, 1.0).astype(np.float32) / r
    nx *= sg; ny *= sg; nz *= sg
    shade = (0.30 + 0.55 * np.clip(nx * L1[0] + ny * L1[1] + nz * L1[2], 0, 1)
             + 0.18 * np.clip(nx * L2[0] + ny * L2[1] + nz * L2[2], 0, 1) + 0.12 * np.clip(nz, 0, 1))
    ld = np.log(ds + 1e-3)                      # per half-res pixel: twice the source-pixel step, so x1.0
    edge = np.clip((np.abs(cv2.Sobel(ld, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(ld, cv2.CV_32F, 0, 1, ksize=3))) * 1.0 - 0.08, 0, 0.35)
    gg = np.clip(shade - edge, 0, 1) * 215
    fog = np.clip((dd - 150) / 600, 0, 1)
    gg = gg * (1 - fog) + 205 * fog
    gg[inv] = 225
    gg = cv2.resize(gg.astype(np.uint8)[:, CROP_X // 2:w - CROP_X // 2], (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(gg, cv2.COLOR_GRAY2RGB)


# ---------------------------------------------------------------- skeleton
def entity_rows(s, key):
    """(pos (N,3), quat (N,4) or None) of npz entity key from state.csv (n<i> -> npc.list.<i>, m<i> -> monster.list.<i>)"""
    kind, idx = key[0], key[1:].split("_", 1)[0]
    base = {"n": "npc.list", "m": "monster.list"}.get(kind)
    if base is None: return None, None
    pos = np.full((s.N, 3), np.nan); q = np.full((s.N, 4), np.nan)
    for i in range(s.N):
        p, r = s.value(i, f"{base}.{idx}.pos"), s.value(i, f"{base}.{idx}.rot")
        if isinstance(p, tuple) and len(p) == 3: pos[i] = p
        if isinstance(r, tuple) and len(r) == 4: q[i] = r
    return pos, q


def joint_sides(s, key):
    """'l' / 'r' / 'c' per joint: median lateral offset of the joint from the entity's position along its right
    axis (forward = local +z, verified against the walking direction; right = forward x up, world y-up)."""
    X = s.skel[key]; J = X.shape[1]
    if key[0] == "w": return ["c"] * J
    pos, q = entity_rows(s, key)
    ok = np.isfinite(q).all(1) & (np.linalg.norm(np.nan_to_num(q), axis=1) > 0.5) & np.isfinite(pos).all(1) & np.isfinite(X).all((1, 2))
    if ok.sum() < 5: return ["c"] * J
    R = Rotation.from_quat(q[ok]).as_matrix()
    fwd = R @ np.array([0, 0, 1.0])
    right = np.cross(fwd, np.array([0, 1.0, 0]))
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-9)
    lat = np.median(np.einsum("njk,nk->nj", X[ok] - pos[ok][:, None, :], right), 0)
    thr = 0.08 * max(np.percentile(np.abs(lat), 95), 1e-3)
    return ["r" if l > thr else "l" if l < -thr else "c" for l in lat]


def draw_entity(img, p_cv, K4, edges, sides, kind, dots):
    """p_cv (J,3) OpenCV camera coords (source); bones clipped at the near plane like the dataset's own renderer."""
    fx, fy, cx, cy = K4
    NEAR = 0.05
    zc = np.nanmedian(p_cv[:, 2]) if np.isfinite(p_cv[:, 2]).any() else 99
    th = 3 if (kind == "m" or zc < 15) else 2
    cols = ENT_COL[kind]
    for a, b in edges:
        if a >= len(p_cv) or b >= len(p_cv): continue
        pa, pb = p_cv[a].copy(), p_cv[b].copy()
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()): continue
        if pa[2] <= NEAR and pb[2] <= NEAR: continue
        if pa[2] <= NEAR: pa += (pb - pa) * (NEAR - pa[2]) / (pb[2] - pa[2])
        elif pb[2] <= NEAR: pb += (pa - pb) * (NEAR - pb[2]) / (pa[2] - pb[2])
        ua = to_out(fx * pa[0] / pa[2] + cx, fy * pa[1] / pa[2] + cy)
        ub = to_out(fx * pb[0] / pb[2] + cx, fy * pb[1] / pb[2] + cy)
        if max(abs(ua[0]), abs(ua[1]), abs(ub[0]), abs(ub[1])) > 1e5: continue
        side = sides[a] if sides[a] == sides[b] else "c"
        cv2.line(img, (int(round(ua[0])), int(round(ua[1]))), (int(round(ub[0])), int(round(ub[1]))), cols[side], th, cv2.LINE_AA)
    if dots:
        for k in range(len(p_cv)):
            p = p_cv[k]
            if np.isfinite(p).all() and p[2] > NEAR:
                uo, vo = to_out(fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy)
                if 0 <= uo < W and 0 <= vo < H:
                    cv2.circle(img, (int(round(uo)), int(round(vo))), 2 if zc < 15 else 1, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def skeleton_on_depth(s, i, g):
    """per kind: (n in-image joints, fraction not floating).  A joint floats when the nearest surface in a 9x9 window
    around its pixel is farther than z*(1+rel)+abs: there is nothing at that pixel near the joint, so the skeleton
    does not project onto what the frame shows (occluded joints have a nearer surface and pass)."""
    gmin = cv2.erode(g, np.ones((9, 9), np.uint8))
    out = {}
    for key, X in s.skel.items():
        kind = key[0]
        if kind == "w": continue
        u, v, z = s.project(i, X[i])
        ok = np.isfinite(u) & np.isfinite(v) & (z > 0.5) & (u >= 0) & (u < SRC_W) & (v >= 0) & (v < SRC_H)
        if not ok.any(): continue
        gm = gmin[v[ok].astype(int), u[ok].astype(int)]
        dm = np.where(gm >= G_FAR, 1e4, g_to_m(gm))
        tol = z[ok] * SKEL_TOL_REL + SKEL_TOL_ABS
        good = dm <= z[ok] + tol
        at = np.abs(dm - z[ok]) <= tol                   # on the visible surface (not occluded): information only
        n0, f0, a0 = out.get(kind, (0, 0.0, 0.0))
        out[kind] = (n0 + int(ok.sum()), f0 + float(good.sum()), a0 + float(at.sum()))
    return {k: (n, f / n, a / n) for k, (n, f, a) in out.items()}


# ---------------------------------------------------------------- windows
def plan_windows(N):
    K = N // SRC_WIN
    off = (N - K * SRC_WIN) // 2
    j = np.arange(FPS * WIN_S)
    rel = np.floor(j * (SRC_FPS / FPS) + 0.5).astype(int)
    return [(k, off + k * SRC_WIN, off + k * SRC_WIN + rel) for k in range(K)]


def frozen_frames(s):
    """bool (N,): frame belongs to a run of >= MAX_FROZEN_RUN frames with a bit-identical camera pose"""
    same = np.zeros(s.N, bool)
    raw = [(r[s.col["camera.pos"]], r[s.col["camera.rot"]]) for r in s.rows]
    same[1:] = [raw[i] == raw[i - 1] for i in range(1, s.N)]
    out = np.zeros(s.N, bool); i = 0
    while i < s.N:
        if same[i]:
            j = i
            while j < s.N and same[j]: j += 1
            if j - i + 1 >= MAX_FROZEN_RUN: out[i - 1:j] = True
            i = j
        else: i += 1
    return out


def state_frame(s, i):
    out = {}
    for c, k in s.col.items():
        if c in ("camera.K", "camera.pos", "camera.rot"): continue
        v = s.rows[i][k] if k < len(s.rows[i]) else ""
        x = _cell(v)
        if isinstance(x, tuple): x = [None if (isinstance(t, float) and t != t) else t for t in x]
        elif isinstance(x, float) and x != x: x = None
        elif isinstance(x, float) and x.is_integer() and abs(x) < 2 ** 31: x = int(x)
        out[c] = x
    return out


class Frames:
    """lock-step sequential decode of rgb.mp4 and depth.mkv (raw Y plane of the depth)"""
    def __init__(self, d):
        self.c1 = av.open(f"{d}/rgb.mp4"); self.c2 = av.open(f"{d}/depth.mkv")
        for c in (self.c1, self.c2):
            st = c.streams.video[0]; st.thread_type = "AUTO"; st.codec_context.thread_count = 4
        self.g1 = self.c1.decode(video=0); self.g2 = self.c2.decode(video=0)
        self.i = -1; self.cur = None

    def get(self, i):
        while self.i < i:
            f1 = next(self.g1, None); f2 = next(self.g2, None)
            if f1 is None: raise EOFError(f"rgb.mp4 ended at frame {self.i + 1}")
            if f2 is None: raise EOFError(f"depth.mkv ended at frame {self.i + 1}")
            self.i += 1
            if self.i == i:
                rgb = f1.to_ndarray(format="rgb24")
                y = np.frombuffer(f2.planes[0], np.uint8).reshape(f2.planes[0].height, f2.planes[0].line_size)[:SRC_H, :SRC_W]
                if rgb.shape[:2] != (SRC_H, SRC_W) or y.shape != (SRC_H, SRC_W):
                    raise RuntimeError(f"unexpected frame size rgb {rgb.shape} depth {y.shape}")
                self.cur = (rgb, depth_g(y))
        return self.cur

    def close(self):
        self.c1.close(); self.c2.close()


def convert(s, fr, sample, cid, out_dir, win, ctx):
    t_start = time.time()
    k_win, s0, idx = win
    reasons = []
    frozen, jumps, sides, edges = ctx["frozen"], ctx["jumps"], ctx["sides"], ctx["edges"]
    cam_bad = int((~s.cam_ok[s0:s0 + SRC_WIN]).sum())
    if cam_bad: reasons.append(f"camera_missing({cam_bad}_src_frames)")
    nfro = int(frozen[s0:s0 + SRC_WIN].sum())
    if nfro: reasons.append(f"camera_frozen({nfro}_src_frames)")
    njump = int(jumps[s0:s0 + SRC_WIN - 1].sum())
    if njump: reasons.append(f"camera_cut({njump})")
    wr = {st: ffmpeg_writer(f"{out_dir}/{cid}_{st}.mp4") for st in STREAMS}
    lum, lstd, flat, farf, frames_json, sk, sk_at = [], [], [], [], [], {"n": [], "m": []}, {"n": [], "m": []}
    dmed = []
    still_j = set(json.loads(os.environ.get("WW_STILL_FRAMES", "[]")))
    stills = {}
    for j, i in enumerate(idx.tolist()):
        rgb, g = fr.get(i)
        K4 = s.K[i] if s.cam_ok[i] else np.array([937.832, 937.832, 640.0, 360.0])
        render = fit(rgb, cv2.INTER_LINEAR)
        wb = white_box(g, *K4)
        sk_img = np.zeros((H, W, 3), np.uint8)
        if s.cam_ok[i]:
            for key, X in s.skel.items():
                draw_entity(sk_img, s.to_cam(i, X[i]), K4, edges[key], sides[key], key[0], dots=key[0] == "n")
            for kind, (n, f, a) in skeleton_on_depth(s, i, g).items():
                if n >= 5: sk[kind].append(f); sk_at[kind].append(a)
        for st, a in zip(STREAMS, (render, wb, sk_img)): wr[st].stdin.write(np.ascontiguousarray(a).tobytes())
        if j in still_j: stills[j] = dict(render=render, white_box=wb, skeleton=sk_img, src=i, g=g)
        gr = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
        lum.append(float(gr.mean())); lstd.append(float(gr.std()))
        flat.append(float(g.std())); farf.append(float((g >= G_FAR).mean())); dmed.append(float(g_to_m(np.median(g[::8, ::8]))))
        frames_json.append(dict(frame=j, src_frame=i, src_t_s=round(i / SRC_FPS, 4),
                                c2w=np.round(s.c2w_cv(i), 6).tolist() if s.cam_ok[i] else None,
                                K=K_out(s.K[i]) if s.cam_ok[i] else None,
                                state=state_frame(s, i),
                                joints3d={k: [[None if not np.isfinite(c) else round(float(c), 3) for c in p] for p in X[i]]
                                          for k, X in s.skel.items()}))
    for st in STREAMS:
        wr[st].stdin.close()
        if wr[st].wait() != 0: raise RuntimeError(f"ffmpeg {st} failed")
    black = [j for j, (a, b) in enumerate(zip(lum, lstd)) if a < 5 or (a < 12 and b < 3)]
    if black: reasons.append(f"black_frames({len(black)})")
    # a depth frame with no structure: the camera is inside / against geometry (flat and nearer than 1 m: the frame
    # shows a wall of rock or monster) or the depth is broken (flat farther away, or > 90 % far / sky)
    occl = [j for j, (a, m) in enumerate(zip(flat, dmed)) if a < 1.0 and m < 1.0]
    bad_depth = [j for j, (a, b, m) in enumerate(zip(flat, farf, dmed)) if (a < 1.0 and m >= 1.0) or b > 0.9]
    if len(occl) > MAX_FLAT_DEPTH_FRAMES: reasons.append(f"camera_occluded({len(occl)}_frames)")
    if len(bad_depth) > MAX_FLAT_DEPTH_FRAMES: reasons.append(f"depth_invalid({len(bad_depth)}_frames)")
    skc = {}
    for kind, v in sk.items():
        if len(v) >= MIN_MEASURED:
            med = float(np.median(v)); badf = float(np.mean(np.array(v) < 0.5))
            skc[kind] = dict(frames=len(v), on_surface_median=round(med, 3), on_surface_p10=round(float(np.percentile(v, 10)), 3),
                             frames_below_0_5=round(badf, 3), at_surface_median=round(float(np.median(sk_at[kind])), 3))
            name = {"n": "player", "m": "monster"}[kind]
            if med < MIN_SKEL_ON: reasons.append(f"skeleton_off_depth({name},median={med:.2f})")
            elif badf > MAX_BAD_SKEL_FRAMES: reasons.append(f"skeleton_off_depth({name},bad_frames={badf:.2f})")
        else:
            skc[kind] = dict(frames=len(v))
    cp = s.cam_pos[idx]
    meta = dict(
        clip_id=cid, generator=GEN, license=LICENSE,
        source=dict(dataset="wildworld", sample_id=sample.split("/")[-1], part=sample.split("/")[0], location=f"{SRC}/{sample}",
                    src_frames_total=s.N, src_fps=SRC_FPS, src_resolution=[SRC_W, SRC_H],
                    src_duration_s=round(s.N / SRC_FPS, 3), state_columns=s.header),
        width=W, height=H, fps=FPS, frames=len(idx), duration_s=round(len(idx) / FPS, 4),
        window=dict(index=k_win, src_start=int(s0), src_end=int(s0 + SRC_WIN), start_s=round(s0 / SRC_FPS, 4),
                    end_s=round((s0 + SRC_WIN) / SRC_FPS, 4), note="non-overlapping 8 s (240 source frame) windows, the group centred in the sample"),
        resample="output frame j shows source frame src_start + floor(1.25 j + 0.5) (frames[j].src_frame): 30 -> 24 fps, "
                 "one source frame in five is skipped",
        src_frame_indices=idx.tolist(),
        crop="source 1280x720: 10 px cropped left and right (1260x720), scaled by 16/15 to 1344x768",
        camera=dict(convention="c2w: OpenCV camera (x right, y down, z forward) to WildWorld world (y up, metres). K: output "
                               "pixels (1344x768), u = fx*x/z + cx. Source: state.csv camera.K (fx, fy, cx, cy at 1280x720), "
                               "camera.pos, camera.rot (camera-to-world quaternion xyzw, OpenGL camera frame).",
                    fx_out_range=[round(float(np.nanmin(s.K[idx, 0])) * SC, 2), round(float(np.nanmax(s.K[idx, 0])) * SC, 2)]),
        state_note="frames[j].state: every state.csv column of the shown source frame except the camera (player npc.list.<i>.* "
                   "and monster monster.list.<i>.*: type ids, pos, rot quaternion xyzw, motion_bank_id / motion_id / motion_frame "
                   "= the animation (action) being played, hp / sp / attack / defence stats, monster cond); tuples are lists, nan = null. "
                   "Column meanings: the WildWorld README.",
        skeleton=dict(entities={k: dict(kind={"m": "monster", "n": "player", "w": "weapon"}[k[0]], joints=int(X.shape[1]),
                                         edges_key=Sample.EDGE_KEY_OF(k), bones=len(edges[k]), sides="".join(sides[k]))
                                for k, X in s.skel.items()},
                      joints_note="frames[j].joints3d[<key>][k] = world xyz (m) of joint k from skeleton.npz (null = entity absent); "
                                  "bones: skeleton_edges.json of the dataset",
                      colours=ENT_COL_NOTE, drawing="all bones, no occlusion test, near-plane clipped"),
        white_box_note="clay shading computed from engine depth (8-bit, dithered; de-quantised by multi-scale smoothing that stops at depth steps; normals from depth); "
                       "far / sky (z > ~640 m) = flat 225 grey",
        depth_decode=dict(formula="g = (Y - 16) * 255 / 219; z_m = C * g / (B - g)", B=DEPTH_B, C=DEPTH_C,
                          note="planar depth, fitted on projected engine joints (5-40 m, rms 0.7 code)"),
        checks=dict(camera_missing_src_frames=cam_bad, camera_frozen_src_frames=nfro, camera_cuts=njump,
                    camera_path_m=round(float(np.nansum(np.linalg.norm(np.diff(cp, axis=0), axis=1))), 2),
                    luma_mean_min=round(min(lum), 1), luma_mean_median=round(float(np.median(lum)), 1), black_frames=black[:20],
                    depth_std_code_min=round(min(flat), 2), depth_far_frac_max=round(max(farf), 4), depth_invalid_frames=bad_depth[:20], camera_occluded_frames=occl[:20],
                    skeleton_on_depth=skc, passed=not reasons, fail_reasons=reasons),
        streams=list(STREAMS), timing_s=round(time.time() - t_start, 1),
        frames_meta=frames_json)
    return meta, reasons, stills


Sample.EDGE_KEY_OF = staticmethod(lambda k: {"m": "em_{t}", "n": "npc", "w": "wp_{t}"}[k[0]].format(t=k.split("_", 1)[1]))


def context(s):
    frozen = frozen_frames(s)
    step = np.linalg.norm(np.diff(s.cam_pos, axis=0), axis=1)
    jumps = np.nan_to_num(step, nan=0.0) > MAX_CAM_JUMP_M
    sides = {k: joint_sides(s, k) for k in s.skel}
    edges = {k: s.edges(k) for k in s.skel}
    return dict(frozen=frozen, jumps=jumps, sides=sides, edges=edges)


if __name__ == "__main__":
    NAME = sys.argv[1].strip("/")                     # data_part1/<sample_id>
    sid = NAME.split("/")[-1]
    base = f"ww_{sid}"
    if os.environ.get("WW_FORCE") != "1" and exists_remote(f"{SAMPLES}/{sid}.json"):
        print(f"{NAME} already done"); sys.exit(3)
    D = f"{WORK}/c/{sid}"
    shutil.rmtree(D, ignore_errors=True); os.makedirs(D)
    try:
        t = time.time()
        SD = os.environ.get("WW_SRC_DIR") or f"{D}/src"        # tests: an already downloaded sample dir
        if not os.environ.get("WW_SRC_DIR"): store("--numworkers", "4", "cp", f"{SRC}/{NAME}/*", f"{SD}/")
        miss = [f for f in ("rgb.mp4", "depth.mkv", "state.csv", "skeleton.npz") if not os.path.exists(f"{SD}/{f}")]
        if miss: raise RuntimeError(f"incomplete sample, missing {miss}")
        t_dl = time.time() - t
        s = Sample(SD)
        bad = [k for k, X in s.skel.items() if X.shape[0] != s.N]
        if bad: raise RuntimeError(f"skeleton rows != state rows for {bad}")
        wins = plan_windows(s.N)
        nfr = {}
        for f in ("rgb.mp4", "depth.mkv"):
            c = av.open(f"{SD}/{f}"); st = c.streams.video[0]
            nfr[f] = st.frames or int(round(float(c.duration or 0) / 1e6 * SRC_FPS)); c.close()
        n_ok = min(nfr.values())
        done = {}
        for pre, kind in ((DST, "ok"), (REJ, "rejected")):
            r = subprocess.run([STORE, "ls", f"{pre}/{base}_*.json"], capture_output=True, text=True)
            for l in r.stdout.splitlines():
                n = l.split()[-1][:-5]
                if n.rsplit("_", 1)[0] == base: done[n] = kind
        summ = dict(sample=NAME, src_frames=s.N, stream_frames=nfr, duration_s=round(s.N / SRC_FPS, 3), windows={}, generator=GEN)
        MAXW = int(os.environ.get("WW_MAX_WIN", "0"))
        todo = wins[:MAXW] if MAXW else wins
        ctx = context(s) if todo else None
        fr = Frames(SD)
        nup = 0
        for win in todo:
            cid = f"{base}_{win[0]}"
            if cid in done and os.environ.get("WW_FORCE") != "1":
                summ["windows"][cid] = done[cid]; continue
            if win[1] + SRC_WIN > n_ok:                     # stream shorter than state.csv: no frames for this window
                reasons = [f"stream_short(rgb={nfr['rgb.mp4']},depth={nfr['depth.mkv']},state={s.N})"]
                meta = dict(clip_id=cid, generator=GEN, license=LICENSE,
                            source=dict(dataset="wildworld", sample_id=sid, part=NAME.split("/")[0], location=f"{SRC}/{NAME}", src_frames_total=s.N),
                            window=dict(index=win[0], src_start=int(win[1]), src_end=int(win[1] + SRC_WIN)),
                            checks=dict(passed=False, fail_reasons=reasons, skeleton_on_depth={}), timing_s=0)
            else:
                meta, reasons, _ = convert(s, fr, NAME, cid, D, win, ctx)
            meta["timing_s"] = dict(download=round(t_dl, 1), convert=meta["timing_s"])
            json.dump(meta, open(f"{D}/{cid}.json", "w"), separators=(",", ":"))
            c = meta["checks"]
            if os.environ.get("WW_NO_UPLOAD") == "1": pass
            elif not reasons:
                p = f"{D}/_s5.txt"
                open(p, "w").write("\n".join(f"cp {D}/{cid}_{st}.mp4 {DST}/{cid}_{st}.mp4" for st in STREAMS) + "\n")
                r = subprocess.run([STORE, "--numworkers", "4", "run", p], capture_output=True, text=True)
                if r.returncode != 0: raise RuntimeError(f"upload failed: {r.stderr[-400:]}")
                store("cp", f"{D}/{cid}.json", f"{DST}/{cid}.json")         # json last: marks the clip complete
                nup += 1
            else:
                store("cp", f"{D}/{cid}.json", f"{REJ}/{cid}.json")
            summ["windows"][cid] = "ok" if not reasons else "rejected: " + " ".join(reasons)
            log(cid, "done", "OK" if not reasons else "REJECT " + " ".join(reasons),
                f"skel={json.dumps(c['skeleton_on_depth'])} conv={meta['timing_s']['convert']}s")
            if os.environ.get("WW_KEEP") != "1":
                for st in STREAMS:
                    if os.path.exists(f"{D}/{cid}_{st}.mp4"): os.remove(f"{D}/{cid}_{st}.mp4")
        fr.close()
        if not wins: summ["note"] = f"shorter than 8 s ({s.N} frames): no window"
        if os.environ.get("WW_NO_UPLOAD") != "1" and not (MAXW and MAXW < len(wins)):
            json.dump(summ, open(f"{D}/_sample.json", "w"), indent=1)
            store("cp", f"{D}/_sample.json", f"{SAMPLES}/{sid}.json")
        log(f"sample done: {len(wins)} windows, {nup} uploaded now, dl {t_dl:.0f}s")
        sys.exit(0 if nup else 3)
    finally:
        if os.environ.get("WW_KEEP") != "1": shutil.rmtree(D, ignore_errors=True)
