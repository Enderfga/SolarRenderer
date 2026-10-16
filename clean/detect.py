"""Black-bar and overlay (watermark / logo / text / HUD) evidence for one clip, CPU only.

usage: detect.py JOBS.jsonl SHARD NSHARDS NAME            (driver: batches, prefetch, pool)
env:   SOLAR_CLEAN_WORK (required: storage prefix for the outputs), STORAGE_CLI (default "solar-storage"),
       CLEAN_TMP (local scratch, default <system temp>/solar_clean), WORKERS (pool size, 64), PULL (download workers, 16)
Output: $CLEAN_TMP/det/NAME.s<SHARD>.jsonl (one row per clip), uploaded at the end to $SOLAR_CLEAN_WORK/det/;
        frames to $SOLAR_CLEAN_WORK/frames/<source>/<id with / -> __>_{a,b,c}.jpg and _static.png after every batch.
JOBS rows: {"id", "source", "target": url, "streams": {name: url} (aligned observations, may be empty)}

Per clip, on 24 evenly spaced frames of the target:
  bars    - rows/columns from each edge that are dark (<=BAR_Y) in every sampled frame, accepted only when the
            content boundary is a straight line (the first bright row of most columns sits on it) - a dark night
            sky has a ragged lower edge and is not a bar.
  static  - pixels whose gradient is strong in >=80% of frames while their brightness hardly changes: an overlay
            rides on the screen while the picture moves under it. Connected components -> candidate boxes.
            When the picture itself barely moves (static camera), this map is flagged unreliable.
  ocr     - RapidOCR (PP-OCRv4) on 6 frames; text boxes that recur at the same place in >=2 frames.
  subject - union / percentile box of the aligned mask stream, when the job lists one, in target pixels.
Three frames (15/50/85 %) are written as JPEG for the VLM pass and the review page.
All boxes are [x0, y0, x1, y1] in the target clip's pixels.
"""
import sys, os, json, subprocess, shutil, threading, queue, time, traceback, tempfile
import numpy as np, cv2

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
BUCKET = os.environ.get("SOLAR_CLEAN_WORK", "").rstrip("/")
TMP = os.path.join(os.environ.get("CLEAN_TMP", os.path.join(tempfile.gettempdir(), "solar_clean")), "det")
NF = 24          # frames for bars / static map
AW = 640         # analysis width for the static map
BAR_Y = 26       # a bar pixel is at most this bright (0-255) in every sampled frame
BATCH = 256

def safe(i): return i.replace("/", "__")

def probe(f):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=width,height,nb_read_packets,r_frame_rate", "-of", "json", f],
                       capture_output=True, text=True).stdout
    s = json.loads(o)["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), int(s.get("nb_read_packets") or 0), float(n) / float(d)

def frames_at(f, W, H, idx, scale=None):
    sel = "+".join(f"eq(n\\,{int(i)})" for i in idx)
    vf = f"select={sel}" + (f",scale={scale[0]}:{scale[1]}:flags=area" if scale else "")
    w, h = scale if scale else (W, H)
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-vf", vf, "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE).stdout
    n = len(b) // (w * h * 3)
    return np.frombuffer(b[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)

def all_frames_gray(f, w, h):
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       stdout=subprocess.PIPE).stdout
    n = len(b) // (w * h)
    return np.frombuffer(b[: n * w * h], np.uint8).reshape(n, h, w)

# ---------------------------------------------------------------- bars
def edge_bar(Y, axis_len, from_end, vertical):
    """Y: T,H,W gray. Thickness of a dark band at one edge, plus straightness of its inner boundary."""
    # per-line "ever bright" fraction: max over frames, fraction of pixels above BAR_Y
    mx = Y.max(0)                                    # H,W   brightest each pixel ever gets
    if not vertical:
        mx = mx.T                                    # treat columns as rows
    L = mx.shape[0]
    frac = (mx > BAR_Y).mean(1)                      # per line
    order = range(L - 1, -1, -1) if from_end else range(L)
    t = 0
    for i in order:
        if frac[i] <= 0.002:
            t += 1
        else:
            break
    if t == 0 or t >= L * 0.45:
        return 0, 0.0, 0.0
    # straightness: per column (of this orientation) the first line that is ever bright, counted from the edge
    m = mx[::-1] if from_end else mx
    br = m > BAR_Y
    first = np.where(br.any(0), br.argmax(0), L)
    straight = float(((first >= t) & (first <= t + 2)).mean())
    # contrast across the boundary: the brightest the first content lines get vs the bar
    inner = m[t:t + 3].astype(np.float32)
    contrast = float(np.percentile(inner, 75))
    return t, straight, contrast

def bars(Y):
    T, H, W = Y.shape
    out = {}
    for name, fe, vert in (("top", False, True), ("bottom", True, True), ("left", False, False), ("right", True, False)):
        t, s, c = edge_bar(Y, H if vert else W, fe, vert)
        out[name] = {"px": int(t), "straight": round(s, 3), "contrast": round(c, 1)}
    # accept: thick enough, and straight (or a symmetric pair that is at least fairly straight)
    acc = {}
    for a, b, L in (("top", "bottom", H), ("left", "right", W)):
        for k, o in ((a, b), (b, a)):
            t, s = out[k]["px"], out[k]["straight"]
            sym = abs(out[k]["px"] - out[o]["px"]) <= max(4, 0.006 * L) and out[o]["px"] > 0
            ok = t >= max(4, 0.004 * L) and (s >= 0.5 or (sym and s >= 0.2))
            acc[k] = int(t) if ok else 0
            out[k]["accepted"] = bool(ok)
    return out, acc

# ---------------------------------------------------------------- static overlay map
def static_map(F, content, sx, sy):
    """F: T,h,w,3 uint8 at analysis res. content: [x0,y0,x1,y1] in analysis px. Returns components + stats."""
    L = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in F]).astype(np.float32)
    keep = L.reshape(len(L), -1).mean(1) > 8           # skip black (fade) frames
    if keep.sum() >= 6:
        L = L[keep]
    T, h, w = L.shape
    x0, y0, x1, y1 = content
    G = np.stack([cv2.magnitude(cv2.Sobel(cv2.GaussianBlur(l, (3, 3), 0), cv2.CV_32F, 1, 0),
                                cv2.Sobel(cv2.GaussianBlur(l, (3, 3), 0), cv2.CV_32F, 0, 1)) for l in L])
    persist = (G > 40).mean(0)
    sd = L.std(0)
    cm = np.zeros((h, w), bool); cm[y0 + 3:y1 - 3, x0 + 3:x1 - 3] = True
    stat = (persist >= 0.8) & (sd < 7) & cm
    carea = max(1, cm.sum())
    still_frac = float(((sd < 3) & cm).sum() / carea)            # how much of the picture never changes
    motion = float(np.abs(np.diff(L, axis=0)).mean()) if T > 1 else 0.0
    stat_frac = float(stat.sum() / carea)
    comps = []
    m = cv2.dilate(stat.astype(np.uint8), np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    for k in range(1, n):
        x, y, ww, hh, a = st[k]
        core = int((stat & (lab == k)).sum())
        if core < 15:
            continue
        comps.append({"box": [round(x / sx), round(y / sy), round((x + ww) / sx), round((y + hh) / sy)],
                      "px": core, "fill": round(core / float(ww * hh), 3)})
    comps.sort(key=lambda c: -c["px"])
    return comps[:20], {"still_frac": round(still_frac, 4), "motion": round(motion, 2), "static_edge_frac": round(stat_frac, 5)}, stat

# ---------------------------------------------------------------- OCR
_ocr = None
def ocr_engine():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
    return _ocr

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0

def ocr_boxes(frames):
    eng = ocr_engine(); per = []
    for f in frames:
        res, _ = eng(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        bx = []
        for pts, txt, conf in (res or []):
            p = np.array(pts); bx.append({"box": [int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())],
                                          "text": txt, "conf": round(float(conf), 3)})
        per.append(bx)
    # cluster across frames
    clusters = []
    for fi, bx in enumerate(per):
        for b in bx:
            for c in clusters:
                if iou(c["box"], b["box"]) >= 0.4:
                    c["frames"].add(fi); c["texts"].append(b["text"]); c["confs"].append(b["conf"])
                    c["box"] = [min(c["box"][0], b["box"][0]), min(c["box"][1], b["box"][1]), max(c["box"][2], b["box"][2]), max(c["box"][3], b["box"][3])]
                    break
            else:
                clusters.append({"box": list(b["box"]), "frames": {fi}, "texts": [b["text"]], "confs": [b["conf"]]})
    out = []
    for c in clusters:
        out.append({"box": c["box"], "n": len(c["frames"]), "text": max(set(c["texts"]), key=c["texts"].count),
                    "conf": round(float(np.mean(c["confs"])), 3)})
    out.sort(key=lambda c: (-c["n"], -c["conf"]))
    return out[:20], len(frames)

# ---------------------------------------------------------------- subject (mask stream)
def subject(maskf, W, H):
    mw, mh, _, _ = probe(maskf)
    s = 4
    M = all_frames_gray(maskf, mw // s, mh // s) > 127
    if M.size == 0 or not M.any():
        return None
    sx, sy = W / (mw // s), H / (mh // s)
    ys, xs = np.nonzero(M.any(0))
    union = [int(xs.min() * sx), int(ys.min() * sy), int((xs.max() + 1) * sx), int((ys.max() + 1) * sy)]
    tt, yy, xx = np.nonzero(M)
    p = [int(np.percentile(xx, 0.5) * sx), int(np.percentile(yy, 0.5) * sy), int((np.percentile(xx, 99.5) + 1) * sx), int((np.percentile(yy, 99.5) + 1) * sy)]
    per = []
    for t in np.linspace(0, len(M) - 1, 12).round().astype(int):
        y_, x_ = np.nonzero(M[t])
        per.append([int(x_.min() * sx), int(y_.min() * sy), int((x_.max() + 1) * sx), int((y_.max() + 1) * sy)] if len(x_) else None)
    # heat at 1/16 of the target for coverage of any candidate crop later
    hm = cv2.resize(M.mean(0).astype(np.float32), (max(1, W // 16), max(1, H // 16)), interpolation=cv2.INTER_AREA)
    return {"union": union, "p99": p, "per_frame": per, "mask_size": [mw, mh], "frac_frames": round(float(M.any((1, 2)).mean()), 3),
            "heat16": np.round(hm / max(hm.sum(), 1e-9), 5).tolist()}

# ---------------------------------------------------------------- one clip
def one(job, local):
    tgt = local[job["target"]]
    W, H, N, fps = probe(tgt)
    rec = {"id": job["id"], "source": job["source"], "target": job["target"], "catalog_target": job.get("catalog_target", job["target"]), "W": W, "H": H, "frames": N, "fps": round(fps, 3)}
    # grids of the aligned observations
    grids = {}
    for k, u in job.get("streams", {}).items():
        if u in local:
            w_, h_, n_, _ = probe(local[u]); grids[k] = [w_, h_, n_]
    rec["grids"] = grids
    rec["grid_mismatch"] = sorted(k for k, g in grids.items() if g[:2] != [W, H])
    N = max(N, 2)
    idx = np.linspace(0, N - 1, NF).round().astype(int)
    F = frames_at(tgt, W, H, idx)
    if len(F) < 4:
        rec["error"] = f"decoded {len(F)} frames"; return rec, None
    Y = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in F])
    raw, acc = bars(Y)
    rec["bars_raw"] = raw; rec["bars"] = acc
    content = [acc["left"], acc["top"], W - acc["right"], H - acc["bottom"]]
    rec["content"] = content
    ah = max(2, int(round(H * AW / W))) if W >= H else 640
    aw = AW if W >= H else max(2, int(round(W * 640 / H)))
    sx, sy = aw / W, ah / H
    Fa = np.stack([cv2.resize(f, (aw, ah), interpolation=cv2.INTER_AREA) for f in F])
    ca = [int(content[0] * sx), int(content[1] * sy), int(np.ceil(content[2] * sx)), int(np.ceil(content[3] * sy))]
    comps, stats, smap = static_map(Fa, ca, sx, sy)
    rec["static"] = comps; rec["motion"] = stats
    # OCR on 6 frames
    oi = [int(round(i)) for i in np.linspace(2, NF - 3, 6)]
    rec["ocr"], _ = ocr_boxes([F[i] for i in oi])
    # subject
    mk = job.get("streams", {}).get("mask")
    if mk and mk in local and "mask" not in rec["grid_mismatch"]:
        try:
            rec["subject"] = subject(local[mk], W, H)
        except Exception as e:
            rec["subject_error"] = str(e)[:200]
    # frames for VLM / review
    jp = {}
    for tag, q in (("a", 0.15), ("b", 0.5), ("c", 0.85)):
        f = F[int(round(q * (len(F) - 1)))]
        sc = min(1.0, 1280.0 / max(W, H))
        im = cv2.resize(f, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else f
        jp[tag] = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    sm = cv2.imencode(".png", (smap * 255).astype(np.uint8))[1].tobytes()
    rec["jpg_scale"] = round(min(1.0, 1280.0 / max(W, H)), 5)
    return rec, (jp, sm)

def work(args):
    job, local, outdir = args
    try:
        rec, imgs = one(job, local)
        if imgs:
            jp, sm = imgs
            for tag, b in jp.items():
                open(f"{outdir}/frames/{safe(job['id'])}_{tag}.jpg", "wb").write(b)
            open(f"{outdir}/frames/{safe(job['id'])}_static.png", "wb").write(sm)
        return rec
    except Exception as e:
        return {"id": job["id"], "source": job["source"], "error": traceback.format_exc()[-600:]}

def fetch(batch, d):
    os.makedirs(d, exist_ok=True)
    local, cmds = {}, []
    for j in batch:
        urls = [j["target"]] + list(j.get("streams", {}).values())
        for u in urls:
            if u in local: continue
            p = f"{d}/{len(local)}_{os.path.basename(u)}"; local[u] = p; cmds.append(f"cp {u} {p}")
    open(f"{d}/cmds", "w").write("\n".join(cmds) + "\n")
    subprocess.run([STORE, "--numworkers", str(int(os.environ.get("PULL", 16))), "run", f"{d}/cmds"], capture_output=True)
    return {u: p for u, p in local.items() if os.path.exists(p)}

def main():
    jobs_f, shard, nsh, prefix = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    if not BUCKET:
        sys.exit("set SOLAR_CLEAN_WORK to the storage prefix for frames/ and det/")
    import multiprocessing as mp
    jobs = [json.loads(l) for l in open(jobs_f)][shard::nsh]
    outf = f"{TMP}/{prefix}.s{shard}.jsonl"
    os.makedirs(TMP, exist_ok=True)
    done = set()
    if os.path.exists(outf):
        done = {json.loads(l)["id"] for l in open(outf)}
    jobs = [j for j in jobs if j["id"] not in done]
    batches = [jobs[i:i + BATCH] for i in range(0, len(jobs), BATCH)]
    q = queue.Queue(maxsize=1)
    def pre():
        for k, b in enumerate(batches):
            d = f"{TMP}/dl_{prefix}_{shard}_{k}"
            q.put((b, d, fetch(b, d)))
        q.put(None)
    threading.Thread(target=pre, daemon=True).start()
    fo = open(outf, "a"); n = 0; t0 = time.time()
    with mp.get_context("spawn").Pool(int(os.environ.get("WORKERS", 64))) as pool:
        while True:
            it = q.get()
            if it is None: break
            b, d, local = it
            od = f"{TMP}/out_{prefix}_{shard}"; os.makedirs(f"{od}/frames", exist_ok=True)
            todo = [(j, local, od) for j in b if j["target"] in local]
            for j in b:
                if j["target"] not in local:
                    fo.write(json.dumps({"id": j["id"], "source": j["source"], "error": "download failed"}) + "\n")
            for rec in pool.imap_unordered(work, todo, chunksize=1):
                fo.write(json.dumps(rec) + "\n"); n += 1
            fo.flush()
            src = b[0]["source"]
            subprocess.run([STORE, "--numworkers", "8", "cp", f"{od}/frames/*", f"{BUCKET}/frames/{src}/"], capture_output=True)
            shutil.rmtree(od, ignore_errors=True); shutil.rmtree(d, ignore_errors=True)
            print(f"{prefix} s{shard}: {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    fo.close()
    subprocess.run([STORE, "cp", outf, f"{BUCKET}/det/{prefix}.s{shard}.jsonl"], capture_output=True)
    print("DETECT_DONE", prefix, shard, n, flush=True)

if __name__ == "__main__":
    main()
