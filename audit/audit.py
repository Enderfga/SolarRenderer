"""Per-clip audit of a SolarRenderer source: final stream vs its condition stream. CPU only.

usage: audit.py JOBS.jsonl SHARD NSHARDS OUT_PREFIX [NPROC] [NDL]
env:   STORAGE_CLI (default "solar-storage"), AUDIT_TMP (local scratch, default <system temp>/solar_audit)
JOBS rows: {"id", "source", "final": url, "cond": url, "extra": {name: url}}   (extra = small json side files)
Writes OUT_PREFIX/<source>/shard_<SHARD>.jsonl (one metrics row per clip) to storage every BATCH clips.

Per clip, both streams are decoded once to grey at analysis width (CARLA 448x256, Magpie 480x270):
  motion   e[t] = mean |f[t]-f[t-1]|; Pearson corr of final vs cond curves at lags -6..6 -> best lag, corr at 0
  edge lag on 8 anchor frames: recall of cond Canny edges within 2 px of final edges, at lags -6..6;
           baseline = same with the final mirrored left-right (a pair that is not aligned scores ~ its baseline)
  dups     share of adjacent pairs with e[t] < DUP (per stream), and how often the dups coincide
  static   median e of the final
  cuts     e[t] spikes (> max(25, 8 x median))
  exposure per-frame mean luma, share of black (<8) / flat (std<4) frames, share of pixels >= 250
  cond     share of non-black (>12) pixels per frame: mean, min
  hash     64-bit dHash of the final at 4 frames (0, 1/3, 2/3, end) for the near-duplicate pass
  probe    width/height/frames/fps of both streams (ffprobe)
"""
import sys, os, json, subprocess, shutil, threading, queue, time, traceback, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
cv2.setNumThreads(1)

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
TMP = os.environ.get("AUDIT_TMP", os.path.join(tempfile.gettempdir(), "solar_audit"))
BATCH = 200
LAGS = range(-6, 7)
DUP = 0.35  # mean abs diff (0-255 grey) below which two adjacent frames count as the same frame


def probe(f):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=width,height,nb_read_packets,r_frame_rate", "-of", "json", f], capture_output=True, text=True).stdout
    s = json.loads(o)["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return {"w": int(s["width"]), "h": int(s["height"]), "n": int(s.get("nb_read_packets") or 0), "fps": round(float(n) / float(d), 3)}


def grey(f, w, h):
    p = subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", f, "-vf", f"scale={w}:{h}:flags=area",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    b = p.stdout
    n = len(b) // (w * h)
    return np.frombuffer(b[: n * w * h], np.uint8).reshape(n, h, w), p.stderr.decode(errors="replace").strip()[:300]


def dhash(img):
    s = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def corr(a, b):
    if len(a) < 8 or a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def lagcorr(ef, ec):
    out = {}
    T = min(len(ef), len(ec))
    ef, ec = ef[:T], ec[:T]
    for L in LAGS:  # cond shifted by L: compare ef[t] with ec[t+L]
        if L >= 0:
            out[L] = corr(ef[: T - L], ec[L:])
        else:
            out[L] = corr(ef[-L:], ec[: T + L])
    return out


def edges(img):
    v = np.median(img)
    return cv2.Canny(img, int(max(10, 0.66 * v)), int(min(255, max(30, 1.33 * v))))


def analyse(job):
    t0 = time.time()
    r = {"id": job["id"], "source": job["source"]}
    d = job["_dir"]
    try:
        ff, fc = os.path.join(d, "final.mp4"), os.path.join(d, "cond.mp4")
        if not os.path.exists(ff) or os.path.getsize(ff) == 0:
            r["error"] = "final missing"; return r
        if not os.path.exists(fc) or os.path.getsize(fc) == 0:
            r["error"] = "cond missing"; return r
        pf, pc = probe(ff), probe(fc)
        r["probe_final"], r["probe_cond"] = pf, pc
        w = 480 if pf["w"] >= 1900 else 448
        h = int(round(w * pf["h"] / pf["w"] / 2) * 2)
        F, errf = grey(ff, w, h)
        C, errc = grey(fc, w, h)
        r["decoded"] = [len(F), len(C)]
        if errf: r["decode_err_final"] = errf
        if errc: r["decode_err_cond"] = errc
        T = min(len(F), len(C))
        if T < 16:
            r["error"] = "too few frames"; return r
        F, C = F[:T], C[:T]
        Ff, Cf = F.astype(np.float32), C.astype(np.float32)
        ef = np.abs(Ff[1:] - Ff[:-1]).mean((1, 2))
        ec = np.abs(Cf[1:] - Cf[:-1]).mean((1, 2))
        r["motion_final_median"] = float(np.median(ef)); r["motion_final_mean"] = float(ef.mean())
        r["motion_cond_median"] = float(np.median(ec)); r["motion_cond_mean"] = float(ec.mean())
        lc = lagcorr(ef, ec)
        r["mcorr"] = {str(k): (None if np.isnan(v) else round(v, 4)) for k, v in lc.items()}
        good = {k: v for k, v in lc.items() if not np.isnan(v)}
        if good:
            bl = max(good, key=lambda k: good[k])
            r["mcorr_best_lag"] = int(bl); r["mcorr_best"] = round(good[bl], 4); r["mcorr_0"] = None if np.isnan(lc[0]) else round(lc[0], 4)
        # dups
        df, dc = ef < DUP, ec < DUP
        r["dup_final"] = float(df.mean()); r["dup_cond"] = float(dc.mean())
        r["dup_both"] = float((df & dc).mean())
        r["dup_final_maxrun"] = int(max((len(s) for s in "".join("1" if x else "0" for x in df).split("0")), default=0))
        # cuts
        med = float(np.median(ef))
        r["cuts_final"] = int((ef > max(25.0, 8 * med)).sum())
        r["cuts_cond"] = int((ec > max(25.0, 8 * float(np.median(ec)))).sum())
        r["max_jump_final"] = float(ef.max())
        # exposure
        mf = Ff.mean((1, 2)); sf = Ff.std((1, 2))
        r["luma_mean"] = float(mf.mean()); r["luma_min_frame"] = float(mf.min()); r["contrast"] = float(sf.mean())
        r["black_frames"] = float((mf < 8).mean()); r["flat_frames"] = float((sf < 4).mean())
        r["over_px"] = float((F[:: max(1, T // 24)] >= 250).mean())
        r["under_px"] = float((F[:: max(1, T // 24)] <= 5).mean())
        mc = Cf.mean((1, 2))
        r["cond_black_frames"] = float((mc < 3).mean())
        cov = (C > 12).mean((1, 2))
        r["cond_cover_mean"] = float(cov.mean()); r["cond_cover_min"] = float(cov.min())
        # edge alignment on 8 anchors
        anc = np.linspace(8, T - 8, 8).astype(int)
        k = np.ones((5, 5), np.uint8)
        rec = {L: [] for L in LAGS}; base = []
        for t in anc:
            Ef = edges(F[t]); Efd = cv2.dilate(Ef, k) > 0; Efm = Efd[:, ::-1]
            for L in LAGS:
                Ec = edges(C[t + L]) > 0
                n = Ec.sum()
                rec[L].append(float((Ec & Efd).sum() / n) if n > 50 else np.nan)
            Ec0 = edges(C[t]) > 0; n = Ec0.sum()
            base.append(float((Ec0 & Efm).sum() / n) if n > 50 else np.nan)
        with np.errstate(all="ignore"):
            er = {L: float(np.nanmean(v)) if not np.all(np.isnan(v)) else np.nan for L, v in rec.items()}
            eb = float(np.nanmean(base)) if not np.all(np.isnan(base)) else np.nan
        r["erec"] = {str(L): (None if np.isnan(v) else round(v, 4)) for L, v in er.items()}
        okl = {L: v for L, v in er.items() if not np.isnan(v)}
        if okl:
            bl = max(okl, key=lambda L: okl[L])
            r["erec_best_lag"] = int(bl); r["erec_best"] = round(okl[bl], 4)
            r["erec_0"] = None if np.isnan(er[0]) else round(er[0], 4)
        r["erec_base"] = None if np.isnan(eb) else round(eb, 4)
        r["curve_final"] = [round(float(x), 2) for x in ef]
        r["curve_cond"] = [round(float(x), 2) for x in ec]
        r["luma_curve"] = [round(float(x), 1) for x in mf]
        r["cover_curve"] = [round(float(x), 3) for x in cov]
        r["hash"] = [format(dhash(F[i]), "016x") for i in (0, T // 3, 2 * T // 3, T - 1)]
        r["T"] = T
        r["sec"] = round(time.time() - t0, 2)
        return r
    except Exception as e:
        r["error"] = "exception: " + repr(e)[:300]
        return r
    finally:
        shutil.rmtree(d, ignore_errors=True)


def fetch(job, d):
    os.makedirs(d, exist_ok=True)
    job["_dir"] = d
    files = [("final.mp4", job["final"]), ("cond.mp4", job["cond"])] + [(f"{k}.json", u) for k, u in job.get("extra", {}).items()]
    if len(files) == 2:  # big files (Magpie): one at a time, multipart
        for name, url in files:
            subprocess.run([STORE, "--log", "error", "cp", "--concurrency", "4", url, os.path.join(d, name)], capture_output=True)
    else:  # several small files: one storage command, two at a time (its start-up dominates otherwise)
        cmds = "".join(f"cp --concurrency 1 {url} {os.path.join(d, name)}\n" for name, url in files)
        subprocess.run([STORE, "--log", "error", "--numworkers", "2", "run"], input=cmds, text=True, capture_output=True)
    for k in job.get("extra", {}):
        p = os.path.join(d, f"{k}.json")
        try:
            job["_extra_" + k] = json.load(open(p))
        except Exception:
            job["_extra_" + k] = None
    return job


def side(job, r):
    """Small facts from the json side files (CARLA: scene metadata)."""
    if job.get("_extra_meta") is not None:
        m = job["_extra_meta"]
        r["meta"] = {k: (m.get("job") or {}).get(k, m.get(k)) for k in ("map", "tod", "weather", "cam", "fog")}


def main():
    jobs_f, shard, nshards, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 48
    ndl = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    jobs = [json.loads(l) for i, l in enumerate(open(jobs_f)) if i % nshards == shard]
    src = jobs[0]["source"] if jobs else "x"
    out_url = f"{outp}/{src}/shard_{shard:03d}.jsonl"
    local = f"{TMP}/out_{src}_{shard:03d}.jsonl"
    os.makedirs(TMP, exist_ok=True)
    done = set()
    # resume from what is already uploaded
    o = subprocess.run([STORE, "cat", out_url], capture_output=True, text=True).stdout
    if o.strip():
        open(local, "w").write(o)
        done = {json.loads(l)["id"] for l in o.splitlines() if l.strip()}
    jobs = [j for j in jobs if j["id"] not in done]
    print(f"shard {shard}: {len(jobs)} to do, {len(done)} done", flush=True)
    q = queue.Queue(maxsize=nproc * 2)
    it = iter(enumerate(jobs)); lock = threading.Lock()

    def dl():
        while True:
            with lock:
                try:
                    i, j = next(it)
                except StopIteration:
                    return
            q.put(fetch(j, f"{TMP}/w_{src}_{shard:03d}_{i}"))

    def feeder():
        n = 0
        while n < len(jobs):
            yield q.get(); n += 1
    fo = open(local, "a"); k = 0; t0 = time.time()
    # pool workers come from a forkserver started before any download thread exists (fork + threads deadlocks)
    with ProcessPoolExecutor(nproc, mp_context=mp.get_context("forkserver")) as ex:
        list(ex.map(int, range(nproc * 2)))
        ths = [threading.Thread(target=dl, daemon=True) for _ in range(ndl)]
        for t in ths: t.start()
        pend = {}
        gen = feeder()
        import concurrent.futures as cf
        for j in gen:
            pend[ex.submit(analyse, {kk: v for kk, v in j.items() if not kk.startswith("_extra")} | {"_dir": j["_dir"]})] = j
            while len(pend) >= nproc * 2:
                dn, _ = cf.wait(pend, return_when=cf.FIRST_COMPLETED)
                for f in dn:
                    jj = pend.pop(f); r = f.result(); side(jj, r); fo.write(json.dumps(r) + "\n"); k += 1
                    if k % BATCH == 0:
                        fo.flush(); subprocess.run([STORE, "--log", "error", "cp", local, out_url], capture_output=True)
                        print(f"{k}/{len(jobs)} {k / (time.time() - t0):.2f}/s", flush=True)
        for f in cf.as_completed(list(pend)):
            jj = pend.pop(f); r = f.result(); side(jj, r); fo.write(json.dumps(r) + "\n"); k += 1
    fo.close()
    subprocess.run([STORE, "--log", "error", "cp", local, out_url], capture_output=True)
    print(f"DONE {k} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
