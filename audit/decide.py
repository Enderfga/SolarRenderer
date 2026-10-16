"""decide.py SRC JOBS.jsonl METRICS_PREFIX [--nofetch] [--upload DEST_URL]: keep / drop for every clip of a source
from the metrics shards that audit.py wrote under METRICS_PREFIX/<SRC>/.

JOBS.jsonl is the full job list of the source (the catalog of clips; make_jobs.py output).
Writes $AUDIT_METRICS_DIR/<SRC>_audit.jsonl (and copies it to DEST_URL with --upload):
one row per catalog clip: id, metrics (scalars), keep, reasons. Clips with no metrics row are keep=null, ["not_audited"].
env: STORAGE_CLI (default "solar-storage"), AUDIT_METRICS_DIR (local copy of the metrics, default <system temp>/solar_audit_m)
"""
import json, subprocess, glob, os, sys, collections, tempfile
import numpy as np
STORE = os.environ.get("STORAGE_CLI", "solar-storage")
M = os.environ.get("AUDIT_METRICS_DIR", os.path.join(tempfile.gettempdir(), "solar_audit_m"))
SRC, JOBS, METRICS = sys.argv[1], sys.argv[2], sys.argv[3]
if "--nofetch" not in sys.argv:
    os.makedirs(M, exist_ok=True)
    subprocess.run([STORE, "--log", "error", "cp", f"{METRICS.rstrip('/')}/{SRC}/*", f"{M}/{SRC}/"], capture_output=True)

DUP = 0.35  # same as audit.py: adjacent frames with mean |diff| below this are the same frame
C = dict(
    static_motion=0.3,   # mean |frame diff| of the final (0-255 grey) below this: nothing moves
    sync_motion=0.6,     # motion-energy sync is only judged when the final moves at least this much
    lag_gain=0.10,       # a non-zero lag must beat lag 0 by this much in motion correlation,
    lag_min_corr=0.5,    #   reach this correlation, and the edge test must not prefer lag 0
    weak_corr=0.2,       # motion correlation at lag 0 below this on a moving clip: condition does not follow the picture
    black_frame=0.02,    # share of frames with mean luma < 8
    flat_frame=0.02,     # share of frames with luma std < 4
    cond_cover=0.15,     # mean share of non-black condition pixels
    cond_cover_min=0.02, # a frame of the condition is (almost) empty
    cond_black=0.02,     # share of black condition frames
    cond_frozen=0.05,    # share of frames where the condition is bit-identical to the previous (< 0.05) while the final moves (> 2.0)
)
S = {
    "magpie": dict(fmt=(1920, 1080, 60.0), frames_min=470, dark=25, over=0.25, edge_ratio=0.97),
    "carla": dict(fmt=(1344, 768, 24.0), frames_min=192, dark=35, crushed=0.40, contrast=15, over=0.25, edge_ratio=None),
}


def load(src):
    rows = {}
    for f in glob.glob(f"{M}/{src}/shard_*.jsonl"):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if "curve_final" in r or r.get("error") or r["id"] not in rows:  # rows of the 24-clip smoke test lack curves
                rows[r["id"]] = r
    return {i: r for i, r in rows.items() if "curve_final" in r or r.get("error")}


def ref_level(e, t):
    """motion level around frame t: the larger of the medians of the moving (non-repeat) frames on each side (+-6)."""
    lv = []
    for nb in (e[max(0, t - 6): t], e[t + 1: t + 7]):
        nb = nb[nb >= DUP]
        lv.append(float(np.median(nb)) if len(nb) >= 2 else 0.0)
    return max(1.0, max(lv))


def spikes(e, floor=20.0, ratio=4.0, end=None):
    """isolated jumps: e[t] > floor and > ratio x the motion level on BOTH sides (a camera whip that starts and keeps
    going is not a jump; a cut or a glitch frame is). The last 3 transitions are ignored (decoder padding at the end)."""
    e = np.asarray(e, float)
    end = min(len(e), end or len(e)) - 3
    return [int(t) for t in np.nonzero(e > floor)[0] if t < end and e[t] > ratio * ref_level(e, t)]


def rises(e, t):
    """does stream e also change markedly at t (+-1)? its amplitude may differ (flat grey boxes vs textures)."""
    e = np.asarray(e, float)
    return any(0 <= u < len(e) and (e[u] > 2.5 * ref_level(e, u) or e[u] > 8) for u in (t - 1, t, t + 1))


def curve_facts(r):
    ef, ec = np.asarray(r["curve_final"]), np.asarray(r["curve_cond"])
    n = min(r["probe_final"]["n"], r["probe_cond"]["n"]) - 1
    sf, sc = spikes(ef, end=n), spikes(ec, end=n)
    cuts = [t for t in sf if rises(ec, t)]
    fo = [t for t in sf if not rises(ec, t)]
    co = [u for u in sc if not rises(ef, u)]
    T = min(len(ef), len(ec))
    return {"cuts": len(cuts), "jumps_final_only": len(fo), "jumps_cond_only": len(co),
            "cond_frozen_frac": round(float(((ec[:T] < 0.05) & (ef[:T] > 2.0)).mean()), 4)}


def near_dups(rows, ids):
    """Twins: two clips with two or more frame pairs (any of the 4 sampled frames 0, 1/3, 2/3, end of one against any of
    the other) whose 64-bit dHash differ by <= 4 bits, mean luma within 8, and the per-frame luma curves agree (mean |diff| < 3)
    over >= 1/3 of a clip at the offset of one hit. Candidates come from exact matches on one
    of five bit bands (pigeonhole: <= 4 differing bits leave one band equal); buckets above 300 (flat frames) are skipped.
    Returns {id: first id (sorted) of its group} for every clip that is not the first of its group."""
    ids = sorted(i for i in ids if rows[i].get("hash"))
    H = [[int(h, 16) for h in rows[i]["hash"]] for i in ids]
    L = [rows[i]["luma_mean"] for i in ids]
    LC = [np.asarray(rows[i]["luma_curve"], float) for i in ids]

    def luma_follows(la, lb, k, l):
        """the frame-mean luma of the two clips agrees (mean |diff| < 3) over >= 1/3 of a clip at the offset of a hash hit."""
        Ta, Tb = len(la), len(lb)
        pa, pb = [0, Ta // 3, 2 * Ta // 3, Ta - 1][k], [0, Tb // 3, 2 * Tb // 3, Tb - 1][l]
        o = pb - pa  # la[t] ~ lb[t + o]
        t0, t1 = max(0, -o), min(Ta, Tb - o)
        return t1 - t0 >= min(Ta, Tb) // 3 and float(np.abs(la[t0:t1] - lb[t0 + o:t1 + o]).mean()) < 3.0
    bands = [(0, 13), (13, 26), (26, 39), (39, 52), (52, 64)]
    buck = collections.defaultdict(set)
    for n, hs in enumerate(H):
        for h in hs:
            for bi, (lo, hi) in enumerate(bands):
                buck[(bi, (h >> lo) & ((1 << (hi - lo)) - 1))].add(n)
    cand = set()
    for v in buck.values():
        if 1 < len(v) <= 300:
            v = sorted(v)
            cand.update((v[x], v[y]) for x in range(len(v)) for y in range(x + 1, len(v)))
    n = len(ids); parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    pairs = 0
    for a, b in cand:
        if abs(L[a] - L[b]) >= 8:
            continue
        hits = [(k, l) for k, x in enumerate(H[a]) for l, y in enumerate(H[b]) if bin(x ^ y).count("1") <= 4]
        if len(hits) >= 2 and any(luma_follows(LC[a], LC[b], k, l) for k, l in hits):
            pairs += 1
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    out = {ids[k]: ids[find(k)] for k in range(n) if find(k) != k}
    print(f"near-dup candidates {len(cand)}, pairs {pairs}, clips dropped as twins {len(out)}")
    return out


def judge(r, th):
    why = []
    if r.get("error"):
        return [{"final missing": "final_missing", "cond missing": "cond_missing"}.get(r["error"], "broken")]
    pf, pc = r["probe_final"], r["probe_cond"]
    W, H, FPS = th["fmt"]
    if (pf["w"], pf["h"]) != (W, H) or abs(pf["fps"] - FPS) > 0.01:
        why.append("format")
    if (pc["w"], pc["h"]) != (pf["w"], pf["h"]) or abs(pc["fps"] - pf["fps"]) > 0.01 or pc["n"] != pf["n"]:
        why.append("cond_format_mismatch")
    if min(pf["n"], r["T"]) < th["frames_min"]:
        why.append("too_short")
    if r.get("decode_err_final") or r.get("decode_err_cond"):
        why.append("decode_error")
    mot = r["motion_final_mean"]
    if mot < C["static_motion"]:
        why.append("static")
    # time alignment
    cf = curve_facts(r); r.update(cf)
    m0, mb, ml = r.get("mcorr_0"), r.get("mcorr_best"), r.get("mcorr_best_lag")
    el, eb, e0 = r.get("erec_best_lag"), r.get("erec_best"), r.get("erec_0")
    if SRC == "carla":  # rain / spray move the picture but not the labels: judge time by label edges, not motion energy
        if el is not None and el != 0 and eb - e0 >= 0.02:
            why.append("time_offset")
    elif mot >= C["sync_motion"] and m0 is not None:
        if ml != 0 and mb - m0 >= C["lag_gain"] and mb >= C["lag_min_corr"] and el != 0:
            why.append("time_offset")
        elif m0 < C["weak_corr"]:
            why.append("weak_sync")
    if SRC != "carla" and cf["cond_frozen_frac"] > C["cond_frozen"]:
        why.append("cond_frozen")
    if cf["cuts"] > 0:
        why.append("cut")
    if cf["jumps_final_only"] > 0 or cf["jumps_cond_only"] > 0:
        why.append("jump_unmatched")
    # picture
    if r["black_frames"] > C["black_frame"]:
        why.append("black_frames")
    if r["flat_frames"] > C["flat_frame"]:
        why.append("flat_frames")
    if r["luma_mean"] < th["dark"] or (th.get("crushed") and r["under_px"] > th["crushed"]):
        why.append("too_dark")
    if th.get("contrast") and r["contrast"] < th["contrast"]:
        why.append("low_contrast")
    if r["over_px"] > th["over"]:
        why.append("overexposed")
    # condition
    if r["cond_cover_mean"] < C["cond_cover"] or r["cond_cover_min"] < C["cond_cover_min"]:
        why.append("cond_empty")
    if r["cond_black_frames"] > C["cond_black"]:
        why.append("cond_black_frames")
    er = th.get("edge_ratio")
    if er and r.get("erec_0") and r.get("erec_base") and r["erec_0"] / r["erec_base"] < er:
        why.append("structure_mismatch")
    return why


def main():
    rows = load(SRC)
    th = S[SRC]
    cat = [json.loads(l)["id"] for l in open(JOBS)]
    have = [i for i in cat if i in rows]
    reasons = {i: judge(rows[i], th) for i in have}
    dup = near_dups(rows, [i for i in have if not reasons[i]])  # twins among otherwise clean clips
    for i in dup:
        reasons[i].append("near_duplicate")
    drop_keys = ("curve_final", "curve_cond", "luma_curve", "cover_curve")
    out = f"{M}/{SRC}_audit.jsonl"
    cnt = collections.Counter(); kept = 0
    with open(out, "w") as f:
        for i in cat:
            if i not in rows:
                f.write(json.dumps({"id": i, "keep": None, "reasons": ["not_audited"]}) + "\n"); continue
            r = {k: v for k, v in rows[i].items() if k not in drop_keys}
            r["keep"] = not reasons[i]; r["reasons"] = reasons[i]
            if i in dup: r["near_duplicate_of"] = dup[i]
            kept += r["keep"]; cnt.update(reasons[i])
            f.write(json.dumps(r) + "\n")
    print(f"{SRC}: catalog {len(cat)}, audited {len(have)}, keep {kept}, drop {len(have) - kept}")
    for k, v in cnt.most_common():
        print(f"   {k:22s} {v}")
    only = collections.Counter(reasons[i][0] for i in have if len(reasons[i]) == 1)
    print("  sole reason:", dict(only.most_common()))
    if "--upload" in sys.argv:
        subprocess.run([STORE, "--log", "error", "cp", out, sys.argv[sys.argv.index("--upload") + 1]], check=True)
        print("uploaded")


if __name__ == "__main__":
    main()
