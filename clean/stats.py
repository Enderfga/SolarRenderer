"""Counts per source. usage: stats.py DEC_DIR       (DEC_DIR holds the decide.py outputs as <source>.jsonl)"""
import sys, json, os, collections
import numpy as np
d = sys.argv[1]

def row(name, rs):
    n = len(rs)
    bars = sum(any((r.get("bars") or {}).values()) for r in rs)
    wm = sum(bool(r.get("watermarks")) for r in rs)
    crop = sum((not r["drop"]) and r.get("crop") is not None for r in rs)
    keep = sum(not r["drop"] for r in rs)
    drop = sum(r["drop"] for r in rs)
    rc = collections.Counter(x for r in rs if r["drop"] for x in r["reasons"] if x not in ("black_bars", "watermark"))
    return f"| {name} | {n} | {bars} | {wm} | {crop} | {keep - crop} | {drop} | {', '.join(f'{k} {v}' for k, v in rc.most_common())} |"

H = "| scope | clips | black bars | overlay | kept after crop | kept uncropped | dropped | drop reasons |\n|---|---|---|---|---|---|---|---|"
print(H)
for f in sorted(os.listdir(d)):
    if not f.endswith(".jsonl"): continue
    rs = [json.loads(l) for l in open(f"{d}/{f}")]
    print(row(f[:-6], rs))
    nk = np.array([r["need_keep"] for r in rs if r.get("watermarks") and r.get("need_keep") is not None])
    if len(nk):
        print(f"  {f[:-6]}: crop cost of clips with overlays (kept area / default canvas area), p1/p5/p10/p25/p50:",
              np.percentile(nk, [1, 5, 10, 25, 50]).round(3).tolist())
