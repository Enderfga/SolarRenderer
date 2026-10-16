"""Fuse detections + VLM review into a crop box or a drop per clip.
usage: decide.py SOURCE DET.jsonl[,..] VLM.jsonl[,..] OUT.jsonl [--theta 0.70] [--v2 VLM2.jsonl[,..]]
  DET from detect.py, VLM from vlm.py, optional VLM2 from vlm2.py (second pass over OCR text boxes).

Overlay evidence (boxes in target px):
  vlm    - Qwen3.5 overlays (any type) with confidence >= 0.5, box = union of its box in the early and late frame.
           Accepted when another detector supports it (or the VLM is sure, >= 0.9, and read text in it): (OCR text cluster or static-edge
           component overlapping it, or the same box in both frames).
  ocr    - recurring OCR text is recorded only.
Each accepted overlay is grown by MARGIN and unioned with the OCR / static boxes it overlaps.

Crop: the content box C (inside the accepted bars) is what the clip really holds; the encoder would keep fit(C), the
largest canvas-aspect (1344:768, or 768:1344 for portrait) centred box in C. We search canvas-aspect boxes R inside C,
largest first (scale s of fit(C), step 0.005), placed anywhere on a 4 px grid, that (1) touch no overlay and (2) keep
>= 98 % of the subject mask (time-averaged) when the clip has an aligned mask. Among feasible boxes at the largest s we
take the one closest to the centre of C. R has the canvas aspect, so the encoder's own centre-crop leaves it unchanged.
  drop  when no R with s^2 >= THETA exists (overlay in the middle, or cutting it away costs too much picture).
"""
import sys, json, math, re, collections
import numpy as np

MARGIN_F = 0.012   # overlay margin as a fraction of the short side (min 6 px)
SUBJ_COV = 0.98
SUBJ_TOL = 0.03    # a crop may lose at most 3 points more of the subject than the encoder's centre-crop already does

def area(b): return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
def inter(a, b): return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
def union(a, b): return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
def iou(a, b):
    i = inter(a, b); u = area(a) + area(b) - i
    return i / u if u else 0.0

def vbox(b, W, H):
    if not b or len(b) != 4: return None
    try: x0, y0, x1, y1 = [float(v) for v in b]
    except Exception: return None
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
    r = [x0 / 1000 * W, y0 / 1000 * H, x1 / 1000 * W, y1 / 1000 * H]
    r = [max(0, min(W, r[0])), max(0, min(H, r[1])), max(0, min(W, r[2])), max(0, min(H, r[3]))]
    return [int(math.floor(r[0])), int(math.floor(r[1])), int(math.ceil(r[2])), int(math.ceil(r[3]))] if area(r) > 0 else None

def overlays(d, v, v2=None):
    W, H = d["W"], d["H"]
    still = d.get("motion", {}).get("still_frac", 1.0) > 0.3
    ocr = [o for o in d.get("ocr", []) if o["n"] >= 2]
    stat = d.get("static", [])
    out = []
    for o in (v or {}).get("overlays", []) or []:
        if not isinstance(o, dict): continue
        try: conf = float(o.get("confidence", 0) or 0)
        except Exception: conf = 0.0
        b1, b2 = vbox(o.get("box1"), W, H), vbox(o.get("box2"), W, H)
        if not (b1 or b2) or conf < 0.5: continue
        box = union(b1, b2) if (b1 and b2) else (b1 or b2)
        sup = []
        if b1 and b2 and iou(b1, b2) >= 0.5: sup.append("same_box_both_frames")
        for c in ocr:
            if inter(c["box"], box) > 0.3 * min(area(c["box"]), area(box)):
                sup.append("ocr"); box = union(box, c["box"])
        if not still:
            for c in stat:
                if inter(c["box"], box) > 0.3 * min(area(c["box"]), area(box)) and area(c["box"]) < 4 * area(box):
                    sup.append("static"); box = union(box, c["box"])
        txt = str(o.get("text") or "").strip()
        acc = bool(sup) or (conf >= 0.9 and txt not in ("", "None", "null"))
        out.append({"box": box, "type": str(o.get("type", "")), "text": str(o.get("text", ""))[:60], "conf": round(conf, 2),
                    "support": sorted(set(sup)), "accepted": acc, "early": b1 is not None, "late": b2 is not None, "by": "vlm"})
    # second pass: OCR text judged on six frames
    for b in (v2 or {}).get("boxes", []) or []:
        # the verdict alone is not enough (posters and signs were called overlays): the text must also sit at the same
        # place on screen in >= 2 of the 6 frames, which scene text under a moving camera does not
        if b.get("verdict") == "overlay" and b.get("n", 1) < 2:
            out.append({"box": list(b["box"]), "type": "text_overlay?", "text": str(b.get("text", ""))[:60], "conf": b.get("conf", 0),
                        "support": ["vlm2_single_frame"], "accepted": False, "early": None, "late": None, "by": "vlm2"})
            continue
        if b.get("verdict") == "overlay":
            hit = [o for o in out if o["accepted"] and inter(o["box"], b["box"]) > 0]
            if hit:
                hit[0]["box"] = union(hit[0]["box"], b["box"]); hit[0]["support"] = sorted(set(hit[0]["support"]) | {"vlm2"})
            else:
                out.append({"box": list(b["box"]), "type": "text_overlay",
                            "text": str(b.get("text", ""))[:60], "conf": b.get("conf", 0), "support": ["ocr", "vlm2"],
                            "accepted": True, "early": None, "late": None, "by": "vlm2"})
    return out

def fit_size(Cw, Ch, A):
    return (Ch * A, Ch) if Cw / Ch > A else (Cw, Cw / A)

def search(C, A, forb, heat, W, H, smin, need=SUBJ_COV):
    """Largest canvas-aspect box inside C avoiding forb, keeping the subject. Returns (box, s, coverage) or None."""
    Cw, Ch = C[2] - C[0], C[3] - C[1]
    Wf, Hf = fit_size(Cw, Ch, A)
    if heat is not None:
        hh, hw = heat.shape; sat = np.zeros((hh + 1, hw + 1)); sat[1:, 1:] = heat.cumsum(0).cumsum(1); tot = heat.sum()
    best_any = None
    for s in np.arange(1.0, smin - 1e-9, -0.005):
        h = int(math.floor(Hf * s)); w = int(round(h * A))
        if w > Cw: w = Cw; h = int(round(w / A))
        if h > Ch or w <= 0 or h <= 0: continue
        xs = np.unique(np.r_[np.arange(C[0], C[2] - w + 1, 4), C[2] - w]); ys = np.unique(np.r_[np.arange(C[1], C[3] - h + 1, 4), C[3] - h])
        X, Y = np.meshgrid(xs, ys)
        ok = np.ones(X.shape, bool)
        for b in forb:
            ok &= (X + w <= b[0]) | (X >= b[2]) | (Y + h <= b[1]) | (Y >= b[3])
        if not ok.any(): continue
        if heat is not None and tot > 0:
            gx0 = np.clip(np.round(X / W * hw).astype(int), 0, hw); gx1 = np.clip(np.round((X + w) / W * hw).astype(int), 0, hw)
            gy0 = np.clip(np.round(Y / H * hh).astype(int), 0, hh); gy1 = np.clip(np.round((Y + h) / H * hh).astype(int), 0, hh)
            cov = (sat[gy1, gx1] - sat[gy0, gx1] - sat[gy1, gx0] + sat[gy0, gx0]) / tot
        else:
            cov = np.ones(X.shape)
        good = ok & (cov >= need)
        cx, cy = (C[0] + C[2] - w) / 2, (C[1] + C[3] - h) / 2
        dist = np.hypot(X - cx, Y - cy)
        if good.any():
            k = np.argmin(np.where(good, dist, np.inf))
            i, j = np.unravel_index(k, X.shape)
            return [int(X[i, j]), int(Y[i, j]), int(X[i, j] + w), int(Y[i, j] + h)], float(s), float(cov[i, j])
        if best_any is None:
            k = np.argmax(np.where(ok, cov, -1)); i, j = np.unravel_index(k, X.shape)
            best_any = ([int(X[i, j]), int(Y[i, j]), int(X[i, j] + w), int(Y[i, j] + h)], float(s), float(cov[i, j]))
    return ("subject", best_any) if best_any else None

def default_cov(C, A, heat, W, H):
    if heat is None: return None
    Cw, Ch = C[2] - C[0], C[3] - C[1]; Wf, Hf = fit_size(Cw, Ch, A)
    x0, y0 = C[0] + (Cw - Wf) / 2, C[1] + (Ch - Hf) / 2
    hh, hw = heat.shape
    g = heat[int(round(y0 / H * hh)):int(round((y0 + Hf) / H * hh)), int(round(x0 / W * hw)):int(round((x0 + Wf) / W * hw))]
    return round(float(g.sum() / max(heat.sum(), 1e-9)), 4)

TRUNC = set()

def decide(d, v, theta, v2=None):
    W, H = d["W"], d["H"]
    if d["id"] in TRUNC:
        return {"id": d["id"], "target": d["target"], "W": W, "H": H, "crop": None, "drop": True,
                "reasons": ["watermark", "too_many_overlays"], "watermarks": [], "bars": d.get("bars"), "content": d.get("content")}
    rec = {"id": d["id"], "target": d["target"], "W": W, "H": H}
    if d.get("catalog_target") and d["catalog_target"] != d["target"]:
        rec["catalog_target"] = d["catalog_target"]
    if "error" in d:
        return dict(rec, crop=None, drop=True, reasons=["undecodable"], watermarks=[], bars=None)
    A = 1344 / 768 if W >= H else 768 / 1344
    bars = d["bars"]; C = list(d["content"])
    # keep 2 px off a detected bar edge (anti-aliased boundary)
    if bars["top"]: C[1] += 2
    if bars["bottom"]: C[3] -= 2
    if bars["left"]: C[0] += 2
    if bars["right"]: C[2] -= 2
    ov = overlays(d, v, v2)
    acc = [o for o in ov if o["accepted"]]
    m = max(6, int(round(MARGIN_F * min(W, H))))
    forb = [[o["box"][0] - m, o["box"][1] - m, o["box"][2] + m, o["box"][3] + m] for o in acc]
    reasons = []
    has_bar = any(bars[k] for k in bars)
    if has_bar: reasons.append("black_bars")
    if acc: reasons.append("watermark")
    sub = d.get("subject")
    heat = np.array(sub["heat16"], np.float64) if sub and sub.get("heat16") else None
    rec.update(bars=bars, content=d["content"], watermarks=[{k: o[k] for k in ("box", "type", "text", "conf", "support", "by", "early", "late")} for o in acc],
               unconfirmed=[{k: o[k] for k in ("box", "type", "text", "conf")} for o in ov if not o["accepted"]],
               vlm_letterbox=(v or {}).get("letterbox"), subject_box=sub["p99"] if sub else None,
               subject_cov_default=default_cov(C, A, heat, W, H))
    if v is None: reasons.append("no_vlm_review")
    # forbidden boxes clipped to the content; boxes fully outside C are irrelevant
    forb = [b for b in forb if inter(b, C) > 0]
    if not acc and not has_bar:
        return dict(rec, crop=None, drop=False, reasons=reasons or ["clean"], keep=1.0, keep_frame=1.0)
    # the subject may not lose more than the encoder's own centre-crop already loses (or 2 %, whichever is less strict)
    dc = rec["subject_cov_default"]
    need = min(SUBJ_COV, dc - SUBJ_TOL) if dc is not None else SUBJ_COV
    # searched down to a quarter of the picture so the cost of every crop is known; THETA decides afterwards
    r = search(C, A, forb, heat, W, H, 0.5, need)
    if r is None:
        return dict(rec, crop=None, drop=True, reasons=reasons + ["overlay_uncroppable"], need_keep=None)
    if r[0] == "subject":
        box, s, cov = r[1]
        return dict(rec, crop=None, drop=True, reasons=reasons + ["subject_would_be_cut"], best_box=box, best_cov=round(cov, 3), need_keep=None)
    box, s, cov = r
    if s * s < theta - 1e-9:
        return dict(rec, crop=None, drop=True, reasons=reasons + ["crop_too_costly"], need_keep=round(s * s, 4), best_box=box)
    x0, y0, x1, y1 = box
    rec.update(crop=[x0, x1, y0, y1], drop=False, reasons=reasons, keep=round(s * s, 4), need_keep=round(s * s, 4), keep_frame=round(area(box) / (W * H), 4),
               subject_cov=round(cov, 4) if heat is not None else None)
    return rec

def main():
    src, dets, vlms, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4]
    theta = float(sys.argv[sys.argv.index("--theta") + 1]) if "--theta" in sys.argv else 0.70
    D = {}
    for f in dets:
        for l in open(f):
            r = json.loads(l); D[r["id"]] = r
    V = {}
    global TRUNC
    for f in vlms:
        if not f: continue
        for l in open(f):
            r = json.loads(l)
            if "error" not in r and isinstance(r.get("overlays"), list): V[r["id"]] = r
            elif '"box1"' in (r.get("raw") or ""): TRUNC.add(r["id"])   # answer cut off by the token cap: a screen full of overlays
    V2 = {}
    if "--v2" in sys.argv:
        for f in sys.argv[sys.argv.index("--v2") + 1].split(","):
            for l in open(f):
                r = json.loads(l)
                if "boxes" in r: V2[r["id"]] = r
    with open(out, "w") as fo:
        for i, d in sorted(D.items()):
            fo.write(json.dumps(decide(d, V.get(i), theta, V2.get(i))) + "\n")

if __name__ == "__main__":
    main()
