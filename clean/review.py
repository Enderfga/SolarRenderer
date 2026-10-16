"""Review page: per category a few clips, the 50 % frame with every box drawn, and the same frame as the encoder will
see it after the crop. usage: review.py OUTDIR DEC_DIR DET_DIR      (flat files: solar-clean-*.jpg + solar-clean-review.html)
Boxes: red = accepted overlay (with margin applied in the crop), yellow = VLM overlay not acted on, cyan = content
(inside black bars), green = subject (mask p99 box), white = crop box sent to the encoder.
env: SOLAR_CLEAN_WORK (required: the storage prefix detect.py wrote frames/ to), STORAGE_CLI (default "solar-storage")."""
import sys, os, json, random, subprocess, html, collections
import numpy as np, cv2

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
W3 = os.environ["SOLAR_CLEAN_WORK"].rstrip("/")
out, decd, detd = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
SRC = sorted(f[:-6] for f in os.listdir(decd) if f.endswith(".jsonl"))
dec = {s: [json.loads(l) for l in open(f"{decd}/{s}.jsonl")] for s in SRC if os.path.exists(f"{decd}/{s}.jsonl")}
det = {}
for f in os.listdir(detd):
    if f.endswith(".jsonl") and not f.startswith("pilot"):
        for l in open(f"{detd}/{f}"):
            try: r = json.loads(l); det[r["id"]] = r
            except Exception: pass
rng = random.Random(7)

def cat_of(r):
    R = set(r["reasons"])
    if "undecodable" in R: return None
    if r["drop"]:
        if "overlay_uncroppable" in R: return "drop_uncroppable"
        if "crop_too_costly" in R: return "drop_costly"
        if "subject_would_be_cut" in R: return "drop_subject"
        return "drop_other"
    if r.get("unconfirmed"): return "unsure_unconfirmed"
    if ("black_bars" in R) != (r.get("vlm_letterbox") not in (None, "none")):
        return "unsure_bars_disagree"
    if "black_bars" in R and "watermark" in R: return "crop_bars_and_mark"
    if "watermark" in R: return "crop_mark"
    if "black_bars" in R: return "crop_bars"
    return "clean"

TITLE = {"crop_bars_and_mark": "Black bars + overlay -> cropped, kept", "crop_mark": "Overlay only -> cropped, kept", "crop_bars": "Black bars only -> cropped, kept",
         "drop_uncroppable": "Dropped: overlay in the middle of the picture / cannot be cropped away", "drop_costly": "Dropped: cropping the overlay away costs more than the threshold",
         "drop_subject": "Dropped: the crop would cut the subject", "drop_other": "Dropped: other",
         "unsure_unconfirmed": "Unsure: overlay reported by the VLM without enough support, not acted on", "unsure_bars_disagree": "Unsure: bar detector and VLM disagree",
         "clean": "Clean (nothing detected) -> not cropped"}
ORDER = list(TITLE)
items = collections.defaultdict(list)
for s, rows in dec.items():
    by = collections.defaultdict(list)
    for r in rows:
        c = cat_of(r)
        if c: by[c].append(r)
    for c, rs in by.items():
        pick = rng.sample(rs, min(4 if c == "clean" else 8, len(rs)))
        items[(s, c)] = [(r, len(rs)) for r in pick]

need = []
for (s, c), lst in items.items():
    for r, _ in lst:
        need.append(f"cp {W3}/frames/{s}/{r['id'].replace('/', '__')}_b.jpg {out}/_f_{r['id'].replace('/', '__')}.jpg")
open(f"{out}/_cmds", "w").write("\n".join(need) + "\n")
subprocess.run([STORE, "--numworkers", "16", "run", f"{out}/_cmds"], capture_output=True)

def draw(r, d, img, sc):
    def R(b, col, t=2):
        cv2.rectangle(img, (int(b[0] * sc), int(b[1] * sc)), (int(b[2] * sc), int(b[3] * sc)), col, t)
    if r.get("content") and any((r.get("bars") or {}).values()): R([r["content"][0], r["content"][1], r["content"][2], r["content"][3]], (255, 255, 0), 2)
    for o in r.get("unconfirmed", []): R(o["box"], (0, 220, 255), 2)
    for o in r.get("watermarks", []): R(o["box"], (0, 0, 255), 3)
    if r.get("subject_box"): R(r["subject_box"], (0, 255, 0), 2)
    c = r.get("crop")
    if c: R([c[0], c[2], c[1], c[3]], (255, 255, 255), 3)
    elif r.get("best_box"): R(r["best_box"], (160, 160, 160), 2)
    return img

body = []
for s in SRC:
    cats = [c for c in ORDER if (s, c) in items]
    if not cats: continue
    body.append(f"<h2>{s}</h2>")
    for c in cats:
        lst = items[(s, c)]
        body.append(f"<h3>{TITLE[c]} <small>({lst[0][1]} clips in this category, {len(lst)} shown)</small></h3><div class=g>")
        for k, (r, _) in enumerate(lst):
            d = det.get(r["id"], {}); sid = r["id"].replace("/", "__")
            f = f"{out}/_f_{sid}.jpg"
            if not os.path.exists(f): continue
            img = cv2.imread(f); sc = d.get("jpg_scale", img.shape[1] / r["W"])
            a = draw(r, d, img.copy(), sc)
            na = f"solar-clean-{s}-{c}-{k:02d}-o.jpg"; cv2.imwrite(f"{out}/{na}", a, [cv2.IMWRITE_JPEG_QUALITY, 85])
            box = r.get("crop") and [r["crop"][0], r["crop"][2], r["crop"][1], r["crop"][3]] or r.get("best_box")
            nb = None
            if box:
                cr = img[int(box[1] * sc):int(box[3] * sc), int(box[0] * sc):int(box[2] * sc)]
                if cr.size:
                    cr = cv2.resize(cr, (448, 256) if r["W"] >= r["H"] else (256, 448), interpolation=cv2.INTER_AREA)
                    nb = f"solar-clean-{s}-{c}-{k:02d}-c.jpg"; cv2.imwrite(f"{out}/{nb}", cr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            wm = "; ".join(f"{o['type']}:{html.escape(o['text'][:30])}({o['conf']},{'+'.join(o['support']) or '-'})" for o in r.get("watermarks", []))
            un = "; ".join(f"{o['type']}:{html.escape(str(o['text'])[:30])}({o['conf']})" for o in r.get("unconfirmed", []))
            bars = r.get("bars") or {}
            info = (f"<b>{html.escape(r['id'])}</b> {r['W']}x{r['H']}<br>reasons: {', '.join(r['reasons'])}"
                    f"<br>bars top {bars.get('top',0)} bottom {bars.get('bottom',0)} left {bars.get('left',0)} right {bars.get('right',0)} | VLM letterbox: {r.get('vlm_letterbox')}"
                    f"<br>overlays: {wm or '-'}" + (f"<br>not acted on: {un}" if un else "") +
                    f"<br>crop: {r.get('crop')} keep (of default canvas)={r.get('keep', r.get('need_keep'))} keep (of frame)={r.get('keep_frame')} subject coverage={r.get('subject_cov')}/{r.get('subject_cov_default')}")
            body.append(f"<div class=c><img src='{na}' loading=lazy>" + (f"<img class=k src='{nb}' loading=lazy>" if nb else "") + f"<p>{info}</p></div>")
        body.append("</div>")
legend = ("Red = accepted overlay (the crop avoids it, with a margin); yellow = reported by the VLM without enough support, not acted on; "
          "cyan = content area inside the black bars; green = subject (p99 box of the mask stream); white = final crop box (what the encoder sees); "
          "grey = best candidate box of a dropped clip. The small image is the picture inside the crop box.")
page = f"""<!doctype html><meta charset=utf-8><title>SolarRenderer overlay / black-bar review</title>
<style>body{{font:13px system-ui;margin:16px;background:#111;color:#ddd}} .g{{display:flex;flex-wrap:wrap;gap:10px}}
.c{{background:#1c1c1c;padding:6px;width:640px}} .c img{{max-width:640px;max-height:420px;display:block}} .c img.k{{margin-top:4px;max-width:448px}}
p{{margin:4px 0;word-break:break-all}} h3 small{{color:#999}}</style>
<h1>SolarRenderer training data: overlay and black-bar review</h1><p>{legend}</p>{''.join(body)}"""
open(f"{out}/solar-clean-review.html", "w").write(page)
for f in os.listdir(out):
    if f.startswith("_"): os.remove(f"{out}/{f}")
print("REVIEW_OK", len(os.listdir(out)))
