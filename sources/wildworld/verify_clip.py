"""verify_clip.py OUT_DIR [clip_id ...]: check uploaded WildWorld SolarRenderer clips (all of clips/ if none given): each
stream 1344x768, 24 fps, frames == json frames; json fields present.  Writes a contact sheet per clip
(render | white_box | skeleton at frames 0, mid, last) to OUT_DIR when OUT_DIR is not '-'.
env: WW_DST (storage prefix of the clips, required), WW_WORK (local scratch, default wildworld_work), STORAGE_CLI."""
import json, os, subprocess, sys, tempfile
import av, numpy as np, cv2
STORE = os.environ.get("STORAGE_CLI", "solar-storage"); DST = os.environ["WW_DST"]
WORK = os.environ.get("WW_WORK", "wildworld_work"); os.makedirs(WORK, exist_ok=True)
out = sys.argv[1]; ids = sys.argv[2:]
if not ids:
    ids = [l.split()[-1][:-5] for l in subprocess.run([STORE, "ls", f"{DST}/*.json"], capture_output=True, text=True).stdout.splitlines()]
bad = 0
for cid in ids:
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        subprocess.run([STORE, "cp", f"{DST}/{cid}*", f"{td}/"], capture_output=True)
        m = json.load(open(f"{td}/{cid}.json"))
        errs, rows = [], []
        for st in m["streams"]:
            p = f"{td}/{cid}_{st}.mp4"
            if not os.path.exists(p): errs.append(f"missing {st}"); continue
            c = av.open(p); s = c.streams.video[0]
            fr = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
            if (s.width, s.height) != (1344, 768) or round(float(s.average_rate)) != 24 or len(fr) != m["frames"]:
                errs.append(f"{st}: {s.width}x{s.height} {s.average_rate} {len(fr)} frames vs {m['frames']}")
            rows.append([fr[k] for k in (0, len(fr) // 2, len(fr) - 1)])
        need = ("license", "source", "frames_meta", "src_frame_indices", "camera", "checks")
        errs += [f"json lacks {k}" for k in need if k not in m]
        fm = m["frames_meta"]
        if len(fm) != m["frames"] or any(f["c2w"] is None or f["K"] is None for f in fm): errs.append("frames_meta camera incomplete")
        print(cid, "OK" if not errs else errs, m["checks"]["skeleton_on_depth"].get("n", {}).get("on_surface_median"), flush=True)
        bad += bool(errs)
        if out != "-" and len(rows) == 3:
            sheet = np.concatenate([np.concatenate([cv2.resize(r[k], (448, 256)) for r in rows], 1) for k in range(3)], 0)
            cv2.imwrite(f"{out}/{cid}.jpg", sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
print("bad", bad, "of", len(ids))
