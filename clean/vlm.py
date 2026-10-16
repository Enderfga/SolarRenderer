"""VLM review of overlays with Qwen3.5-397B (served by vLLM, see vlm_serve.sh). Two frames per clip (15 % and 85 %).
usage: vlm.py DET.jsonl[,DET2...] OUT.jsonl [LIMIT]       (frames read from $SOLAR_CLEAN_WORK/frames/<source>/<id>_{a,c}.jpg)
env:   SOLAR_CLEAN_WORK (required), VLM_URL (default http://127.0.0.1:8000), VLM_MODEL (served model name, default qwen3.5),
       CONC (parallel requests, 96), STORAGE_CLI (default "solar-storage"), CLEAN_TMP (local scratch, default <system temp>/solar_clean)
Each row: {"id", "overlays": [{"type","text","box1","box2","in_scene"?, "confidence"}], "letterbox", "raw"} with boxes in
0-1000 normalised coordinates of the frame."""
import sys, os, json, base64, subprocess, re, time, tempfile, concurrent.futures as cf
import requests

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
W = os.environ.get("SOLAR_CLEAN_WORK", "").rstrip("/")
URL = os.environ.get("VLM_URL", "http://127.0.0.1:8000").rstrip("/") + "/v1/chat/completions"
MODEL = os.environ.get("VLM_MODEL", "qwen3.5")
PROMPT = """Image 1 and image 2 are two frames (early and late) of the same short video clip.

List every graphic that is overlaid on top of the picture and is NOT part of the filmed or rendered scene itself, for example:
- an artist / creator name, credit line, signature or handle
- a watermark, channel / studio / brand logo, website or social-media handle
- subtitles, captions, title text, a timestamp, a frame counter
- game UI / HUD (health bar, minimap, speedometer, crosshair, menu, button prompts)
- a picture-in-picture inset, a decorative border or frame drawn around the picture.

Do NOT list text or logos that physically exist inside the scene (signs, posters, screens, writing on objects or clothing) - those move and change perspective with the scene. Do NOT list plain black letterbox / pillarbox bars; report those only in the "letterbox" field.

Answer with JSON only, no prose:
{"overlays": [{"type": "credit|watermark|logo|subtitle|title|timestamp|hud|inset|border|other", "text": "<text if any>", "box1": [x0, y0, x1, y1], "box2": [x0, y0, x1, y1] or null, "confidence": 0.0-1.0}], "letterbox": "none|top_bottom|left_right|all_sides"}
Boxes are tight, in coordinates normalised to 0-1000 of each image (x right, y down). box1 is in image 1, box2 in image 2 (null if the overlay is absent there). Return {"overlays": [], "letterbox": "none"} if there is nothing."""

def b64(p): return base64.b64encode(open(p, "rb").read()).decode()

def ask(a, c):
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 700,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(a)}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(c)}"}},
                {"type": "text", "text": PROMPT}]}]}
    for k in range(3):
        try:
            r = requests.post(URL, json=body, timeout=300).json()
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            err = str(e); time.sleep(5)
    return "ERROR " + err

def parse(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def main():
    dets, out = sys.argv[1].split(","), sys.argv[2]
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    rows = []
    for d in dets:
        if not os.path.exists(d): continue
        for l in open(d):
            try: rows.append(json.loads(l))
            except Exception: pass
    rows = [r for r in rows if "error" not in r][:lim]
    done = set()
    if os.path.exists(out): done = {json.loads(l)["id"] for l in open(out)}
    rows = [r for r in rows if r["id"] not in done]
    if not W:
        sys.exit("set SOLAR_CLEAN_WORK to the storage prefix detect.py wrote frames/ to")
    tmp = os.path.join(os.environ.get("CLEAN_TMP", os.path.join(tempfile.gettempdir(), "solar_clean")), "vlmf"); os.makedirs(tmp, exist_ok=True)
    fo = open(out, "a"); t0 = time.time(); n = 0
    for i in range(0, len(rows), 1024):
        chunk = rows[i:i + 1024]
        cmds = []
        for r in chunk:
            s = r["id"].replace("/", "__")
            for t in "ac":
                cmds.append(f"cp {W}/frames/{r['source']}/{s}_{t}.jpg {tmp}/{s}_{t}.jpg")
        open(f"{tmp}/cmds", "w").write("\n".join(cmds) + "\n")
        subprocess.run([STORE, "--numworkers", "16", "run", f"{tmp}/cmds"], capture_output=True)
        def job(r):
            s = r["id"].replace("/", "__"); a, c = f"{tmp}/{s}_a.jpg", f"{tmp}/{s}_c.jpg"
            if not (os.path.exists(a) and os.path.exists(c)):
                return {"id": r["id"], "error": "frames missing"}
            t = ask(a, c); p = parse(t)
            for f in (a, c): os.remove(f)
            return {"id": r["id"], "source": r["source"], **(p or {}), "raw": t if p is None else None}
        with cf.ThreadPoolExecutor(int(os.environ.get("CONC", 96))) as ex:
            for res in ex.map(job, chunk):
                if res.get("error") == "frames missing": continue      # not uploaded yet; retried next round
                fo.write(json.dumps(res) + "\n"); n += 1
        fo.flush()
        print(f"vlm {n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    print("VLM_DONE", n, flush=True)

if __name__ == "__main__":
    main()
