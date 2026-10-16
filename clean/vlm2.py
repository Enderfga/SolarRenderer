"""Second VLM pass: is each OCR text box an overlay or part of the scene?
The first pass looked at two frames; text that shows up only for a moment (an intro title, a card at the end of a clip)
or that the first pass judged ambiguous is asked about here, on the six OCR frames with the boxes numbered on them.
Candidates: OCR clusters with conf >= 0.85 and >= 5 letters.
usage: vlm2.py DET.jsonl[,..] OUT.jsonl
env:   VLM_URL (default http://127.0.0.1:8000), VLM_MODEL (served model name, default qwen3.5), STORAGE_CLI (default "solar-storage"),
       CLEAN_TMP (local scratch, default <system temp>/solar_clean)
The target clips are downloaded again from the "target" urls in the detection rows."""
import sys, os, json, re, subprocess, base64, time, threading, queue, tempfile, concurrent.futures as cf
import numpy as np, cv2, requests

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
URL = os.environ.get("VLM_URL", "http://127.0.0.1:8000").rstrip("/") + "/v1/chat/completions"
MODEL = os.environ.get("VLM_MODEL", "qwen3.5")
TMP = os.path.join(os.environ.get("CLEAN_TMP", os.path.join(tempfile.gettempdir(), "solar_clean")), "v2")
PROMPT = """The six images are frames of one short video clip, in time order (top-left to bottom-right: frames 1-6). Numbered red boxes mark text that OCR found in some of the frames.

For each numbered box decide whether that text is
- "overlay": a 2D graphic composited on top of the video - a title, credit, caption, subtitle, watermark, logo, social handle, an on-screen UI / HUD, a game-card frame and its text, a software interface or screenshot, a text slide; it is not lit or occluded by the scene and keeps its position on screen;
- "scene": text that physically exists inside the filmed or rendered 3D world - a sign, poster, billboard, screen, label, licence plate, writing on an object or clothing; it moves, scales and is occluded with the scene.

Reply with one JSON object and nothing else, e.g. {"1": "overlay", "2": "scene"}. No explanation."""

def letters(t): return len(re.sub(r"[^A-Za-z\u4e00-\u9fff]", "", t))

def cands(d):
    W, H = d["W"], d["H"]
    out = []
    for o in d.get("ocr", []):
        b = o["box"]
        if o["conf"] >= 0.85 and letters(o["text"]) >= 5:
            # one box per place: clusters that overlap are the same text seen in different frames
            for q in out:
                ix = max(0, min(q["box"][2], b[2]) - max(q["box"][0], b[0])) * max(0, min(q["box"][3], b[3]) - max(q["box"][1], b[1]))
                if ix > 0.3 * min((q["box"][2] - q["box"][0]) * (q["box"][3] - q["box"][1]), (b[2] - b[0]) * (b[3] - b[1])):
                    q["box"] = [min(q["box"][0], b[0]), min(q["box"][1], b[1]), max(q["box"][2], b[2]), max(q["box"][3], b[3])]
                    q["n"] = max(q["n"], o["n"]); q["text"] += " | " + o["text"]
                    break
            else:
                out.append(dict(o))
    return out[:6]

def frames(f, N, W, H):
    idx24 = np.linspace(0, max(N, 2) - 1, 24).round().astype(int)
    oi = [idx24[int(round(i))] for i in np.linspace(2, 21, 6)]
    sel = "+".join(f"eq(n\\,{i})" for i in oi)
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-vf", f"select={sel}", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       stdout=subprocess.PIPE).stdout
    n = len(b) // (W * H * 3)
    return np.frombuffer(b[: n * W * H * 3], np.uint8).reshape(n, H, W, 3)

def tile(F, boxes, W, H):
    tw = 640 if W >= H else 360
    sc = tw / W; th = int(round(H * sc))
    ims = []
    for f in F:
        im = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        for k, b in enumerate(boxes, 1):
            p0, p1 = (int(b[0] * sc) - 2, int(b[1] * sc) - 2), (int(b[2] * sc) + 2, int(b[3] * sc) + 2)
            cv2.rectangle(im, p0, p1, (0, 0, 255), 2)
            cv2.putText(im, str(k), (p0[0], max(14, p0[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        ims.append(im)
    while len(ims) < 6: ims.append(np.zeros_like(ims[0]))
    cols = 3 if W >= H else 6
    rows = [np.hstack(ims[i:i + cols]) for i in range(0, 6, cols)]
    return np.vstack(rows)

def ask(img):
    b = base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()).decode()
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 120, "response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}},
                                                      {"type": "text", "text": PROMPT}]}]}
    for _ in range(3):
        try:
            return requests.post(URL, json=body, timeout=300).json()["choices"][0]["message"]["content"]
        except Exception as e:
            err = str(e); time.sleep(5)
    return "ERROR " + err

def main():
    dets, out = sys.argv[1].split(","), sys.argv[2]
    todo = []
    for p in dets:
        for l in open(p):
            d = json.loads(l)
            if "error" in d: continue
            c = cands(d)
            if c: todo.append((d, c))
    done = set()
    if os.path.exists(out): done = {json.loads(l)["id"] for l in open(out)}
    todo = [t for t in todo if t[0]["id"] not in done]
    print("candidates", len(todo), flush=True)
    os.makedirs(TMP, exist_ok=True)
    q = queue.Queue(maxsize=2)
    def pre():
        for i in range(0, len(todo), 256):
            ch = todo[i:i + 256]; dd = f"{TMP}/b{i}"; os.makedirs(dd, exist_ok=True)
            open(f"{dd}/cmds", "w").write("".join(f"cp {d['target']} {dd}/{k}.mp4\n" for k, (d, _) in enumerate(ch)))
            subprocess.run([STORE, "--numworkers", "16", "run", f"{dd}/cmds"], capture_output=True)
            q.put((ch, dd))
        q.put(None)
    threading.Thread(target=pre, daemon=True).start()
    fo = open(out, "a"); n = 0; t0 = time.time()
    def job(a):
        k, (d, c), dd = a
        f = f"{dd}/{k}.mp4"
        if not os.path.exists(f): return {"id": d["id"], "error": "download"}
        F = frames(f, d["frames"], d["W"], d["H"])
        if len(F) == 0: return {"id": d["id"], "error": "decode"}
        t = ask(tile(F, [o["box"] for o in c], d["W"], d["H"]))
        m = re.search(r"\{.*\}", t, re.S); v = None
        try: v = json.loads(m.group(0)) if m else None
        except Exception: v = None
        return {"id": d["id"], "boxes": [{"box": o["box"], "text": o["text"], "n": o["n"], "conf": o["conf"],
                                          "verdict": (v or {}).get(str(i)) } for i, o in enumerate(c, 1)], "raw": None if v else t}
    with cf.ThreadPoolExecutor(64) as ex:
        while True:
            it = q.get()
            if it is None: break
            ch, dd = it
            for r in ex.map(job, [(k, x, dd) for k, x in enumerate(ch)]):
                fo.write(json.dumps(r) + "\n"); n += 1
            fo.flush(); subprocess.run(["rm", "-rf", dd])
            print(f"vlm2 {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("VLM2_DONE", n, flush=True)

if __name__ == "__main__":
    main()
