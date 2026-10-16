"""free_caption.py: write the free parts of SolarRenderer prompts with a vision-language model.

    python free_caption.py clips.jsonl out_dir [--server URL] [--model NAME] [--workers N] [--window 6.583]

clips.jsonl has one {"id": ..., "video": <local path, http(s) URL or object-store URL>} per line. For every clip the model sees
the finished video only - the first `window` seconds (the 158 frames SolarRenderer generates), at 2 fps, longer side
672 - and answers with JSON: summary, detailed_description (350-500 words), overall_soundscape, non_diegetic_music.
The answer is checked (check.py: fields, length, English only, no input labels, shot cuts); a failing answer is sent
back with the problems, up to three times. out_dir/<id>.json holds the fields, the checks and the model's usage; a clip
that still fails goes to out_dir/_failed/<id>.json. Clips already written are skipped, so a run can be restarted.

The server is any OpenAI-compatible chat endpoint that accepts images. We used Kimi K2.6 on vLLM 0.30 (TP8 on one
8-GPU node: `vllm serve moonshotai/Kimi-K2.6 --tensor-parallel-size 8 --trust-remote-code --max-model-len 65536`),
about 4,000 clips an hour at 96 concurrent requests. The video goes in as timestamped frames, since vLLM's chat path
takes no video for Kimi K2.6.
"""
import argparse, base64, glob, json, os, shutil, subprocess, sys, tempfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check import check_free, problems  # noqa: E402

SYSTEM = """You write the training caption of a short video clip for a video generation model. The video model will be given your text together with other inputs (such as the first frame or a structure video) that are attached later; your text must describe the finished clip completely and concretely on its own.

Answer with ONE JSON object and nothing else:
{
  "subject": "<if one main character (person, creature or vehicle) is clearly the centre of the clip: a short identity of a few words, e.g. 'the runner in the orange sleeveless hoodie'; otherwise null>",
  "summary": "<one short paragraph: who is where doing what, and how the camera moves>",
  "detailed_description": "<350-500 words, see below>",
  "overall_soundscape": "<1-4 sentences: ambience and physical sounds across the whole clip, no dialogue>",
  "non_diegetic_music": "<background score audible only to the audience; you cannot hear the clip, so write N/A unless the clip is clearly a scored trailer>"
}

detailed_description:
- First one or two sentences on the visual style and light, e.g. "The target video is a realistic 3D CG game render in bright midday sun with crisp shadows." Then "[Shot 1]" followed by the composition, subjects and scene of the first frame, then what happens next in playback order (first-frame state, action onset, continuous development, result).
- 350-500 English words in total (count them; aim for about 420).
- Carry everything a structure-only rendering would lack, precisely: materials and surfaces with their colours and wear; lighting (direction, colour, hardness, time of day, shadows, reflections, emissive lights); weather and atmosphere; effects (sparks, dust, smoke, fire, particles, glows, motion blur); characters' full appearance (outfit pieces with colours and materials, gear, hair, skin, weapons); small objects; events and state changes; camera motion.
- Events in natural playback order, timed in words ("midway", "near the end"). Do NOT write timestamps inside a shot ("From 00:01.000 to 00:02.000" and "At about 00:06.000" are wrong). A timestamp appears only at a hard cut, which starts a new shot: "[Shot 2] At 00:03.500, the shot cuts to ...", times strictly increasing and inside the clip length. Most clips are a single shot: then write [Shot 1] only.
- Camera motion as natural English: type (static, pan, tilt, truck, push in, pull out, tracking, arc, pedestal, shake, POV) plus amplitude and speed when they matter ("the camera tracks behind him at fast speed").
- Do not use any reference labels such as <Picture 1>, <Video 1> or <Subject 1>; they are added later.
- English only. Text visibly written in the scene is quoted exactly in double quotes in its original language ("营业中"); nothing else may be in another language.
- Describe only what is visible; no guesses about story, names or intentions, no mood words standing in for visible detail, no mention of datasets, game titles you are guessing, or the video model."""

RETRIES = 3
LOCK = threading.Lock()
STATS = {"ok": 0, "failed": 0, "error": 0, "t0": time.time()}


def fetch(src: str, dst: str) -> None:
    if src.startswith(("http://", "https://")):
        urllib.request.urlretrieve(src, dst)
    elif "://" in src:  # an object-store URL, fetched with the storage client
        subprocess.run([os.environ.get("STORAGE_CLI", "solar-storage"), "cp", src, dst], check=True, capture_output=True)
    else:
        shutil.copy(src, dst)


def frames(path: str, window: float, fps: float = 2.0) -> tuple[list, float]:
    """The clip's first `window` seconds as timestamped 2 fps frames (longer side 672), and the length described."""
    tmp = tempfile.mkdtemp()
    try:
        vf = f"fps={fps},scale='if(gte(iw,ih),672,-2)':'if(gte(iw,ih),-2,672)',setsar=1"
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", path, "-t", str(window), "-vf", vf,
                        "-q:v", "3", f"{tmp}/%04d.jpg"], check=True)
        parts = []
        for i, f in enumerate(sorted(glob.glob(f"{tmp}/*.jpg"))):
            t = i / fps
            parts.append({"type": "text", "text": f"{int(t // 60):02d}:{t % 60:06.3f}"})
            parts.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(open(f, "rb").read()).decode()}})
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    path], capture_output=True, text=True).stdout.strip())
        return parts, round(min(dur, window), 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ask(a, messages):
    body = {"model": a.model, "messages": messages, "temperature": 0.6, "top_p": 0.95, "max_tokens": 3000,
            "response_format": {"type": "json_object"}, "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(a.server, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read())
    return j["choices"][0]["message"]["content"] or "", j.get("usage", {})


def one(a, row):
    cid = row["id"]
    out_ok, out_bad = f"{a.out}/{cid}.json", f"{a.out}/_failed/{cid}.json"
    tmp = tempfile.mkdtemp()
    try:
        local = f"{tmp}/clip{os.path.splitext(row['video'])[1] or '.mp4'}"
        fetch(row["video"], local)
        video, duration = frames(local, a.window)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [{"type": "text", "text": f"The clip ({duration:.2f} s):"}] + video +
                     [{"type": "text", "text": (row.get("note", "") + "\n" if row.get("note") else "") + "Write the JSON."}]}]
        attempts, fields, checks = [], None, None
        for _ in range(1 + RETRIES):
            text, usage = ask(a, messages)
            attempts.append(usage)
            try:
                fields, checks = check_free(json.loads(text), duration)
                bad = problems(checks)
            except (ValueError, TypeError, AttributeError) as e:
                fields, checks, bad = None, {"pass": False}, [f"not the JSON object asked for ({e})"]
            if not bad:
                break
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Fix these problems and answer with the whole JSON again: " + "; ".join(bad)}]
        ok = bool(checks and checks.get("pass"))
        rec = {"id": cid, "video": row["video"], "duration": duration, "model": a.model, "window_s": a.window,
               "fields": fields, "checks": checks, "usage": attempts}
        dest = out_ok if ok else out_bad
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        json.dump(rec, open(dest + ".part", "w"), ensure_ascii=False)
        os.replace(dest + ".part", dest)
        with LOCK:
            STATS["ok" if ok else "failed"] += 1
    except Exception as e:  # one clip must not stop the run
        with LOCK:
            STATS["error"] += 1
        print("error", cid, type(e).__name__, str(e)[:200], flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("--server", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="kimi")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--window", type=float, default=6.583, help="seconds described: 158 frames at 24 fps")
    ap.add_argument("--shard", default="0/1", help="k/n: this process takes every n-th clip from the k-th")
    a = ap.parse_args()
    k, n = map(int, a.shard.split("/"))
    rows = [json.loads(l) for l in open(a.clips) if l.strip()][k::n]
    done = {os.path.basename(p)[:-5] for p in glob.glob(f"{a.out}/*.json") + glob.glob(f"{a.out}/_failed/*.json")}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} clips, {len(todo)} to caption", flush=True)

    def report():
        while True:
            time.sleep(300)
            s = dict(STATS); el = time.time() - s["t0"]
            print(time.strftime("%T"), f"ok={s['ok']} failed={s['failed']} error={s['error']} "
                  f"rate={(s['ok'] + s['failed']) / el * 3600:.0f}/h", flush=True)
    threading.Thread(target=report, daemon=True).start()
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(lambda r: one(a, r), todo))
    print("done", STATS, flush=True)


if __name__ == "__main__":
    main()
