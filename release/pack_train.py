"""pack_train.py (CPU only): the released SolarRenderer training clips, as the encoder saw them.

    pack_train.py <shard k> <shards n> --index <storage>/.../train.jsonl --enc <storage>/.../enc_solar_v2.jsonl
                  [--out <storage>/...] [--preencode tools/omni_preencode.py] [--per-source N] [--workers W] [--verify]

Release scope: Magpie (Apache-2.0), CARLA and UnrealZoo (rendered by us) go out as clips; the clips of the
other sources do not.

Reads the final run's training index (--index) joined with its encode list (--enc). Every clip gets render.mp4,
condition.mp4 and first_frame.png, made by the encoder's own functions (tools/omni_preencode.py of the main branch:
canvas, read_video, fit, prepared, aligned_image) - on 24 fps, the first 158 frames, LANCZOS to 1344x768 - so before
the mp4 encode the pixels are the ones the model trained on. The condition is written for every clip, also where
training gave only the first frame or only text; the index says which inputs training gave (``training_input``) and
the prompt is the one written for that combination.

Output: <out>/train/<source>/<clip>/{render.mp4,condition.mp4,first_frame.png}, one index line per clip in
<out>/index/train_<k>.jsonl, failures in <out>/index/failed_<k>.jsonl. A clip already in the shard's index is skipped,
so a restarted shard goes on where it stopped.

Environment: STORAGE_CLI (storage command, default ``solar-storage``), SOLAR_RELEASE_ROOT (default for --out), SOLAR_PREENCODE
(default for --preencode), SOLAR_PYTHONPATH (extra import dirs for omni_preencode.py, os.pathsep-separated), SOLAR_SOURCE_ALIAS (``catalog=name,...``:
the name a catalog source is released under).
"""
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile, traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
OUT = os.environ.get("SOLAR_RELEASE_ROOT")
PREENCODE = os.environ.get("SOLAR_PREENCODE", "tools/omni_preencode.py")
NUM_FRAMES = 158

SOURCES = {
    "magpie": {"condition": "scene grey box (engine white-box)",
               "cond": lambda t: t.replace("_render.mp4", "_white_box.mp4"),
               "license": "Apache-2.0, from Magpie (MogoAI/Magpie_lite)",
               "upstream": "https://huggingface.co/datasets/MogoAI/Magpie_lite"},
    "carla": {"condition": "semantic segmentation (engine export)",
              "cond": lambda t: t.replace("_rgb.mp4", "_semantic.mp4"),
              "license": "rendered by us in CARLA 0.9.16 (code MIT, assets CC-BY); attribute CARLA",
              "upstream": "https://carla.org"},
    "unrealzoo": {"condition": "object masks + character skeletons (engine export)",
                  "cond": lambda t: t.replace("/clips/", "/solar/").replace("_render.mp4", "_maskskel.mp4"),
                  "license": "rendered by us in UnrealZoo environments (code Apache-2.0); research use only",
                  "upstream": "https://unrealzoo.github.io"},
    # benchmark only (pack_heldout.py): the held-out engine-recording clips; their training clips are not released
    "game": {"condition": "near-field shaded geometry (depth within 30 m)",
             "cond": None, "license": "research use only", "upstream": None},
}
ALIAS = dict(x.split("=", 1) for x in os.environ.get("SOLAR_SOURCE_ALIAS", "").split(",") if x)


def alias(source: str) -> str:
    """A catalog source path (``name/...``) under the name it is released as."""
    head, _, rest = source.partition("/")
    return ALIAS.get(head, head) + (f"/{rest}" if rest else "")
TRAIN_SOURCES = ("magpie", "carla", "unrealzoo")
TRAINING_INPUT = {"full": "structure + first frame + text", "structure": "structure + text",
                  "first": "first frame + text", "text": "text only"}
CRF = {"render": 18, "condition": 15}   # structure maps carry hard label edges: a lower CRF keeps them


def store(*argv):
    subprocess.run([STORE, *argv], check=True, capture_output=True)


def store_cat(url):
    out = subprocess.run([STORE, "cat", url], capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def set_preencode(path):
    """Record where omni_preencode.py is; the environment carries it into the worker processes."""
    os.environ["SOLAR_PREENCODE"] = os.path.abspath(path)


def load_preencode():
    path = os.path.abspath(os.environ.get("SOLAR_PREENCODE", PREENCODE))
    # omni_preencode.py imports from the main branch's src/ (its sibling of tools/), plus any extra dirs given
    extra = [p for p in os.environ.get("SOLAR_PYTHONPATH", "").split(os.pathsep) if p]
    sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(path)), "src"), *extra]
    spec = importlib.util.spec_from_file_location("omni_preencode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_mp4(frames: np.ndarray, path: str, crf: int) -> None:
    n, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "24",
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", path]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def pack(job):
    row, out, verify = job[:3]
    sub = job[3] if len(job) > 3 else "train"     # pack_heldout.py writes the held-out clips under benchmark/
    P = load_preencode() if "P" not in globals() else globals()["P"]
    globals()["P"] = P
    from PIL import Image

    source = alias(row["source"]).split("/")[0]
    variant = row["source"].split("/")[2]
    spec = SOURCES[source]
    clip = job[4] if len(job) > 4 and job[4] else row["catalog_id"].split("/", 1)[1]   # pack_heldout.py may rename
    target = row["target"]
    cond = next((o["media"] for o in row.get("observations", []) if o["media"].endswith(".mp4")),
                spec["cond"](target) if spec["cond"] else None)   # None: the clip has no structure stream
    local = []
    try:
        shape = P.canvas(target)
        frames, fps = P.read_video(target, NUM_FRAMES / P.FPS + 0.5)
        render = P.prepared(P.fit(frames, shape), fps, NUM_FRAMES)
        condition = None
        if cond:
            frames, fps = P.read_video(cond, NUM_FRAMES / P.FPS + 0.5)
            condition = P.prepared(P.fit(frames, shape), fps, NUM_FRAMES)
            # the encoder keeps the condition's frames that line up with the target's, and no more
            condition = condition[: len(render)]
        del frames
        # Frame 0 of the released render. Training's own first-frame image is the same for CARLA and UnrealZoo;
        # for Magpie (60 fps) it is raw frame 0 while the 24 fps grid starts at raw frame 1, 1/60 s later.
        first = render[0]
        local = [P.localize(p) for p in (target, cond) if p]
        tmp = tempfile.mkdtemp(prefix="pack_")
        write_mp4(render, f"{tmp}/render.mp4", CRF["render"])
        if condition is not None:
            write_mp4(condition, f"{tmp}/condition.mp4", CRF["condition"])
        Image.fromarray(first).save(f"{tmp}/first_frame.png", optimize=True)
        check = {}
        if verify:   # what the mp4 encode cost, and that frame 0 of the render is the given first frame
            back, _ = P.read_video(f"{tmp}/render.mp4")
            cback, _ = P.read_video(f"{tmp}/condition.mp4") if condition is not None else (render, None)
            psnr = lambda a, b: float(10 * np.log10(255 ** 2 / max(np.mean((a.astype(np.float32) - b) ** 2), 1e-9)))
            check = {"render_psnr": round(psnr(back[: len(render)], render), 2),
                     "condition_psnr": round(psnr(cback[: len(condition)], condition), 2) if condition is not None else None,
                     "frames_back": [int(len(back)), int(len(cback))]}
        dest = f"{out}/{sub}/{source}/{clip}"
        store("cp", f"{tmp}/*", f"{dest}/")
        shutil.rmtree(tmp, ignore_errors=True)
        line = {"clip_id": f"{source}/{clip}", "source": source, "condition": spec["condition"],
                "training_input": TRAINING_INPUT[variant], "prompt": row["text"],
                "frames": int(len(render)), "condition_frames": int(len(condition)) if condition is not None else 0,
                "height": int(render.shape[1]), "width": int(render.shape[2]), "fps": 24,
                "files": {k: f"{sub}/{source}/{clip}/{k}" for k in ("render.mp4", "condition.mp4", "first_frame.png")
                          if condition is not None or k != "condition.mp4"},
                "license": spec["license"], "upstream": spec["upstream"]}
        return {"ok": line, "check": check}
    except Exception:
        return {"failed": {"clip_id": f"{source}/{clip}", "error": traceback.format_exc()[-1500:]}}
    finally:
        for p in local:
            try:
                os.remove(p)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("--index", required=True, help="the final run's training index (train.jsonl, one row per id)")
    ap.add_argument("--enc", required=True, help="the encode list the index was built from (enc_solar_v2.jsonl)")
    ap.add_argument("--per-source", type=int, default=0, help="pilot: first N clips of each source (by id hash)")
    ap.add_argument("--out", default=OUT, required=OUT is None, help="release root (default: $SOLAR_RELEASE_ROOT)")
    ap.add_argument("--preencode", default=PREENCODE,
                    help="path to the main branch's tools/omni_preencode.py (default: $SOLAR_PREENCODE or "
                         "tools/omni_preencode.py)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    set_preencode(a.preencode)

    train_ids = {r["id"] for r in store_cat(a.index)}
    rows = []
    with tempfile.TemporaryFile("w+") as fh:
        subprocess.run([STORE, "cat", a.enc], stdout=fh, check=True)
        fh.seek(0)
        for l in fh:
            r = json.loads(l)
            if r["source"].split("/")[0] in TRAIN_SOURCES and r["id"] in train_ids:
                rows.append(r)
    rows.sort(key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())
    if a.per_source:
        seen = {}
        rows = [r for r in rows if seen.setdefault(r["source"].split("/")[0], []).append(1) or
                len(seen[r["source"].split("/")[0]]) <= a.per_source]
    rows = rows[a.k :: a.n]
    idx_url, fail_url = f"{a.out}/index/train_{a.k}.jsonl", f"{a.out}/index/failed_{a.k}.jsonl"
    done = {l["clip_id"] for l in store_cat(idx_url)} if subprocess.run([STORE, "ls", idx_url], capture_output=True).returncode == 0 else set()
    todo = [r for r in rows if f"{r['source'].split('/')[0]}/{r['catalog_id'].split('/', 1)[1]}" not in done]
    print(f"shard {a.k}/{a.n}: {len(rows)} clips, {len(done)} done, {len(todo)} to pack", flush=True)
    tmpdir = tempfile.gettempdir()
    idx_local, fail_local = f"{tmpdir}/pack_index_{a.k}.jsonl", f"{tmpdir}/pack_failed_{a.k}.jsonl"
    if done:
        store("cp", idx_url, idx_local)
    checks = []
    with ProcessPoolExecutor(a.workers) as pool, open(idx_local, "a") as idx, open(fail_local, "a") as fail:
        for i, res in enumerate(pool.map(pack, [(r, a.out, a.verify) for r in todo], chunksize=1)):
            if "ok" in res:
                idx.write(json.dumps(res["ok"], ensure_ascii=False) + "\n")
                if res["check"]:
                    checks.append({"clip_id": res["ok"]["clip_id"], **res["check"]})
            else:
                fail.write(json.dumps(res["failed"]) + "\n")
            if (i + 1) % 200 == 0 or i + 1 == len(todo):
                idx.flush(); fail.flush()
                store("cp", idx_local, idx_url)
                store("cp", fail_local, fail_url)
                print(f"  {i + 1}/{len(todo)}", flush=True)
    if checks:
        print(json.dumps(checks, indent=0))


if __name__ == "__main__":
    main()
