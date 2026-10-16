"""uz_v2_prep.py (CPU; rerun any time): for every accepted UnrealZoo clip (its json in clips/ is written last), make
the SolarRenderer condition - the object mask with the skeleton drawn over it, one video - and the first frame of the
render, then write the manifest the training-sample builder reads. Skips what is already made.

    <out>/<clip>_maskskel.mp4      condition (skeleton keyed off black, laid over the mask)
    <first_frames>/<clip>.png      frame 0 of the render
    <out>/manifest.jsonl           {id: unrealzoo/<clip>, target, condition, first_frame, map, camera}

usage: uz_v2_prep.py --clips <storage>/.../unrealzoo/clips --out <storage>/.../unrealzoo/solar --first-frames <storage>/.../first_frame
                     [--work DIR]      (STORAGE_CLI = storage command, default solar-storage)
"""
import argparse, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
SRC = OUT = FF = WORK = None


def ls(url):
    out = subprocess.run([STORE, "ls", url], capture_output=True, text=True).stdout
    return [l.split()[-1] for l in out.splitlines() if l.strip()]


def one(clip):
    d = f"{WORK}/{clip}"
    os.makedirs(d, exist_ok=True)
    try:
        for k in ("mask", "skeleton", "render"):
            subprocess.run([STORE, "cp", f"{SRC}/{clip}_{k}.mp4", f"{d}/{k}.mp4"], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", f"{d}/mask.mp4", "-i", f"{d}/skeleton.mp4",
                        "-filter_complex", "[1:v]colorkey=0x000000:0.08:0.0[s];[0:v][s]overlay=format=auto,format=yuv420p[v]",
                        "-map", "[v]", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-an", f"{d}/c.mp4"], check=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", f"{d}/render.mp4", "-frames:v", "1", f"{d}/f.png"], check=True)
        subprocess.run([STORE, "cp", f"{d}/c.mp4", f"{OUT}/{clip}_maskskel.mp4"], check=True, capture_output=True)
        subprocess.run([STORE, "cp", f"{d}/f.png", f"{FF}/{clip}.png"], check=True, capture_output=True)
        return clip, True
    except Exception:
        return clip, False
    finally:
        subprocess.run(["rm", "-rf", d])


def main():
    global SRC, OUT, FF, WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="storage prefix of the recorder clips: <uz_prod.py --dest prefix>/clips")
    ap.add_argument("--out", required=True, help="storage prefix for the conditions and the manifest")
    ap.add_argument("--first-frames", required=True, help="storage prefix for the first-frame PNGs")
    ap.add_argument("--work", default="uz_prep", help="local scratch directory")
    a = ap.parse_args()
    SRC, OUT, FF, WORK = a.clips.rstrip("/"), a.out.rstrip("/"), a.first_frames.rstrip("/"), a.work
    os.makedirs(WORK, exist_ok=True)
    clips = sorted(n[:-5] for n in ls(f"{SRC}/*.json"))
    made = {n[: -len("_maskskel.mp4")] for n in ls(f"{OUT}/*_maskskel.mp4")}
    have_ff = {n[:-4] for n in ls(f"{FF}/*.png")}
    todo = [c for c in clips if c not in made or c not in have_ff]
    print("accepted", len(clips), "todo", len(todo), flush=True)
    with ThreadPoolExecutor(16) as pool:
        failed = [c for c, ok in pool.map(one, todo) if not ok]
    rows = []
    for c in clips:
        if c in failed:
            continue
        rows.append({"id": f"unrealzoo/{c}", "target": f"{SRC}/{c}_render.mp4", "condition": f"{OUT}/{c}_maskskel.mp4",
                     "first_frame": f"{FF}/{c}.png", "map": c.split("_s")[0][3:], "camera": c.rsplit("_", 1)[1]})
    open(f"{WORK}/manifest.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows)
    subprocess.run([STORE, "cp", f"{WORK}/manifest.jsonl", f"{OUT}/manifest.jsonl"], check=True, capture_output=True)
    print("manifest", len(rows), "failed", len(failed), flush=True)


if __name__ == "__main__":
    main()
