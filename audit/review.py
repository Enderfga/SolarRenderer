"""review.py PICK.jsonl OUTDIR NPROC: one triptych mp4 per picked clip -> OUTDIR/<prefix>.mp4
   final | condition | condition edges (red) over the final.   PICK rows: {"id", "final", "cond", "file"}
   env: STORAGE_CLI (default "solar-storage"), REVIEW_TMP (local scratch, default <system temp>/solar_review)"""
import sys, os, json, subprocess, shutil, tempfile
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, cv2
cv2.setNumThreads(1)
STORE = os.environ.get("STORAGE_CLI", "solar-storage")
TMP = os.environ.get("REVIEW_TMP", os.path.join(tempfile.gettempdir(), "solar_review"))
W = 448


def read(f, w, h, step):
    b = subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", f, "-vf", f"select='not(mod(n\\,{step}))',scale={w}:{h}:flags=area",
                        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    n = len(b) // (w * h * 3)
    return np.frombuffer(b[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)


def one(job):
    out = job["out"]
    if os.path.exists(out):
        return out
    d = f"{TMP}/{os.path.basename(out)}"
    os.makedirs(d, exist_ok=True)
    try:
        for n, u in (("f.mp4", job["final"]), ("c.mp4", job["cond"])):
            subprocess.run([STORE, "--log", "error", "cp", "--concurrency", "1", u, f"{d}/{n}"], capture_output=True)
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", f"{d}/f.mp4"],
                            capture_output=True, text=True).stdout
        s = json.loads(pr)["streams"][0]
        fps = eval(s["r_frame_rate"])
        step = 2 if fps > 40 else 1
        h = int(round(W * s["height"] / s["width"] / 2) * 2)
        F = read(f"{d}/f.mp4", W, h, step); C = read(f"{d}/c.mp4", W, h, step)
        T = min(len(F), len(C))
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{3 * W}x{h}", "-r", f"{fps / step:g}", "-i", "-",
                              "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
        for t in range(T):
            g = cv2.cvtColor(C[t], cv2.COLOR_RGB2GRAY)
            v = np.median(g)
            e = cv2.Canny(g, int(max(10, 0.66 * v)), int(min(255, max(30, 1.33 * v)))) > 0
            ov = F[t].copy(); ov[e] = (255, 40, 40)
            p.stdin.write(np.concatenate([F[t], C[t], ov], 1).tobytes())
        p.stdin.close(); p.wait()
        return out
    except Exception as ex:
        return f"ERR {out} {ex!r}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    picks = [json.loads(l) for l in open(sys.argv[1])]
    od = sys.argv[2]; os.makedirs(od, exist_ok=True)
    for j in picks:
        j["out"] = os.path.join(od, j["file"])
    with ProcessPoolExecutor(int(sys.argv[3])) as ex:
        for r in ex.map(one, picks):
            print(r, flush=True)
