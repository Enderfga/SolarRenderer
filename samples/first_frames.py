"""first_frames.py <source> [<source> ...] (CPU only): the first frame of every clip the audit kept, as PNG, for the
first-frame input. Reads $SOLAR_ROOT/solar_audit/<source>.jsonl (keep) and the catalog (the target clip), writes
$SOLAR_ROOT/solar_first_frame/<source>/<name>.png (frame 0 of the target, full resolution - the encoder puts it on
the canvas) and a manifest <source>.jsonl. 32 workers: the store sees at most 32 requests at a time.
Skips frames already written, so it can be rerun after the audit's keep list changes.

Environment:
    SOLAR_ROOT   root of the raw data (required), e.g. <storage>/raw/omni
    STORAGE_CLI     storage command (default: solar-storage)
    SOLAR_WORK      local scratch directory (default: ./solar_work/first_frames)
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
R = os.environ["SOLAR_ROOT"].rstrip("/")
OUT = f"{R}/solar_first_frame"
WORK = os.path.join(os.environ.get("SOLAR_WORK", "solar_work"), "first_frames")


def store_cat(url):
    return [json.loads(l) for l in subprocess.run([STORE, "cat", url], capture_output=True, text=True).stdout.splitlines()
            if l.strip()]


def done(source):
    out = subprocess.run([STORE, "ls", f"{OUT}/{source}/*.png"], capture_output=True, text=True).stdout
    return {l.split()[-1][:-4] for l in out.splitlines() if l.strip()}


def one(job):
    name, target, source = job
    flat = name.replace("/", "__")  # a clip name may hold a folder
    src, png = f"{WORK}/{flat}.mp4", f"{WORK}/{flat}.png"
    try:
        subprocess.run([STORE, "cp", target, src], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", src, "-frames:v", "1", png], check=True)
        subprocess.run([STORE, "cp", png, f"{OUT}/{source}/{name}.png"], check=True, capture_output=True)
        return name, True
    except Exception:
        return name, False
    finally:
        for f in (src, png):
            if os.path.exists(f):
                os.remove(f)


def main():
    os.makedirs(WORK, exist_ok=True)
    for source in sys.argv[1:]:
        keep = {r["id"] for r in store_cat(f"{R}/solar_audit/{source}.jsonl") if r.get("keep")}
        cat = {r["id"]: r["streams"][r["final"]] for r in store_cat(f"{R}/solar_catalog/{source}.jsonl")}
        have = done(source)
        jobs = [(cid.split("/", 1)[1], cat[cid], source) for cid in sorted(keep) if cid in cat]
        todo = [j for j in jobs if j[0] not in have]
        print(source, "kept", len(keep), "todo", len(todo), flush=True)
        failed = []
        with ThreadPoolExecutor(32) as pool:
            for k, (name, ok) in enumerate(pool.map(one, todo)):
                if not ok:
                    failed.append(name)
                if k % 2000 == 0:
                    print(source, k, "/", len(todo), flush=True)
        rows = [{"id": f"{source}/{name}", "target": target, "first_frame": f"{OUT}/{source}/{name}.png"}
                for name, target, _ in jobs if name not in failed]
        local = f"{WORK}/{source}.jsonl"
        open(local, "w").writelines(json.dumps(r) + "\n" for r in rows)
        subprocess.run([STORE, "cp", local, f"{OUT}/{source}.jsonl"], check=True, capture_output=True)
        print(source, "written", len(rows), "failed", len(failed), flush=True)


if __name__ == "__main__":
    main()
