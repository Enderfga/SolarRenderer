"""pack_shards.py (CPU only): the released training clips as WebDataset shards, for the Hub.

    pack_shards.py <stage dir> [--src <storage>/...] [--sources magpie,carla,unrealzoo] [--per-shard 170] [--workers 8]

Reads the per-clip files pack_train.py wrote (<out>/train/<source>/<clip>/{render.mp4,condition.mp4,first_frame.png})
and its index (<out>/index/train_<k>.jsonl). Clips of a source are sorted by id and cut into shards of --per-shard
clips (about 2 GB). A shard holds <key>.render.mp4, <key>.condition.mp4, <key>.first_frame.png and <key>.json, the
clip's index line; the key is the clip id without its source. Writes <stage>/train/<source>/<source>-<k>.tar and
<stage>/train/index.jsonl (one line per clip with its shard). A shard already in the stage dir is skipped.

Environment: STORAGE_CLI (storage command, default ``solar-storage``), SOLAR_RELEASE_ROOT (default for --src, the --out of
pack_train.py), SOLAR_SHM (scratch dir for a shard's files, default /dev/shm).
"""
import argparse, io, json, os, subprocess, tarfile, tempfile
from concurrent.futures import ProcessPoolExecutor

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
OUT = os.environ.get("SOLAR_RELEASE_ROOT")
SHM = os.environ.get("SOLAR_SHM", "/dev/shm")
FILES = ("render.mp4", "condition.mp4", "first_frame.png")


def store(*argv, text=False):
    return subprocess.run([STORE, *argv], capture_output=True, text=text, check=True).stdout


def add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def pack(job):
    path, rows, src = job
    with tempfile.TemporaryDirectory(dir=SHM) as tmp:
        with open(f"{tmp}/cmds", "w") as fh:
            for r in rows:
                key = r["clip_id"].split("/", 1)[1]
                for f in FILES:
                    fh.write(f"cp {src}/{r['files'][f]} {tmp}/{key}.{f}\n")
        store("run", f"{tmp}/cmds")
        part = path + ".part"
        with tarfile.open(part, "w") as tar:
            for r in rows:
                key = r["clip_id"].split("/", 1)[1]
                for f in FILES:
                    add(tar, f"{key}.{f}", open(f"{tmp}/{key}.{f}", "rb").read())
                meta = {k: v for k, v in r.items() if k != "files"}
                add(tar, f"{key}.json", json.dumps(meta, ensure_ascii=False).encode())
        os.replace(part, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--src", default=OUT, required=OUT is None,
                    help="release root pack_train.py wrote to (default: $SOLAR_RELEASE_ROOT)")
    ap.add_argument("--sources", default="magpie,carla,unrealzoo")
    ap.add_argument("--per-shard", type=int, default=170)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        store("cp", f"{a.src}/index/train_*.jsonl", tmp + "/")
        rows = [json.loads(l) for f in sorted(os.listdir(tmp)) for l in open(os.path.join(tmp, f)) if l.strip()]
    assert len({r["clip_id"] for r in rows}) == len(rows), "duplicate clip ids in the index"
    for r in rows:  # the CARLA build used was 0.9.16
        if "license" in r:
            r["license"] = r["license"].replace("CARLA 0.10 ", "CARLA 0.9.16 ")
    jobs, index = [], []
    for source in a.sources.split(","):
        clips = sorted((r for r in rows if r["source"] == source), key=lambda r: r["clip_id"])
        os.makedirs(f"{a.stage}/train/{source}", exist_ok=True)
        for k in range(0, len(clips), a.per_shard):
            shard = f"train/{source}/{source}-{k // a.per_shard:05d}.tar"
            part = []
            for r in clips[k:k + a.per_shard]:
                line = {key: v for key, v in r.items() if key != "files"}
                line["shard"] = shard
                part.append(line | {"files": r["files"]})
                index.append(line)
            if not os.path.exists(f"{a.stage}/{shard}"):
                jobs.append((f"{a.stage}/{shard}", part, a.src))
        print(f"{source}: {len(clips)} clips, {-(-len(clips) // a.per_shard)} shards", flush=True)
    print(f"{len(jobs)} shards to pack", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for n, path in enumerate(ex.map(pack, jobs), 1):
            print(f"{n}/{len(jobs)} {path}", flush=True)
    with open(f"{a.stage}/train/index.jsonl", "w") as fh:
        fh.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in sorted(index, key=lambda l: l["clip_id"]))
    print(f"index: {len(index)} lines", flush=True)


if __name__ == "__main__":
    main()
