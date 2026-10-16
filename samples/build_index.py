"""build_index.py: the final training index - every chosen encoded train clip once, the structure video read at the
full grid (no coarse_observations).

    build_index.py <out-prefix> [--check]

What is kept (each row the encode list holds, held-out units never in train):
  magpie, carla, unrealzoo   every variant (structure + first frame / structure / first / text)
  wildworld, omniworld_v3    first-frame-only rows only (their box / background-box conditions do not line up)
The training index also holds clips of sources that are not published; their rules are not part of this file.
Refuses to write unless every kept listed clip is encoded (--check only prints what is missing). Writes
<out>/train.jsonl, <out>/val.jsonl and <out>/report.json.

Environment:
    SOLAR_ROOT    root of the raw data (required), e.g. <storage>/raw/omni; the encode lists are read from
                     $SOLAR_ROOT/solar_train
    SOLAR_LATENTS    prefix of the encoded set (required), e.g. <storage>/latents/omni/solar_v2; members under
                     <prefix>/samples/, the held-out set under <prefix>_val/samples/
    STORAGE_CLI      storage command (default: solar-storage)
    SOLAR_WORK       local scratch directory (default: ./solar_work)
"""
import collections, hashlib, json, os, random, struct, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

import fsspec

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
PREFIX = os.environ["SOLAR_LATENTS"].rstrip("/")
T = os.environ["SOLAR_ROOT"].rstrip("/") + "/solar_train"
WORK = os.environ.get("SOLAR_WORK", "solar_work")
ALIGNED = ("magpie", "carla", "unrealzoo")


def header(url: str) -> dict:
    """The safetensors header of an encoded member: its first eight bytes give the header size."""
    with fsspec.open(url, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(size))


def members(prefix: str) -> list[str]:
    out = subprocess.run([STORE, "ls", f"{prefix}/samples/*.safetensors"], capture_output=True, text=True).stdout
    return [f"{prefix}/samples/{l.split()[-1]}" for l in out.splitlines() if l.strip()]


def row(member: str) -> dict:
    meta = json.loads(header(member)["__metadata__"]["omni"])
    return {"id": member.rsplit("/", 1)[1][: -len(".safetensors")], "member": member,
            "latent_frames": meta["latent_frames"], "canvas": meta["canvas"],
            "source": meta["source"], "observations": len(meta["observations"])}


CACHE = os.path.join(WORK, "index_cache.jsonl")  # rows already read: a member's header never changes once written


def rows(prefix: str) -> list[dict]:
    """The index row of every encoded member under prefix, read from its header."""
    known = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            if line.strip():
                r = json.loads(line); known[r["member"]] = r
    names = members(prefix)
    fresh = [m for m in names if m not in known]
    with ThreadPoolExecutor(64) as pool:
        got = list(pool.map(row, fresh))
    with open(CACHE, "a") as fh:
        for r in got:
            fh.write(json.dumps(r) + "\n")
    known.update({r["member"]: r for r in got})
    return [known[m] for m in names]


def h(text):
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def chosen(row):
    source, kind, variant = row["source"].split("/")
    if source in ALIGNED:
        return True
    if source in ("wildworld", "omniworld_v3"):
        return variant == "first"
    return False


def store_cat(url):
    out = subprocess.run([STORE, "cat", url], capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def main():
    os.makedirs(WORK, exist_ok=True)
    out = sys.argv[1].rstrip("/")
    check = "--check" in sys.argv
    ls = subprocess.run([STORE, "ls", f"{T}/val_solar_v2_*.jsonl"], capture_output=True, text=True).stdout
    val_ids = set()
    for name in [l.split()[-1] for l in ls.splitlines() if l.strip()]:
        val_ids |= {r["id"] for r in store_cat(f"{T}/{name}")}
    listed = {r["id"]: r["source"] for r in store_cat(f"{T}/enc_solar_v2.jsonl") if r["id"] not in val_ids and chosen(r)}
    encoded = {r["id"]: r for r in rows(PREFIX) if r["id"] in listed}
    ls = subprocess.run([STORE, "ls", f"{PREFIX}/failed/*"], capture_output=True, text=True).stdout
    failed = {l.split()[-1].rsplit(".", 1)[0] for l in ls.splitlines() if l.strip()}
    gave_up = {i for i in listed if i not in encoded and (i in failed or f"solar2_{i}" in failed)}
    missing = collections.Counter(src for i, src in listed.items() if i not in encoded and i not in gave_up)
    report = {"listed": len(listed), "encoded": len(encoded), "failed": len(gave_up), "missing": dict(sorted(missing.items()))}
    if check or missing:
        print(json.dumps(report, indent=1))
        if missing and not check:
            raise SystemExit("not every chosen listed clip is encoded yet")
        return
    train = sorted(encoded.values(), key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())
    random.Random(1008).shuffle(train)
    train = [{k: v for k, v in r.items() if k != "coarse_observations"} for r in train]
    val = [{k: v for k, v in r.items() if k != "coarse_observations"} for r in rows(PREFIX + "_val")
           if chosen(r)]
    for name, items in (("train", train), ("val", val)):
        path = f"{WORK}/index_{name}.jsonl"
        with open(path, "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in items)
        subprocess.run([STORE, "cp", path, f"{out}/{name}.jsonl"], check=True, capture_output=True)
    report["groups"] = dict(sorted(collections.Counter(r["source"] for r in train).items()))
    report["_total"] = {"train_rows": len(train), "val_rows": len(val)}
    print(json.dumps(report, indent=1))
    open(f"{WORK}/index_report.json", "w").write(json.dumps(report, indent=1))
    subprocess.run([STORE, "cp", f"{WORK}/index_report.json", f"{out}/report.json"], check=True, capture_output=True)


if __name__ == "__main__":
    main()
