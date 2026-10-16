"""make_jobs.py CATALOG_PREFIX OUTDIR [N]: audit jobs for magpie / carla from the SolarRenderer catalog
(CATALOG_PREFIX/<source>.jsonl, one row per clip with its stream urls), optionally a seeded sample of N.
Writes OUTDIR/<source>.jsonl. env: STORAGE_CLI (default "solar-storage")."""
import sys, json, subprocess, random, os
STORE = os.environ.get("STORAGE_CLI", "solar-storage")
CAT = sys.argv[1].rstrip("/")
out = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(out, exist_ok=True)
for src in ("magpie", "carla"):
    rows = [json.loads(l) for l in subprocess.run([STORE, "cat", f"{CAT}/{src}.jsonl"], capture_output=True, text=True).stdout.splitlines() if l.strip()]
    jobs = []
    for r in rows:
        fin = r["streams"][r["final"]]
        if src == "magpie":
            cond, extra = r["streams"].get("white_box"), {}
        else:
            cond = r["streams"].get("semantic")
            extra = {"meta": fin[: -len("_rgb.mp4")] + ".json"}
        jobs.append({"id": r["id"], "source": src, "final": fin, "cond": cond, "extra": extra, "seconds": r.get("seconds")})
    random.Random(0).shuffle(jobs)  # shuffled so every shard sees every scene
    if N:
        jobs = jobs[:N]
    with open(f"{out}/{src}.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(src, len(rows), "->", len(jobs), "missing cond:", sum(j["cond"] is None for j in jobs))
