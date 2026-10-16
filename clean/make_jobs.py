"""Job lists for detect.py. usage: make_jobs.py CATALOG_PREFIX OUTDIR SOURCE [SOURCE ...] [--sample N]
CATALOG_PREFIX/<source>.jsonl is the clip catalog of each source (one row per clip with its stream urls). Every row
becomes a job, or N random rows with --sample N (a spot check). Env: STORAGE_CLI (default "solar-storage")."""
import json, os, sys, random, subprocess
STORE = os.environ.get("STORAGE_CLI", "solar-storage")
args = sys.argv[1:]
n = 0
if "--sample" in args:
    i = args.index("--sample"); n = int(args[i + 1]); del args[i:i + 2]
CAT, out, sources = args[0].rstrip("/"), args[1], args[2:]
os.makedirs(out, exist_ok=True)
def cat(name): return [json.loads(l) for l in subprocess.run([STORE, "cat", f"{CAT}/{name}.jsonl"], capture_output=True, text=True).stdout.splitlines() if l.strip()]
for src in sources:
    rows = cat(src); random.Random(1).shuffle(rows)
    J = [{"id": r["id"], "source": src, "target": r["streams"][r["final"]], "streams": {}} for r in (rows[:n] if n else rows)]
    open(f"{out}/jobs_{src}.jsonl", "w").write("".join(json.dumps(j) + "\n" for j in J))
    print(src, len(J))
