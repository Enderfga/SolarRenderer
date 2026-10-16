"""pack_heldout.py (CPU only): the held-out split of the SolarRenderer benchmark, with ground truth.

    pack_heldout.py --inputs <storage>/.../inputs_final.jsonl --val <storage>/.../val_a.jsonl [<storage>/.../val_b.jsonl ...]
                    [--out <storage>/...] [--preencode tools/omni_preencode.py] [--workers W]

The paper's 53 clips are the evaluation input list --inputs restricted to the sources the paper reports (KEEP below)
minus the inputs it drops (DROP below); every clip gives a structure video, its real first frame and a prompt. --val
are the held-out encode lists the clips were drawn from (the raw rows: target, observations, prompt, catalog id).
Each clip gets <out>/benchmark/heldout/clips/<source>/<clip>/{render.mp4 (the ground truth), condition.mp4,
first_frame.png}: the files the paper's held-out numbers were computed from - the structure video (``given``), the
ground truth (``target``) and the first frame (``first``) of the evaluation input list - so eval/ reproduces them.
../release/pack_train.pack runs first for the clip's metadata. <out>/benchmark/heldout/cases.jsonl lists the clips
with their prompts. Engine-recording clips are named game/clipNN in the order
of their recording ids. None of these clips, nor any clip cut from the same recording, scene or map, is in the
training set (CARLA is held out by clip only).

Environment: STORAGE_CLI (storage command, default ``solar-storage``), SOLAR_RELEASE_ROOT (default for --out), SOLAR_PREENCODE /
SOLAR_PYTHONPATH as in pack_train.py.
"""
import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "release"))
from pack_train import PREENCODE, TRAINING_INPUT, alias, pack, store, store_cat, set_preencode  # noqa: E402

OUT = os.environ.get("SOLAR_RELEASE_ROOT")
KEEP = ("carla/", "game/", "magpie/", "unrealzoo/")
DROP = ("game/engine_box/",)
EXPECT = 53
SUB = "benchmark/heldout/clips"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="the evaluation input list of the held-out clips")
    ap.add_argument("--val", nargs="+", required=True, help="the held-out encode lists the clips come from; for an id in two lists the later one wins")
    ap.add_argument("--out", default=OUT, required=OUT is None, help="release root (default: $SOLAR_RELEASE_ROOT)")
    ap.add_argument("--preencode", default=PREENCODE,
                    help="path to the main branch's tools/omni_preencode.py (default: $SOLAR_PREENCODE or "
                         "tools/omni_preencode.py)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    set_preencode(a.preencode)
    raw = {r["id"]: r for v in a.val for r in store_cat(v)}
    rows = [r for r in store_cat(a.inputs) if alias(r["source"]).startswith(KEEP) and not alias(r["source"]).startswith(DROP)]
    assert len(rows) == EXPECT, len(rows)
    missing = [r["id"] for r in rows if r["id"] not in raw]
    assert not missing, missing[:5]
    jobs = {raw[r["id"]]["catalog_id"]: raw[r["id"]] for r in rows}
    games = sorted(c for c in jobs if alias(c).startswith("game/"))
    name = {c: f"clip{i:02d}" for i, c in enumerate(games)}
    results = {}
    with ProcessPoolExecutor(a.workers) as pool:
        work = [(r, a.out, False, SUB, name.get(c)) for c, r in jobs.items()]
        for c, res in zip(jobs, pool.map(pack, work)):
            assert "ok" in res, res
            results[c] = res["ok"]
    for r in rows:  # the evaluation inputs replace the re-encoded copies, byte for byte
        files = results[raw[r["id"]]["catalog_id"]]["files"]
        for key, name in (("given", "condition.mp4"), ("target", "render.mp4"), ("first", "first_frame.png")):
            store("cp", r[key], f"{a.out}/{files[name]}")
    path = os.path.join(tempfile.gettempdir(), "heldout_cases.jsonl")
    with open(path, "w") as fh:
        for r in rows:
            packed = results[raw[r["id"]]["catalog_id"]]
            variant = r["source"].split("/")[2]
            fh.write(json.dumps({"clip_id": packed["clip_id"], "condition": packed["condition"],
                                 "inputs": TRAINING_INPUT[variant], "prompt": raw[r["id"]]["text"],
                                 "ground_truth": packed["files"]["render.mp4"], "files": packed["files"],
                                 "license": packed["license"]}, ensure_ascii=False) + "\n")
    store("cp", path, f"{a.out}/benchmark/heldout/cases.jsonl")
    print(len(jobs), "clips packed")


if __name__ == "__main__":
    main()
