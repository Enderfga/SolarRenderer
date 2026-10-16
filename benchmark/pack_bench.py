"""pack_bench.py (CPU only): the 208-case hard benchmark of the SolarRenderer data release.

    pack_bench.py --src <storage>/.../hardbench [--cases <src>/cases_en.jsonl] [--out <storage>/...]

Copies <src>/cases/<case>/{cond.mp4, first.png, prompt.txt} to <out>/benchmark/hard/<case>/
{condition.mp4, first_frame.png, prompt.txt} and writes <out>/benchmark/hard/cases.jsonl with the metadata in English.
The case list (default <src>/cases_en.jsonl) holds one line per case (case_id, segment, look, source, hard_tags);
source and hard_tags use the labels below. No ground truth exists for these cases: the first frames are image-model renders of the
condition's frame 0.

Environment: STORAGE_CLI (storage command, default ``solar-storage``), SOLAR_RELEASE_ROOT (default for --out).
"""
import argparse, json, os, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
OUT = os.environ.get("SOLAR_RELEASE_ROOT")
# Case labels -> what the release says: tag; source -> (condition, origin, seen in training).
TAGS = {"large_motion": "large motion", "fast_camera": "fast camera", "stylized": "stylized", "night": "night",
        "multi_person": "multi-person", "unseen_condition": "unseen condition"}
SOURCES = {
    "scene_grey_box/agent": ("scene grey box", "built by our agent", True),
    "near_field/generated_video": ("near-field shaded geometry", "estimated from a generated video", True),
    "near_field/agent": ("near-field shaded geometry", "built by our agent", True),
    "mocap_mesh/dirtymocap": ("SMPL body mesh", "DirtyMoCap project video with markers, used as is", False),
    "smpl_mesh/fitted": ("SMPL body mesh", "fitted SMPL motion, white background", True),
    "semantic/carla": ("semantic segmentation", "CARLA engine export", True),
    "semantic/agent": ("semantic segmentation", "built by our agent", True),
    "pose_2d/agent": ("2D pose skeleton", "built by our agent", False),
    "pose_2d/photo": ("2D pose skeleton", "aligned to a photo", False),
    "pencil/agent": ("pencil storyboard", "built by our agent", False),
    "crayon/agent": ("child's crayon drawing", "built by our agent", False),
    "silhouette/agent": ("silhouette", "built by our agent", False),
    "colour_depth/agent": ("colour-mapped depth", "built by our agent", False),
    "wireframe/agent": ("wireframe", "built by our agent", False),
    "point_cloud/agent": ("point cloud", "built by our agent", False),
    "normal_map/agent": ("normal map", "built by our agent", False),
    "voxels/agent": ("voxels", "built by our agent", False),
    "masks_skeletons/agent": ("object masks + skeletons", "built by our agent", True),
    "boxes_3d/agent": ("3D bounding boxes (driving format)", "built by our agent", False),
    "mixamo/retargeted": ("Mixamo character", "public animation retargeted", False),
}
# condition type -> 16 types, 11 of them never seen in training. The DirtyMoCap clips are an SMPL body mesh drawn with
# markers in the project's own rendering: the type was seen in training, these renderings were not, so their cases are
# marked condition_seen_in_training = false (74 such cases in all).


def store(*argv):
    return subprocess.run([STORE, *argv], capture_output=True, text=True, check=True).stdout


def copy(case):
    cid = case["case_id"]
    for a, b in (("cond.mp4", "condition.mp4"), ("first.png", "first_frame.png"), ("prompt.txt", "prompt.txt")):
        store("cp", f"{SRC_}/cases/{cid}/{a}", f"{OUT_}/benchmark/hard/{cid}/{b}")
    condition, origin, seen = SOURCES[case["source"]]
    return {"case_id": cid, "segment": case["segment"], "look": case["look"], "condition": condition,
            "condition_origin": origin, "condition_seen_in_training": seen,
            "tags": [TAGS[t] for t in case["hard_tags"]],
            "files": {k: f"benchmark/hard/{cid}/{k}" for k in ("condition.mp4", "first_frame.png", "prompt.txt")},
            "first_frame_note": "image-model render of the condition's frame 0 in this look, not a real photo",
            "ground_truth": None}


def main():
    global OUT_, SRC_
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the benchmark cases folder (cases/<case>/...)")
    ap.add_argument("--cases", default="", help="the case list (default: <src>/cases_en.jsonl)")
    ap.add_argument("--out", default=OUT, required=OUT is None, help="release root (default: $SOLAR_RELEASE_ROOT)")
    a = ap.parse_args()
    OUT_, SRC_ = a.out, a.src
    cases = [json.loads(l) for l in store("cat", a.cases or f"{SRC_}/cases_en.jsonl").splitlines() if l.strip()]
    missing = {c["source"] for c in cases} - set(SOURCES)
    assert not missing, missing
    with ThreadPoolExecutor(16) as pool:
        lines = list(pool.map(copy, cases))
    path = os.path.join(tempfile.gettempdir(), "bench_hard_cases.jsonl")
    with open(path, "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in lines)
    store("cp", path, f"{OUT_}/benchmark/hard/cases.jsonl")
    print(len(lines), "cases")


if __name__ == "__main__":
    main()
