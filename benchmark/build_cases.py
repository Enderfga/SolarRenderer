"""build_cases.py - the hard cases: hand-made or agent-built conditions, no ground truth.

    build_cases.py --root <storage>/.../cases_root [case ...]

Each case lives at <root>/cases/<case>/:
  condition.mp4   1344x768, 24 fps, 158 frames, frame-aligned structure (optional for a first-frame/text case)
  first.png       optional first frame (first_<look>.png: that look's own first frame, preferred for it)
  spec.json       {"kind": <a DEFINE key of samples/build_samples.py>,
                   "modes": ["structure", "full"],            # which input combinations to render
                   "looks": {"<look>": {"summary": ..., "detailed_description": ...,
                                        "overall_soundscape": ..., "non_diegetic_music": "N/A"}}}
The prompt is assembled exactly as for training (build_samples.prompt: the fixed sentences of the condition type,
the free fields of the look). Writes <root>/enc.jsonl: one encode row per case x look x mode, the condition doubling
as the (unused) target, as for any inference-only encode. Given case names, only those cases are listed.

Environment: STORAGE_CLI (storage command, default ``solar-storage``), SOLAR_SAMPLES_DIR (the directory holding build_samples.py,
which defines DEFINE and prompt; default: ../samples), SOLAR_ROOT (read by build_samples.py on import).
"""
import argparse, json, os, subprocess, sys, tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.environ.get("SOLAR_SAMPLES_DIR", os.path.join(os.path.dirname(_HERE), "samples"))))
from build_samples import DEFINE, prompt  # noqa: E402  samples/build_samples.py of this branch (reads SOLAR_ROOT on import)

# Condition types the model never saw in training (benchmark only): what the clip is, and the same follow-the-structure,
# do-not-show-its-look instruction the trained types use.
_GEOM = ("(frame-by-frame layout, shapes, character poses and motion, camera): attribute_transfer - the layout, shapes, "
         "poses, motion and camera are followed exactly; materials, colours and lighting are not given by {v} and are "
         "the target video's own; {look} of {v} is not shown")
UNSEEN = {
    "wireframe": ("a wireframe rendering of the scene, every object drawn only as thin white mesh edges on black, from "
                  "the same camera as the target video", _GEOM.replace("{look}", "the white wire lines")),
    "normal_map": ("a surface-normal rendering of the scene, the facing direction of every surface shown as a flat RGB "
                   "colour, from the same camera as the target video", _GEOM.replace("{look}", "the normal colours")),
    "point_cloud": ("a sparse 3D point cloud scan of the scene and its characters, white points on black, from the same "
                    "camera as the target video", _GEOM.replace("{look}", "the dotted point look")),
    "line_sketch": ("a black pencil line sketch of the scene on white paper, outlines only, like an animated "
                    "storyboard, from the same camera as the target video", _GEOM.replace("{look}", "the pencil lines and white paper")),
    "pose2d": ("the 2D skeletons of the characters, coloured limb lines and joint dots on black, frame by frame, from the "
               "same camera as the target video", "(frame-by-frame positions, poses and motion of every character, camera): "
               "attribute_transfer - every character keeps its place, pose and motion; their bodies, clothes, the setting, "
               "materials, colours and lighting are not given by {v} and are the target video's own; the skeleton lines "
               "of {v} are not shown"),
    "depth_map": ("a grey depth map of the scene, nearer surfaces brighter and farther ones darker, frame by frame, from "
                  "the same camera as the target video", _GEOM.replace("{look}", "the grey depth shading")),
    "proxy_render": ("a rough proxy rendering of the scene from a simple 3D layout - flat-coloured or depth-shaded shapes, "
                     "people as simple figures or stick figures, objects as boxes - from the same camera as the target video",
                     _GEOM.replace("{look}", "the flat proxy shapes and colours")),
    "edges": ("the edge lines of the scene (outlines and straight line segments) on black, frame by frame, from the same "
              "camera as the target video", _GEOM.replace("{look}", "the edge lines")),
    "box_layout": ("the objects and characters of the scene drawn as coloured boxes or outlines that follow their positions "
                   "and sizes, frame by frame, from the same camera as the target video",
                   "(frame-by-frame positions, sizes and motion of every object and character, camera): attribute_transfer - "
                   "every boxed object keeps its place, size and motion; its exact shape, the setting, materials, colours and "
                   "lighting are not given by {v} and are the target video's own; the boxes and outlines of {v} are not shown"),
    "grayscale": ("a black-and-white (luminance only) version of the scene, frame by frame, from the same camera as the "
                  "target video", _GEOM.replace("{look}", "the grey tones")),
    "silhouette": ("black filled silhouettes of every object and character on a flat white background, frame by frame, "
                   "from the same camera as the target video", _GEOM.replace("{look}", "the black-on-white silhouette look")),
    "depth_turbo": ("a false-colour depth map of the scene, near surfaces red and orange, farther ones green, cyan and blue, "
                    "frame by frame, from the same camera as the target video", _GEOM.replace("{look}", "the false depth colours")),
    "child_drawing": ("a child's wobbly crayon drawing of the scene on white paper - stick figures, simple shapes and "
                      "scribbles - moving frame by frame, from the same camera as the target video",
                      _GEOM.replace("{look}", "the crayon lines and white paper")),
    "mixamo_bot": ("a stylised 3D game mannequin (a smooth white test character with dark grey joint bands) performing "
                   "the motion on a dark grey background with a grid floor, from the same camera as the target video",
                   "(frame-by-frame position, body pose and motion of the performer, camera): attribute_transfer - the "
                   "performer keeps its place, pose and motion; the body, face, clothes, setting, materials, colours and "
                   "lighting are not given by {v} and are the target video's own; the mannequin, the dark background and the grid "
                   "of {v} are not shown"),
    "mocap_markers": ("the optical motion-capture markers of the performer only - small white dots on the body, on black, no "
                      "body drawn - frame by frame, from the same camera as the target video",
                      "(frame-by-frame position, body pose and motion of the performer, camera): attribute_transfer - the "
                      "performer's body follows the markers exactly; the body, face, clothes, setting, materials, colours and "
                      "lighting are not given by {v} and are the target video's own; the dots of {v} are not shown"),
    "game_footage": ("a low-resolution gameplay recording of the same scene, from the same camera as the target video",
                     "(frame-by-frame layout, characters, motion and camera): attribute_transfer - the layout, characters, "
                     "motion and camera are followed exactly; the target video re-renders everything at full quality with its "
                     "own materials and lighting; the blur, low resolution and game look of {v} are not shown"),
    "voxel": ("a blocky voxel rendering of the scene built from flat-coloured cubes, like a block-building game, from the "
              "same camera as the target video", _GEOM.replace("{look}", "the cube blocks and flat colours")),
}
DEFINE.update({k: v for k, v in UNSEEN.items() if k not in DEFINE})

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
ROOT = None   # --root


def ls(pattern):
    out = subprocess.run([STORE, "ls", pattern], capture_output=True, text=True).stdout
    return [l.split()[-1] for l in out.splitlines() if l.strip()]


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder holding cases/<case>/; enc.jsonl is written beside it")
    ap.add_argument("cases", nargs="*", help="only these cases (default: all)")
    a = ap.parse_args()
    ROOT = a.root
    only = set(a.cases)
    rows = []
    for spec_name in ls(f"{ROOT}/cases/*/spec.json"):
        case = spec_name.split("/")[0] if "/" in spec_name else None
        if case is None:  # a wildcard listing prints the path below the prefix
            continue
        if only and case not in only:
            continue
        base = f"{ROOT}/cases/{case}"
        spec = json.loads(subprocess.run([STORE, "cat", f"{base}/spec.json"], capture_output=True, text=True).stdout)
        files = {n.rsplit("/", 1)[-1] for n in ls(f"{base}/*")}
        kind = spec.get("kind", "none")
        if kind != "none" and kind not in DEFINE:
            raise SystemExit(f"{case}: unknown condition kind {kind}")
        for look, fields in spec["looks"].items():
            for mode in spec.get("modes", ["structure"]):
                # a look may bring its own first frame (first_<look>.png, e.g. generated in that look's style)
                first = f"first_{look}.png" if f"first_{look}.png" in files else "first.png"
                if mode in ("full", "first") and first not in files:
                    continue
                if mode in ("full", "structure") and "condition.mp4" not in files:
                    continue
                obs = []
                if mode in ("full", "structure"):
                    obs.append({"media": f"{base}/condition.mp4", "frames": "all"})
                if mode in ("full", "first"):
                    obs.append({"media": f"{base}/{first}", "frames": [0]})
                target = f"{base}/condition.mp4" if "condition.mp4" in files else f"{base}/{spec['target']}"
                rows.append({"id": f"sb_{case}_{look}_{mode}", "target": target,
                             "text": prompt(kind, mode, fields), "observations": obs,
                             "source": f"showcase/{kind}/{mode}"})
    local = os.path.join(tempfile.gettempdir(), "hard_cases_enc.jsonl")
    open(local, "w").writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    subprocess.run([STORE, "cp", local, f"{ROOT}/enc.jsonl"], check=True, capture_output=True)
    print(json.dumps({"rows": len(rows), "cases": sorted({r["id"][3:].rsplit("_", 2)[0] for r in rows})}))


if __name__ == "__main__":
    main()
