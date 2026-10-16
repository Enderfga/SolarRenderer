"""build_samples.py <part> [max]: the encode list - every approved, gated clip whose caption (6.583 s window) is
written, as an omni sample: target, its structure clip and/or its first frame, and a prompt in the six-part form
that names exactly the references the sample carries.

Each clip is ONE sample in ONE variant, drawn on a hash of its id (no observation dropout at training time - a
dropped observation would leave its tokens in the prompt). Sources with a structure video (VARIANTS below):
  full       structure clip + first frame   75 %   [keyframe completion + reference generation]
  structure  structure clip only            15 %   [reference generation]
  first      first frame only                7 %   [keyframe completion]
  text       no reference                    3 %   text-to-video fields
Sources without a structure video (VARIANTS_FREE): first frame only 60 %, text only 40 %.
The first frame is a still the encoder holds as a short clip (so it lands on the target's grid), so the prompt calls
it a video: <Video 2> beside a structure clip, <Video 1> alone. The fixed sentences per condition type (DEFINE) are
the same as in skills/solarrenderer-caption/conditions.py on the main branch and must stay identical to them; the
caption model only wrote the free fields. Held-out units (split below) are written to a separate list. Writes
$SOLAR_ROOT/solar_train/enc_solar_v2_<part>.jsonl (+ val_solar_v2_<part>.jsonl), only clips not in an earlier part,
so it can be rerun as captions land, and the union of all parts as enc_solar_v2.jsonl / val_solar_v2.jsonl.

This file covers Magpie, CARLA, UnrealZoo, WildWorld and OmniWorld. The training index also holds clips of other
sources; their sample lists are not part of this file.

Special parts (each re-lists clips on a different input, with ids that sit beside the earlier rows):
  owfirst   OmniWorld clips drawn with their structure (full / structure) again as first frame only
  wwfirst   WildWorld no longer as structure: up to two windows per sample, first frame only, about 5,000 in all

Environment:
    SOLAR_ROOT    root of the raw data (required), e.g. <storage>/raw/omni
    STORAGE_CLI      storage command (default: solar-storage)
    SOLAR_WORK       local scratch directory (default: ./solar_work)
"""
import hashlib, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

STORE = os.environ.get("STORAGE_CLI", "solar-storage")
R = os.environ["SOLAR_ROOT"].rstrip("/")
CAP = f"{R}/solar_caption_v2/w158"  # 6.583 s window captions; the 8 s ones beside it are not used
T = f"{R}/solar_train"
WORK = os.environ.get("SOLAR_WORK", "solar_work")


# Hold-out units: whole scenes, maps, samples or source videos, never single clips, so a test
# clip never shares a place or a take with a training clip.
def unit(source: str, cid: str) -> str:
    """The hold-out unit of a clip id (catalog ids are '<source>/<name>')."""
    name = cid.split("/", 1)[1]
    if source == "magpie":
        return name.split("_")[0]                       # scene
    if source == "wildworld":
        return name.rsplit("_", 1)[0]                    # sample
    if source == "unrealzoo":
        return name.split("_s")[0]                       # map
    if source == "omniworld_v3":
        return name.split("_s")[0]                       # source video
    return name                                          # carla: single clips are independent renders


HELD_UNITS = {"magpie": {"scene09", "scene19"}}
HELD_FRACTION = {"carla": 0.02, "wildworld": 0.03, "omniworld_v3": 0.05, "unrealzoo": 0.12}


def split(source: str, cid: str) -> str:
    u = unit(source, cid)
    if u in HELD_UNITS.get(source, ()):
        return "heldout"
    if h(f"{source}:{u}") < HELD_FRACTION.get(source, 0.0):
        return "heldout"
    return "train"

# condition type -> (what <V> is, what of it is followed / not given / not shown)
DEFINE = {
    "scene_graybox": ("an untextured grey block-out of the scene, rendered from the same camera as the target video, "
                      "giving the layout, the large geometry, the characters' bodies, their motion and the camera of every frame",
                      "(frame-by-frame layout, geometry, motion and camera): attribute_transfer - the layout, large shapes, "
                      "character bodies, motion and camera are followed exactly; small objects, materials, colours and "
                      "lighting are not given by {v} and are the target video's own; the grey look of {v} is not shown"),
    "engine_box": ("a grey box proxy of the scene from the game engine, rendered from the same camera as the target video: "
                   "every character is a set of boxes on its body parts, the surroundings are coarse blocks and the ground "
                   "is a grey mesh, giving the positions, sizes and motion of the characters, the rough layout of the "
                   "surroundings and the camera of every frame",
                   "(frame-by-frame positions, sizes, motion and camera): attribute_transfer - the characters' positions, "
                   "body proportions, poses and motion, the rough layout and the camera are followed; the exact shapes, "
                   "materials, colours and lighting are not given by {v} and are the target video's own; the grey boxes "
                   "of {v} are not shown"),
    "semantic": ("a semantic segmentation of the scene, one flat colour per class (road, sidewalk, building, vegetation, "
                 "vehicle, pedestrian, sky and so on), frame by frame, from the same camera as the target video",
                 "(frame-by-frame layout of every class and camera): attribute_transfer - every region keeps its class, "
                 "outline and place; materials, colours and lighting are not given by {v}; the flat class colours are "
                 "not shown"),
    "scene_whitebox": ("an untextured grey rendering of the scene and the main character, from the same camera "
                       "as the target video, giving the layout of the scene, the character's pose and motion, and the "
                       "camera of every frame",
                       "(frame-by-frame layout, character pose and motion, camera): attribute_transfer - the layout, the "
                       "character's pose and motion and the camera are followed; materials, colours, lighting and any "
                       "added detail are the target video's own; the grey look of {v} is not shown"),
    "mask_skeleton": ("the objects of the scene as flat-coloured masks, one colour per object, with the skeletons of the "
                      "characters drawn on top, frame by frame, from the same camera as the target video",
                      "(frame-by-frame object outlines, character poses and camera): attribute_transfer - every object "
                      "keeps its outline and place, every character keeps its pose and motion; materials, colours and "
                      "lighting are not given by {v}; the flat mask colours and skeleton lines are not shown"),
    # YouTube: a GVHMR body fit of the main character, rendered as a grey mesh on white, frame-aligned with the
    # target (same crop, 1344x768, 24 fps)
    "smpl_mesh": ("a grey 3D body mesh of the main character on a plain white background, rendered from the same camera "
                  "as the target video; it gives the character's position, size, body pose and motion in every frame, "
                  "and nothing of the surroundings",
                  "(frame-by-frame body pose and motion of the main character): attribute_transfer - the main character's "
                  "position, size, pose and motion are followed; the character's clothing, face and hair, the surroundings, "
                  "the camera's view of the scene, materials, colours and lighting are not given by {v} and are the target "
                  "video's own; the grey mesh and the white background of {v} are not shown"),
    # engine recordings: the depth itself, near range only
    "near_depth": ("a grey shaded rendering of the game engine's depth within about 30 metres of the camera, from the "
                   "same camera as the target video; everything farther, and the sky, is black. It gives the exact "
                   "shapes, positions and motion of the nearby scene, characters and vehicles, and the camera of every "
                   "frame",
                   "(frame-by-frame near geometry, motion and camera): attribute_transfer - the shapes, positions and "
                   "motion of everything within the near range and the camera are followed; the far scene, materials, "
                   "colours and lighting are not given by {v} and are the target video's own; the grey shading and the "
                   "black background of {v} are not shown"),
    "bg_box": ("a grey block proxy of the scene's surroundings rebuilt from the game's depth, rendered from the same camera "
               "as the target video, giving the rough layout of the surroundings, the ground and the camera of every "
               "frame; characters and moving objects are not drawn",
               "(frame-by-frame layout of the surroundings and camera): attribute_transfer - the rough layout, the ground "
               "and the camera are followed; characters, moving objects, exact shapes, materials, colours and lighting "
               "are not given by {v}; the grey blocks of {v} are not shown"),
}
# Structured sources: mostly structure + first frame. The unconditioned share is carried by YouTube (first frame or
# text only, no structure at all - its prompts never name a structure clip); a little stays here so "no structure"
# is not tied to the YouTube look.
VARIANTS = (("full", 0.75), ("structure", 0.90), ("first", 0.97), ("text", 1.00))
VARIANTS_FREE = (("first", 0.60), ("text", 1.00))


def store_cat(url):
    return [json.loads(l) for l in subprocess.run([STORE, "cat", url], capture_output=True, text=True).stdout.splitlines()
            if l.strip()]


def h(text):
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def variant(cid, kind):
    x = h("v2variant:" + cid)
    return next(name for name, edge in (VARIANTS_FREE if kind == "none" else VARIANTS) if x < edge)


def prompt(kind, var, f):
    detail = f["detailed_description"].strip()
    sound = (f.get("overall_soundscape") or "N/A").strip() or "N/A"
    music = (f.get("non_diegetic_music") or "N/A").strip() or "N/A"
    summary = f["summary"].strip()
    if var == "text":  # T2VA: the three core fields, style inside [Shot 1]
        body = "[Shot 1] " + re.sub(r"\[Shot 1\]\s*", "", detail, count=1)  # T2VA puts the style inside [Shot 1]
        return f"integrated_multimodal_description: {body}\n\noverall_soundscape: {sound}\n\nnon_diegetic_music: {music}"
    lines, keep, tasks = [], [], []
    structure = var in ("full", "structure")
    first = var in ("full", "first")
    s_lab = "<Video 1>" if structure else None
    f_lab = ("<Video 2>" if structure else "<Video 1>") if first else None
    if structure:
        what, kept = DEFINE[kind]
        lines.append(f"{s_lab} is {what}.")
        keep.append(f"{s_lab} {kept.format(v=s_lab)}.")
        tasks.append("reference generation")
    if first:  # listed after the structure clip, in label order
        lines.append(f"{f_lab} is the first frame of [Shot 1], held still.")
        keep.append(f"{f_lab} ([Shot 1] first frame): fully_preserved - the composition, subjects, materials, colours "
                       f"and lighting of {f_lab} are the target video's first frame.")
        tasks.insert(0, "keyframe completion")
    lead = []
    if first:
        lead.append(f"The target video begins from {f_lab}")
    if structure:
        lead.append(f"follows the frame-by-frame structure and motion of {s_lab}" if first else
                    f"The target video follows the frame-by-frame structure and motion of {s_lab}")
    head = (" and ".join(lead) + ". ") if lead else ""
    if first:  # the opening shot starts from the first frame
        detail = re.sub(r"\[Shot 1\]\s*", f"[Shot 1] The shot begins from {f_lab}. ", detail, count=1) \
            if "[Shot 1]" in detail else f"[Shot 1] The shot begins from {f_lab}. " + detail
    return ("subject_definitions:\n" + "\n".join(lines) + "\n\n"
            f"summary:\n[{' + '.join(tasks)}] {head}{summary}\n\n"
            "retention_analysis:\n" + "\n".join(keep) + "\n\n"
            f"detailed_description:\n{detail}\n\n"
            f"overall_soundscape:\n{sound}\n\nnon_diegetic_music:\n{music}")


WW_HELD = set()  # WildWorld windows its box manifest holds out


def approved():
    """(catalog id, kind, target, condition, first frame, crop) of every gated clip, by source."""
    out = []
    cat = {s: {r["id"]: r for r in store_cat(f"{R}/solar_catalog/{s}.jsonl")} for s in ("magpie", "carla")}
    ff = lambda s, cid: f"{R}/solar_first_frame/{s}/{cid.split('/', 1)[1]}.png"
    for r in store_cat(f"{R}/solar_audit/magpie.jsonl"):
        if r.get("keep") and r["id"] in cat["magpie"]:
            c = cat["magpie"][r["id"]]
            out.append((r["id"], "scene_graybox", c["streams"]["render"], c["streams"]["white_box"], ff("magpie", r["id"]), None))
    for r in store_cat(f"{R}/solar_audit/carla.jsonl"):
        if r.get("keep") and r["id"] in cat["carla"]:
            c = cat["carla"][r["id"]]
            out.append((r["id"], "semantic", c["streams"]["rgb"], c["streams"]["semantic"], ff("carla", r["id"]), None))
    for r in store_cat(f"{R}/wildworld_solar/box_manifest.jsonl"):
        if not r.get("drop"):  # its held-out windows go to val by the manifest's own split (see main)
            if r.get("split") == "heldout":
                WW_HELD.add(r["id"])
            out.append((r["id"], "engine_box", r["target"], r["condition"], r["first_frame"], None))
    for r in store_cat(f"{R}/unrealzoo_v2/solar/manifest.jsonl"):  # accepted recordings
        out.append((r["id"], "mask_skeleton", r["target"], r["condition"], r["first_frame"], None))
    for r in store_cat(f"{R}/solar_ow_bgbox/manifest.jsonl"):
        if not r.get("drop"):
            out.append((r["id"], "bg_box", r["target"], r["condition"], r["first_frame"], None))
    return out


FIRST = ("magpie", "carla", "unrealzoo")  # sources encoded before the rest of the list
CAP_DIR = {"omniworld_v3": "omniworld"}  # the caption queue names these by dataset


def captions(source):
    ls = subprocess.run([STORE, "ls", f"{CAP}/{CAP_DIR.get(source, source)}/*.json"], capture_output=True, text=True).stdout
    return {l.split()[-1][:-5] for l in ls.splitlines() if l.strip()}


def main():
    os.makedirs(WORK, exist_ok=True)
    part = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    earlier = set()
    ls = subprocess.run([STORE, "ls", f"{T}/enc_solar_v2_*.jsonl"], capture_output=True, text=True).stdout
    for name in [l.split()[-1] for l in ls.splitlines() if l.strip()]:
        earlier |= {r["catalog_id"] for r in store_cat(f"{T}/{name}")}
    clips = approved()
    have = {}
    rows = {"train": [], "val": []}
    stats = {}
    todo = []
    # part "owfirst" (OmniWorld's box condition is a static background with the people masked out): the OmniWorld clips encoded with their structure (full / structure) again as first frame only; those
    # already drawn as first frame or text only are kept as they are.
    owfirst = part == "owfirst"
    # Part "wwfirst" (WildWorld no longer as structure, first frame at most): up to two windows per WildWorld sample,
    # first frame only, about 5,000 in all.
    wwfirst = part == "wwfirst"
    per = {}  # wwfirst: windows taken per WildWorld sample
    for c in sorted(clips, key=lambda c: h(("wwfirst:" if wwfirst else "") + c[0])):
        source, name = c[0].split("/", 1)
        if owfirst:
            if source != "omniworld_v3" or variant(c[0], c[1]) not in ("full", "structure"):
                continue
        elif wwfirst:
            if source != "wildworld" or variant(c[0], c[1]) == "first":
                continue  # those already drawn as first frame keep their rows
        elif c[0] in earlier:
            continue
        if source not in have:
            have[source] = captions(source)
        if name in have[source]:
            if wwfirst:
                sample = c[0].rsplit("_", 1)[0]  # the WildWorld sample the window was cut from
                if per.get(sample, 0) >= 2 or len(todo) >= 5000:
                    continue
                per[sample] = per.get(sample, 0) + 1
            todo.append(c)
    # One caption at a time takes many minutes for a few thousand clips and leaves the encoders waiting; read them
    # 24 at a time (the store allows 32 requests at a time).
    fetch = lambda c: json.loads(subprocess.run([STORE, "cat", f"{CAP}/{CAP_DIR.get(c[0].split('/', 1)[0], c[0].split('/', 1)[0])}/{c[0].split('/', 1)[1]}.json"],
                                                capture_output=True, text=True).stdout or "{}")
    with ThreadPoolExecutor(24) as pool:
        caps = list(pool.map(fetch, todo))
    for (cid, kind, target, cond, first, crop), cap in zip(todo, caps):
        source, name = cid.split("/", 1)
        if cap.get("window_s") != 6.583 or not cap.get("checks", {}).get("fields", True):
            continue  # the 8 s captions describe seconds the encoder never keeps
        var = "first" if (owfirst or wwfirst) else variant(cid, kind)
        obs = []
        if var in ("full", "structure"):
            obs.append({"media": cond, "frames": "all"})
        if var in ("full", "first"):
            obs.append({"media": first, "frames": [0]})
        stem = name.replace('/', '_')
        row = {"id": f"solar2_{source}_{stem}_{var}", "catalog_id": cid, "target": target,
               "text": prompt(kind, var, cap["fields"]), "observations": obs, "source": f"{source}/{kind}/{var}"}
        if crop:
            row["crop"] = crop
        held = cid in WW_HELD if source == "wildworld" else split(source, cid) == "heldout"
        part_of = "val" if held else "train"
        rows[part_of].append(row)
        stats[row["source"]] = stats.get(row["source"], 0) + 1
        if limit and len(rows["train"]) >= limit:
            break
    for p, items in rows.items():
        local = f"{WORK}/enc_solar_v2_{part}{'' if p == 'train' else '_val'}.jsonl"
        open(local, "w").writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in items)
        subprocess.run([STORE, "cp", local, f"{T}/{'enc_solar_v2_' + part if p == 'train' else 'val_solar_v2_' + part}.jsonl"],
                       check=True, capture_output=True)
    # The encode queue reads one list per set and rereads it every pass: keep the union of every part in one file.
    for stem in ("enc_solar_v2", "val_solar_v2"):
        ls = subprocess.run([STORE, "ls", f"{T}/{stem}_*.jsonl"], capture_output=True, text=True).stdout
        allrows = []
        for name in sorted(l.split()[-1] for l in ls.splitlines() if l.strip()):
            allrows += subprocess.run([STORE, "cat", f"{T}/{name}"], capture_output=True, text=True).stdout.splitlines()
        # Encoders take a list in order. The groups that hold the training index back go first.
        allrows = [l for l in allrows if l.strip()]
        allrows.sort(key=lambda l: not any(f'"source": "{g}/' in l for g in FIRST))
        local = f"{WORK}/{stem}.jsonl"
        open(local, "w").write("\n".join(allrows) + "\n")
        subprocess.run([STORE, "cp", local, f"{T}/{stem}.jsonl"], check=True, capture_output=True)
    print(json.dumps({"part": part, "train": len(rows["train"]), "val": len(rows["val"]), "by": stats}, indent=0))


if __name__ == "__main__":
    main()
