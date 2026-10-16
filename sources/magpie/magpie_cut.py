"""Cut Magpie Lite's twenty-minute trajectories into 8 s windows the encoder can take whole.

Both streams of a trajectory are frame-aligned and put a keyframe on the same frames, so a window
that starts on a shared keyframe is cut by copying packets: no re-encode, and the cut decodes to the
very frames of the source. A window starts every SPACING seconds after the first SKIP seconds (the
spawn), and a sample of the cuts is decoded and compared with the source frame for frame.

Environment:
  MAGPIE_ROOT   the Magpie Lite download: <root>/scene*/<player>/{render,white_box}.mp4
  MAGPIE_CLIPS  output directory for <scene>_<player>_<start frame>_{render,white_box}.mp4 and the windows file
  SPACING       minimum seconds between window starts (default 25). SPACING=8 with EXISTING=<windows.jsonl
                of an earlier pass> cuts the windows in between, none overlapping one already cut.
  TRAJS         optional file naming the trajectories to take (one path per line), e.g. one shard of them
  WINDOWS       name of the windows file written in MAGPIE_CLIPS (default windows.jsonl)
  CUT_WORKERS   parallel trajectories (default 24)
"""
import concurrent.futures as cf, glob, json, os, random, subprocess, sys
import av, numpy as np

ROOT = os.environ.get("MAGPIE_ROOT", "magpie")
OUT = os.environ.get("MAGPIE_CLIPS", "magpie_clips")
SKIP, LENGTH = 8.0, 8.0
SPACING = float(os.environ.get("SPACING", "25"))
EXISTING = os.environ.get("EXISTING", "")
TAKEN = {}


def window_starts(shared, fps, frames, spacing, taken=()):
    """Shared keyframes a window can start on: past the spawn, SPACING apart, ending inside the
    trajectory and overlapping none of the windows already taken."""
    width = LENGTH * fps
    starts, last = [], -1e9
    for k in shared:
        if k / fps < SKIP or k / fps - last < spacing or k + width > frames:
            continue
        if any(k < t + width and t < k + width for t in taken):
            continue
        starts.append(k)
        last = k / fps
    return starts


def keyframes(path):
    c = av.open(path, metadata_errors="ignore"); s = c.streams.video[0]
    t0 = s.start_time or 0
    ks = sorted(int(round(float((p.pts - t0) * s.time_base) * float(s.average_rate)))
                for p in c.demux(s) if p.is_keyframe and p.pts is not None)
    n, fps = s.frames, float(s.average_rate); c.close()
    return ks, n, fps

def first_frame(path, index=None, fps=None):
    c = av.open(path, metadata_errors="ignore"); s = c.streams.video[0]
    if index:
        c.seek(int(index / fps / s.time_base) + (s.start_time or 0), stream=s, backward=True)
    for f in c.decode(s):
        if index is None or int(round(float((f.pts - (s.start_time or 0)) * s.time_base) * fps)) >= index:
            a = f.to_ndarray(format="rgb24"); c.close(); return a

def one(traj):
    scene, player = traj.split("/")[-2:]
    r, w = f"{traj}/render.mp4", f"{traj}/white_box.mp4"
    kr, nr, fps = keyframes(r); kw, nw, _ = keyframes(w)
    if nr != nw:
        return traj, [], f"frame counts differ {nr} {nw}"
    shared = sorted(set(kr) & set(kw))
    starts = window_starts(shared, fps, nr, SPACING, TAKEN.get(f"{scene}/{player}", ()))
    rows = []
    for k in starts:
        name = f"{scene}_{player}_{k:06d}"
        for src, tag in ((r, "render"), (w, "white_box")):
            out = f"{OUT}/{name}_{tag}.mp4"
            if not os.path.exists(out):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{k / fps + 0.001:.6f}", "-i", src,
                                "-t", f"{LENGTH}", "-c", "copy", "-an", "-avoid_negative_ts", "make_zero", out + ".part.mp4"], check=True)
                os.replace(out + ".part.mp4", out)
        rows.append({"name": name, "scene": scene, "player": player, "start_frame": k, "fps": fps})
    # One window in ten is decoded and checked against the source, both streams.
    bad = 0
    for row in random.Random(traj).sample(rows, max(1, len(rows) // 10)) if rows else []:
        for src, tag in ((r, "render"), (w, "white_box")):
            if not np.array_equal(first_frame(f"{OUT}/{row['name']}_{tag}.mp4"), first_frame(src, row["start_frame"], fps)):
                bad += 1
    return traj, rows, f"bad {bad}" if bad else ""

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if EXISTING:
        for line in open(EXISTING):
            if line.strip():
                row = json.loads(line)
                TAKEN.setdefault(f"{row['scene']}/{row['player']}", []).append(row["start_frame"])
    named = os.environ.get("TRAJS")
    trajs = sorted(d.rstrip("/") for d in (
        [f"{l.strip()}/" for l in open(named) if l.strip()] if named else glob.glob(f"{ROOT}/scene*/*/"))
                   if os.path.exists(f"{d}render.mp4") and os.path.exists(f"{d}white_box.mp4"))
    print(len(trajs), "trajectories", flush=True)
    allrows, problems = [], []
    with cf.ProcessPoolExecutor(int(os.environ.get("CUT_WORKERS", "24"))) as pool:
        for traj, rows, note in pool.map(one, trajs):
            allrows += rows
            if note: problems.append((traj, note)); print("PROBLEM", traj, note, flush=True)
    with open(f"{OUT}/{os.environ.get('WINDOWS', 'windows.jsonl')}", "w") as h:
        for row in allrows: h.write(json.dumps(row) + "\n")
    print(f"CUT DONE {len(allrows)} windows from {len(trajs)} trajectories; problems {len(problems)}", flush=True)
