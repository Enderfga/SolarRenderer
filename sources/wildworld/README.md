# WildWorld

Input: [WildWorld](https://huggingface.co/datasets/AlayaLab/WildWorld) (gated; non-commercial research use, no
redistribution). Each sample `data_part<x>/<id>/` has `rgb.mp4` (1280x720, 30 fps), `depth.mkv`, `state.csv`
(per-frame camera) and `skeleton.npz` (world-space joints). `ww_io.py` reads them; it also needs the dataset's
`skeleton_edges.json` next to it (or set `WW_EDGES`).

`ww_clip.py <part>/<id>` cuts a sample into back-to-back 8 s windows and writes, per window, `{render, white_box,
skeleton}.mp4` (1344x768, 24 fps, 192 frames) and a json with the source frame of every output frame
(`src_frame_indices`), the per-frame camera and the checks. Geometry: 10 px cropped left and right, then scaled by
16/15; output frame j of a window starting at source frame s0 shows source frame s0 + floor(1.25 j + 0.5). The white
box is clay shading from the dataset's depth; the skeleton stream projects the engine joints through the camera.

`wildworld_dl.py` copies the dataset into storage (`HF_TOKEN`, `WW_SRC`); `convert.sh <shard> <num_shards>` (or
`launch.sh [N]` for N shards on one machine) runs `ww_clip.py` over every complete sample, with `WW_SRC`,
`WW_DST` (ending in `/clips`) and `WW_WORK`. `verify_clip.py OUT_DIR [clip ...]` checks the uploaded clips. The
released data carries only captions and these frame indices (training used the first frame and the caption of
these clips), so the clips can be rebuilt from your own copy.
