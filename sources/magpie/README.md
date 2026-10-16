# Magpie Lite

Input: [Magpie Lite](https://huggingface.co/datasets/MogoAI/Magpie_lite), gameplay trajectories in Unreal Engine
scenes, each recorded as a finished render (`render.mp4`) and the engine's untextured white-box view (`white_box.mp4`)
of the same frames. The white-box stream comes with the dataset; nothing is rendered here.

`magpie_cut.py` cuts every trajectory into 8 s windows that start on a keyframe both streams share, by copying
packets (no re-encode), so a window decodes to exactly the source frames. One window in ten is decoded and compared
with the source frame for frame. Run it on a local copy:

    MAGPIE_ROOT=./magpie_lite MAGPIE_CLIPS=./magpie_clips SPACING=8 python3 magpie_cut.py

It writes `<scene>_<player>_<start frame>_{render,white_box}.mp4` and `windows.jsonl` (one line per window: name,
scene, player, start frame, fps). Clips keep the dataset's resolution and frame rate (60 fps); they are
resampled to 24 fps by nearest timestamp and resized to 1344x768 when encoded for training. `magpie_dense.sh <shard> <num_shards>` runs the same cutter over a
copy kept in storage (`MAGPIE_OUT`), a batch of trajectories at a time, skipping windows an earlier pass already cut.
