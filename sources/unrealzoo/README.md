# UnrealZoo

Input: the [UnrealZoo](https://unrealzoo.github.io) UE5 environments (Linux release v3.1.0 from
https://huggingface.co/datasets/UnrealZoo/UnrealZoo_UE5) driven through UnrealCV, with spawn points from the
Track configs of [unrealzoo-gym](https://github.com/UnrealZoo/unrealzoo-gym). One clip is 8 s, 24 fps, 1344x768,
192 frames, third-person follow (`tps`) or first-person (`fps`), with the streams
`<clip>_{render,white_box,mask,depth,normals,skeleton}.mp4`, `<clip>_mid_depth.npz` and `<clip>.json` (camera pose,
intrinsics, 21 joints per character per frame, quality-gate stats); clip id `uz_<map>_s<seed>_<idx>_<tps|fps>`.

Capture: the game is paused and stepped one tick per frame (`vrun slomo 1.25` makes a tick 1/24 s), and
`use_fast_capture 0` keeps lit / mask / normal on the same tick as depth. The UnrealCV server accepts one client
only, so `bridge.py` holds that connection and the scripts talk to it (`uc.py`). The white box is clay shading
computed from engine depth (no unlit capture exists). Clips that fail a gate (`GATES` in `uz_prod.py`) go to `rejected/`.

Run (as root, Linux with NVIDIA GPUs): `UZ_PKG=<storage>/... deploy.sh`, then
`UZ_OUT=<storage>/.../unrealzoo UZ_MACHINE=<k> prod.sh start` (`drain`, `stop`, `status`). The released clips were
recorded in task-list mode, `UZ_TASKS=tasks_v2_catalog.txt` (map, seed per line; `UZ_QUEUE` on a shared
filesystem lets several machines share it). `uz_v2_prep.py` then builds the condition video (mask with the skeleton
drawn over it, `<clip>_maskskel.mp4`) and the first frame; `verify_align.py DIR CLIP` checks frame alignment.
`uz_record.py` (+ `run_batch.sh`) is the first recorder; `uz_prod.py` imports its helpers.
