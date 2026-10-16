# Blender (procedural game scenes)

Input: [Blender](https://www.blender.org) 4.5.3 and CC0 asset packs (KayKit, Kenney, Quaternius; listed in
`biomes.py`) plus CC0 [Poly Haven](https://polyhaven.com) texture sets (`pbr_sets.py`, downloaded by
`fetch_pbr.py`). `catalog.py` indexes the asset packs into `catalog.json` (bounding boxes, armatures, actions).

`gen_clip.py` builds and renders one clip from a seed (biome, layout, characters and their actions, camera, time
of day; same seed, same clip), and `post.py` draws the skeleton stream, makes the pixel-art style, runs the checks
and encodes. One clip is `bg<seed>_{render,white_box,skeleton,box,class}.mp4` plus extra final looks of the same
frames (`render_toon`, `render_realistic`, `render_pixel`) and `bg<seed>.json`; 1344x768, 24 fps, 192 frames (8 s).

One clip by hand: `run_clip.sh <seed> <gpu> [work_dir]`. Many clips: `deploy.sh` installs Blender and the assets from
`BGEN_ASSETS`, then on each machine
`BGEN_ASSETS=<storage>/... BGEN_OUT=<storage>/.../clips BGEN_ENGINE=cycles BGEN_SAMPLES=48 BGEN_PER_GPU=2 bash run.sh <shard> <num_shards>`
(the settings of the main production run); accepted clips go to `BGEN_OUT`, rejected clips' json to the sibling
`rejected/`. `audit.py <clips prefix>` summarises what was made.
