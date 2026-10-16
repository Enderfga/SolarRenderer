# Benchmark scripts

How the two evaluation parts of the release were built.

**Hard benchmark** (`benchmark/hard/`): 208 cases over 16 structure types, 11 of them never seen in training. No ground truth exists; each first frame is an image-model render of the condition's frame 0 in the case's look.

- `build_cases.py --root <cases folder> [case ...]` — reads each `<root>/cases/<case>/` (hand-made or agent-built `condition.mp4`, optional `first.png` / `first_<look>.png`, and `spec.json` with the condition type, input modes and per-look caption fields), assembles the prompt exactly as in training, and writes `<root>/enc.jsonl` (one row per case x look x mode). Imports `DEFINE` and `prompt` from `../samples/build_samples.py` (override the folder with `SOLAR_SAMPLES_DIR`); that module reads `SOLAR_ROOT` on import, so set it.
- `pack_bench.py --src <hardbench folder> [--cases <case list>] --out <release root>` — reads the case list (default `<src>/cases_en.jsonl`; source and tag labels as listed in the script), copies each case to `benchmark/hard/<case>/{condition.mp4, first_frame.png, prompt.txt}` and writes `benchmark/hard/cases.jsonl` (condition type, origin, seen-in-training flag, tags, files).

**Held-out split** (`benchmark/heldout/`): 53 clips with ground truth from CARLA, engine recordings with near-range depth, Magpie and UnrealZoo; every
clip gives a structure video, its real first frame and a prompt.

- `pack_heldout.py --inputs <.../inputs_final.jsonl> --val <held-out list> [...] --out <release root>` — writes `benchmark/heldout/clips/<source>/<clip>/{render.mp4, condition.mp4, first_frame.png}`, the structure, ground truth and first frame the paper's held-out numbers were computed from (taken byte for byte from the evaluation input list; `../release/pack_train.py`'s `pack` runs first for each clip's metadata), and writes `benchmark/heldout/cases.jsonl`; gameplay-video clips are named `game/clipNN` (`SOLAR_SOURCE_ALIAS` maps their catalog source to `game`). It imports `pack_train.py` from `../release/`, which in turn loads `tools/omni_preencode.py` from the `main` branch (`--preencode`, default `tools/omni_preencode.py`, i.e. run from a `main` checkout; or set `SOLAR_PREENCODE`).

Environment variables (all scripts): `STORAGE_CLI` (storage command, default `solar-storage`), `SOLAR_RELEASE_ROOT` (default for `--out`), `SOLAR_PREENCODE` / `SOLAR_PYTHONPATH` (where `omni_preencode.py` and any extra import dirs live), `TMPDIR` (local scratch). Inputs and outputs may be storage URLs (`<scheme>://...`); `ffmpeg` must be on PATH.

The case list, each case's prompt and its first frame were assembled with `build_cases.py` and checked
by hand: first frames were made with an image model (Nano Banana) from frame 0 of the structure video, and the 49
near-field shaded geometry cases that are not built in an engine are shaded from Depth Anything 3 depth of clips
generated with MiniMax-H3.
