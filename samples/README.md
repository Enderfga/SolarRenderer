# samples

From audited, captioned clips to the training index. All scripts read and write a storage location given by environment variables.

1. `first_frames.py <source> ...` takes frame 0 of every clip the audit kept and stores it as a PNG; this is the first-frame input.
2. `build_samples.py <part> [max]` turns each clip into one training sample. The input combination is fixed per clip by a hash of its id: structure + first frame, structure only, first frame only, or text only. The six-part prompt (`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`) names exactly the inputs the sample carries. Clips from held-out units go to a separate validation list.
3. The samples are encoded into latents by `tools/omni_preencode.py` on the `main` branch of this repository (not copied here).
4. `build_index.py <out-prefix> [--check]` writes the final training index: every chosen encoded clip once, held-out units never in train. It refuses to write while chosen clips are still unencoded; `--check` prints what is missing.

The final index has 136,289 rows. Input combinations: structure + first frame 67.1%, structure only 16.7%, first frame only 13.2%, text only 2.9%.

The fixed sentences per condition type (`DEFINE` in `build_samples.py`) are word for word those of `skills/solarrenderer-caption/conditions.py` on the `main` branch for every type the two share; `build_samples.py` adds the sentences of its box-proxy entries (`engine_box`, `bg_box`).

Environment variables:

- `SOLAR_ROOT` (required): root of the raw data, e.g. `<storage>/raw`. Audits, catalogs, captions, first frames and sample lists live under it.
- `SOLAR_LATENTS` (`build_index.py`): prefix of the encoded set, e.g. `<storage>/latents`.
- `STORAGE_CLI` (default `solar-storage`), `SOLAR_WORK` (local scratch directory, default `./solar_work`).

`build_index.py` reads member headers through `fsspec` (install the filesystem package for your storage); `first_frames.py` needs `ffmpeg`.
