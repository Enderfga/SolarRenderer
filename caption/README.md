# caption

Writes the free parts of the SolarRenderer training prompts from the finished clips.

Every prompt has six parts: `subject_definitions`, `summary`, `retention_analysis`, `detailed_description` (350-500 words), `overall_soundscape` and `non_diegetic_music`. The captioning model writes only the free parts (`summary`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`) and sees only the finished clip. The parts that name the inputs (`subject_definitions`, `retention_analysis` and the task tag of `summary`) are fixed sentences per condition type, added later by `samples/build_samples.py`.

- `serve_kimi.sh` starts Kimi K2.6 on vLLM 0.30.0 (tensor parallel over 8 GPUs on one node) as an OpenAI-compatible server on port 8000.
- `free_caption.py` sends each clip to the server as timestamped frames (first 6.583 s, 2 fps, longer side 672) and asks for the free parts as JSON.
- `check.py` checks each answer: all fields present, length, no Chinese, Japanese or Korean text outside double-quoted on-screen text, no input labels, shot cuts well formed. A failing answer goes back to the model with the problems listed, up to three times. It also checks finished six-part prompts: `python check.py prompt.txt`.

Run:

```
bash serve_kimi.sh                          # or: bash serve_kimi.sh /path/to/Kimi-K2.6
python free_caption.py clips.jsonl out/ --server http://127.0.0.1:8000/v1/chat/completions --workers 96
```

`clips.jsonl` holds one `{"id": ..., "video": ...}` per line; the video can be a local path, an http(s) URL or an object-store URL (fetched with the storage command, `$STORAGE_CLI`, default `solar-storage`). Output: `out/<id>.json` with the fields, the check results and the token usage; clips that still fail after three rewrites go to `out/_failed/<id>.json`. Clips already written are skipped, so a run can be restarted. Use `--shard k/n` to split a list across several processes.

The prompt-writing skill and `conditions.py` (the fixed sentences per condition type and the assembly of the six parts) are on the `main` branch under `skills/solarrenderer-caption/`.
