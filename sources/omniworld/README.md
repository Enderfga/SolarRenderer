# OmniWorld-Game

Input: the OmniWorld-Game part of [OmniWorld](https://huggingface.co/datasets/InternRobotics/OmniWorld), game
footage split into segments with per-frame `color/<frame>.png`. Its videos are already public, so the release
ships one line per training clip instead of video: the upstream `uid` and `split`, the source frame rate, the exact
upstream frames the clip shows (`source_frames`) and the caption. Training used the first frame and the caption only.

To rebuild a clip: read `color/<frame>.png` for each entry of `source_frames` in order, scale it with area
interpolation (OpenCV `INTER_AREA`) so that it covers 1344x768, centre-crop to 1344x768, and write the frames at
24 fps.

The conversion script that picked the frames is not included here.
