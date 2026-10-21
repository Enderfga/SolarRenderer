<p align="center"><img src="assets/teaser.gif" width="100%"/></p>

<h1 align="center">SolarRenderer: Few-Step Generative Reference Rendering</h1>

<p align="center">
  🌐 <a href="https://enderfga.cn/SolarRenderer">Project Page</a> •
  📄 <a href="https://arxiv.org/abs/2610.28630">Paper</a> •
  🤗 <a href="https://huggingface.co/Enderfga/SolarRenderer">Weights</a> •
  🤗 <a href="https://huggingface.co/datasets/Enderfga/SolarRenderer-Data">Data</a> •
  💻 <a href="https://github.com/Enderfga/SolarRenderer/tree/data">Data code</a>
</p>

**SolarRenderer** turns a structure video from a game engine or an agent (grey box, semantic map, skeletons, depth,
body mesh, or a sketch it never saw in training) into a finished video, with an optional first frame and a prompt
deciding the look. It runs on **MiniMax-H3** with two LoRAs and renders a clip in **four steps**.
A block-causal pair of the same LoRAs streams the clip block by block.
SolarRenderer is built on [SolarWM](https://github.com/Junchao-cs/SolarWM), the open framework for
training and inference of video world models ([paper](https://arxiv.org/abs/2609.02886)).

## News

- **[2026-10]** Code, weights, the shareable training data and the benchmark released.

## Highlights

- 🧱 **Any structure, one model**: no extra branch and no condition-type label; the prompt names the structure.
- ⚡ **Four steps**: structure-guided post-training, 12.3× faster sampling than 50 steps.
- 🎨 **Any look**: the same structure renders as photoreal, anime, pixel art or ink, set by the first frame and prompt.
- 🌊 **Streaming**: a block-causal version of both LoRAs renders the video block by block, each block shown as soon as
  it is done ([guide](docs/inference.md#stream-a-clip)).
- 📊 **Open benchmark**: 208 hard cases and 53 held-out clips, with scoring code, our renders and every judge verdict.

## Gallery

Structure video (left) and SolarRenderer (right).

| Scene grey box | Semantic segmentation |
|:---:|:---:|
| <img src="assets/gallery/greybox.gif" width="100%"> | <img src="assets/gallery/semantic.gif" width="100%"> |
| **Object masks + skeletons** | **SMPL body mesh** |
| <img src="assets/gallery/masks_skeletons.gif" width="100%"> | <img src="assets/gallery/smpl.gif" width="100%"> |
| **Near-field shaded geometry** | **Pencil storyboard** (unseen) |
| <img src="assets/gallery/near_depth.gif" width="100%"> | <img src="assets/gallery/unseen_pencil.gif" width="100%"> |
| **Point cloud** (unseen) | **Child's crayon drawing** (unseen) |
| <img src="assets/gallery/unseen_pointcloud.gif" width="100%"> | <img src="assets/gallery/unseen_crayon.gif" width="100%"> |
| **Voxels** (unseen) | **Wireframe** (unseen) |
| <img src="assets/gallery/unseen_voxels.gif" width="100%"> | <img src="assets/gallery/unseen_wireframe.gif" width="100%"> |

More structure types, styles and comparisons are on the [project page](https://enderfga.cn/SolarRenderer/#structures).

## Quick Start

```bash
git clone https://github.com/Enderfga/SolarRenderer && cd SolarRenderer
pip install -e . && pip install flash-attn==2.7.4.post1 --no-build-isolation
```

```bash
python scripts/prepare_h3.py --dir ./MiniMax-H3
hf download Enderfga/SolarRenderer solarrenderer_control_lora.safetensors solarrenderer_4step_lora.safetensors \
    --local-dir ./weights
```

```bash
python scripts/render.py --h3 ./MiniMax-H3 \
    --control weights/solarrenderer_control_lora.safetensors \
    --fewstep weights/solarrenderer_4step_lora.safetensors \
    --structure examples/sem-highway_synthwave/structure.mp4 \
    --first examples/sem-highway_synthwave/first.png \
    --prompt examples/sem-highway_synthwave/prompt.txt --out render.mp4
```

The example is a [synthwave highway run](examples/sem-highway_synthwave) (semantic map in, neon render out).
Add `--causal` with the two causal LoRAs to stream the clip block by block
([details](docs/inference.md#stream-a-clip)).

- ✍️ **Prompts and first frames**: give [`skills/solarrenderer-caption`](skills/solarrenderer-caption) to your coding
  agent ([guide](docs/prompts.md)).
- 🎬 **Inference details**: inputs, sizes and timing ([guide](docs/inference.md)).
- 🏋️ **Training** on one 8-GPU node ([guide](docs/training.md)).
- 📊 **Benchmark scoring** ([guide](eval/README.md)).

## Ways to run

| | For | Start here |
|---|---|---|
| **Reference script** | The reference implementation: one GPU, a clip at a time | `scripts/render.py` above |
| **ComfyUI** | A browser page and node workflows with no code; the low-memory workflow fits a 16 GB GPU (e.g. RTX 4090) | [`comfyui/`](comfyui/README.md) |
| **Streaming on [Reactor](https://reactor.inc)** | The fastest path: a four-GPU streaming service with a browser demo, validated by the Reactor team on four B200 GPUs | [`reactor/`](reactor/README.md) |

The Reactor integration is a community contribution; thanks to [Rising0321](https://github.com/Rising0321) and the Reactor team.

## Citation

If you use SolarRenderer in your research, please cite SolarRenderer and SolarWM, the framework on which it is built.

```bibtex
@misc{fang2026solarrenderer,
  title  = {SolarRenderer: Few-Step Generative Reference Rendering},
  author = {Fang, Guian and Qian, Shengju and Lu, Dongyue and
            Huang, Junchao and Jiang, Lutao and Wang, Xin and
            Song, Yiren and Jiang, Li and Shou, Mike Zheng},
  year   = {2026},
  eprint = {2610.28630},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2610.28630}
}
```

[SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models](https://arxiv.org/abs/2609.02886):

```bibtex
@misc{huang2026solarwmopendatascalable,
      title={SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models},
      author={Junchao Huang and Guian Fang and Shengju Qian and Xianghao Kong and Zhuoran Zhao and Wei Huang and Yihua Du and Zixin Zhang and Justin Cui and Yuchao Gu and Yukang Chen and Xinting Hu and Tianyu He and Shaoshuai Shi and Zhuotao Tian and Xin Wang and Mike Zheng Shou and Li Jiang},
      year={2026},
      eprint={2609.02886},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.02886},
}
```

<details>
<summary>Acknowledgement</summary>

SolarRenderer builds on [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), starts its four-step LoRA from
[DMAD](https://huggingface.co/ZhengmingYu/DMAD), and shares its training framework with
[SolarWM](https://github.com/Junchao-cs/SolarWM).
</details>

## License

Code: Apache 2.0. Weights: [MiniMax H3 Community License](https://huggingface.co/Enderfga/SolarRenderer/blob/main/LICENSE),
including its territorial restrictions (EU, UK, Republic of Korea and US excluded) and Acceptable Use Policy; the
four-step LoRA is fine-tuned from DMAD under the same license.
