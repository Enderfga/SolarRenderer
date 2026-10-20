"""Resolve deployment paths once, before starting the model worker."""

import subprocess
from pathlib import Path

import yaml
from solarrenderer_model import SolarSettings

SOURCE_REVISION = "a2371eaecff9a8ab378b5ec36199c5863d138074"


def read_settings(config_path: Path, weights_root: Path) -> SolarSettings:
    config = yaml.safe_load(config_path.read_text())
    source = Path(config["source"])
    if not source.is_absolute():
        source = (config_path.parent / source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"Expected upstream source {SOURCE_REVISION}; found {revision}."
        )
    root = weights_root.resolve()
    scratch = root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return SolarSettings(
        source=source,
        h3=root / "MiniMax-H3",
        control=root / "solarrenderer_causal_control_lora.safetensors",
        fewstep=root / "solarrenderer_causal_4step_lora.safetensors",
        scratch=scratch,
    )
