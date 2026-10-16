"""upload_hf.py: push a staged folder to the Hub dataset repo, resumably.

    HF_TOKEN=... upload_hf.py <stage dir> [--repo Enderfga/SolarRenderer-Data] [--workers 16]

upload_large_folder keeps its progress in <stage>/.cache/huggingface, so a run that stops can simply be started again.
"""
import argparse
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("stage")
ap.add_argument("--repo", default="Enderfga/SolarRenderer-Data")
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
HfApi().upload_large_folder(repo_id=a.repo, folder_path=a.stage, repo_type="dataset", private=True,
                            num_workers=a.workers, ignore_patterns=["*.part"])
print("uploaded", a.stage, flush=True)
