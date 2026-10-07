"""Race-day mode: capture a 2×2 feed, learn the track, show results (docs/RACE_DAY_PLAN.md)."""

import os
from pathlib import Path


def _clip_cached() -> bool:
    """Is the CLIP model (open_clip ViT-B-32 'openai', from Hugging Face Hub) already downloaded?"""
    hub = os.environ.get("HF_HUB_CACHE") or str(Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
    return (Path(hub) / "models--timm--vit_base_patch32_clip_224.openai").exists()


# Loading CLIP makes Hugging Face check online for a newer model version — once per analysed run.
# At a race the internet may be slow or missing, so when the model is already on this laptop, load it
# from disk only (no requests, no waiting). Without the model it stays online so it can download once.
# Set before anything imports open_clip / huggingface_hub; an explicit HF_HUB_OFFLINE in the
# environment wins.
if _clip_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
