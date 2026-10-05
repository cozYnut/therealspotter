"""
Heat and pilot names for a race-day run.

A provider is asked when a heat starts. The website provider (fril live)
comes later; without it — or without internet — pilots get a sequential name
from the heat number and channel, e.g. H012-R3.
"""

from typing import Dict, List, Optional


class HeatInfoProvider:
    def heat_info(self, wall_time: float, channels: List[str]) -> Optional[dict]:
        """{"heat": int|None, "round": str|None, "pilots": {channel: name}} or None."""
        return None


def fallback_name(heat: int, channel: str) -> str:
    return f"H{heat:03d}-{channel}"


def pilot_name(info: Optional[dict], heat: int, channel: str) -> str:
    name = ((info or {}).get("pilots") or {}).get(channel)
    return name or fallback_name(heat, channel)
