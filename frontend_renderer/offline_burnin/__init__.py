"""
Offline Burn-in Package for Automated Caption with Intention (CWI)
"""

from .ass_compiler import compile_captionspec_to_ass, hex_rgb_to_ass_bgr, ms_to_ass_timestamp
from .sync_engine import DeterministicSyncEngine
from .ffmpeg_burnin import burn_captions_into_video

__all__ = [
    "compile_captionspec_to_ass",
    "hex_rgb_to_ass_bgr",
    "ms_to_ass_timestamp",
    "DeterministicSyncEngine",
    "burn_captions_into_video",
]
