"""
FFmpeg libass Offline Burn-In Wrapper
Reference: docs/07_client_and_rendering/ffmpeg_ass_burnin.md (Section 3)
"""

from pathlib import Path
import subprocess
from typing import List

try:
    from backend.schemas.caption_spec import CaptionSpec
except ImportError:
    from typing import Any
    CaptionSpec = Any  # type: ignore

from frontend_renderer.offline_burnin.ass_compiler import compile_captionspec_to_ass


def burn_captions_into_video(
    input_video_path: str,
    output_video_path: str,
    specs: List[CaptionSpec],
    fonts_dir: str = "frontend_renderer/assets/fonts",
    video_width: int = 1920,
    video_height: int = 1080,
) -> Path:
    """
    Executes ffmpeg -vf "ass=...:fontsdir=..." to burn-in deterministic CWI captions.
    """
    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path = out_path.with_suffix(".ass")

    ass_content = compile_captionspec_to_ass(
        specs=specs, video_width=video_width, video_height=video_height
    )
    ass_path.write_text(ass_content, encoding="utf-8")

    vf_filter = f"ass={ass_path.as_posix()}:fontsdir={Path(fonts_dir).as_posix()}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video_path),
        "-vf",
        vf_filter,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path
