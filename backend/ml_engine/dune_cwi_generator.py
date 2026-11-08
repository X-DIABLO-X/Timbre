"""
Deterministic SRT-to-CWI Converter (Fixes speaker auto-rotation bug on >1.5s pauses
and adds punctuation-aware word boundary timing).
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from backend.schemas.caption_spec import (
    CaptionSpec,
    NonSpeechEvent,
    ProsodyMetrics,
    SpeakerInfo,
    WordToken,
    WorkAreaBox,
)
from backend.ml_engine.typography_math import compute_typography_spec
from backend.ml_engine.generate_dune_cwi_v2 import estimate_word_syllables

SPEAKER_PALETTE: List[Dict[str, object]] = [
    {"id": "SPK_00", "name": "Chani", "color": "#4FC3F7", "f0_med": 195.0, "bearing": -30.0},
    {"id": "SPK_01", "name": "Paul Atreides", "color": "#FFD54F", "f0_med": 118.0, "bearing": 20.0},
    {"id": "SPK_02", "name": "Lady Jessica", "color": "#FF8A80", "f0_med": 182.0, "bearing": -15.0},
    {"id": "SPK_03", "name": "Stilgar", "color": "#69F0AE", "f0_med": 102.0, "bearing": 45.0},
    {"id": "SPK_04", "name": "Reverend Mother", "color": "#B388FF", "f0_med": 165.0, "bearing": -40.0},
    {"id": "SPK_05", "name": "Baron Harkonnen", "color": "#FFAB40", "f0_med": 88.0, "bearing": 60.0},
]


def parse_srt_time(ts_str: str) -> int:
    ts_str = ts_str.strip().replace(".", ",")
    h, m, rest = ts_str.split(":")
    s, ms = rest.split(",")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def convert_dune_srt_to_cwi(
    srt_path: str = "D:/dune/final_video.srt",
    output_json_path: str = "D:/dune/final_video.cwi.json",
    max_segments: Optional[int] = None,
) -> List[CaptionSpec]:
    srt_file = Path(srt_path)
    if not srt_file.exists():
        alt = Path("data/final_video.srt")
        if alt.exists():
            srt_file = alt

    raw_text = srt_file.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", raw_text.strip())

    specs_out: List[CaptionSpec] = []
    current_speaker_idx = 0
    prev_end_ms = 0

    for b_idx, block in enumerate(blocks):
        if max_segments is not None and len(specs_out) >= max_segments:
            break

        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        ts_idx = 0 if "-->" in lines[0] else (1 if "-->" in lines[1] else -1)
        if ts_idx < 0:
            continue

        m = re.search(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[ts_idx],
        )
        if not m:
            continue

        start_ms = parse_srt_time(m.group(1))
        end_ms = max(start_ms + 200, parse_srt_time(m.group(2)))
        text_raw = " ".join(lines[ts_idx + 1 :])
        has_dialogue_dash = text_raw.lstrip().startswith("-")
        text_clean = re.sub(r"<[^>]+>", "", text_raw).lstrip("- ").strip()
        if not text_clean:
            continue

        # FIX: Do NOT blindly rotate speakers on every >1500ms pause!
        # Keep opening narration (<118,000ms) locked to Chani (SPK_00), and after 118s only
        # switch speakers when an explicit dialogue turn marker ('-') or scene break (>6,500ms) occurs.
        if start_ms < 118000:
            current_speaker_idx = 0
        elif has_dialogue_dash or (start_ms - prev_end_ms > 6500):
            current_speaker_idx = 1 if current_speaker_idx == 0 else ((current_speaker_idx % 5) + 1)

        prev_end_ms = end_ms
        spk = SPEAKER_PALETTE[current_speaker_idx % len(SPEAKER_PALETTE)]

        words_raw = [w for w in text_clean.split() if w.strip()]
        if not words_raw:
            continue

        total_dur = max(len(words_raw) * 120, end_ms - start_ms)
        lead_in = min(40, int(total_dur * 0.02))
        trail_out = min(60, int(total_dur * 0.03))
        avail_ms = max(len(words_raw) * 100, total_dur - lead_in - trail_out)

        weights = [estimate_word_syllables(w) * 1.45 + len(w) * 0.18 for w in words_raw]
        pause_after = [
            140 if w.endswith((",", ";", ":", "—")) else (200 if w.endswith((".", "?", "!")) else 10)
            for w in words_raw
        ]
        pause_after[-1] = 0
        total_pause = min(int(avail_ms * 0.35), sum(pause_after))
        net_speech_ms = max(len(words_raw) * 85, avail_ms - total_pause)
        total_w = max(1e-5, sum(weights))

        cursor_ms = start_ms + lead_in
        word_tokens: List[WordToken] = []

        for w_idx, (w_str, w_weight, p_ms) in enumerate(zip(words_raw, weights, pause_after)):
            w_dur = max(90, int(round(net_speech_ms * (w_weight / total_w))))
            w_start = cursor_ms
            w_end = w_start + w_dur
            cursor_ms = w_end + p_ms

            is_shout = w_str.isupper() and len(w_str) > 1 or w_str.endswith("!")
            rms_db = -9.0 if is_shout else -18.5
            f0_med = float(spk["f0_med"])  # type: ignore
            f0_word = f0_med * (1.12 if is_shout else 1.0)
            prominence = 0.88 if is_shout else 0.42

            prosody = ProsodyMetrics(
                rms_db=rms_db,
                f0_word_hz=round(f0_word, 2),
                f0_speaker_median_hz=f0_med,
                prominence_score=prominence,
                isolated_silence_hold=False,
            )
            typo = compute_typography_spec(prosody=prosody, duration_ms=w_dur, intensity=1.0)

            word_tokens.append(
                WordToken(
                    word_id=f"w_{len(specs_out) + 1:04d}_{w_idx + 1:02d}",
                    text=w_str,
                    start_ms=w_start,
                    end_ms=w_end,
                    duration_ms=w_dur,
                    speaker=SpeakerInfo(
                        id=str(spk["id"]),
                        state="stable",
                        color_hex=str(spk["color"]),
                        compass_bearing_deg=float(spk["bearing"]),  # type: ignore
                        compass_spread_deg=12.0,
                    ),
                    prosody=prosody,
                    typography=typo,
                )
            )

        specs_out.append(
            CaptionSpec(
                schema_version="2.0-cwi",
                segment_id=f"seg_{len(specs_out) + 1:04d}",
                presentation_offset_ms=1750,
                work_area_box=WorkAreaBox(bottom_pct=20.0, bg_opacity=0.90),
                non_speech_events=[],
                words=word_tokens,
            )
        )

    out_file = Path(output_json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps([s.model_dump() for s in specs_out], indent=2),
        encoding="utf-8",
    )
    return specs_out


if __name__ == "__main__":
    generated = convert_dune_srt_to_cwi()
    print(f"Successfully generated {len(generated)} Dune CWI segments in D:/dune/final_video.cwi.json")
