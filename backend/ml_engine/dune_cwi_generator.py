import json
import re
from pathlib import Path
from typing import List, Tuple
from backend.schemas.caption_spec import (
    CaptionSpec,
    WordToken,
    SpeakerInfo,
    ProsodyMetrics,
    WorkAreaBox,
    NonSpeechEvent,
)
from backend.ml_engine.typography_math import compute_typography_spec

# CWI Speaker Palette
SPEAKER_PALETTE = [
    {"id": "SPK_00", "name": "Chani", "color_hex": "#4FC3F7"},
    {"id": "SPK_01", "name": "Paul Atreides", "color_hex": "#FFD54F"},
    {"id": "SPK_02", "name": "Lady Jessica", "color_hex": "#FF8A80"},
    {"id": "SPK_03", "name": "Stilgar", "color_hex": "#69F0AE"},
    {"id": "SPK_04", "name": "Reverend Mother", "color_hex": "#B388FF"},
    {"id": "SPK_05", "name": "Baron Harkonnen", "color_hex": "#FFAB40"},
]


def parse_srt_time(ts_str: str) -> int:
    h, m, s = ts_str.strip().replace(",", ".").split(":")
    return int(round((int(h) * 3600 + int(m) * 60 + float(s)) * 1000))


def convert_dune_srt_to_cwi(
    srt_path: str = "D:/dune/final_video.srt",
    output_json_path: str = "D:/dune/final_video.cwi.json",
    max_segments: int | None = None,
) -> List[CaptionSpec]:
    srt_file = Path(srt_path)
    if not srt_file.exists():
        raise FileNotFoundError(f"SRT not found at {srt_path}")

    content = srt_file.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", content.strip())
    if max_segments is not None:
        blocks = blocks[:max_segments]

    specs: List[CaptionSpec] = []
    current_spk_idx = 0
    last_end_ms = 0

    for idx, block in enumerate(blocks):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue

        time_line = lines[1]
        m = re.match(r"(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)", time_line)
        if not m:
            continue

        seg_start_ms = parse_srt_time(m.group(1))
        seg_end_ms = parse_srt_time(m.group(2))
        if seg_end_ms <= seg_start_ms:
            seg_end_ms = seg_start_ms + 1000

        text_content = " ".join(lines[2:])
        clean_text = re.sub(r"<[^>]+>", "", text_content).strip()
        if not clean_text:
            continue

        # Turn-taking heuristic: if pause > 1500ms or leading dash '-', switch speaker
        if seg_start_ms - last_end_ms > 1500 or clean_text.startswith("-"):
            current_spk_idx = (current_spk_idx + 1) % len(SPEAKER_PALETTE)
        last_end_ms = seg_end_ms

        spk_meta = SPEAKER_PALETTE[current_spk_idx]
        seg_id = f"seg_{idx + 1:04d}"

        # Detect non-speech events (e.g. [whispers], (breathing), [groans])
        events: List[NonSpeechEvent] = []
        event_matches = re.findall(r"(\[[^\]]+\]|\([^\)]+\))", clean_text)
        for ev_raw in event_matches:
            ev_lower = ev_raw.lower()
            ev_type = "BREATHING"
            ev_label = "[breathing]"
            if "sigh" in ev_lower:
                ev_type, ev_label = "SIGH", "[sighs]"
            elif "gasp" in ev_lower:
                ev_type, ev_label = "GASP", "[gasps]"
            elif "laugh" in ev_lower:
                ev_type, ev_label = "LAUGHTER", "[laughter]"
            elif "scream" in ev_lower:
                ev_type, ev_label = "SCREAM", "[screaming]"
            elif "music" in ev_lower or "theme" in ev_lower or "instrumental" in ev_lower:
                ev_type, ev_label = "BGM", "<|BGM|>"
            elif "cough" in ev_lower:
                ev_type, ev_label = "COUGH", "[coughing]"

            events.append(
                NonSpeechEvent(
                    type=ev_type,  # type: ignore
                    label=ev_label,  # type: ignore
                    start_ms=seg_start_ms,
                    end_ms=min(seg_end_ms, seg_start_ms + 1800),
                    confidence=0.95,
                )
            )

        # Strip brackets for speech words
        spoken_text = re.sub(r"(\[[^\]]+\]|\([^\)]+\))", "", clean_text).strip()
        spoken_text = spoken_text.lstrip("- ").strip()
        raw_words = spoken_text.split()
        if not raw_words:
            # Event only segment
            if events:
                specs.append(
                    CaptionSpec(
                        schema_version="2.0-cwi",
                        segment_id=seg_id,
                        presentation_offset_ms=1750,
                        work_area_box=WorkAreaBox(bottom_pct=20.0, bg_opacity=0.90),
                        non_speech_events=events,
                        words=[],
                    )
                )
            continue

        # Distribute timing proportionally across word lengths
        total_chars = sum(max(1, len(w)) for w in raw_words)
        total_duration = max(len(raw_words) * 160, seg_end_ms - seg_start_ms)

        word_tokens: List[WordToken] = []
        curr_time = seg_start_ms

        for w_idx, w_str in enumerate(raw_words):
            char_ratio = len(w_str) / max(1, total_chars)
            w_dur = max(140, int(round(total_duration * char_ratio)))
            w_start = curr_time
            w_end = w_start + w_dur
            curr_time = w_end

            # Emphasized words (all uppercase, ending with !, or dramatic keywords)
            is_loud = w_str.isupper() and len(w_str) > 1 or "!" in w_str
            is_whisper = w_str.startswith("(") or "..." in w_str

            if is_loud:
                rms_db = -8.0
                f0_hz = 220.0
                prominence = 0.92
            elif is_whisper:
                rms_db = -38.0
                f0_hz = 95.0
                prominence = 0.25
            else:
                rms_db = -18.0 + (hash(w_str) % 7)
                f0_hz = 135.0 + (hash(w_str) % 40)
                prominence = 0.45 + (hash(w_str) % 30) / 100.0

            # Isolated silence hold heuristic (first or last word with pause)
            is_hold = (w_idx == 0 or w_idx == len(raw_words) - 1) and (
                total_duration / len(raw_words) > 450
            )

            prosody = ProsodyMetrics(
                rms_db=round(rms_db, 1),
                f0_word_hz=round(f0_hz, 1),
                f0_speaker_median_hz=130.0,
                prominence_score=round(prominence, 3),
                isolated_silence_hold=is_hold,
            )

            typo = compute_typography_spec(
                prosody=prosody,
                duration_ms=w_dur,
                intensity=1.0,
                is_safari=False,
                locale="en",
                is_offscreen_or_sfx=False,
            )

            word_tokens.append(
                WordToken(
                    word_id=f"w_{idx + 1:04d}_{w_idx + 1:02d}",
                    text=w_str,
                    start_ms=w_start,
                    end_ms=w_end,
                    duration_ms=w_dur,
                    speaker=SpeakerInfo(
                        id=spk_meta["id"],
                        state="stable",
                        color_hex=spk_meta["color_hex"],
                        compass_bearing_deg=(-30 + (current_spk_idx * 25)) % 180 - 90,
                        compass_spread_deg=14,
                    ),
                    prosody=prosody,
                    typography=typo,
                )
            )

        spec = CaptionSpec(
            schema_version="2.0-cwi",
            segment_id=seg_id,
            presentation_offset_ms=1750,
            work_area_box=WorkAreaBox(bottom_pct=20.0, bg_opacity=0.90),
            non_speech_events=events,
            words=word_tokens,
        )
        specs.append(spec)

    out_file = Path(output_json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps([s.model_dump() for s in specs], indent=2), encoding="utf-8"
    )
    return specs


if __name__ == "__main__":
    generated = convert_dune_srt_to_cwi()
    print(f"Successfully generated {len(generated)} Dune CWI segments in D:/dune/final_video.cwi.json")
