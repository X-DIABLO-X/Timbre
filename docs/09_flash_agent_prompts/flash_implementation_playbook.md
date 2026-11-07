# Turnkey Flash Model Implementation Playbook & Agent Prompts (`flash_implementation_playbook.md`)

> **Purpose:** This file provides **copy-paste system prompts and deterministic build checklists** for feeding this documentation archive to a smaller, fast code-generation model (such as Gemini Flash) so it writes the entire production codebase without needing to invent any equations, constants, schemas, or file structures.

---

## 1. Master System Prompt for a Single Flash Model (Full-Repo Generation)

Copy and paste the prompt below when giving this unzipped documentation folder to a Flash model:

```markdown
You are a deterministic code-generation engine implementing the Automated "Caption with Intention" (CWI) Platform (`v2.0-cwi`).
All architectural decisions, mathematical equations, neural network layer tap points, CSS keyframes, and file paths have already been designed for you in the documentation directory.

STRICT EXECUTION RULES:
1. DO NOT invent new file paths, rename classes, or alter JSON keys. Follow `00_overview/folders.md` and `01_contracts_and_schemas/caption_spec_schema.md` verbatim.
2. Enforce every mathematical constant in `01_contracts_and_schemas/typography_and_math_constants.md`:
   - Screen work area: strictly lower `20%` (`bottom_pct = 20`)
   - Caption box backdrop: `90%` opacity black (`rgba(0, 0, 0, 0.90)`, `.ass` `&H1A000000&`)
   - Dual-Clock Read-Ahead Buffer: `presentation_offset_ms = 1750` (`-1.75s`), minimum floor `420 ms`
   - Immutable Post-Onset Text Rule: once `t_present >= word.start_ms`, freeze DOM text and layout geometry
   - Channel 1 (Colour Turn): horizontal `background-clip: text` wipe from `t_start` to `t_end`
   - Channel 2 (Pop): `1.15x` scale impulse at word onset `t_0`
   - Channel 3 (Crest): `0.72x` to `1.62x` scale (`3%` to `12%` screen height; `5%` normal = `1.0x`), settling to `0.70` of peak
   - Channel 4 (Weight): `80 Hz -> wght 900`, `250 Hz -> wght 340`, anchored to speaker running median F0 + Layer-9 prominence (plus Safari `wght: 404` workaround)
   - Channel 5 (Hold & Width): `-0.525em` vertical lift when isolated by >= 250ms silence on both sides; `wdth` `25..151` (`100` for Korean `Noto Sans KR`)
3. In `extension/offscreen/offscreen.js`, ALWAYS connect `sourceNode.connect(audioCtx.destination)` immediately after `createMediaStreamSource` so `chrome.tabCapture` never mutes the user's browser tab.
4. Generate files in the exact build order below and run `pytest` on `tests/test_all_256_invariants.py` to verify all 256 unit test invariants pass.
```

---

## 2. Prompts for Spawning Four Parallel Flash Sub-Agents

If running four concurrent Flash coding agents, assign each agent its dedicated context slice:

### Agent 1 Prompt — Parallel Audio ML & Prosody Backend (`backend/` & `wearable_pi_node/`)
- **Context Files to Provide:**
  - `00_overview/folders.md`
  - `01_contracts_and_schemas/caption_spec_schema.md`
  - `01_contracts_and_schemas/typography_and_math_constants.md`
  - `02_tools_and_environment/tools.md`
  - `02_tools_and_environment/hardware_profiles.md`
  - `03_models_and_weights/models.md`
  - `04_pipelines/pipelines.md`
  - `04_pipelines/stream_0_audio_vad_separation.md`
  - `04_pipelines/stream_1_asr_alignment.md`
  - `04_pipelines/stream_2_diarization.md`
  - `04_pipelines/stream_3_prosody_and_stress.md`
  - `04_pipelines/stream_4_paralinguistics_affect.md`
  - `07_client_and_rendering/wearable_spatial_haptics.md`
- **Files Agent 1 Must Emit:**
  1. `pyproject.toml` & `shared_schemas/caption_spec.schema.json` & `shared_schemas/speaker_palette.json`
  2. `backend/config.py`
  3. `backend/schemas/caption_spec.py`
  4. `backend/ml_engine/typography_math.py`
  5. `backend/ml_engine/stage0_vad_separator.py`
  6. `backend/ml_engine/stream1_asr.py`
  7. `backend/ml_engine/stream2_diarization.py`
  8. `backend/ml_engine/stream3_prosody_stress.py`
  9. `backend/ml_engine/stream4_paralinguistics.py`
  10. `backend/ml_engine/joiner.py`
  11. `backend/ml_engine/orchestrator.py`
  12. `backend/main.py` & `backend/native_host/launcher.py` & `wearable_pi_node/pi_neckband_daemon.py`

### Agent 2 Prompt — Custom Model Training & ONNX Distillation (`training/`)
- **Context Files to Provide:**
  - `00_overview/folders.md`
  - `03_models_and_weights/models.md`
  - `05_training_and_distillation/training.md`
  - `05_training_and_distillation/datasets.md`
  - `06_optimizations/optimizations.md`
- **Files Agent 2 Must Emit:**
  1. `training/models/whistress_layer9_head.py`
  2. `training/models/emotion2vec_multitask_head.py`
  3. `training/datasets/tinystress_burnc_dataset.py`
  4. `training/datasets/affect_vocalization_dataset.py`
  5. `training/train_whistress_head.py`
  6. `training/train_emotion2vec_head.py`
  7. `training/export_onnx_hybrid.py`

### Agent 3 Prompt — Chrome Manifest V3 Extension & WebGPU Worker (`extension/`)
- **Context Files to Provide:**
  - `00_overview/folders.md`
  - `01_contracts_and_schemas/caption_spec_schema.md`
  - `06_optimizations/optimizations.md`
  - `07_client_and_rendering/chrome_extension_mv3.md`
  - `07_client_and_rendering/cwi_dom_renderer.md`
- **Files Agent 3 Must Emit:**
  1. `extension/manifest.json`
  2. `extension/background.js`
  3. `extension/offscreen/offscreen.html` & `extension/offscreen/offscreen.js`
  4. `extension/offscreen/pcm-worklet.js`
  5. `extension/workers/webgpu_worker.js`
  6. `extension/content/content.js` & `extension/content/overlay.css`
  7. `extension/popup/popup.html` & `extension/popup/popup.js`

### Agent 4 Prompt — Deterministic `CwiRenderer`, React 19 Player & FFmpeg `.ass` Burn-In (`frontend_renderer/` & `tests/`)
- **Context Files to Provide:**
  - `00_overview/folders.md`
  - `01_contracts_and_schemas/caption_spec_schema.md`
  - `01_contracts_and_schemas/typography_and_math_constants.md`
  - `07_client_and_rendering/cwi_dom_renderer.md`
  - `07_client_and_rendering/ffmpeg_ass_burnin.md`
  - `07_client_and_rendering/wearable_spatial_haptics.md`
  - `08_testing_and_verification/test_suite_256_invariants.md`
- **Files Agent 4 Must Emit:**
  1. `frontend_renderer/src/core/types.ts`
  2. `frontend_renderer/src/core/dual_clock.ts`
  3. `frontend_renderer/src/core/cwi_renderer.ts`
  4. `frontend_renderer/src/core/voice_compass.ts`
  5. `frontend_renderer/src/player/App.tsx`
  6. `frontend_renderer/offline_burnin/ass_compiler.py`
  7. `frontend_renderer/offline_burnin/ffmpeg_burnin.py`
  8. `tests/test_all_256_invariants.py`
