# Product Vision, Perceptual Pillars & HCI Design Specification (`idea.md`)

> **Primary References:** [Caption With Intention Official Site](https://www.captionwithintention.org/), [Weave (SNU, 2026)](https://xlaude2040.com/weave/), [CwiRenderer Automated Project](https://stardance.hackclub.com/projects/63123), [Cannes Lions Grand Prix Coverage](https://screenmag.com/fcb-chicagos-caption-with-intention-wins-two-grand-prix-honors-at-cannes-lions/), [The Clios Winner Dossier](https://clios.com/winners-gallery/details/195157/)

---

## 1. Executive Summary & Historical Context

Traditional closed captioning has remained functionally stagnant for over 50 years since its 1971 introduction and the first live-captioned Academy Awards broadcast in 1982 ([Caption With Intention](https://www.captionwithintention.org/)). Standard white-text-on-black-box captions treat cinema and live dialogue as a flat text utility, stripping away vocal delivery, speaker identity, comedic timing, and dramatic tension for the **466 million people worldwide** living with hearing disabilities ([Caption With Intention](https://www.captionwithintention.org/)).

Developed over 10 to 11 months by **FCB Chicago** (led by Chief Creative Officer Andres Ordonez, Executive Creative Director Bruno Mazzotti—who was raised by two deaf parents—and Associate Creative Directors Gustavo Dallegrave, Zach Frazer, Frederik, and So A Ryu), production company **[Rakish Entertainment](https://clios.com/winners-gallery/details/195157/)**, and a **125-person Deaf and Hard of Hearing (DHH) testing panel** from the **Chicago Hearing Society**, **[Caption with Intention©](https://www.captionwithintention.org/)** won two Grand Prix honors at the 2025 Cannes Lions (Design and Digital Craft) and was recognized by the Academy of Motion Picture Arts and Sciences as a RAISE partner ([Screen Magazine](https://screenmag.com/fcb-chicagos-caption-with-intention-wins-two-grand-prix-honors-at-cannes-lions/)).

While the original campaign was authored manually using Adobe After Effects templates and the **[Roboto Flex](https://www.csstypestudio.com/library/font/roboto-flex)** variable font, our engineering blueprint automates the complete specification deterministically in real time and offline VOD modes by synthesizing:
1. The official [captionwithintention.org](https://www.captionwithintention.org/) design rules,
2. The [Cannes Lions](https://screenmag.com/fcb-chicagos-caption-with-intention-wins-two-grand-prix-honors-at-cannes-lions/), [Clio](https://clios.com/winners-gallery/details/195157/), and [LIA 2025](https://2025.liaentries.com/winners/?id_entry=101992&id_medium=1&id_submedium=21&id_category=0&view=details&range=w&page=55&keyword&medium&category&award&country&title_brand&credits&company_name&city&proceed_simple_search=false&proceed_advanced_search=false) award dossiers,
3. Glyph-by-glyph measurements of the official After Effects template and PDF guidelines documented in the **[Weave](https://xlaude2040.com/weave/)** (SNU, 2026) and **[CwiRenderer](https://stardance.hackclub.com/projects/63123)** engineering implementations.

---

## 2. The Three Core Perceptual Pillars & Canonical Film Benchmarks

Every feature in the platform serves one of three perceptual pillars validated on classic cinema benchmarks ([Caption With Intention](https://www.captionwithintention.org/)):

| Perceptual Pillar | Problem Solved in Traditional Captions | Automated Engineering Mechanism | Canonical Film Benchmarks |
| :--- | :--- | :--- | :--- |
| **1. Attribution** *(Speaker Identification via Color)* | Multi-character, overlapping, or off-screen dialogue fails to indicate who is speaking without clunky `[NAME]:` text prefixes that waste reading time. | Assigns each distinct speaker a persistent, high-contrast color from a curated palette via streaming speaker diarization ([Streaming Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) + ERes2Net/CAM++). Undecided speakers default to neutral grey. | *Toy Story* (character-specific color wheel attribution for Buzz Lightyear) & *Ferris Bueller's Day Off* ([Caption With Intention](https://www.captionwithintention.org/)). |
| **2. Synchronization** *(Word-Onset Color Wipe & Read-Ahead)* | Static block captions appear all at once, ruining punchlines and dramatic reveals when hearing audiences react seconds before a Deaf viewer finishes reading the block. | Pre-renders upcoming words in neutral white/grey (`1.75s` read-ahead layer, `420 ms` minimum floor). At the exact millisecond of vocal onset ($t_0$), each word wipes horizontally letter-by-letter into the speaker's color with a `1.15x` scale Pop impulse ([Weave](https://xlaude2040.com/weave/)). | *Back to the Future* (Doc Brown speaking *"Thank you"* to Marty McFly with exact onset synchronization) ([Caption With Intention](https://www.captionwithintention.org/)). |
| **3. Intonation** *(Variable Typography for Volume, Pitch & Harmonics)* | Uniform font size and weight flatten whispers, shouts, deep voices, high-pitched cries, and vocal tremor into identical text. | Maps RMS loudness to **Crest** font size (`3%` to `12%` of screen height; `0.72x` to `1.62x` scale), speaker-normalized $F_0$ pitch + prominence inversely to **Roboto Flex `wght`** (`900` at `80 Hz` down to `340` at `250 Hz`), and pitch contour to **`wdth`** (`25%`–`151%`) ([Weave](https://xlaude2040.com/weave/)). | *Pulp Fiction* (Samuel L. Jackson's monologue as Jules Winnfield) & *Titanic* (emotional exchanges between Rose and Jack) ([Caption With Intention](https://www.captionwithintention.org/)). |

---

## 3. The Five Deterministic Word-Level Kinetic Animation Channels & Container Rules

To translate qualitative design intent into deterministic code, every rendered caption container and word token must obey the following exact mathematical invariants extracted by [Weave](https://xlaude2040.com/weave/) and [CwiRenderer](https://stardance.hackclub.com/projects/63123):

### 3.1 Container & Timing Invariants
1. **Screen Work Area (`work_area_box.bottom_pct = 20`):** The caption container is strictly restricted to the **lower 20% work area** (`top >= 80%` of the video viewport height) so captions never occlude actors' faces or lip movements ([CwiRenderer](https://stardance.hackclub.com/projects/63123)).
2. **Caption Box Backdrop (`work_area_box.bg_opacity = 0.90`):** Captions render inside a **90% opacity black bounding box** (`background-color: rgba(0, 0, 0, 0.90)`) to guarantee WCAG high-contrast legibility across all speaker colors and video backgrounds ([CwiRenderer](https://stardance.hackclub.com/projects/63123)).
3. **Read-Ahead Layer (`presentation_offset_ms = 1750`, floor `420 ms`):** Words appear in neutral uncolored white/grey (`#B0B5BC` / `rgba(255, 255, 255, 0.65)`) **1.75 seconds before** their acoustic onset ($t_{\text{start}}$), with a strict minimum read-ahead floor of **420 ms** per word ([Weave](https://xlaude2040.com/weave/)).
4. **Immutable Post-Onset Text Rule:** Once the visual presentation playhead ($t_{\text{present}} = t_{\text{acoustic}} - 1.75\text{ s}$) crosses a word's onset timestamp ($t_{\text{start}}$), that word's DOM text string and layout geometry are **permanently frozen** ([Weave](https://xlaude2040.com/weave/)). Late ASR endpoint corrections from Parakeet or CrisperWhisper may only mutate tokens still waiting inside the pre-onset read-ahead zone.

### 3.2 The Five Word-Level Kinetic Channels
1. **Channel 1 — Colour Turn (`background-clip: text` Sweep):**
   - Triggers at exact word onset ($t_0 = t_{\text{start}}$) and wipes horizontally **letter-by-letter** (from `0%` to `100%` width) across the word over its exact spoken duration ($\Delta t = t_{\text{end}} - t_{\text{start}}$), transitioning from neutral read-ahead white/grey to the speaker's assigned hex color ([Caption With Intention](https://www.captionwithintention.org/), [Weave](https://xlaude2040.com/weave/)).
2. **Channel 2 — Pop (`transform: scale()` Synchronization Impulse):**
   - Applies a universal **1.15× scale impulse** (`pop_scale = 1.15`) to every word at its exact onset moment ($t_0$), providing a crisp visual synchronization beat before returning to rest ([Weave](https://xlaude2040.com/weave/)).
3. **Channel 3 — Crest (Loudness-Driven Size Envelope):**
   - Normal conversational speech renders at **5% of screen height** (`1.0×` baseline scale).
   - Across whispers to shouts, font size spans **3% to 12% of screen height**, corresponding to a **0.72× to 1.62× scale multiplier** (`crest_scale`).
   - **Envelope Trajectory:** At word onset ($t_0$), Crest overshoots to its peak `crest_scale`, settles to **0.70 of its peak** (`crest_sustain_scale = 1.0 + (crest_scale - 1.0) * 0.70` or `0.70 * crest_scale` per envelope mode), sustains across the spoken vowel core, and releases smoothly at word offset ($t_{\text{end}}$) ([Weave](https://xlaude2040.com/weave/)).
4. **Channel 4 — Weight (`font-variation-settings: 'wght'`):**
   - Continuous inverse linear mapping between **340 (lightest)** at **250 Hz** and **900 (heaviest)** at **80 Hz** ([Weave](https://xlaude2040.com/weave/)).
   - **Speaker-Median Anchoring:** Weight is anchored to the **speaker's running median pitch ($F_{0,\text{median}}$) plus word prominence** ([WhiStress](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf)), preventing a deep-voiced speaker who shouts (which naturally doubles instantaneous $F_0$) from erroneously rendering in thin, high-pitched text ([Weave](https://xlaude2040.com/weave/)).
5. **Channel 5 — Hold & Width (`translateY()` Lift & `'wdth'` Modulation):**
   - **Hold Lift:** Any word isolated by silence on both sides (pre-word silence $\ge 250\text{ ms}$ and post-word silence $\ge 250\text{ ms}$) receives a vertical lift (**Hold**) of up to **0.525 em** (`transform: translateY(-0.525em)`) ([Weave](https://xlaude2040.com/weave/)).
   - **Width (`wdth`):** [Roboto Flex](https://www.csstypestudio.com/library/font/roboto-flex)'s width axis (`25%` to `151%`, default `100%`) modulates along the word's internal pitch contour and vocal texture to express harmonic richness ([Weave](https://xlaude2040.com/weave/)).
6. **SFX & Non-Speech Formatting Rules:**
   - Non-verbal human vocalizations (`[sighs]`, `[gasps]`, `[laughter]`, `[crying]`, `[coughing]`) and background audio events (`<|BGM|>`, `<|Applause|>`) detected by [SenseVoice-Small](https://fun-audio-llm.github.io/) and [emotion2vec+](https://huggingface.co/emotion2vec/emotion2vec_plus_large) render with dedicated bracketed/tagged formatting and slant (`slnt: -10`) inside the caption container ([CwiRenderer](https://stardance.hackclub.com/projects/63123)).

---

## 4. Resolving the HCI Design Conflict: Speaker Attribution vs. Emotional Valence

A critical architectural decision in expressive captioning is how to allocate visual channels without degrading reading speed or causing visual fatigue ([Caption Royale, CHI 2024](https://www.caluapataca.com/files/caption_royale_preprint.pdf); [Tactile Emotions, CHI 2025](https://saadh.info/papers/pataca-chi-25/pataca-chi25.pdf)):

1. **Academic HCI Background:**
   - In **[Caption Royale](https://www.caluapataca.com/files/caption_royale_preprint.pdf)** (Pataca et al., CHI 2024), researchers evaluated nine typographic parameters (*background-color, baseline shift, emotional typeface, font-color, font-size, font-weight, letter spacing, opacity,* and *shadow-color*) across three studies with DHH participants, finding a four-way tie in Study 2 and selecting **font-color + font-weight** and **font-color + font-size** in Study 3 when using color to depict emotional **valence** (pleasantness vs. unpleasantness on Russell's circumplex model) and weight/size to depict **arousal**.
   - In **[Tactile Emotions](https://saadh.info/papers/pataca-chi-25/pataca-chi25.pdf)** (Pataca et al., CHI 2025), researchers kept valence on font color while offloading arousal to a 75 Hz wrist-worn haptic device.
   - Conversely, Google's Android **[Expressive Captions](https://www.cnet.com/tech/home-entertainment/ai-powered-closed-captions-could-open-up-new-possibilities-and-pitfalls/)** (2025) restricts all text to black-and-white, conveying intensity through `ALL-CAPS` (*"HAPPY BIRTHDAY!"*), letter-stretching (*"amaaazing shot!"*), and sound labels (`[applause]`, `[whistling]`).
2. **Why Our Architecture Reserves Color Strictly for Speaker Attribution:**
   - Following **[Caption with Intention](https://www.captionwithintention.org/)** and **[Weave](https://xlaude2040.com/weave/)**, our engine reserves font color **exclusively for Speaker Attribution** rather than emotional valence.
   - Both systems cannot occupy the color channel simultaneously, and in multi-speaker cinema scenes or live group conversations, not knowing *who* is talking is a far more severe comprehension failure than omitting a mood color ([Weave](https://xlaude2040.com/weave/)).
   - Furthermore, making algorithmic claims about a speaker's psychological emotion can polarize viewers—as shown in [2026 STEM adaptive captioning evaluations (arXiv:2603.15977)](https://arxiv.org/pdf/2603.15977), where Prototype 4 (emojis and multimodal emotion cues) achieved `34.62%` emotional conveyance and `57.69%` readability but drew criticism from some Deaf viewers as *"childish"* or *"unnecessary fluff"*.
   - By contrast, our system guarantees that **every typographic modulation is a direct physical measurement of the voice** (who spoke, when they spoke, how loud in dB RMS, and at what fundamental frequency $F_0$ in Hz) ([Weave](https://xlaude2040.com/weave/)).
3. **User-Selectable Customization Profiles & Intensity Slider:**
   - Because [STEM DHH/neurodivergent studies](https://arxiv.org/pdf/2603.15977) show that Hard of Hearing (HoH) viewers favor clean separation of text and video (Prototype 1, `58.62%` readability) while neurodivergent/ADHD viewers favor keyword prominence highlighting (Prototype 3, which significantly lowered NASA-TLX mental demand compared to video-occluding overlays, $p < 0.05$), our renderer implements FCB Chicago's planned customization controls ([CwiRenderer](https://stardance.hackclub.com/projects/63123)):
     - **Animation Intensity Slider (`0.0` to `1.0`):** Continuously scales kinetic channels from `0%` (static color-attributed text) to `100%` (full 5-channel kinetic typography).
     - **Selectable Exception Profiles:** `Default CWI`, `HoH Clean Separation`, `ADHD Prominence Spotlight`, and `High-Contrast Minimal`.
