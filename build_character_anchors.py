import json
import re
import soundfile as sf
import numpy as np

# Load SRT
with open(r'D:\dune\final_video.srt', 'r', encoding='utf-8') as f:
    srt_text = f.read()

def parse_srt_time(ts_str):
    ts_str = ts_str.strip().replace('.', ',')
    h, m, rest = ts_str.split(':')
    s, ms = rest.split(',')
    return (int(h)*3600 + int(m)*60 + int(s))*1000 + int(ms)

blocks = re.split(r'\n\s*\n', srt_text.strip())
cards = {}
for b in blocks:
    lines = [l.strip() for l in b.splitlines() if l.strip()]
    if len(lines) < 2: continue
    idx = int(lines[0]) if lines[0].isdigit() else len(cards)+1
    ts_line = lines[1] if '-->' in lines[1] else lines[0]
    m = re.search(r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', ts_line)
    if not m: continue
    text_lines = lines[2:] if '-->' in lines[1] else lines[1:]
    clean_text = re.sub(r'<[^>]+>', '', ' '.join(text_lines)).strip()
    cards[idx] = {
        'card': idx,
        'start_ms': parse_srt_time(m.group(1)),
        'end_ms': parse_srt_time(m.group(2)),
        'start_str': m.group(1),
        'end_str': m.group(2),
        'text': clean_text
    }

wav_path = r'D:\dune\final_video_16k.wav'

def get_audio_metrics(start_ms, end_ms):
    sr = 16000
    s_idx = int(start_ms / 1000.0 * sr)
    e_idx = int(end_ms / 1000.0 * sr)
    data, _ = sf.read(wav_path, start=s_idx, frames=e_idx - s_idx, dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
    rms = np.sqrt(np.mean(data**2) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-6)
    peak_db = 20 * np.log10(np.max(np.abs(data)) + 1e-6)
    
    frame_len = int(0.05 * sr)
    hop = int(0.025 * sr)
    frames = [data[i:i+frame_len] for i in range(0, len(data)-frame_len, hop)]
    if frames:
        f_rms = [np.sqrt(np.mean(f**2) + 1e-12) for f in frames]
        floor = np.percentile(f_rms, 10)
        p90 = np.percentile(f_rms, 90)
        snr_est = 20 * np.log10((p90 + 1e-6)/(floor + 1e-6))
    else:
        snr_est = 0.0
        
    return {
        'duration_s': round((end_ms - start_ms) / 1000.0, 3),
        'rms_dbfs': round(float(rms_db), 2),
        'peak_dbfs': round(float(peak_db), 2),
        'snr_db': round(float(snr_est), 2)
    }

def make_anchor_entry(start_card, end_card, label, rationale):
    s_ms = cards[start_card]['start_ms']
    e_ms = cards[end_card]['end_ms']
    metrics = get_audio_metrics(s_ms, e_ms)
    dialogue = " ".join([cards[c]['text'] for c in range(start_card, end_card + 1)])
    return {
        'label': label,
        'start_card': start_card,
        'end_card': end_card,
        'card_span': f"{start_card}-{end_card}" if start_card != end_card else str(start_card),
        'start_ms': s_ms,
        'end_ms': e_ms,
        'start_timestamp': cards[start_card]['start_str'],
        'end_timestamp': cards[end_card]['end_str'],
        'duration_s': metrics['duration_s'],
        'rms_dbfs': metrics['rms_dbfs'],
        'peak_dbfs': metrics['peak_dbfs'],
        'snr_db': metrics['snr_db'],
        'transcript': dialogue,
        'rationale': rationale
    }

# Build anchor profiles for 12 major characters
anchors = {
    "Chani": {
        "character": "Chani",
        "actor": "Zendaya",
        "role": "Fremen warrior / Paul's prophetic love interest / Opening narrator",
        "primary_spk_id": "SPK_00",
        "palette_color": "#4FC3F7",
        "compass_bearing": -30.0,
        "primary_seed_anchor": make_anchor_entry(1, 4, "Opening Narration - Desert Beauty & Spice",
            "Opening soliloquy over sweeping Arrakis dunes with minimal ambient score, close-mic vocal presence, and zero cross-talk. High SNR (~19.9 dB)."),
        "secondary_anchors": [
            make_anchor_entry(1281, 1286, "Crysknife Presentation to Paul",
                "Quiet, reverent dialogue in rock shelter handing Paul her aunt's crysknife before the Jamis duel. Intimate whispered vocal delivery."),
            make_anchor_entry(1177, 1180, "Desert Prophetic Vision",
                "Soft, clear poetic monologue in Paul's vision ('A friend will help you. Follow the friend...')."),
            make_anchor_entry(1323, 1323, "Final Movie Dialogue",
                "Final line of film ('This is only the beginning.') turning toward camera.")
        ]
    },
    "Paul_Atreides": {
        "character": "Paul Atreides",
        "actor": "Timothée Chalamet",
        "role": "Duke's heir / Kwisatz Haderach / Muad'Dib",
        "primary_spk_id": "SPK_01",
        "palette_color": "#FFD54F",
        "compass_bearing": 20.0,
        "primary_seed_anchor": make_anchor_entry(1067, 1075, "Ecological Station - Landsraad Challenge to Kynes",
            "Clear, steady, articulate acoustic monologue explaining political strategy to Kynes. Controlled room acoustic, low noise floor, excellent SNR (~25.9 dB)."),
        "secondary_anchors": [
            make_anchor_entry(1313, 1321, "Desert Climax - Road into the Desert Speech to Stilgar",
                "Resolute, mature vocal declaration to Stilgar and Sietch Tabr after defeating Jamis. High vocal authority."),
            make_anchor_entry(26, 27, "Caladan Breakfast Inquiry",
                "Clean, crisp morning conversational register with Lady Jessica in dining hall."),
            make_anchor_entry(1007, 1010, "Desert Tent Spice Vision Outburst",
                "Intense, emotive, high-salience vocal performance projecting panic over the future galactic jihad.")
        ]
    },
    "Lady_Jessica": {
        "character": "Lady Jessica",
        "actor": "Rebecca Ferguson",
        "role": "Bene Gesserit sister / Leto's concubine / Paul's mother",
        "primary_spk_id": "SPK_02",
        "palette_color": "#FF8A80",
        "compass_bearing": -15.0,
        "primary_seed_anchor": make_anchor_entry(449, 453, "Arrakeen Residency - Interrogation of Shadout Mapes",
            "Calm, icy, commanding Bene Gesserit vocal timbre identifying Mapes' concealed crysknife. High SNR (~26.4 dB), ultra-clean interior acoustic."),
        "secondary_anchors": [
            make_anchor_entry(30, 31, "Caladan Breakfast - Command to Use the Voice",
                "Authoritative maternal instructional tone ('If you want it, make me give it to you. Use the Voice.'). High SNR (~30.0 dB)."),
            make_anchor_entry(301, 307, "Gom Jabbar Exterior - Litany Against Fear (Part 1)",
                "Whispered, trembling, highly distinctive rhythmic recitation of the Litany outside the testing chamber."),
            make_anchor_entry(361, 365, "Caladan Grounds - Explanation of Bene Gesserit Politics",
                "Thoughtful, hushed explanation of secret breeding and political steering.")
        ]
    },
    "Duke_Leto_Atreides": {
        "character": "Duke Leto Atreides",
        "actor": "Oscar Isaac",
        "role": "Head of House Atreides / Duke of Caladan & Arrakis",
        "primary_spk_id": "SPK_03",
        "palette_color": "#64B5F6",
        "compass_bearing": 10.0,
        "primary_seed_anchor": make_anchor_entry(644, 651, "Arrakis Council - Fief Agreement with Stilgar",
            "Diplomatic, stately, baritone speech granting eternal sovereignty to Fremen sietches. Outstanding SNR (~35.9 dB), perfectly balanced room resonance."),
        "secondary_anchors": [
            make_anchor_entry(175, 180, "Caladan Graveyard - Cultivating Desert Power",
                "Intimate, resolute father-son guidance regarding Fremen alliance. Pristine SNR (~30.0 dB) in quiet graveyard."),
            make_anchor_entry(189, 197, "Caladan Graveyard - Leadership & Fatherhood",
                "Deep emotional tenderness ('A great man doesn't seek to lead... my son')."),
            make_anchor_entry(89, 93, "Caladan Ceremony of Change - Acceptance Declaration",
                "Formal military/state proclamation accepting stewardship of Arrakis.")
        ]
    },
    "Baron_Vladimir_Harkonnen": {
        "character": "Baron Vladimir Harkonnen",
        "actor": "Stellan Skarsgård",
        "role": "Ruler of House Harkonnen / Arch-nemesis of House Atreides",
        "primary_spk_id": "SPK_05",
        "palette_color": "#FFAB40",
        "compass_bearing": 60.0,
        "primary_seed_anchor": make_anchor_entry(538, 544, "Giedi Prime Mud Bath - 'My desert. My Arrakis. My Dune.'",
            "Iconic, rumbling, ultra-deep sub-bass vocal timbre. Zero music mask, stark sub-harmonic chest resonance, high SNR (~22.3 dB)."),
        "secondary_anchors": [
            make_anchor_entry(530, 533, "Giedi Prime - Word of Honour to Reverend Mother",
                "Sinuous, deceitful political vow with distinctive low vocal fry."),
            make_anchor_entry(1151, 1161, "Floating Bath to Rabban - Demand for Spice Income ('Squeeze hard')",
                "Gravelly, wheezing baritone commanding economic oppression after surviving poison gas."),
            make_anchor_entry(982, 986, "Arrakeen Palace to Paralyzed Leto - Fall of House Atreides",
                "Mocking, triumphalist taunt standing over the fallen Duke.")
        ]
    },
    "Reverend_Mother_Gaius_Helen_Mohiam": {
        "character": "Reverend Mother Gaius Helen Mohiam",
        "actor": "Charlotte Rampling",
        "role": "Bene Gesserit Truthsayer to the Padishah Emperor",
        "primary_spk_id": "SPK_04",
        "palette_color": "#B388FF",
        "compass_bearing": -40.0,
        "primary_seed_anchor": make_anchor_entry(285, 290, "Gom Jabbar Chamber - The Poison Needle Rule",
            "Chilling, razor-sharp diction explaining the Gom Jabbar test. Absolute quiet room, phenomenal SNR (~33.5 dB), dry direct vocal presence."),
        "secondary_anchors": [
            make_anchor_entry(314, 319, "Gom Jabbar Chamber - Sifting Sand / Humanity Test",
                "Philosophical explanation of Bene Gesserit natural selection ('we sift people'). High SNR (~24.1 dB)."),
            make_anchor_entry(337, 344, "Caladan Ship Landing - Reprimand of Lady Jessica",
                "Harsh, imperious condemnation of Jessica's pride in breeding a son."),
            make_anchor_entry(525, 529, "Giedi Prime - Protection of Jessica and Paul",
                "Stern decree to Baron Harkonnen enforcing Bene Gesserit covenant.")
        ]
    },
    "Stilgar": {
        "character": "Stilgar",
        "actor": "Javier Bardem",
        "role": "Naib of Sietch Tabr / Fremen leader",
        "primary_spk_id": "SPK_06",
        "palette_color": "#69F0AE",
        "compass_bearing": 45.0,
        "primary_seed_anchor": make_anchor_entry(636, 642, "Arrakis Council - Fremen Desert Boundary Demand",
            "Gravelly, earthy, intense Fremen vocal timbre warning the Outworlders to stay behind the Shield Wall. Outstanding SNR (~34.2 dB)."),
        "secondary_anchors": [
            make_anchor_entry(1225, 1228, "Desert Rock Confrontation - Praise of the Brave Crossing",
                "Deep authoritative Fremen dialect acknowledging Paul and Leto's courage to Jamis. High SNR (~29.5 dB)."),
            make_anchor_entry(1237, 1241, "Desert Rock Confrontation - Sanctuary Verdict",
                "Harsh desert pragmatism judging Paul vs Jessica."),
            make_anchor_entry(1305, 1307, "Post-Duel - Acceptance of Paul into Sietch Tabr",
                "Ceremonial Fremen decree ('A life for a life. Come with us to Sietch Tabr.').")
        ]
    },
    "Gurney_Halleck": {
        "character": "Gurney Halleck",
        "actor": "Josh Brolin",
        "role": "Warmaster of House Atreides / Paul's combat trainer",
        "primary_spk_id": "SPK_07",
        "palette_color": "#FF7043",
        "compass_bearing": 30.0,
        "primary_seed_anchor": make_anchor_entry(230, 237, "Shield Training Room - Harkonnen Brutality Monologue",
            "Passionate, gritty, battle-scarred baritone lecturing Paul on Harkonnen violence. Phenomenal SNR (~32.3 dB), dry acoustic, powerful vocal projection."),
        "secondary_anchors": [
            make_anchor_entry(227, 229, "Shield Training Room - Gravity of Situation",
                "Stern, urgent wake-up call to Paul ('You don't get it, do you?'). SNR (~32.2 dB)."),
            make_anchor_entry(216, 219, "Shield Training Room - Necessity Over Mood",
                "Classic combat instruction ('What's mood to do with it? You fight when the necessity arises...')."),
            make_anchor_entry(204, 206, "Shield Training Room Entrance",
                "Gruff, wry dialogue challenging Paul to choose his blade.")
        ]
    },
    "Duncan_Idaho": {
        "character": "Duncan Idaho",
        "actor": "Jason Momoa",
        "role": "Swordmaster of the Ginaz / Advance scout to the Fremen",
        "primary_spk_id": "SPK_08",
        "palette_color": "#26A69A",
        "compass_bearing": 15.0,
        "primary_seed_anchor": make_anchor_entry(601, 607, "Arrakis Council Room - Report on Fremen Lethality",
            "Warm, rugged, dynamic storytelling tone reporting on four weeks with the Fremen. Phenomenal acoustic clarity, SNR (~34.4 dB)."),
        "secondary_anchors": [
            make_anchor_entry(587, 591, "Arrakis Council Room - Introduction of Stilgar",
                "Formal military intelligence report to Duke Leto. Pristine SNR (~30.4 dB)."),
            make_anchor_entry(622, 624, "Arrakis Council Room - Acceptance of Stilgar's Moisture",
                "Calm, diplomatic explanation preventing bloodshed over Fremen spitting custom. SNR (~29.1 dB)."),
            make_anchor_entry(674, 680, "Arrakis Ornithopter - Reverence for the Desert",
                "Affectionate, awe-struck reflection on Fremen harmony with Arrakis. SNR (~27.6 dB)."),
            make_anchor_entry(134, 139, "Caladan Hangar - Wisdom on Dreams and Action",
                "Big-brother mentoring Paul before departure ('Dreams make good stories... Put on some muscle?').")
        ]
    },
    "Thufir_Hawat": {
        "character": "Thufir Hawat",
        "actor": "Stephen McKinley Henderson",
        "role": "Master of Assassins / Mentat of House Atreides",
        "primary_spk_id": "SPK_09",
        "palette_color": "#AB47BC",
        "compass_bearing": -20.0,
        "primary_seed_anchor": make_anchor_entry(554, 556, "Arrakis Council Room - Harkonnen Spice Ledger Analysis",
            "Analytical, rhythmic, Mentat computational cadence breaking down 10 billion solaris spice extraction. Excellent room clarity, SNR (~23.1 dB)."),
        "secondary_anchors": [
            make_anchor_entry(508, 509, "Residency - Resignation Offer to Duke Leto",
                "Grave, sorrowful, honourable delivery after hunter-seeker breach. SNR (~17.9 dB)."),
            make_anchor_entry(75, 76, "Caladan Ceremony - Guild Navigator Cost Calculation",
                "Rapid computational trances ('Three Guild Navigators. A total of 1.46 million, 62 solaris...')."),
            make_anchor_entry(571, 573, "Arrakis Council - Briefing on Dr. Liet Kynes",
                "Terse intelligence summary of the Imperial Ecologist.")
        ]
    },
    "Herald_of_the_Change": {
        "character": "Herald of the Change",
        "actor": "Robert Higgins",
        "role": "Padishah Imperial Emissary / Master of Ceremonies",
        "primary_spk_id": "SPK_10",
        "palette_color": "#D4E157",
        "compass_bearing": 0.0,
        "primary_seed_anchor": make_anchor_entry(81, 84, "Caladan Ceremony of the Change - Witness Invocation",
            "Magnificent, theatrical, boomingly resonant heraldic cadence witnessed by Imperial Court and Spacing Guild. High acoustic clarity, SNR (~22.7 dB)."),
        "secondary_anchors": [
            make_anchor_entry(85, 88, "Caladan Ceremony of the Change - Imperial Mandate & Challenge",
                "Imperial decree transfer of Arrakis to House Atreides ('Do you accept?'). SNR (~19.2 dB)."),
            make_anchor_entry(77, 80, "Caladan Ceremony of the Change - Sovereign Title Proclamation",
                "Formal heraldic opening invoking Shaddam IV of House Corrino.")
        ]
    },
    "Dr_Wellington_Yueh": {
        "character": "Dr. Wellington Yueh",
        "actor": "Chang Chen",
        "role": "Suk Doctor of House Atreides (Imperial Conditioning broken)",
        "primary_spk_id": "SPK_11",
        "palette_color": "#8D6E63",
        "compass_bearing": -25.0,
        "primary_seed_anchor": make_anchor_entry(968, 973, "Arrakeen Palace - Deliverance Demand to Baron Harkonnen",
            "Tense, desperate, tragic delivery demanding his wife Wanna's release. High SNR (~23.2 dB), stark interior acoustic."),
        "secondary_anchors": [
            make_anchor_entry(268, 271, "Caladan Quarters - Checking Paul's Vitals",
                "Gentle, respectful, professional bedside manner before Gom Jabbar test."),
            make_anchor_entry(921, 928, "Arrakeen Infirmary - Confession of Betrayal to Leto",
                "Tortured, weeping confession explaining Wanna's torture by the Harkonnens."),
            make_anchor_entry(929, 937, "Arrakeen Infirmary - Poison Peg Tooth Plan",
                "Whispered tactical instructions for assassinating the Baron with cyanide gas.")
        ]
    },
    "Dr_Liet_Kynes": {
        "character": "Dr. Liet Kynes",
        "actor": "Sharon Duncan-Brewster",
        "role": "Imperial Ecologist / Judge of the Change / Fremen leader",
        "primary_spk_id": "SPK_12",
        "palette_color": "#26C6DA",
        "compass_bearing": 35.0,
        "primary_seed_anchor": make_anchor_entry(700, 706, "Arrakeen Testing Room - Stillsuit Technical Briefing",
            "Clear, precise, scientific expository speech detailing the Fremen filtration system. Outstanding SNR (~25.7 dB), minimal background noise."),
        "secondary_anchors": [
            make_anchor_entry(1086, 1089, "Testing Station - Confronting Paul on Imperial Realities",
                "Sharp, piercing critique challenging Paul's ambition for the throne. High SNR (~27.8 dB)."),
            make_anchor_entry(707, 710, "Arrakeen Testing Room - Suit Water Retention Mechanics",
                "Clinical description of thimbleful water loss in desert conditions. SNR (~25.4 dB)."),
            make_anchor_entry(1053, 1060, "Testing Station - History of Arrakis Ecology & Spice",
                "Wistful, tragic recounting of how spice exploitation ruined the greening of Arrakis."),
            make_anchor_entry(1128, 1129, "Desert Dunes - Final Defiance to Sardaukar",
                "Solemn, unwavering last words ('I serve only one master. His name is Shai-Hulud.').")
        ]
    }
}

# Key multi-speaker scene mappings
scene_mappings = {
    "scene_01_caladan_breakfast": {
        "title": "Caladan Breakfast Scene (The Voice Practice)",
        "card_span": "Cards 20-41",
        "time_span": f"{cards[20]['start_str']} --> {cards[41]['end_str']}",
        "speakers": ["Lady Jessica", "Paul Atreides"],
        "turn_boundaries": [
            {"card": 20, "speaker": "Lady Jessica", "text": cards[20]['text']},
            {"card": 21, "speaker": "Lady Jessica", "text": cards[21]['text']},
            {"card": 22, "speaker": "Lady Jessica", "text": cards[22]['text']},
            {"card": 23, "speaker": "Paul Atreides", "text": cards[23]['text']},
            {"card": 24, "speaker": "Paul Atreides", "text": cards[24]['text']},
            {"card": 25, "speaker": "Lady Jessica", "text": cards[25]['text']},
            {"card": 26, "speaker": "Paul Atreides", "text": cards[26]['text']},
            {"card": 27, "speaker": "Paul Atreides", "text": cards[27]['text']},
            {"card": 28, "speaker": "Lady Jessica", "text": cards[28]['text']},
            {"card": 29, "speaker": "Paul Atreides", "text": cards[29]['text']},
            {"card": 30, "speaker": "Lady Jessica", "text": cards[30]['text']},
            {"card": 31, "speaker": "Lady Jessica", "text": cards[31]['text']},
            {"card": 32, "speaker": "Paul Atreides", "text": cards[32]['text']},
            {"card": 33, "speaker": "Paul Atreides", "text": cards[33]['text']},
            {"card": 34, "speaker": "Lady Jessica", "text": cards[34]['text']},
            {"card": 35, "speaker": "Paul Atreides (using The Voice)", "text": cards[35]['text']},
            {"card": 36, "speaker": "Lady Jessica", "text": cards[36]['text']},
            {"card": 37, "speaker": "Paul Atreides", "text": cards[37]['text']},
            {"card": 38, "speaker": "Lady Jessica", "text": cards[38]['text']},
            {"card": 39, "speaker": "Lady Jessica", "text": cards[39]['text']},
            {"card": 40, "speaker": "Lady Jessica", "text": cards[40]['text']},
            {"card": 41, "speaker": "Paul Atreides", "text": cards[41]['text']}
        ]
    },
    "scene_02_caladan_herald_ceremony": {
        "title": "Caladan Herald of the Change Ceremony",
        "card_span": "Cards 71-98",
        "time_span": f"{cards[71]['start_str']} --> {cards[98]['end_str']}",
        "speakers": ["Duke Leto Atreides", "Gurney Halleck", "Thufir Hawat", "Herald of the Change", "Atreides Army", "Paul Atreides", "Lady Jessica"],
        "turn_boundaries": [
            {"card": 71, "speaker": "Duke Leto Atreides", "text": cards[71]['text']},
            {"card": 72, "speaker": "Gurney Halleck", "text": cards[72]['text']},
            {"card": 73, "speaker": "Duke Leto Atreides", "text": cards[73]['text']},
            {"card": 74, "speaker": "Duke Leto Atreides", "text": cards[74]['text']},
            {"card": 75, "speaker": "Thufir Hawat", "text": cards[75]['text']},
            {"card": 76, "speaker": "Thufir Hawat", "text": cards[76]['text']},
            {"cards": "77-88", "speaker": "Herald of the Change", "text": " ".join([cards[c]['text'] for c in range(77, 89)])},
            {"cards": "89-93", "speaker": "Duke Leto Atreides", "text": " ".join([cards[c]['text'] for c in range(89, 94)])},
            {"cards": "94-95", "speaker": "Atreides Army Chorus", "text": "Atreides! Atreides! Atreides!"},
            {"card": 96, "speaker": "Herald of the Change", "text": cards[96]['text']},
            {"card": 97, "speaker": "Paul Atreides", "text": cards[97]['text']},
            {"card": 98, "speaker": "Lady Jessica", "text": cards[98]['text']}
        ]
    },
    "scene_03_shield_training": {
        "title": "Shield Training Room (Paul vs Gurney)",
        "card_span": "Cards 198-237",
        "time_span": f"{cards[198]['start_str']} --> {cards[237]['end_str']}",
        "speakers": ["Gurney Halleck", "Paul Atreides"],
        "turn_boundaries": [
            {"cards": "198-199", "speaker": "Gurney Halleck", "text": " ".join([cards[c]['text'] for c in (198, 199)])},
            {"cards": "200-201", "speaker": "Paul Atreides", "text": " ".join([cards[c]['text'] for c in (200, 201)])},
            {"card": 202, "speaker": "Gurney Halleck", "text": cards[202]['text']},
            {"card": 203, "speaker": "Paul Atreides", "text": cards[203]['text']},
            {"cards": "204-206", "speaker": "Gurney Halleck", "text": " ".join([cards[c]['text'] for c in range(204, 207)])},
            {"cards": "207-209", "speaker": "Paul Atreides", "text": " ".join([cards[c]['text'] for c in range(207, 210)])},
            {"cards": "210-211", "speaker": "Gurney Halleck", "text": " ".join([cards[c]['text'] for c in (210, 211)])},
            {"card": 212, "speaker": "Paul Atreides", "text": cards[212]['text']},
            {"card": 213, "speaker": "Gurney Halleck", "text": cards[213]['text']},
            {"card": 214, "speaker": "Paul Atreides", "text": cards[214]['text']},
            {"card": 215, "speaker": "Gurney Halleck / Paul Atreides", "text": "- Mood? - Yeah."},
            {"cards": "216-220", "speaker": "Gurney Halleck", "text": " ".join([cards[c]['text'] for c in range(216, 221)])},
            {"card": 221, "speaker": "Paul Atreides", "text": cards[221]['text']},
            {"cards": "222-224", "speaker": "Gurney Halleck", "text": " ".join([cards[c]['text'] for c in range(222, 225)])},
            {"card": 225, "speaker": "Paul Atreides", "text": cards[225]['text']},
            {"card": 226, "speaker": "Paul Atreides", "text": cards[226]['text']},
            {"cards": "227-237", "speaker": "Gurney Halleck", "text": " ".join([cards[c]['text'] for c in range(227, 238)])}
        ]
    },
    "scene_04_gom_jabbar": {
        "title": "Gom Jabbar Scene (Reverend Mother vs Paul vs Jessica)",
        "card_span": "Cards 254-356",
        "time_span": f"{cards[254]['start_str']} --> {cards[356]['end_str']}",
        "speakers": ["Lady Jessica", "Paul Atreides", "Dr. Wellington Yueh", "Reverend Mother Gaius Helen Mohiam"],
        "turn_boundaries": [
            {"cards": "254-255", "speaker": "Lady Jessica", "text": "Paul. Paul, wake up."},
            {"card": 256, "speaker": "Paul Atreides", "text": cards[256]['text']},
            {"card": 257, "speaker": "Lady Jessica", "text": cards[257]['text']},
            {"card": 258, "speaker": "Paul Atreides", "text": cards[258]['text']},
            {"cards": "259-262", "speaker": "Lady Jessica", "text": " ".join([cards[c]['text'] for c in range(259, 263)])},
            {"card": 263, "speaker": "Paul Atreides", "text": cards[263]['text']},
            {"card": 264, "speaker": "Lady Jessica", "text": cards[264]['text']},
            {"cards": "265-266", "speaker": "Paul Atreides", "text": " ".join([cards[c]['text'] for c in (265, 266)])},
            {"card": 267, "speaker": "Lady Jessica", "text": cards[267]['text']},
            {"cards": "268-269", "speaker": "Dr. Wellington Yueh", "text": " ".join([cards[c]['text'] for c in (268, 269)])},
            {"card": 270, "speaker": "Paul Atreides", "text": cards[270]['text']},
            {"card": 271, "speaker": "Dr. Wellington Yueh", "text": cards[271]['text']},
            {"card": 272, "speaker": "Lady Jessica", "text": cards[272]['text']},
            {"card": 273, "speaker": "Paul Atreides", "text": cards[273]['text']},
            {"cards": "274-276", "speaker": "Reverend Mother Gaius Helen Mohiam", "text": " ".join([cards[c]['text'] for c in range(274, 277)])},
            {"cards": "277-278", "speaker": "Lady Jessica", "text": " ".join([cards[c]['text'] for c in (277, 278)])},
            {"card": 279, "speaker": "Paul Atreides", "text": cards[279]['text']},
            {"cards": "280-281", "speaker": "Reverend Mother Gaius Helen Mohiam (using The Voice)", "text": "Come here. Kneel."},
            {"card": 282, "speaker": "Paul Atreides", "text": cards[282]['text']},
            {"card": 283, "speaker": "Reverend Mother Gaius Helen Mohiam", "text": cards[283]['text']},
            {"card": 284, "speaker": "Reverend Mother Gaius Helen Mohiam", "text": cards[284]['text']},
            {"cards": "285-290", "speaker": "Reverend Mother Gaius Helen Mohiam", "text": " ".join([cards[c]['text'] for c in range(285, 291)])},
            {"card": 291, "speaker": "Paul Atreides", "text": cards[291]['text']},
            {"card": 292, "speaker": "Reverend Mother Gaius Helen Mohiam", "text": cards[292]['text']},
            {"cards": "293-295", "speaker": "Reverend Mother Gaius Helen Mohiam", "text": " ".join([cards[c]['text'] for c in range(293, 296)])},
            {"card": 296, "speaker": "Paul Atreides", "text": cards[296]['text']},
            {"cards": "297-299", "speaker": "Reverend Mother Gaius Helen Mohiam", "text": " ".join([cards[c]['text'] for c in range(297, 300)])},
            {"card": 300, "speaker": "Reverend Mother Gaius Helen Mohiam", "text": cards[300]['text']},
            {"cards": "301-312", "speaker": "Lady Jessica (Hallway - Litany Against Fear)", "text": " ".join([cards[c]['text'] for c in range(301, 313)])},
            {"card": 313, "speaker": "Reverend Mother Gaius Helen Mohiam", "text": cards[313]['text']},
            {"cards": "314-319", "speaker": "Reverend Mother Gaius Helen Mohiam", "text": " ".join([cards[c]['text'] for c in range(314, 320)])},
            {"card": 320, "speaker": "Paul Atreides", "text": cards[320]['text']},
            {"cards": "321-322", "speaker": "Reverend Mother Gaius Helen Mohiam", "text": " ".join([cards[c]['text'] for c in (321, 322)])},
            {"card": 323, "speaker": "Reverend Mother Gaius Helen Mohiam", "text": cards[323]['text']},
            {"card": 324, "speaker": "Reverend Mother Gaius Helen Mohiam", "text": cards[324]['text']},
            {"card": 325, "speaker": "Reverend Mother Gaius Helen Mohiam", "text": cards[325]['text']},
            {"card": 326, "speaker": "Paul Atreides / Reverend Mother Mohiam", "text": "- I had one tonight. - What did you see?"},
            {"cards": "327-328", "speaker": "Paul Atreides", "text": "A girl. On Arrakis."},
            {"card": 329, "speaker": "Reverend Mother Gaius Helen Mohiam", "text": cards[329]['text']},
            {"card": 330, "speaker": "Paul Atreides", "text": cards[330]['text']},
            {"cards": "331-332", "speaker": "Reverend Mother Gaius Helen Mohiam", "text": " ".join([cards[c]['text'] for c in (331, 332)])},
            {"card": 333, "speaker": "Paul Atreides", "text": cards[333]['text']},
            {"cards": "334-335", "speaker": "Reverend Mother Gaius Helen Mohiam", "text": " ".join([cards[c]['text'] for c in (334, 335)])},
            {"card": 336, "speaker": "Lady Jessica", "text": cards[336]['text']},
            {"cards": "337-344", "speaker": "Reverend Mother Gaius Helen Mohiam", "text": " ".join([cards[c]['text'] for c in range(337, 345)])},
            {"card": 345, "speaker": "Lady Jessica", "text": cards[345]['text']},
            {"cards": "346-351", "speaker": "Reverend Mother Gaius Helen Mohiam", "text": " ".join([cards[c]['text'] for c in range(346, 352)])},
            {"card": 352, "speaker": "Lady Jessica", "text": cards[352]['text']},
            {"cards": "353-356", "speaker": "Reverend Mother Gaius Helen Mohiam", "text": " ".join([cards[c]['text'] for c in range(353, 357)])}
        ]
    },
    "scene_05_arrakis_council": {
        "title": "Arrakis Council Room (Operations & Spice Strategy)",
        "card_span": "Cards 545-580",
        "time_span": f"{cards[545]['start_str']} --> {cards[580]['end_str']}",
        "speakers": ["Gurney Halleck", "Paul Atreides", "Duke Leto Atreides", "Thufir Hawat"],
        "turn_boundaries": [
            {"cards": "545-548", "speaker": "Gurney Halleck", "text": " ".join([cards[c]['text'] for c in range(545, 549)])},
            {"card": 549, "speaker": "Paul Atreides", "text": cards[549]['text']},
            {"card": 550, "speaker": "Gurney Halleck", "text": cards[550]['text']},
            {"card": 551, "speaker": "Atreides Guard Officer", "text": cards[551]['text']},
            {"cards": "552-553", "speaker": "Duke Leto Atreides", "text": " ".join([cards[c]['text'] for c in (552, 553)])},
            {"cards": "554-556", "speaker": "Thufir Hawat", "text": " ".join([cards[c]['text'] for c in range(554, 557)])},
            {"cards": "557-558", "speaker": "Gurney Halleck (quoting scripture)", "text": " ".join([cards[c]['text'] for c in (557, 558)])},
            {"cards": "559-560", "speaker": "Thufir Hawat", "text": " ".join([cards[c]['text'] for c in (559, 560)])},
            {"card": 561, "speaker": "Duke Leto Atreides", "text": cards[561]['text']},
            {"cards": "562-563", "speaker": "Thufir Hawat", "text": " ".join([cards[c]['text'] for c in (562, 563)])},
            {"card": 564, "speaker": "Paul Atreides", "text": cards[564]['text']},
            {"cards": "565-566", "speaker": "Thufir Hawat", "text": " ".join([cards[c]['text'] for c in (565, 566)])},
            {"cards": "567-570", "speaker": "Duke Leto Atreides", "text": " ".join([cards[c]['text'] for c in range(567, 571)])},
            {"cards": "571-573", "speaker": "Thufir Hawat", "text": " ".join([cards[c]['text'] for c in range(571, 574)])},
            {"card": 574, "speaker": "Duke Leto Atreides", "text": cards[574]['text']},
            {"card": 575, "speaker": "Thufir Hawat", "text": cards[575]['text']},
            {"cards": "576-578", "speaker": "Duke Leto Atreides", "text": " ".join([cards[c]['text'] for c in range(576, 579)])},
            {"card": 579, "speaker": "Gurney Halleck", "text": cards[579]['text']},
            {"card": 580, "speaker": "Duke Leto Atreides", "text": cards[580]['text']}
        ]
    },
    "scene_06_stilgar_meeting": {
        "title": "Arrakis Meeting with Stilgar (Water Gift & Sietch Pact)",
        "card_span": "Cards 581-661",
        "time_span": f"{cards[581]['start_str']} --> {cards[661]['end_str']}",
        "speakers": ["Duncan Idaho", "Paul Atreides", "Duke Leto Atreides", "Thufir Hawat", "Gurney Halleck", "Stilgar"],
        "turn_boundaries": [
            {"card": 581, "speaker": "Guard Attendant", "text": cards[581]['text']},
            {"cards": "582-585", "speaker": "Duncan Idaho", "text": "Take care of them. They are friends. / My boy! / I swear you're getting taller."},
            {"card": 586, "speaker": "Paul Atreides", "text": cards[586]['text']},
            {"cards": "587-591", "speaker": "Duncan Idaho", "text": " ".join([cards[c]['text'] for c in range(587, 592)])},
            {"card": 592, "speaker": "Duke Leto Atreides", "text": cards[592]['text']},
            {"cards": "593-594", "speaker": "Duncan Idaho", "text": " ".join([cards[c]['text'] for c in (593, 594)])},
            {"card": 595, "speaker": "Duke Leto Atreides", "text": cards[595]['text']},
            {"cards": "596-598", "speaker": "Duncan Idaho", "text": " ".join([cards[c]['text'] for c in range(596, 599)])},
            {"card": 599, "speaker": "Duke Leto Atreides", "text": cards[599]['text']},
            {"card": 600, "speaker": "Thufir Hawat", "text": cards[600]['text']},
            {"cards": "601-607", "speaker": "Duncan Idaho", "text": " ".join([cards[c]['text'] for c in range(601, 608)])},
            {"card": 608, "speaker": "Duke Leto Atreides", "text": cards[608]['text']},
            {"cards": "609-610", "speaker": "Duke Leto Atreides / Duncan Idaho", "text": "- Duncan. Well done. - Thank you, my lord."},
            {"cards": "611-612", "speaker": "Atreides Guard", "text": " ".join([cards[c]['text'] for c in (611, 612)])},
            {"card": 613, "speaker": "Duncan Idaho", "text": cards[613]['text']},
            {"card": 614, "speaker": "Duke Leto Atreides", "text": cards[614]['text']},
            {"cards": "615-616", "speaker": "Gurney Halleck", "text": "Stop there. Hey! Hey!"},
            {"cards": "617-620", "speaker": "Duke Leto Atreides", "text": " ".join([cards[c]['text'] for c in range(617, 621)])},
            {"card": 621, "speaker": "Duncan Idaho", "text": cards[621]['text']},
            {"cards": "622-624", "speaker": "Duncan Idaho", "text": " ".join([cards[c]['text'] for c in range(622, 625)])},
            {"cards": "625-627", "speaker": "Duke Leto Atreides", "text": " ".join([cards[c]['text'] for c in range(625, 628)])},
            {"cards": "628-630", "speaker": "Stilgar", "text": " ".join([cards[c]['text'] for c in range(628, 631)])},
            {"card": 631, "speaker": "Duke Leto Atreides", "text": cards[631]['text']},
            {"cards": "632-635", "speaker": "Duke Leto Atreides", "text": " ".join([cards[c]['text'] for c in range(632, 636)])},
            {"cards": "636-642", "speaker": "Stilgar", "text": " ".join([cards[c]['text'] for c in range(636, 643)])},
            {"card": 643, "speaker": "Gurney Halleck", "text": cards[643]['text']},
            {"cards": "644-651", "speaker": "Duke Leto Atreides", "text": " ".join([cards[c]['text'] for c in range(644, 652)])},
            {"cards": "652-653", "speaker": "Stilgar", "text": " ".join([cards[c]['text'] for c in (652, 653)])},
            {"cards": "654-655", "speaker": "Duke Leto Atreides", "text": " ".join([cards[c]['text'] for c in (654, 655)])},
            {"card": 656, "speaker": "Stilgar", "text": cards[656]['text']},
            {"card": 657, "speaker": "Gurney Halleck", "text": cards[657]['text']},
            {"card": 658, "speaker": "Duncan Idaho", "text": cards[658]['text']},
            {"card": 659, "speaker": "Duke Leto Atreides", "text": cards[659]['text']},
            {"cards": "660-661", "speaker": "Duncan Idaho", "text": "Yes. It will take time."}
        ]
    }
}

full_data = {
    "title": "Dune Part 1 - Character Anchor Embeddings & Scene Turn Boundary Ground Truth",
    "total_subtitle_cards": len(cards),
    "characters": anchors,
    "key_scenes": scene_mappings
}

out_path_1 = r'D:\dune\character_anchors.json'
out_path_2 = r'D:\CAPTIONING\dune_character_anchors.json'

with open(out_path_1, 'w', encoding='utf-8') as f:
    json.dump(full_data, f, indent=2, ensure_ascii=False)
with open(out_path_2, 'w', encoding='utf-8') as f:
    json.dump(full_data, f, indent=2, ensure_ascii=False)

print(f"Successfully generated and wrote anchor database to:")
print(f"1. {out_path_1}")
print(f"2. {out_path_2}")
