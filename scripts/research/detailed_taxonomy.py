import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

with open(DISTILLATION_DIR / "terra_failed_samples_repair.jsonl", "r", encoding="utf-8") as f:
    terra_records = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

with open(DISTILLATION_DIR / "train_distilled_trajectories.jsonl", "r", encoding="utf-8") as f:
    luna_records = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

categories = {
    "CAT_1_DEFAULT_AND_INFERENCE": [], # Inferred default (e.g. acc_01, default beneficiary, bill amount matching)
    "CAT_2_SPOKEN_ENTITY_PARSING": [], # Spoken phone number, OTP, currency words, complex spelling
    "CAT_3_TEMPORAL_MATH_AND_MULTI_STAGE": [], # Time calculation offset (45 min before, 15 min after), multi-step chaining
    "CAT_4_CROSS_DOMAIN_CONDITION": [], # Weather condition, smart home device condition
    "CAT_5_OVERTHINKING_FRONTIER": [], # Frontier model refused because schema had slight limitation (recurring vs one-time, restore vs create)
    "CAT_6_HIDDEN_CONTEXT_CROSS_APP": [], # Context hidden in another app (reminders, recent calls)
    "CAT_7_STRING_FORMAT_NOISE": [], # Spaces in phone number, minor string format
    "CAT_8_AMBIGUITY_AND_EDGE": [] # Linguistic ambiguity (e.g. may lanh de nguyen)
}

for sid, t_rec in terra_records.items():
    l_rec = luna_records.get(sid, {})
    audio = t_rec.get("audio_text", "")
    t_match = t_rec["state_match"]
    l_match = l_rec.get("state_match", False)
    
    t_calls = [c["name"] for c in t_rec["predicted_tool_calls"]]
    g_calls = [c["name"] for c in t_rec["gold_tool_calls"]]
    l_calls = [c["name"] for c in l_rec.get("predicted_tool_calls", [])]
    
    # Classify based on signature
    if not t_match:
        # One of the 26
        thought = t_rec["turns"][-1]["thought"] if t_rec["turns"] else ""
        if "quy tắc" in thought or "hằng ngày" in thought or "khôi phục" in thought or "Không thể hẹn" in thought or "Không có công cụ" in thought:
            categories["CAT_5_OVERTHINKING_FRONTIER"].append((sid, audio, t_match, thought))
        elif "0983 214 6755" in str(t_rec["predicted_tool_calls"]):
            categories["CAT_7_STRING_FORMAT_NOISE"].append((sid, audio, t_match, "Phone format spaces"))
        elif "máy lạnh" in audio and "đèn" in audio:
            categories["CAT_8_AMBIGUITY_AND_EDGE"].append((sid, audio, t_match, thought))
        elif "sys.ask_missing_info" in g_calls or "sys.ask_missing_info" in t_calls:
            categories["CAT_6_HIDDEN_CONTEXT_CROSS_APP"].append((sid, audio, t_match, thought))
        else:
            categories["CAT_8_AMBIGUITY_AND_EDGE"].append((sid, audio, t_match, thought))
    else:
        # Repaired by Terra (61)
        # Check what was the reason Luna failed:
        if "sys.ask_missing_info" in l_calls and "sys.ask_missing_info" not in g_calls:
            categories["CAT_1_DEFAULT_AND_INFERENCE"].append((sid, audio, t_match, "Luna asked missing info; Terra inferred"))
        elif any(w in audio for w in ["không chín", "số không", "năm không", "bảy sáu", "tám ba", "09", "03", "08", "07"]):
            categories["CAT_2_SPOKEN_ENTITY_PARSING"].append((sid, audio, t_match, "Spoken digits / phone"))
        elif any(w in audio for w in ["trước", "sớm hơn", "muộn hơn", "sau khi", "phút", "tiếng", "lúc"]):
            categories["CAT_3_TEMPORAL_MATH_AND_MULTI_STAGE"].append((sid, audio, t_match, "Temporal math / multi-stage"))
        elif any(w in audio for w in ["mưa", "nắng", "nếu", "thì", "đang bật", "đang tắt"]):
            categories["CAT_4_CROSS_DOMAIN_CONDITION"].append((sid, audio, t_match, "Condition / smart device"))
        else:
            categories["CAT_1_DEFAULT_AND_INFERENCE"].append((sid, audio, t_match, "Complex domain inference"))

for cat, items in categories.items():
    print(f"\n=== {cat}: {len(items)} samples ===")
    for sid, audio, match, note in items[:3]:
        print(f"  [{sid}] Match:{match} | {audio[:70]}... -> {note[:60]}")

