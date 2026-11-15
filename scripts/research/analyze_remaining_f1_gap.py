import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
from collections import Counter, defaultdict

with open(DISTILLATION_DIR / "harmonized_sft_1526_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

# Also load gold labels from train/train.jsonl
with open(DATA_ROOT / "train/train.jsonl", "r", encoding="utf-8") as f:
    train_samples = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

total = len(samples)
perfect = [s for s in samples if s["score"] == 1.0]
imperfect = [s for s in samples if s["score"] < 1.0]

print(f"Total samples: {total}")
print(f"Perfect 1.00:  {len(perfect)} ({len(perfect)/total*100:.2f}%)")
print(f"Imperfect:     {len(imperfect)} ({len(imperfect)/total*100:.2f}%)")

# Categorize the discrepancies
categories = defaultdict(list)

for s in imperfect:
    sid = s["id"]
    gold_data = train_samples[sid]
    gold_calls = gold_data.get("label", {}).get("tool_calls", [])
    
    # Extract pred calls from messages
    pred_calls = []
    for m in s["messages"]:
        if m["role"] == "assistant":
            try:
                p = json.loads(m["content"])
                pred_calls.extend(p.get("tool_calls", []))
            except Exception:
                pass
                
    p_names = [c["name"] for c in pred_calls]
    g_names = [c["name"] for c in gold_calls]
    
    p_count = Counter(p_names)
    g_count = Counter(g_names)
    
    # 1. Extra read tool calls
    if len(p_names) > len(g_names) and set(g_names).issubset(set(p_names)):
        extra_tools = list((p_count - g_count).elements())
        categories["1_EXTRA_READ_CALLS"].append((sid, extra_tools, s["score"], s["audio_text"]))
    # 2. Missing tool calls in Pred vs Gold
    elif len(p_names) < len(g_names) and set(p_names).issubset(set(g_names)):
        missing_tools = list((g_count - p_count).elements())
        categories["2_MISSING_TOOL_CALLS"].append((sid, missing_tools, s["score"], s["audio_text"]))
    # 3. Different tool names entirely
    elif set(p_names) != set(g_names):
        categories["3_DIFFERENT_TOOL_CHOICE"].append((sid, (p_names, g_names), s["score"], s["audio_text"]))
    # 4. Same tool names, but argument difference
    else:
        # Investigate argument diff
        arg_diffs = []
        for p, g in zip(pred_calls, gold_calls):
            if p["name"] == g["name"]:
                p_args = p.get("arguments", {})
                g_args = g.get("arguments", {})
                if p_args != g_args:
                    arg_diffs.append((p["name"], p_args, g_args))
        categories["4_ARGUMENT_MISMATCH"].append((sid, arg_diffs, s["score"], s["audio_text"]))

print("\n" + "=" * 80)
print("📊 BREAKDOWN OF THE REMAINING IMPERFECT CASES (520 samples):")
for cat, items in sorted(categories.items()):
    print(f"\n--- {cat}: {len(items)} samples ({len(items)/total*100:.2f}%) ---")
    for sid, detail, score, audio in items[:4]:
        print(f"  [{sid}] (Score: {score:.2f}) Req: {audio[:60]}...")
        if cat == "1_EXTRA_READ_CALLS":
            print(f"       -> Extra tools called: {detail}")
        elif cat == "2_MISSING_TOOL_CALLS":
            print(f"       -> Tools missing vs Gold: {detail}")
        elif cat == "3_DIFFERENT_TOOL_CHOICE":
            print(f"       -> Pred: {detail[0]} vs Gold: {detail[1]}")
        elif cat == "4_ARGUMENT_MISMATCH":
            for t_name, p_a, g_a in detail[:2]:
                print(f"       -> Tool {t_name}:")
                print(f"          Pred: {p_a}")
                print(f"          Gold: {g_a}")

