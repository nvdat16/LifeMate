import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import sys
from pathlib import Path
import json

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.evaluation.metrics import compute_sample_score

with open(DISTILLATION_DIR / "harmonized_sft_1526_samples.json", "r", encoding="utf-8") as f:
    harm_samples = {s["id"]: s for s in json.load(f)}

with open(DATA_ROOT / "train/train.jsonl", "r", encoding="utf-8") as f:
    train_samples = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

factory = ZaicSandboxEnvFactory(DATA_ROOT)

def harmonize_single_sample(sid):
    s_harm = harm_samples[sid]
    s_gold = train_samples[sid]
    gold_calls = s_gold.get("label", {}).get("tool_calls", [])
    gold_ask = [c for c in gold_calls if c["name"] == "sys.ask_missing_info"][0]
    
    msgs = list(s_harm["messages"])
    
    # Update system prompt
    UPGRADED_RULE = """4. AMBIGUITY & MISSING INFO PROTOCOL (STRICT MANDATORY TOOL CALL):
   - Whenever an entity search returns MULTIPLE candidates (e.g. 2 contacts named "Minh", 2 chats named "Lan"), or when a required parameter cannot be determined with certainty:
   - You MUST EXPLICITLY INVOKE `sys.ask_missing_info` with the intended write tool name and the ambiguous/missing parameter(s):
     {"name": "sys.ask_missing_info", "arguments": {"tool_name": "<intended_tool_name>", "params": ["<param_name>"]}}
   - NEVER ask the user only via natural language (final_response) without invoking `sys.ask_missing_info`! Calling `sys.ask_missing_info` is the obligatory system event to trigger user clarification."""
    
    OLD_RULE = "4. MISSING INFO: If mandatory information is missing and cannot be retrieved via read tools, call `sys.ask_missing_info`."
    if OLD_RULE in msgs[0]["content"]:
        msgs[0]["content"] = msgs[0]["content"].replace(OLD_RULE, UPGRADED_RULE)
        
    prompt_idx = None
    for i, m in enumerate(msgs):
        if m["role"] == "user" and "User Voice Request:" in m["content"]:
            prompt_idx = i
            break
    if prompt_idx is None:
        prompt_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"][-1]
        
    # Get last assistant message for final_response text
    last_asst = [m for m in msgs[prompt_idx:] if m["role"] == "assistant"][-1]
    last_data = json.loads(last_asst["content"])
    final_text = last_data.get("final_response") or "Tôi cần thêm thông tin để thực hiện yêu cầu của bạn."
    
    # Identify non-ask tools in gold
    gold_non_ask = [c for c in gold_calls if c["name"] != "sys.ask_missing_info"]
    
    # Build new current turn messages:
    # 1. User Voice Request (keep as is)
    new_curr = [msgs[prompt_idx]]
    
    session = factory.create_session("train", sid)
    
    if gold_non_ask:
        # Step 1: execute non-ask tools
        step1_obs = []
        for c in gold_non_ask:
            res = session.call(c["name"], c.get("arguments", {}))
            step1_obs.append({"tool_name": c["name"], "ok": res.ok, "result": res.result})
            
        thought_1 = f"Người dùng yêu cầu thực hiện tác vụ. Tuân thủ READ-BEFORE-WRITE và quy tắc tối giản tham số, thực hiện các thao tác cần thiết: {', '.join(c['name'] for c in gold_non_ask)}."
        new_curr.append({
            "role": "assistant",
            "content": json.dumps({
                "thought": thought_1,
                "tool_calls": gold_non_ask,
                "is_finished": False,
                "final_response": None
            }, ensure_ascii=False)
        })
        new_curr.append({
            "role": "user",
            "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(step1_obs, ensure_ascii=False, indent=2)}\n\nPlease provide your next action or final response."
        })
        
    # Step 2: execute sys.ask_missing_info
    thought_ask = "Phát hiện thông tin còn mơ hồ hoặc tìm thấy nhiều ứng viên / thiếu tham số bắt buộc. Tuân thủ nghiêm ngặt AMBIGUITY & MISSING INFO PROTOCOL, thực hiện gọi sys.ask_missing_info để kích hoạt giao thức hỏi lại người dùng trước khi thực thi."
    new_curr.append({
        "role": "assistant",
        "content": json.dumps({
            "thought": thought_ask,
            "tool_calls": [gold_ask],
            "is_finished": False,
            "final_response": None
        }, ensure_ascii=False)
    })
    
    ask_res = session.call(gold_ask["name"], gold_ask.get("arguments", {}))
    new_curr.append({
        "role": "user",
        "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps([{'tool_name': 'sys.ask_missing_info', 'ok': True, 'result': {'ok': True, 'waiting_for_user': True}}], ensure_ascii=False, indent=2)}\n\nPlease provide your next action or final response."
    })
    
    # Step 3: final response
    thought_final = "Đã phát lệnh sys.ask_missing_info và ghi nhận trạng thái chờ làm rõ. Cung cấp câu hỏi làm rõ tự nhiên cho người dùng."
    new_curr.append({
        "role": "assistant",
        "content": json.dumps({
            "thought": thought_final,
            "tool_calls": [],
            "is_finished": True,
            "final_response": final_text
        }, ensure_ascii=False)
    })
    
    final_messages = msgs[:prompt_idx] + new_curr
    
    # Evaluate in sandbox
    pred_hash = session.get_state_hash()
    gold_session = factory.create_session("train", sid)
    for c in gold_calls:
        gold_session.call(c["name"], c.get("arguments", {}))
    gold_hash = gold_session.get_state_hash()
    
    executed_calls = gold_non_ask + [gold_ask]
    eval_res = compute_sample_score(executed_calls, gold_calls, pred_hash, gold_hash)
    
    return {
        "id": sid,
        "messages": final_messages,
        "score": eval_res["score"],
        "state_match": eval_res["state_match"],
        "f1": eval_res["f1"]
    }

missing_ids = ['train_0042', 'train_0068', 'train_0065', 'train_0086', 'train_0605', 'train_0652', 'train_0775', 'train_0857', 'train_0974', 'train_1064', 'train_1103', 'train_1109', 'train_1147', 'train_1223', 'train_1237', 'train_1339', 'train_1427', 'train_1543', 'train_1570', 'train_1579', 'train_1620', 'train_1630', 'train_1679', 'train_1687', 'train_1688', 'train_1700', 'train_1723', 'train_1735', 'train_1743', 'train_0228', 'train_0243', 'train_0245', 'train_0273', 'train_0285', 'train_0331', 'train_0334', 'train_0341', 'train_0365', 'train_0366', 'train_0414', 'train_0420', 'train_0444']

all_res = []
for test_id in missing_ids:
    res = harmonize_single_sample(test_id)
    all_res.append(res)
    if res['score'] < 1.0 or not res['state_match']:
        print(f"Sample {test_id}: Score={res['score']:.2f}, F1={res['f1']:.2f}, State Match={res['state_match']}")

avg_s = sum(r['score'] for r in all_res) / len(all_res)
perfect_c = sum(1 for r in all_res if r['score'] == 1.0)
all_match = all(r['state_match'] for r in all_res)

print("=" * 70)
print(f"Total 42 samples evaluated:")
print(f"  Avg Score: {avg_s:.4f}")
print(f"  Perfect 1.00 count: {perfect_c} / {len(all_res)}")
print(f"  All State Match: {all_match}")
