import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
import sys
from pathlib import Path
import tiktoken
import numpy as np

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
from run_terra_on_failed_samples import extract_full_installed_schemas, build_system_prompt

enc = tiktoken.get_encoding("cl100k_base")

reader = JsonlDatasetReader(DATA_ROOT)
all_samples = {s.id: s for s in reader.read_samples("train")}
catalog = json.load(open(DATA_ROOT / "train" / "train_tools_catalog.json", encoding="utf-8"))

master_file = DISTILLATION_DIR / "master_train_distilled.jsonl"
trajectories = []
with open(master_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            trajectories.append(json.loads(line))

print(f"Loaded {len(trajectories)} trajectories.")

system_prompt_tokens = []
user_prompt_tokens = []
turn_1_input_tokens = []
max_turn_input_tokens = []
output_tokens_per_sample = []
total_trajectory_tokens = [] # Full multi-turn conversation length for SFT

for traj in trajectories:
    sid = traj["id"]
    sample = all_samples.get(sid)
    if not sample:
        continue
    
    context = sample.context
    tool_schemas = extract_full_installed_schemas(catalog, context.installed_apps)
    sys_prompt = build_system_prompt(context, tool_schemas)
    
    n_sys = len(enc.encode(sys_prompt))
    system_prompt_tokens.append(n_sys)
    
    # History
    hist_text = ""
    for turn in sample.history:
        if turn.role == "user":
            hist_text += turn.audio_text or turn.text or ""
        elif turn.role == "assistant":
            hist_text += turn.text or ""
            if turn.tool_calls:
                hist_text += json.dumps([tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in turn.tool_calls], ensure_ascii=False)
            if turn.tool_results:
                hist_text += json.dumps(turn.tool_results, ensure_ascii=False)
    
    user_req = f"User Voice Request: \"{sample.audio_text or ''}\""
    n_user = len(enc.encode(user_req + hist_text))
    user_prompt_tokens.append(n_user)
    
    t1_in = n_sys + n_user
    turn_1_input_tokens.append(t1_in)
    
    # Trajectory turns
    curr_context = t1_in
    max_in = t1_in
    sample_outputs = 0
    full_tokens = t1_in
    
    turns = traj.get("turns", [])
    for t in turns:
        # model generation
        resp_obj = {
            "thought": t.get("thought", ""),
            "tool_calls": t.get("tool_calls", []),
            "is_finished": t.get("is_finished", False),
            "final_response": t.get("final_response")
        }
        resp_text = json.dumps(resp_obj, ensure_ascii=False)
        n_out = len(enc.encode(resp_text))
        sample_outputs += n_out
        full_tokens += n_out
        
        # observation if any
        obs = t.get("observations", [])
        if obs:
            obs_text = f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(obs, ensure_ascii=False, indent=2)}\n\nPlease provide your next action or final response."
            n_obs = len(enc.encode(obs_text))
            curr_context += n_out + n_obs
            full_tokens += n_obs
            if curr_context > max_in:
                max_in = curr_context
                
    max_turn_input_tokens.append(max_in)
    output_tokens_per_sample.append(sample_outputs)
    total_trajectory_tokens.append(full_tokens)

def stats(arr, name):
    print(f"\n--- {name} (Tokens) ---")
    print(f"  Average (Mean):   {np.mean(arr):.1f}")
    print(f"  Median (50th):    {np.median(arr):.1f}")
    print(f"  Min:              {np.min(arr)}")
    print(f"  Max:              {np.max(arr)}")
    print(f"  95th Percentile:  {np.percentile(arr, 95):.1f}")
    print(f"  99th Percentile:  {np.percentile(arr, 99):.1f}")
    print(f"  Exceeds 16k (16384): {sum(1 for x in arr if x > 16384)} samples ({sum(1 for x in arr if x > 16384)/len(arr)*100:.2f}%)")

stats(system_prompt_tokens, "1. System Prompt (Context + Tools)")
stats(turn_1_input_tokens, "2. Turn 1 Input Tokens (Prompt + User Req)")
stats(max_turn_input_tokens, "3. Max Input Tokens at any Turn (Peak Inference Window)")
stats(output_tokens_per_sample, "4. Total Generated Tokens (Thought + Calls)")
stats(total_trajectory_tokens, "5. Full Trajectory Sequence Length (Full SFT Multi-turn Context)")
