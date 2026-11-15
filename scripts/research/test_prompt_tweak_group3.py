import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import os
import sys
import json
import time
from pathlib import Path
import openai

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from lifemate.domain.entities import SampleContext, Sample
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
from lifemate.evaluation.metrics import compute_sample_score
from run_terra_on_failed_samples import extract_full_installed_schemas, extract_json_from_raw_text

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def require_openai_api_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")
    return OPENAI_API_KEY


# UPGRADED SYSTEM PROMPT WITH STRICT AMBIGUITY & MISSING INFO PROTOCOL
def build_upgraded_prompt(context: SampleContext, tool_schemas: list[dict]) -> str:
    compact_schemas = json.dumps(tool_schemas, ensure_ascii=False, separators=(',', ':'))
    return f"""You are LifeMate, an autonomous voice assistant for Vietnamese users.
You translate natural Vietnamese voice requests into concrete, precise tool calls to manage user apps and devices.

### CURRENT CONTEXT:
- Current Time: {context.current_time}
- Timezone: {context.timezone}
- Location: {json.dumps(context.location, ensure_ascii=False)}
- User Profile: {json.dumps(context.user_profile, ensure_ascii=False)}

### AVAILABLE TOOLS (FULL SCHEMAS):
Below are the exact schemas of all installed tools authorized for this session. You MUST strictly adhere to parameter names, types, and formats.
{compact_schemas}

### REACT OPERATING PROTOCOL:
You operate in an iterative closed-loop:
1. Analyze user request, conditional logic (e.g. if weather is rainy, reschedule; otherwise do nothing), relative time references ("mai", "3 giờ chiều"), and context.
2. READ-BEFORE-WRITE: If a write action requires specific entity IDs (e.g. event_id, contact_id, device_id), search/list them first using [read] tools.
3. IN-SCOPE vs OUT-OF-SCOPE: If user request cannot be fulfilled by any installed tool, return empty tool_calls with is_finished=true.
4. AMBIGUITY & MISSING INFO PROTOCOL (STRICT MANDATORY TOOL CALL):
   - Whenever an entity search returns MULTIPLE candidates (e.g. 2 contacts named "Minh", 2 chats named "Lan"), or when a required parameter cannot be determined with certainty:
   - You MUST EXPLICITLY INVOKE `sys.ask_missing_info` with the intended write tool name and the ambiguous/missing parameter(s):
     {{"name": "sys.ask_missing_info", "arguments": {{"tool_name": "<intended_tool_name>", "params": ["<param_name>"]}}}}
   - NEVER ask the user only via natural language (final_response) without invoking `sys.ask_missing_info`! Calling `sys.ask_missing_info` is the obligatory system event to trigger user clarification.

### PARAMETER INVOCATION RULES (GENERIC MINIMAL ARGUMENTS CONVENTION):
1. MINIMAL ARGUMENT CALLS: Only pass parameters that are strictly required by the schema (`required`) or specifically demanded by the user. Never pass optional parameters with default values.
2. IMPLICIT ENUM[0] DEFAULT: For any optional parameter with an `enum` list, the first element (`enum[0]`) is the platform's implicit default value. NEVER pass an optional parameter if its value matches `enum[0]`.
3. UNFILTERED READ TOOLS: For data query and inspection tools (e.g., tools whose names begin with `list_` or `get_status`), invoke them with an empty argument object {{}} so the environment returns the complete collection; perform all necessary filtering in memory.
4. MEMORY-FIRST & HISTORY REUSE: If requested information was already observed in conversation history, reuse it directly without re-querying read tools.
5. CONCISE FREE-TEXT: Use concise text without polite conversational padding.

### OUTPUT FORMAT:
You MUST respond STRICTLY in JSON format:
{{
  "thought": "<Detailed step-by-step reasoning explaining the logic, conditions checked, and tools selected>",
  "tool_calls": [
    {{
      "name": "<tool_name>",
      "arguments": {{
        "<parameter_name>": <value>
      }}
    }}
  ],
  "is_finished": false,
  "final_response": null
}}

When all actions are completed, or no actions are needed, return:
{{
  "thought": "<Explanation that all tasks are finished or out of scope>",
  "tool_calls": [],
  "is_finished": true,
  "final_response": "<Natural Vietnamese reply to the user>"
}}
"""

def test_samples():
    target_ids = ["train_0042", "train_0065", "train_0605", "train_0068"]
    reader = JsonlDatasetReader(DATA_ROOT)
    all_samples = {s.id: s for s in reader.read_samples("train")}
    catalog = json.load(open(DATA_ROOT / "train" / "train_tools_catalog.json", encoding="utf-8"))
    factory = ZaicSandboxEnvFactory(DATA_ROOT)
    client = openai.OpenAI(api_key=require_openai_api_key())

    print("=" * 80)
    print("🧪 TESTING UPGRADED PROMPT ON GROUP 3 SAMPLES (sys.ask_missing_info)")
    print(f"Target IDs: {target_ids}")
    print("=" * 80)

    for sid in target_ids:
        sample = all_samples[sid]
        context = sample.context
        tool_schemas = extract_full_installed_schemas(catalog, context.installed_apps)
        system_prompt = build_upgraded_prompt(context, tool_schemas)

        messages = [{"role": "system", "content": system_prompt}]
        for turn in sample.history:
            if turn.role == "user":
                messages.append({"role": "user", "content": turn.audio_text or turn.text or ""})
            elif turn.role == "assistant":
                if turn.tool_calls:
                    tc_dicts = [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in turn.tool_calls]
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps({"thought": "History action", "tool_calls": tc_dicts, "is_finished": False}, ensure_ascii=False)
                    })
                    if turn.tool_results:
                        messages.append({
                            "role": "user",
                            "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(turn.tool_results, ensure_ascii=False)}"
                        })
                if turn.text:
                    messages.append({"role": "assistant", "content": turn.text})

        messages.append({"role": "user", "content": f"User Voice Request: \"{sample.audio_text or ''}\""})

        session = factory.create_session("train", sid)
        executed_calls = []

        for cur_turn in range(4):
            resp = client.chat.completions.create(
                model="gpt-5.6-luna",
                messages=messages
            )
            raw_out = resp.choices[0].message.content or ""
            parsed = extract_json_from_raw_text(raw_out)
            thought = parsed.get("thought", "")
            tool_calls = parsed.get("tool_calls", [])
            is_finished = parsed.get("is_finished", False)

            observations = []
            if tool_calls:
                for c in tool_calls:
                    c_name = c.get("name", "")
                    c_args = c.get("arguments", {})
                    obs = session.call(c_name, c_args)
                    executed_calls.append({"name": c_name, "arguments": c_args})
                    observations.append({"tool_name": c_name, "ok": obs.ok, "result": obs.result})

            if is_finished or not tool_calls:
                break

            messages.append({"role": "assistant", "content": raw_out})
            messages.append({
                "role": "user",
                "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(observations, ensure_ascii=False, indent=2)}\n\nPlease provide your next action or final response."
            })

        pred_hash = session.get_state_hash()
        gold_calls = sample.label.get("tool_calls", []) if sample.label else []
        gold_session = factory.create_session("train", sid)
        for gc in gold_calls:
            gold_session.call(gc["name"], gc.get("arguments", {}))
        gold_hash = gold_session.get_state_hash()

        eval_res = compute_sample_score(executed_calls, gold_calls, pred_hash, gold_hash)

        print("\n" + "=" * 70)
        print(f"Sample {sid}:")
        print(f"  User Request:  {sample.audio_text}")
        print(f"  Pred Calls:    {executed_calls}")
        print(f"  Gold Calls:    {gold_calls}")
        print(f"  Score:         {eval_res['score']:.2f}")
        print(f"  F1:            {eval_res['f1']:.2f}")
        print(f"  State Match:   {eval_res['state_match']}")

if __name__ == "__main__":
    test_samples()
