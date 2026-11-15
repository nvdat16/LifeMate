"""
Single Agent ReAct Pipeline with gpt-6.1-terra (reasoning_effort=medium).

Features:
- Single Agent architecture (ReAct loop).
- Dynamic full schema extraction for all installed apps.
- Native OpenAI API call to gpt-6.1-terra with reasoning_effort='medium'.
- Closed-loop execution against the official ToolEnv Sandbox.
- Prints full raw text output, thoughts, tool calls, observations, and final response.
- Verifies state_hash against Gold Label.
"""

import os
import sys
import json
import re
import time
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))
from paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR, LLM_OUTPUT_DIR

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(DATA_ROOT / "public_test"))

import openai
from lifemate.domain.entities import SampleContext
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def require_openai_api_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")
    return OPENAI_API_KEY


def extract_full_installed_schemas(catalog: dict, installed_apps: dict) -> list[dict]:
    """
    Extracts the FULL parameter and property schemas for every tool
    belonging to installed apps in the sample context.
    """
    allowed_tool_names = set()
    for app_id, app_info in installed_apps.items():
        if isinstance(app_info, dict) and "tools" in app_info:
            allowed_tool_names.update(app_info["tools"])
        elif isinstance(app_info, list):
            allowed_tool_names.update(app_info)

    # Always include sys.ask_missing_info if available in catalog
    allowed_tool_names.add("sys.ask_missing_info")

    full_schemas = []
    for app in catalog.get("apps", []):
        app_id = app.get("id", "")
        for tool in app.get("tools", []):
            name = tool.get("name", "")
            if name in allowed_tool_names:
                full_schemas.append({
                    "name": name,
                    "app": app_id,
                    "kind": tool.get("kind", "write"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                    "returns": tool.get("returns", "")
                })
    return full_schemas

def build_system_prompt(context: SampleContext, tool_schemas: list[dict]) -> str:
    """Builds comprehensive system prompt with complete tool schemas and ReAct protocol."""
    return f"""You are LifeMate, an autonomous voice assistant for Vietnamese users.
You translate natural Vietnamese voice requests into concrete, precise tool calls to manage user apps and devices.

### CURRENT CONTEXT:
- Current Time: {context.current_time}
- Timezone: {context.timezone}
- Location: {json.dumps(context.location, ensure_ascii=False)}
- User Profile: {json.dumps(context.user_profile, ensure_ascii=False)}

### AVAILABLE TOOLS (FULL SCHEMAS):
Below are the exact schemas of all installed tools authorized for this session. You MUST strictly adhere to parameter names, types, and formats.
{json.dumps(tool_schemas, ensure_ascii=False, indent=2)}

### REACT OPERATING PROTOCOL:
You operate in an iterative closed-loop:
1. Analyze user request, conditional logic (e.g. if weather is rainy, reschedule; otherwise do nothing), relative time references ("mai", "3 giờ chiều"), and context.
2. READ-BEFORE-WRITE: If a write action requires specific entity IDs (e.g. event_id, contact_id, device_id), search/list them first using [read] tools.
3. IN-SCOPE vs OUT-OF-SCOPE: If user request cannot be fulfilled by any installed tool, return empty tool_calls with is_finished=true.
4. MISSING INFO: If mandatory information is missing and cannot be retrieved via read tools, call `sys.ask_missing_info`.

### OUTPUT FORMAT:
You MUST respond STRICTLY in JSON format (enclosed optionally in ```json ... ```):
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

def extract_json_from_raw_text(text: str) -> dict:
    """Extracts valid JSON object from model's raw text response."""
    # Strip thinking tags if any
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Try direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Try code block extraction
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except Exception:
            pass

    # Try bracket matching
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {"thought": cleaned, "tool_calls": [], "is_finished": True, "final_response": cleaned}

def run_single_agent(sample_index: int = 0, model_name: str = "gpt-6.1-terra"):
    catalog_path = DATA_ROOT / "train" / "train_tools_catalog.json"
    catalog = json.load(open(catalog_path, encoding="utf-8"))

    # Load target sample from train.jsonl
    with open(DATA_ROOT / "train" / "train.jsonl", "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    
    sample = samples[sample_index]
    sample_id = sample["id"]
    audio_text = sample.get("audio_text", "")
    raw_ctx = sample["context"]
    gold_label = sample.get("label", {})

    context = SampleContext(
        current_time=raw_ctx.get("current_time", ""),
        timezone=raw_ctx.get("timezone", ""),
        location=raw_ctx.get("location", {}),
        user_profile=raw_ctx.get("user_profile", {}),
        installed_apps=raw_ctx.get("installed_apps", {})
    )

    print("=" * 80)
    print(f"🤖 SINGLE AGENT REACT LOOP (MODEL: {model_name} | reasoning_effort=medium)")
    print("=" * 80)
    print(f"Sample ID:        {sample_id}")
    print(f"User Request:     {audio_text}")
    print(f"Current Time:     {context.current_time} ({context.timezone})")
    print(f"User Location:    {context.location}")
    print(f"Installed Apps:   {list(context.installed_apps.keys())}")
    print("=" * 80)

    # 1. Extract Full Schemas for installed apps
    full_schemas = extract_full_installed_schemas(catalog, context.installed_apps)
    print(f"📋 Loaded {len(full_schemas)} tool schemas for installed apps.")

    # 2. Setup Sandbox Session
    factory = ZaicSandboxEnvFactory(DATA_ROOT)
    session = factory.create_session("train", sample_id)
    initial_hash = session.get_state_hash()
    print(f"🔒 Initial Sandbox state_hash: {initial_hash}")

    # 3. Setup OpenAI Client
    client = openai.OpenAI(api_key=require_openai_api_key())

    system_prompt = build_system_prompt(context, full_schemas)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Voice Request: \"{audio_text}\""}
    ]

    all_executed_tool_calls = []
    max_turns = 5
    turn = 0

    while turn < max_turns:
        turn += 1
        print(f"\n" + "-" * 70)
        print(f"🔄 [TURN {turn}] Calling {model_name} (reasoning_effort='medium')...")
        print("-" * 70)

        start_t = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            reasoning_effort="medium"
        )
        elapsed = time.time() - start_t

        raw_output = response.choices[0].message.content
        print(f"⏱️ Model responded in {elapsed:.2f}s")
        print("\n📝 --- RAW MODEL OUTPUT ---")
        print(raw_output)
        print("----------------------------\n")

        parsed = extract_json_from_raw_text(raw_output)
        thought = parsed.get("thought", "")
        tool_calls = parsed.get("tool_calls", [])
        is_finished = parsed.get("is_finished", False)
        final_resp = parsed.get("final_response")

        print(f"🧠 Thought: {thought}")
        print(f"🛠️  Tool Calls to execute ({len(tool_calls)}): {json.dumps(tool_calls, ensure_ascii=False)}")
        print(f"🏁 Finished: {is_finished}")
        if final_resp:
            print(f"💬 Final Response: {final_resp}")

        # If model decided it's finished or no more tool calls
        if is_finished or not tool_calls:
            print("\n🎯 Agent signaled completion.")
            break

        # Execute tool calls in the Sandbox
        observations = []
        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_args = call.get("arguments", {})
            print(f"\n⚙️  [SANDBOX EXECUTION] Calling `{tool_name}` with args: {json.dumps(tool_args, ensure_ascii=False)}")
            
            obs = session.call(tool_name, tool_args)
            print(f"   -> Result: ok={obs.ok}, data={json.dumps(obs.result, ensure_ascii=False)[:200]}")
            
            all_executed_tool_calls.append({
                "name": tool_name,
                "arguments": tool_args
            })
            observations.append({
                "tool_name": tool_name,
                "ok": obs.ok,
                "result": obs.result
            })

        # Append assistant response & observations to messages for next turn
        messages.append({"role": "assistant", "content": raw_output})
        messages.append({
            "role": "user",
            "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(observations, ensure_ascii=False, indent=2)}\n\nPlease provide your next action or final response."
        })

    # Final Sandbox State Check
    final_hash = session.get_state_hash()
    print("\n" + "=" * 80)
    print("🏆 CLOSED-LOOP EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Turns:               {turn}")
    print(f"Total Tool Calls Executed: {len(all_executed_tool_calls)}")
    for i, c in enumerate(all_executed_tool_calls, 1):
        print(f"  {i}. {c['name']} -> {json.dumps(c['arguments'], ensure_ascii=False)}")

    print(f"\nGenerated state_hash:     {final_hash}")

    if gold_label:
        gold_calls = gold_label.get("tool_calls", [])
        print(f"\nGold Tool Calls ({len(gold_calls)}):")
        for i, c in enumerate(gold_calls, 1):
            print(f"  {i}. {c['name']} -> {json.dumps(c['arguments'], ensure_ascii=False)}")
        
        # Compare state_hash by creating a temporary session and replaying gold calls
        gold_session = factory.create_session("train", sample_id)
        for c in gold_calls:
            gold_session.call(c["name"], c["arguments"])
        gold_hash = gold_session.get_state_hash()
        
        print(f"Gold state_hash:          {gold_hash}")
        if final_hash == gold_hash:
            print("🎉 STATE_HASH MATCH: 100% IDENTICAL! (Full state points awarded!)")
        else:
            print(f"⚠️ STATE_HASH MISMATCH! (Got: {final_hash[:16]}... Expected: {gold_hash[:16]}...)")

    print("=" * 80)

if __name__ == "__main__":
    sample_idx = 0
    model_name = "gpt-5.6-luna"
    if len(sys.argv) > 1:
        try:
            sample_idx = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    run_single_agent(sample_idx, model_name=model_name)

