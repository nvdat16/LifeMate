import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import json

from lifemate.domain.entities import SampleContext
from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
from lifemate.workflow.single_agent_pipeline import LifeMateSingleAgentPipeline
from lifemate.adapters.asr.mock_asr import MockASREngine
from lifemate.adapters.llm.mock_llm import MockLLMEngine
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory

catalog = json.load(open(DATA_ROOT / "train/train_tools_catalog.json", encoding="utf-8"))
factory = ZaicSandboxEnvFactory(DATA_ROOT)
asr = MockASREngine()
llm = MockLLMEngine()
pipe = LifeMateSingleAgentPipeline(asr, llm, factory, catalog)

reader = JsonlDatasetReader(DATA_ROOT)
s = reader.get_sample_by_id("train", "train_0008")

# 1. System Prompt
schemas = pipe.extract_full_installed_schemas(s.context)
system_prompt = pipe.build_system_prompt(s.context, schemas)

# 2. Initial Messages
messages = pipe._build_initial_messages(s, s.audio_text)

print(f"System Prompt Length: {len(system_prompt)} chars")
print(f"Total Initial Messages: {len(messages)}")

print("\n" + "="*80)
print("SYSTEM PROMPT (Snippet):")
print("="*80)
print(system_prompt[:500] + "\n... [Full Tool Schemas of 10 Apps] ...")

print("\n" + "="*80)
print("INITIAL MESSAGES SENT TO AGENT SERVICE (OpenAI / LangChain / vLLM):")
print("="*80)
for idx, m in enumerate(messages):
    role = m["role"]
    content = m["content"]
    print(f"\n--- [Message {idx}] role: '{role}' ---")
    print(content)
