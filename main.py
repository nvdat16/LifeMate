"""Main CLI entrypoint for LifeMate Voice Agent system."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src and current directory to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
from lifemate.adapters.storage.jsonl_exporter import JsonlSubmissionExporter
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.adapters.asr.mock_asr import MockASREngine
from lifemate.adapters.asr.whisper_adapter import WhisperASREngine
from lifemate.adapters.llm.mock_llm import MockLLMEngine
from lifemate.workflow.orchestrator import LifeMatePipeline
from lifemate.workflow.graph_orchestrator import LifeMateGraphPipeline
from lifemate.workflow.single_agent_pipeline import LifeMateSingleAgentPipeline
from lifemate.evaluation.scorer import LocalScorer

def run_inference(
    split: str,
    limit: int | None,
    output_file: str,
    mode: str = "mock",
    model_name: str | None = None,
    base_url: str | None = None,
    asr_mode: str = "whisper",
    asr_model: str = "nunu-willump/PhoWhisper-medium-ct2",
    vieneu_voice_id: str | None = None,
    force_asr: bool = False,
    pipeline_type: str = "graph",
    data_root: str | Path | None = None
):
    data_root = Path(data_root or os.environ.get("ZAIC_DATA_ROOT", WORKSPACE_ROOT)).expanduser().resolve()

    api_key = "EMPTY"
    if mode == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to the environment/Kaggle Secrets; "
                "never put the key in source code or a command-line argument."
            )
        if not model_name:
            raise ValueError("Choose an OpenAI model explicitly with --model.")
        base_url = base_url or "https://api.openai.com/v1"
    elif mode in ("ollama", "llm"):
        model_name = model_name or "qwen2.5:1.5b"
        base_url = base_url or "http://localhost:11434/v1"

    print(f"\n🚀 Running LifeMate Multi-Agent Pipeline on split '{split}'...")
    print(f"📁 Data root: {data_root}")
    start_time = time.time()

    # 1. Load catalog
    catalog_path = data_root / split / f"{split}_tools_catalog.json"
    if not catalog_path.is_file():
        catalog_path = data_root / "train" / "train_tools_catalog.json"
    catalog = json.load(open(catalog_path, encoding="utf-8")) if catalog_path.is_file() else {}

    # 2. Setup Adapters
    reader = JsonlDatasetReader(data_root)
    exporter = JsonlSubmissionExporter()
    sandbox_factory = ZaicSandboxEnvFactory(data_root)

    # Local and hosted providers share the OpenAI-compatible chat-completions adapter.
    if mode in ("ollama", "llm", "openai"):
        provider = "OpenAI API" if mode == "openai" else "local LLM server"
        print(f"Connecting to {provider} model '{model_name}' at {base_url} ...")
        from lifemate.adapters.llm.local_llm_adapter import OpenAICompatibleLLMEngine
        llm_engine = OpenAICompatibleLLMEngine(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            parallel_tool_calls=False if mode == "openai" else None,
            timeout=120.0 if mode == "openai" else None
        )
    else:
        llm_engine = MockLLMEngine()

    if asr_mode in ("whisper", "vieneu") and split == "train" and not force_asr:
        print("🎙️ [Skip ASR] Train split has ground-truth audio_text; using it directly.")
        asr_engine = MockASREngine()
    elif asr_mode == "whisper":
        print(f"🎙️ Using PhoWhisper ASR model: '{asr_model}'")
        asr_engine = WhisperASREngine(model_size_or_path=asr_model, data_root=data_root)
    elif asr_mode == "vieneu":
        from lifemate.adapters.asr.vieneu_asr import VieNeuASREngine
        if not vieneu_voice_id:
            raise ValueError("Set --vieneu-voice-id to a valid VieNeu voice ID when using --asr vieneu.")
        print("🎙️ Using VieNeu transcription via its dubbing API (dubbing is billed)")
        asr_engine = VieNeuASREngine(voice_id=vieneu_voice_id, data_root=data_root)
    else:
        print("🎙️ Using Mock ASR Engine")
        asr_engine = MockASREngine()

    # 3. Instantiate Pipeline (Single Agent vs Dynamic Graph vs Classic)
    if pipeline_type == "classic":
        print("⚙️  Using Classic Pipeline")
        pipeline = LifeMatePipeline(
            asr_engine=asr_engine,
            llm_engine=llm_engine,
            sandbox_factory=sandbox_factory,
            catalog=catalog,
            force_asr=force_asr
        )
    elif pipeline_type == "graph":
        print("🕸️  Using Dynamic Planner & Parallel Worker Pool (LangGraph)")
        pipeline = LifeMateGraphPipeline(
            asr_engine=asr_engine,
            llm_engine=llm_engine,
            sandbox_factory=sandbox_factory,
            catalog=catalog,
            force_asr=force_asr
        )
    else:
        print("🤖 Using Single Agent ReAct Pipeline (High-Speed & End-to-End)")
        pipeline = LifeMateSingleAgentPipeline(
            asr_engine=asr_engine,
            llm_engine=llm_engine,
            sandbox_factory=sandbox_factory,
            catalog=catalog,
            force_asr=force_asr
        )

    # 4. Process Samples
    results = []
    sample_count = 0

    print(f"Reading samples from: {split}/{split}.jsonl")
    for sample in reader.read_samples(split, limit=limit):
        res = pipeline.process_sample(sample, split=split)
        results.append(res)
        sample_count += 1
        if sample_count % 10 == 0:
            print(f"  Processed {sample_count} samples...")

    # 5. Export to submission.jsonl
    out_path = exporter.export(results, output_file)
    elapsed = time.time() - start_time
    avg_time = (elapsed / sample_count) if sample_count > 0 else 0

    print(f"\n✅ Completed {sample_count} samples in {elapsed:.2f}s ({avg_time*1000:.1f}ms/sample)!")
    print(f"📄 Saved submission file to: {out_path}")
    print(f"   Matches BTC format: [id, tool_calls, state_hash]\n")

def run_evaluation(submission_file: str, data_root: str | Path | None = None):
    print(f"\n📊 Evaluating submission '{submission_file}' against train ground truth...")
    data_root = Path(data_root or os.environ.get("ZAIC_DATA_ROOT", WORKSPACE_ROOT)).expanduser().resolve()
    scorer = LocalScorer(data_root)
    metrics = scorer.evaluate(submission_file)
    print("\n" + "="*50)
    print("🏆 LOCAL EVALUATION METRICS REPORT")
    print("="*50)
    for k, v in metrics.items():
        print(f"  - {k}: {v}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="LifeMate Voice Agent CLI")
    parser.add_argument("--split", type=str, default="public_test", help="Dataset split ('train', 'public_test')")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--output", type=str, default="submission.jsonl", help="Output file path")
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "ollama", "llm", "openai"], help="LLM backend: mock, local Ollama/vLLM, or hosted OpenAI API")
    parser.add_argument("--model", type=str, default=None, help="Model name (local defaults to qwen2.5:1.5b; required for --mode openai)")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible endpoint; defaults to Ollama locally or https://api.openai.com/v1 for --mode openai")
    parser.add_argument("--asr", type=str, default="whisper", choices=["mock", "whisper", "vieneu"], help="ASR mode ('mock', 'whisper', 'vieneu')")
    parser.add_argument("--asr-model", type=str, default="nunu-willump/PhoWhisper-medium-ct2", help="HuggingFace model for PhoWhisper / faster-whisper")
    parser.add_argument("--vieneu-voice-id", type=str, default=None, help="VieNeu voice ID required by its dubbing-based transcription API")
    parser.add_argument("--force-asr", action="store_true", help="Force running ASR even if ground truth audio_text exists in train set")
    parser.add_argument("--pipeline", type=str, default="single", choices=["single", "graph", "classic"], help="Pipeline architecture ('single': Fast Single Agent ReAct, 'graph': Dynamic LangGraph, 'classic': Imperative)")
    parser.add_argument("--eval", type=str, default=None, help="Evaluate a submission file against train ground truth")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory containing train/ and public_test/ (or set ZAIC_DATA_ROOT)")
    args = parser.parse_args()

    if args.eval:
        run_evaluation(args.eval, data_root=args.data_root)
    else:
        run_inference(
            args.split,
            args.limit,
            args.output,
            mode=args.mode,
            model_name=args.model,
            base_url=args.base_url,
            asr_mode=args.asr,
            asr_model=args.asr_model,
            vieneu_voice_id=args.vieneu_voice_id,
            force_asr=args.force_asr,
            pipeline_type=args.pipeline,
            data_root=args.data_root
        )

if __name__ == "__main__":
    main()

