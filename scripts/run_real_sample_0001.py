import json
import time
import sys
import os
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("ZAIC_DATA_ROOT", WORKSPACE_ROOT)).expanduser().resolve()
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(DATA_ROOT / "public_test"))

from lifemate.domain.entities import Sample, SampleContext
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.adapters.llm.local_llm_adapter import LocalLLMEngine
from lifemate.workflow.agents.dynamic_planner import DynamicPlannerAgent
from lifemate.workflow.agents.worker_agent import WorkerAgent
from lifemate.workflow.guardrail.schema_checker import GuardrailInterceptor

def run_real_sample():
    catalog_path = DATA_ROOT / "train" / "train_tools_catalog.json"
    catalog = json.load(open(catalog_path, encoding="utf-8"))

    # Load train_0001
    with open(DATA_ROOT / "train" / "train.jsonl", "r", encoding="utf-8") as f:
        raw_sample = json.loads(f.readline())

    sample_id = raw_sample["id"]
    audio_text = raw_sample["audio_text"]
    raw_ctx = raw_sample["context"]
    context = SampleContext(
        current_time=raw_ctx.get("current_time", ""),
        timezone=raw_ctx.get("timezone", ""),
        location=raw_ctx.get("location", {}),
        user_profile=raw_ctx.get("user_profile", {}),
        installed_apps=raw_ctx.get("installed_apps", {})
    )

    print("=" * 80)
    print(f"RUNNING REAL SAMPLE: {sample_id}")
    print(f"INPUT AUDIO TEXT: {audio_text}")
    print(f"CURRENT TIME: {context.current_time}")
    print(f"USER LOCATION: {context.location}")
    print("=" * 80)

    # Initialize LocalLLMEngine with no timeout (CPU execution)
    llm = LocalLLMEngine(model="qwen2.5:1.5b", base_url="http://127.0.0.1:11434/v1", timeout=None)
    planner = DynamicPlannerAgent(llm_engine=llm, catalog=catalog)
    worker = WorkerAgent(llm_engine=llm, catalog=catalog)
    guardrail = GuardrailInterceptor(catalog=catalog, allowed_tools=context.allowed_tools)

    # Official Sandbox Session for train_0001
    factory = ZaicSandboxEnvFactory(DATA_ROOT)
    session = factory.create_session("train", sample_id)

    abstract_tools = planner.extract_abstract_catalog(context)
    observations = []
    completed_tasks = []
    iteration = 0
    max_iterations = 4

    history_log = []

    while iteration < max_iterations:
        print(f"\n>>>>>>>>>>>>>>> [CYCLE {iteration + 1} - PLANNER] <<<<<<<<<<<<<<<")
        t0 = time.time()
        
        # Planner planning
        plan_result = planner.plan_next_batch(
            transcript=audio_text,
            context=context,
            abstract_tools=abstract_tools,
            observations=observations,
            completed_tasks=completed_tasks,
            iteration=iteration
        )
        t_plan = time.time() - t0

        planner_sys_prompt = planner.last_system_prompt
        planner_messages = planner.last_messages
        planner_raw_thought = planner.last_raw_thought

        print("\n--- [PLANNER SYSTEM PROMPT] ---")
        print(planner_sys_prompt)
        print("\n--- [PLANNER USER INPUT (MESSAGES)] ---")
        print(json.dumps(planner_messages, ensure_ascii=False, indent=2))
        print(f"\n--- [PLANNER RAW OUTPUT from LLM ({t_plan:.2f}s)] ---")
        print(planner_raw_thought)
        print("\n--- [PARSED PLAN RESULT] ---")
        print(json.dumps(plan_result, ensure_ascii=False, indent=2))

        history_log.append({
            "cycle": iteration + 1,
            "node": "PLANNER",
            "system_prompt": planner_sys_prompt,
            "messages": planner_messages,
            "raw_output": planner_raw_thought,
            "parsed_plan": plan_result,
            "duration": t_plan
        })

        if plan_result.get("is_finished", False):
            print("\n>> Planner flagged is_finished=True. Terminating loop.")
            break

        tasks = plan_result.get("parallel_tasks", [])
        if not tasks:
            print("\n>> No parallel tasks emitted. Terminating loop.")
            break

        print(f"\n>>>>>>>>>>>>>>> [CYCLE {iteration + 1} - WORKER POOL ({len(tasks)} tasks)] <<<<<<<<<<<<<<<")
        for idx, task in enumerate(tasks, 1):
            tool_name = task.get("tool_name", "")
            task_id = task.get("task_id", f"T{idx}")
            desc = task.get("description", tool_name)

            print(f"\n--- [WORKER TASK {task_id}: {tool_name}] ---")
            t_w0 = time.time()
            
            resolved_args, worker_sys_prompt, worker_messages, worker_raw = worker.resolve_arguments(
                task=task,
                transcript=audio_text,
                context=context,
                observations=observations
            )
            t_work = time.time() - t_w0

            print("\n[WORKER SYSTEM PROMPT]")
            print(worker_sys_prompt)
            print("\n[WORKER INPUT (MESSAGES)]")
            print(json.dumps(worker_messages, ensure_ascii=False, indent=2))
            print(f"\n[WORKER RAW OUTPUT from LLM ({t_work:.2f}s)]")
            print(worker_raw)
            print(f"\n[WORKER RESOLVED ARGUMENTS]: {json.dumps(resolved_args, ensure_ascii=False)}")

            # Validate & Run in Sandbox
            is_valid, sanitized_args, err = guardrail.intercept(tool_name, resolved_args, session)
            print(f"[GUARDRAIL STATUS]: {'PASSED' if is_valid else f'REJECTED ({err})'}")

            if is_valid:
                obs = session.call(tool_name, sanitized_args)
                print(f"[SANDBOX EXECUTION RESULT]: ok={obs.ok}, result={obs.result}")
                obs_record = {
                    "task_id": task_id,
                    "tool_name": tool_name,
                    "result": obs.result,
                    "ok": obs.ok
                }
                observations.append(obs_record)
                completed_tasks.append({
                    "task_id": task_id,
                    "tool_name": tool_name,
                    "arguments": sanitized_args,
                    "status": "COMPLETED"
                })
            else:
                obs_record = {
                    "task_id": task_id,
                    "tool_name": tool_name,
                    "result": {"error": err},
                    "ok": False
                }
                observations.append(obs_record)

            history_log.append({
                "cycle": iteration + 1,
                "node": "WORKER",
                "task_id": task_id,
                "tool_name": tool_name,
                "system_prompt": worker_sys_prompt,
                "messages": worker_messages,
                "raw_output": worker_raw,
                "resolved_arguments": resolved_args,
                "observation": obs_record,
                "duration": t_work
            })

        iteration += 1

    # Save detailed execution log to disk
    with open("scripts/sample_0001_real_run_log.json", "w", encoding="utf-8") as f:
        json.dump(history_log, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("FINISHED REAL EXECUTION RUN.")
    print("=" * 80)

if __name__ == "__main__":
    run_real_sample()
