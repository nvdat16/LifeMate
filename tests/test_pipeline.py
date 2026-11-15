"""Unit and integration tests for LifeMate Clean Architecture pipeline."""

import unittest
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "train"))

from lifemate.domain.entities import Sample, SampleContext, ToolCall
from lifemate.domain.scenario import ScenarioType
from lifemate.workflow.guardrail.type_caster import TypeCaster
from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.adapters.asr.mock_asr import MockASREngine
from lifemate.adapters.llm.mock_llm import MockLLMEngine
from lifemate.workflow.orchestrator import LifeMatePipeline
from lifemate.workflow.graph_orchestrator import LifeMateGraphPipeline
from lifemate.workflow.single_agent_pipeline import LifeMateSingleAgentPipeline

class TestLifeMatePipeline(unittest.TestCase):

    def test_type_caster_integers_and_time(self):
        tool_spec = {
            "parameters": {
                "properties": {
                    "temperature": {"type": "integer"},
                    "time": {"type": "string", "format": "time"}
                }
            }
        }
        args = {"temperature": "26 độ", "time": "7:00"}
        sanitized = TypeCaster.sanitize_arguments(args, tool_spec)
        self.assertEqual(sanitized["temperature"], 26)
        self.assertEqual(sanitized["time"], "07:00")

    def test_dataset_reader_loads_sample(self):
        reader = JsonlDatasetReader(WORKSPACE_ROOT)
        samples = list(reader.read_samples("train", limit=3))
        self.assertEqual(len(samples), 3)
        self.assertTrue(samples[0].id.startswith("train_"))
        self.assertIn("weather", samples[0].context.installed_apps)

    def test_sandbox_session_creation(self):
        factory = ZaicSandboxEnvFactory(WORKSPACE_ROOT)
        session = factory.create_session("train", "train_0001")
        state_hash = session.get_state_hash()
        self.assertTrue(len(state_hash) == 64)

        # Schema validate check
        val = session.validate("smart_home.light_set", {"device_id": "light_1", "power": "on"})
        # Should return ok (True or False), but not raise
        self.assertIn("ok", val)

    def test_pipeline_end_to_end_on_train_sample(self):
        reader = JsonlDatasetReader(WORKSPACE_ROOT)
        sample = reader.get_sample_by_id("train", "train_0001")

        factory = ZaicSandboxEnvFactory(WORKSPACE_ROOT)
        catalog = {
            "apps": [
                {
                    "id": "weather",
                    "tools": [{"name": "weather.get_current", "parameters": {}}]
                }
            ]
        }
        pipeline = LifeMatePipeline(
            asr_engine=MockASREngine(),
            llm_engine=MockLLMEngine(),
            sandbox_factory=factory,
            catalog=catalog
        )

        res = pipeline.process_sample(sample, split="train")
        self.assertEqual(res.id, "train_0001")
        self.assertEqual(len(res.state_hash), 64)
        self.assertIsNotNone(res.trajectory)
        self.assertTrue(len(res.trajectory.steps) > 0)

    def test_langgraph_pipeline_end_to_end(self):
        reader = JsonlDatasetReader(WORKSPACE_ROOT)
        sample = reader.get_sample_by_id("train", "train_0001")

        factory = ZaicSandboxEnvFactory(WORKSPACE_ROOT)
        catalog = {
            "apps": [
                {
                    "id": "weather",
                    "tools": [{"name": "weather.get_current", "parameters": {}}]
                }
            ]
        }
        graph_pipeline = LifeMateGraphPipeline(
            asr_engine=MockASREngine(),
            llm_engine=MockLLMEngine(),
            sandbox_factory=factory,
            catalog=catalog
        )

        res = graph_pipeline.process_sample(sample, split="train")
        self.assertEqual(res.id, "train_0001")
        self.assertEqual(len(res.state_hash), 64)
        self.assertIsNotNone(res.trajectory)
        self.assertTrue(len(res.trajectory.steps) > 0)

    def test_single_agent_pipeline_end_to_end(self):
        reader = JsonlDatasetReader(WORKSPACE_ROOT)
        sample = reader.get_sample_by_id("train", "train_0001")

        factory = ZaicSandboxEnvFactory(WORKSPACE_ROOT)
        catalog = {
            "apps": [
                {
                    "id": "weather",
                    "tools": [{"name": "weather.get_current", "parameters": {}}]
                }
            ]
        }
        single_pipeline = LifeMateSingleAgentPipeline(
            asr_engine=MockASREngine(),
            llm_engine=MockLLMEngine(),
            sandbox_factory=factory,
            catalog=catalog
        )

        res = single_pipeline.process_sample(sample, split="train")
        self.assertEqual(res.id, "train_0001")
        self.assertEqual(len(res.state_hash), 64)
        self.assertIsNotNone(res.trajectory)
        self.assertTrue(len(res.trajectory.steps) > 0)

if __name__ == "__main__":
    unittest.main()

