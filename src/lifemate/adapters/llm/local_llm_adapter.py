"""Adapter for OpenAI-compatible chat-completion endpoints.

Supports local servers such as Ollama and vLLM, as well as hosted APIs such as
OpenAI. Credentials are supplied by the caller and are never logged.
"""

import json
import urllib.request
import urllib.error
from typing import Any, Optional
from ...domain.entities import ToolCall
from ...ports.llm_port import AbstractLLMEngine, LLMDecision

class LocalLLMEngine(AbstractLLMEngine):
    """
    Connects to a local or hosted OpenAI-compatible inference server.
    Uses Python standard library (urllib) to guarantee zero extra dependencies and fast execution.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:1.5b",
        api_key: str = "EMPTY",
        timeout: Optional[float] = None,
        parallel_tool_calls: Optional[bool] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.parallel_tool_calls = parallel_tool_calls

    def predict_action(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        temperature: float = 0.0
    ) -> LLMDecision:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Build full message history with system prompt
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        # Build function calling tool schemas (compatible with Ollama & vLLM)
        formatted_tools = []
        for t in tools_schema:
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}})
                }
            })

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
        }
        if formatted_tools:
            payload["tools"] = formatted_tools
            payload["tool_choice"] = "auto"
            if self.parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = self.parallel_tool_calls

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            timeout_args = {} if self.timeout is None else {"timeout": self.timeout}
            with urllib.request.urlopen(req, **timeout_args) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data["choices"][0]["message"]
                thought = choice.get("content") or ""

                tool_call_obj = None
                is_done = True

                # 1. Native function tool_calls structure from local model
                if choice.get("tool_calls"):
                    tc = choice["tool_calls"][0]
                    fn = tc["function"]
                    name = fn["name"]
                    args_raw = fn.get("arguments", "{}")
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    tool_call_obj = ToolCall(name=name, arguments=args)
                    is_done = False

                # 2. Fallback: Parse embedded tool_call XML or JSON from local model output text
                elif thought and "<tool_call>" in thought:
                    import re
                    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", thought, re.DOTALL)
                    if match:
                        try:
                            parsed_tc = json.loads(match.group(1))
                            name = parsed_tc.get("name")
                            args = parsed_tc.get("arguments", {})
                            if name:
                                tool_call_obj = ToolCall(name=name, arguments=args)
                                is_done = False
                        except Exception:
                            pass

                return LLMDecision(
                    thought=thought,
                    tool_call=tool_call_obj,
                    is_done=is_done,
                    raw_response=str(choice)
                )
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"LLM endpoint returned HTTP {exc.code} at {url}{suffix}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not connect to LLM endpoint {url}: {exc.reason}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"LLM endpoint returned an invalid chat-completion response: {exc}") from exc

# Explicit aliases for the supported endpoint types.
OpenSourceLLMEngine = LocalLLMEngine
OllamaLLMEngine = LocalLLMEngine
OpenAICompatibleLLMEngine = LocalLLMEngine
