"""AAA algorithms as MCP tools for a coding harness (opencode, DeepSeek Harness, ...).

The harness keeps its own loop and its own model; these tools are what it can call when a question deserves more than one pass.
The model behind the algorithms is configured with environment variables, so it can be a different (cheaper, local) one than the harness uses:

    AAA_BASE_URL   an OpenAI-compatible endpoint   (default http://localhost:8001/v1)
    AAA_MODEL      the model id                    (default mlx-community--Qwen3-4B-Instruct-2507-4bit)
    AAA_API_KEY    sent as a bearer token, if the endpoint needs one
"""

import os

from aaa.algos import SelfConsistency, SelfRefine
from aaa.llm import LLMConfig, LLMHandler, OpenAICompatClient
from aaa.runtime import Engine

careful = SelfConsistency(5)  # five samples, majority vote: for questions with one checkable answer
polished = SelfRefine(max_rounds=2)  # draft, critique, rewrite: for text


def engine() -> Engine:
    eng = Engine()
    LLMHandler(
        OpenAICompatClient(
            LLMConfig(
                model=os.environ.get("AAA_MODEL", "mlx-community--Qwen3-4B-Instruct-2507-4bit"),
                base_url=os.environ.get("AAA_BASE_URL", "http://localhost:8001/v1"),
                api_key=os.environ.get("AAA_API_KEY") or None,
                timeout_s=120,
                params={"temperature": 0.2, "max_tokens": 900},
            )
        )
    ).attach(eng)
    return eng
