from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import APIError, BadRequestError, OpenAI, RateLimitError


DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_TEMPERATURE = 0.0

ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "abstained": {"type": "boolean"},
        "abstention_reason": {"type": ["string", "null"]},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["answer", "abstained", "abstention_reason", "citations"],
}


@dataclass(frozen=True)
class LLMResult:
    parsed: dict[str, Any]
    raw_response: dict[str, Any]
    request_id: str | None
    usage: dict[str, Any] | None


class OpenAIGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> None:
        load_dotenv()
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def generate_structured(self, instructions: str, user_input: str, max_retries: int = 3) -> LLMResult:
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self.client.responses.create(**self._request_args(instructions, user_input))
                return parse_response(response)
            except BadRequestError as error:
                if "temperature" not in str(error).lower():
                    raise
                response = self.client.responses.create(
                    **self._request_args(instructions, user_input, include_temperature=False)
                )
                return parse_response(response)
            except (RateLimitError, APIError) as error:
                last_error = error
                if attempt == max_retries - 1:
                    break
                time.sleep(2**attempt)

        assert last_error is not None
        raise last_error

    def _request_args(
        self,
        instructions: str,
        user_input: str,
        include_temperature: bool = True,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": user_input,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "policy_answer",
                    "strict": True,
                    "schema": ANSWER_SCHEMA,
                }
            },
        }
        if include_temperature:
            args["temperature"] = self.temperature
        return args


def parse_response(response: Any) -> LLMResult:
    text = response.output_text
    parsed = json.loads(text)
    return LLMResult(
        parsed=parsed,
        raw_response=response.model_dump(mode="json"),
        request_id=getattr(response, "_request_id", None),
        usage=response.usage.model_dump(mode="json") if getattr(response, "usage", None) else None,
    )
