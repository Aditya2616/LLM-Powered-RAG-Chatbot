"""Generators behind one small interface.

``local`` (default): a small open instruct model through ``transformers``, no key needed.
``openai``: any OpenAI-compatible ``/chat/completions`` endpoint. ``gemini``: the Gemini REST API.
``extractive``: no model at all; it returns the retrieved sentence that best matches the question.
Hosted model ids come from the environment only. The hosted adapters are exercised in the tests
against a stubbed HTTP transport; no hosted provider has been called for real.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .config import ConfigError, Settings, pick_device
from .embeddings import ProviderError, post_json
from .text import ABSTAIN_TEXT, content_tokens, stem


@dataclass
class Prompt:
    """What the pipeline asks a generator. Hosted and local models use ``system`` and ``user``;
    the extractive generator uses ``question`` and ``passages`` directly."""

    system: str
    user: str
    question: str = ""
    passages: list[str] = field(default_factory=list)
    messages: list[dict[str, str]] | None = None  # optional few-shot turns placed before ``user``

    def chat(self) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system}]
        msgs.extend(self.messages or [])
        msgs.append({"role": "user", "content": self.user})
        return msgs


class LLM(Protocol):
    name: str

    def generate(self, prompt: Prompt, max_new_tokens: int, temperature: float) -> str: ...

    def stream(self, prompt: Prompt, max_new_tokens: int, temperature: float) -> Iterator[str]: ...


class ExtractiveLLM:
    """Returns the single retrieved sentence with the most question words in it, or abstains."""

    name = "extractive"

    def __init__(self, min_overlap: int = 2) -> None:
        self.min_overlap = min_overlap

    def generate(self, prompt: Prompt, max_new_tokens: int = 0, temperature: float = 0.0) -> str:
        q = {stem(t) for t in content_tokens(prompt.question)}
        best, best_score = "", (0, 0)
        for passage in prompt.passages:
            for sent in re.split(r"(?<=[.!?])\s+|\n+", passage):
                overlap = len(q & {stem(w) for w in content_tokens(sent)})
                if (overlap, len(sent)) > best_score:  # ties go to the longer, fuller sentence
                    best, best_score = sent.strip(), (overlap, len(sent))
        return best if best_score[0] >= self.min_overlap else ABSTAIN_TEXT

    def stream(
        self, prompt: Prompt, max_new_tokens: int = 0, temperature: float = 0.0
    ) -> Iterator[str]:
        text = self.generate(prompt, max_new_tokens, temperature)
        words = text.split(" ")
        for i, w in enumerate(words):
            yield w if i == 0 else " " + w


class LocalLLM:
    """A causal LM from the Hugging Face hub, loaded on first use (float16 on CUDA, else float32)."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.name = f"local:{model_name}"
        self.device_request = device
        self._tok: Any = None
        self._model: Any = None
        self.device = ""
        self.dtype = ""

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ConfigError("the local generator needs: pip install -e '.[local]'") from exc
        self.device = pick_device(self.device_request)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.dtype = str(dtype).replace("torch.", "")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        self._model.to(self.device).eval()

    def _inputs(self, prompt: Prompt) -> Any:
        text = self._tok.apply_chat_template(
            prompt.chat(), tokenize=False, add_generation_prompt=True
        )
        return self._tok(text, return_tensors="pt").to(self.device)

    def _kwargs(self, max_new_tokens: int, temperature: float) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tok.eos_token_id,
        }
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        return kw

    def generate(self, prompt: Prompt, max_new_tokens: int, temperature: float) -> str:
        import torch

        self._load()
        inputs = self._inputs(prompt)
        with torch.inference_mode():
            out = self._model.generate(**inputs, **self._kwargs(max_new_tokens, temperature))
        new = out[0][inputs["input_ids"].shape[1] :]
        return self._tok.decode(new, skip_special_tokens=True).strip()

    def stream(self, prompt: Prompt, max_new_tokens: int, temperature: float) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer

        self._load()
        inputs = self._inputs(prompt)
        streamer = TextIteratorStreamer(self._tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(inputs, streamer=streamer, **self._kwargs(max_new_tokens, temperature))

        def run() -> None:
            with torch.inference_mode():
                self._model.generate(**kwargs)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            yield from streamer
        finally:
            thread.join()


def _sse_data(lines: Iterator[str]) -> Iterator[str]:
    for line in lines:
        if line.startswith("data:"):
            data = line[5:].strip()
            if data:
                yield data


class OpenAICompatLLM:
    """``POST {base_url}/chat/completions``, plain or streamed (server-sent events)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.name = f"openai:{model}"
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.Client(transport=transport, timeout=120.0)

    def _body(self, prompt: Prompt, max_new_tokens: int, temperature: float, stream: bool) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": prompt.chat(),
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if stream:
            body["stream"] = True
        return body

    def generate(self, prompt: Prompt, max_new_tokens: int, temperature: float) -> str:
        body = post_json(
            self._client,
            f"{self.base_url}/chat/completions",
            self._body(prompt, max_new_tokens, temperature, False),
            self._headers,
        )
        try:
            return (body["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("unexpected response shape from /chat/completions") from exc

    def stream(self, prompt: Prompt, max_new_tokens: int, temperature: float) -> Iterator[str]:
        url = f"{self.base_url}/chat/completions"
        with self._client.stream(
            "POST",
            url,
            json=self._body(prompt, max_new_tokens, temperature, True),
            headers=self._headers,
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise ProviderError(f"request to {url} failed (HTTP {resp.status_code})")
            for data in _sse_data(resp.iter_lines()):
                if data == "[DONE]":
                    return
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise ProviderError("unexpected event in the completion stream") from exc
                if delta:
                    yield delta


class GeminiLLM:
    """``models/{model}:generateContent`` (and ``streamGenerateContent`` with SSE)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.name = f"gemini:{self.model}"
        self.base_url = base_url.rstrip("/")
        self._headers = {"x-goog-api-key": api_key}
        self._client = httpx.Client(transport=transport, timeout=120.0)

    def _body(self, prompt: Prompt, max_new_tokens: int, temperature: float) -> dict:
        contents = []
        for m in [*(prompt.messages or []), {"role": "user", "content": prompt.user}]:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return {
            "systemInstruction": {"parts": [{"text": prompt.system}]},
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_new_tokens},
        }

    @staticmethod
    def _text(body: dict) -> str:
        try:
            parts = body["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            return ""
        return "".join(p.get("text", "") for p in parts)

    def generate(self, prompt: Prompt, max_new_tokens: int, temperature: float) -> str:
        body = post_json(
            self._client,
            f"{self.base_url}/{self.model}:generateContent",
            self._body(prompt, max_new_tokens, temperature),
            self._headers,
        )
        if "candidates" not in body:
            raise ProviderError("the response has no candidates (blocked or empty)")
        return self._text(body).strip()

    def stream(self, prompt: Prompt, max_new_tokens: int, temperature: float) -> Iterator[str]:
        url = f"{self.base_url}/{self.model}:streamGenerateContent?alt=sse"
        with self._client.stream(
            "POST", url, json=self._body(prompt, max_new_tokens, temperature), headers=self._headers
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise ProviderError(
                    f"request to {url.split('?')[0]} failed (HTTP {resp.status_code})"
                )
            for data in _sse_data(resp.iter_lines()):
                try:
                    text = self._text(json.loads(data))
                except ValueError as exc:
                    raise ProviderError("unexpected event in the generation stream") from exc
                if text:
                    yield text


def get_llm(settings: Settings, transport: httpx.BaseTransport | None = None) -> LLM:
    provider = settings.llm_provider
    if provider == "extractive":
        return ExtractiveLLM()
    if provider == "local":
        return LocalLLM(settings.local_llm_model, settings.device)
    settings.require_hosted()
    if provider == "openai":
        return OpenAICompatLLM(
            settings.openai_api_key, settings.openai_model, settings.openai_base_url, transport
        )
    return GeminiLLM(
        settings.gemini_api_key, settings.gemini_model, settings.gemini_base_url, transport
    )
