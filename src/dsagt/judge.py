"""
The episodic-memory Judge — distills one conversational turn into a few tagged,
≤1-sentence facts for the ``session_memory`` collection.

This is Phase-3's Tier-1 layer (see ``design-notes/memory-plan.md``).  A
:class:`Judge` is the small generative LLM that turns a turn's exchanges into
``[{text, tag}]`` facts: closed-set tag classification over the project's tag
taxonomy + a ≤1-sentence *extractive* condense, with an explicit empty escape
(most turns produce nothing).  Reliability comes from asking it to do little —
classify into a fixed set and copy out one sentence, never "invent a category"
or summarize — which is exactly what a small (1–3B-class) model does well, and a
GBNF grammar pins the output to the JSON-array-of-tagged-facts shape so even a
1.5B model cannot emit malformed JSON or an off-taxonomy tag.

The abstraction mirrors :class:`dsagt.knowledge.Embedder`: ``Judge.create``
selects a backend; :class:`LocalJudge` is the default (an embeddable GGUF via
``llama-cpp-python`` — *local-by-default* so no second API key/cost gates
adoption), :class:`APIJudge` keeps the OpenAI-/Anthropic-compatible door open.
This module imports nothing heavy at import time; the GGUF runtime and the HTTP
client are lazy-loaded at first use so importing :mod:`dsagt.judge` (and merely
constructing a judge) stays cheap.

:class:`LocalJudge` is **live** (grammar-constrained GGUF inference);
:class:`APIJudge`'s ``distill`` is still a no-op (returns ``[]``) pending demand
— the abstraction, factory, prompt builder, and parser are shared, so filling it
in is a fill-in, not a refactor.

Class map — ``▷`` inherits · ``◇`` holds::

    Judge  (ABC)                         backend selector + distill() contract
    ├─▷ LocalJudge                       grammar-constrained GGUF (llama-cpp-python)
    └─▷ APIJudge                         OpenAI/Anthropic-compatible endpoint (stub)

    build_distill_prompt(exchanges, tags) → str     the lean per-turn prompt
    parse_distill_response(text, tags)    → [{text, tag}]   tolerant JSON parse
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction + response parsing (pure — shared by every backend)
# ---------------------------------------------------------------------------


def _render_turn(exchanges: list[dict]) -> str:
    """Flatten a turn's exchanges into a compact ``[role] text`` transcript.

    Reuses the content-block shape :meth:`CanonicalTrace.to_exchanges` emits
    (``new_messages`` / ``response`` lists of Anthropic blocks).  Kept local to
    the judge rather than importing memory's richer renderer: the judge wants a
    terse view (it classifies, it doesn't audit), and the seam stays one-way.
    """
    lines: list[str] = []
    for ex in exchanges:
        for msg in ex.get("new_messages", []):
            role = msg.get("role", "user")
            for block in msg.get("content", []):
                if block.get("type") == "text" and block.get("text"):
                    lines.append(f"[{role}] {block['text']}")
                elif block.get("type") == "tool_use":
                    lines.append(f"[{role} → {block.get('name', 'tool')}]")
        for block in ex.get("response", []):
            if block.get("type") == "text" and block.get("text"):
                lines.append(f"[assistant] {block['text']}")
            elif block.get("type") == "tool_use":
                lines.append(f"[assistant → {block.get('name', 'tool')}]")
    return "\n".join(lines)


def build_distill_prompt(exchanges: list[dict], tags: dict[str, str]) -> str:
    """The lean per-turn distillation prompt.

    Deliberately *not* ``memory.build_extraction_prompt`` (facts + summary +
    insights + classify is too much for a small model).  One job: pick a tag
    from the closed set and copy out ≤1 declarative sentence per fact, or return
    ``[]`` when the turn carries nothing worth remembering.
    """
    tag_list = "\n".join(f"- {name}: {desc}" for name, desc in tags.items())
    transcript = _render_turn(exchanges)
    return f"""\
You extract durable facts from one turn of a data-pipeline assistant session.

Rules:
- Output ONLY a JSON array, no prose, no markdown fences.
- Each element is {{"fact": "<one declarative sentence>", "tag": "<tag>"}}.
- Copy facts out of the turn (extract); do not summarize or infer.
- "tag" MUST be exactly one of the tags below — never invent one.
- Most turns carry nothing durable. When in doubt, output [].

Tags:
{tag_list}

Turn:
{transcript}

JSON array:"""


def parse_distill_response(text: str, tags: dict[str, str]) -> list[dict]:
    """Parse the model's JSON array into validated ``[{text, tag}]`` facts.

    Tolerant of a markdown fence around the array.  Drops malformed elements
    and any fact whose tag is outside the taxonomy (the closed set is what makes
    the small model viable — an off-set tag is a model error, not a new tag).
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    if not cleaned:
        return []

    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        return []

    facts: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        fact = (item.get("fact") or "").strip()
        tag = (item.get("tag") or "").strip()
        if fact and tag in tags:
            facts.append({"text": fact, "tag": tag})
    return facts


# ---------------------------------------------------------------------------
# Judge abstraction
# ---------------------------------------------------------------------------


class Judge(ABC):
    """A small generative LLM that distills a turn into tagged facts.

    Stateless after construction (the model/connection is the only state).  A
    single judge instance is shared by one :class:`~dsagt.memory.MemoryExtractor`
    across every turn it processes.
    """

    #: Short backend tag for span labelling / config display.
    backend: str = "unknown"
    #: Model identifier; subclasses set this in ``__init__``.
    model: str | None = None

    @classmethod
    def create(
        cls,
        backend: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
    ) -> "Judge":
        """Factory: construct the one judge for an extractor, with explicit args.

        Mirrors :meth:`Embedder.create` — each backend pulls only the parameters
        it uses, no ``**kwargs`` splat.  Defaults to ``local`` (the adoption
        bet: no API key required).
        """
        backend = (backend or "local").lower()
        if backend == "local":
            return LocalJudge(model=model)
        if backend == "api":
            return APIJudge(
                model=model, base_url=base_url, api_key=api_key, provider=provider
            )
        raise ValueError(f"Unknown judge backend {backend!r}. Choose from: local, api")

    @abstractmethod
    def distill(self, exchanges: list[dict], tags: dict[str, str]) -> list[dict]:
        """Distill one turn's ``exchanges`` into ``[{text, tag}]`` facts (or ``[]``)."""

    def close(self) -> None:
        pass


class LocalJudge(Judge):
    """Embeddable GGUF inference via ``llama-cpp-python`` — runs fully offline.

    The local weight is real (a generative LLM, GBs on disk + an inference
    runtime), unlike the light ``LocalEmbedder``; it loads lazily on first
    ``distill`` so constructing a ``LocalJudge`` (e.g. when memory is wired but
    Tier-1 stays disabled) costs nothing.
    """

    backend = "local"

    #: A small instruction-tuned, quantized GGUF — the class the plan calls for
    #: (reliable at closed-set classification, cheap to run on CPU; ~1GB Q4).
    #: GBNF grammar guarantees valid JSON regardless of size, so 1.5B is a safe
    #: quality floor for the one-sentence condense rather than a capability bet.
    #: Pulled from the HF hub on first use and cached; override via the
    #: ``judge.model`` config (e.g. a 0.5B for speed, or a local .gguf path).
    DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    DEFAULT_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

    def __init__(self, model: str | None = None):
        # ``model``, when set, is a local .gguf path; otherwise the default HF
        # repo/file pair is fetched lazily.  Stored only — no I/O here.
        self.model = model or self.DEFAULT_REPO
        self._model_path = model
        self._llm = None  # lazily-loaded llama_cpp.Llama
        self._grammars: dict[tuple, object] = {}  # tag-set → compiled LlamaGrammar

    def _ensure_llm(self):
        """Load the GGUF runtime on first use (heavy: native lib + GB weights)."""
        if self._llm is not None:
            return self._llm
        try:
            from llama_cpp import Llama
        except ImportError as e:  # actionable — it's a core dep, so this is an
            # incomplete install, not a missing opt-in.
            raise RuntimeError(
                "LocalJudge needs llama-cpp-python (a core dependency) — "
                "reinstall with `uv sync`."
            ) from e
        if self._model_path:
            self._llm = Llama(model_path=self._model_path, n_ctx=8192, verbose=False)
        else:
            self._llm = Llama.from_pretrained(
                repo_id=self.DEFAULT_REPO,
                filename=self.DEFAULT_FILE,
                n_ctx=8192,
                verbose=False,
            )
        return self._llm

    def _grammar(self, tags: dict[str, str]):
        """A GBNF grammar pinning output to ``[{fact: str, tag: <enum>}]``.

        Compiled from a JSON schema whose ``tag`` is an enum of the taxonomy, so
        the model *cannot* emit malformed JSON or an off-set tag — the single
        biggest reliability lever for a small model (``parse_distill_response``
        is then just defence in depth).  Cached per tag-set since the taxonomy is
        fixed for a project.
        """
        key = tuple(sorted(tags))
        if key not in self._grammars:
            from llama_cpp import LlamaGrammar

            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string"},
                        "tag": {"type": "string", "enum": list(tags)},
                    },
                    "required": ["fact", "tag"],
                    "additionalProperties": False,
                },
            }
            self._grammars[key] = LlamaGrammar.from_json_schema(json.dumps(schema))
        return self._grammars[key]

    def distill(self, exchanges: list[dict], tags: dict[str, str]) -> list[dict]:
        if not exchanges or not tags:
            return []
        llm = self._ensure_llm()
        out = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": build_distill_prompt(exchanges, tags)}
            ],
            max_tokens=512,
            temperature=0.0,  # deterministic extraction, not creative writing
            grammar=self._grammar(tags),
        )
        return parse_distill_response(out["choices"][0]["message"]["content"], tags)


class APIJudge(Judge):
    """OpenAI-/Anthropic-compatible endpoint — keeps the remote door open.

    Reuses the project's LLM gateway settings; carries a credential, so it is
    never the default (see the plan's adoption argument for local-by-default).
    """

    backend = "api"

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
    ):
        self.model = model or self.DEFAULT_MODEL
        self._base_url = base_url
        self._api_key = api_key
        self._provider = provider

    def distill(self, exchanges: list[dict], tags: dict[str, str]) -> list[dict]:
        # Tier-1 not yet wired.  When it is, mirror the old
        # ``memory.call_extraction_llm`` shape (litellm completion against the
        # configured gateway), then ``parse_distill_response`` the result.
        del exchanges, tags
        return []
