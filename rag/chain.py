"""The question-answering chain (LangChain LCEL).

retrieve -> format numbered context -> prompt -> LLM -> parse citations.
Every answer must cite sources as [n]; if the documents don't contain the
answer the model is instructed to say so instead of guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import Settings, settings as default_settings
from .llm import get_llm
from .retriever import HybridRetriever, get_retriever

SYSTEM_PROMPT = """You are a careful research assistant answering questions about a set of \
documents (e.g. State Bank of Pakistan circulars, bank and telecom annual reports).

Rules:
1. Answer ONLY from the numbered context passages below. Do not use outside knowledge.
2. Cite evidence inline with the passage number in square brackets, e.g. "... within six months [2]." \
Every factual sentence needs at least one citation. Cite several passages if they all support a claim.
3. Quote figures, dates, deadlines and circular numbers exactly as written in the passages.
4. If the passages do not contain the answer, reply exactly: "I couldn't find this in the provided documents." \
and then, if useful, say what related information IS available.
5. Be concise: a short paragraph or a few bullet points. No preamble.

Context passages:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    n: int
    source: str
    page: int | None
    chunk: int | None
    text: str
    score: float | None = None

    @property
    def label(self) -> str:
        return f"{self.source}" + (f", p.{self.page}" if self.page else "")


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[Source]
    cited: List[int] = field(default_factory=list)

    def cited_sources(self) -> List[Source]:
        return [s for s in self.sources if s.n in self.cited] or self.sources


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        header = f"[{i}] {m.get('source')}" + (f" (page {m['page']})" if m.get("page") else "")
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts)


def to_sources(docs: List[Document]) -> List[Source]:
    return [
        Source(
            n=i,
            source=str(d.metadata.get("source")),
            page=d.metadata.get("page"),
            chunk=d.metadata.get("chunk"),
            text=d.page_content,
            score=d.metadata.get("rrf_score"),
        )
        for i, d in enumerate(docs, start=1)
    ]


def _history_messages(history: Optional[List[dict]]) -> list:
    """history: [{"role": "user"|"assistant", "content": str}, ...] (last few turns)."""
    msgs = []
    for turn in (history or [])[-6:]:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        msgs.append(cls(content=turn["content"]))
    return msgs


class RAGChain:
    def __init__(self, cfg: Settings = default_settings, retriever: HybridRetriever | None = None, llm=None):
        self.cfg = cfg
        self.retriever = retriever or get_retriever(cfg)
        self.llm = llm or get_llm(cfg)
        self.chain = PROMPT | self.llm | StrOutputParser()

    # -- helpers --------------------------------------------------------------
    def retrieve(self, question: str) -> List[Document]:
        return self.retriever.invoke(question)

    def _inputs(self, question: str, docs: List[Document], history=None) -> dict:
        return {"context": format_context(docs), "question": question, "history": _history_messages(history)}

    # -- public API -----------------------------------------------------------
    def ask(self, question: str, history: Optional[List[dict]] = None) -> Answer:
        docs = self.retrieve(question)
        text = self.chain.invoke(self._inputs(question, docs, history))
        cited = sorted({int(n) for n in _CITE.findall(text) if 0 < int(n) <= len(docs)})
        return Answer(question=question, answer=text.strip(), sources=to_sources(docs), cited=cited)

    def stream(self, question: str, history: Optional[List[dict]] = None) -> tuple[List[Source], Iterator[str]]:
        """Returns (sources, token_iterator) so a UI can show sources while streaming."""
        docs = self.retrieve(question)
        return to_sources(docs), self.chain.stream(self._inputs(question, docs, history))
