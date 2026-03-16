from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests


# ============================================================
# Helpers
# ============================================================

def now_ms() -> float:
    return time.time() * 1000.0


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\sáàâãéèêíìîóòôõúùûç\-#\[\]]", "", text, flags=re.UNICODE)
    return text


def exact_match_contains(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(gold) in normalize_text(pred) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    p_count: Dict[str, int] = {}
    g_count: Dict[str, int] = {}

    for tok in p:
        p_count[tok] = p_count.get(tok, 0) + 1
    for tok in g:
        g_count[tok] = g_count.get(tok, 0) + 1

    common = 0
    for tok, cnt in p_count.items():
        common += min(cnt, g_count.get(tok, 0))

    precision = common / max(1, len(p))
    recall = common / max(1, len(g))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_citations(text: str) -> List[str]:
    return re.findall(r"\[([A-Za-z0-9_\-]+#p\d+)\]", text)


def citations_pr(pred_citations: List[str], gold_citations: List[str]) -> Tuple[float, float]:
    pred = set(pred_citations)
    gold = set(gold_citations)

    if not pred and not gold:
        return 1.0, 1.0
    if not pred:
        return 0.0, 0.0

    correct = len(pred.intersection(gold))
    precision = correct / max(1, len(pred))
    recall = correct / max(1, len(gold))
    return precision, recall


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {"_raw": text}

    return {"_raw": text}


def overlap_score(query: str, text: str) -> float:
    q_tokens = set(normalize_text(query).split())
    t_tokens = set(normalize_text(text).split())
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens.intersection(t_tokens)) / len(q_tokens)


# ============================================================
# Stats / Trace
# ============================================================

@dataclass
class CallStats:
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0
    wall_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_duration_ns": self.total_duration_ns,
            "load_duration_ns": self.load_duration_ns,
            "prompt_eval_duration_ns": self.prompt_eval_duration_ns,
            "eval_duration_ns": self.eval_duration_ns,
            "wall_ms": self.wall_ms,
        }


@dataclass
class TraceItem:
    name: str
    wall_ms: float
    prompt_tokens: int
    output_tokens: int


@dataclass
class RunTrace:
    items: List[TraceItem] = field(default_factory=list)

    def add(self, name: str, stats: CallStats) -> None:
        self.items.append(
            TraceItem(
                name=name,
                wall_ms=stats.wall_ms,
                prompt_tokens=stats.prompt_tokens,
                output_tokens=stats.output_tokens,
            )
        )

    def totals(self) -> Dict[str, float]:
        return {
            "wall_ms": sum(i.wall_ms for i in self.items),
            "prompt_tokens": sum(i.prompt_tokens for i in self.items),
            "output_tokens": sum(i.output_tokens for i in self.items),
        }

    def by_component(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for item in self.items:
            bucket = out.setdefault(
                item.name,
                {"wall_ms": 0.0, "prompt_tokens": 0.0, "output_tokens": 0.0},
            )
            bucket["wall_ms"] += item.wall_ms
            bucket["prompt_tokens"] += item.prompt_tokens
            bucket["output_tokens"] += item.output_tokens
        return out


# ============================================================
# Ollama local client
# ============================================================

class OllamaLocalClient:
    def __init__(
        self,
        host: Optional[str] = None,
        llm_model: str = "phi4",
        embed_model: str = "nomic-embed-text",
        temperature: float = 0.0,
        timeout: int = 180,
    ):
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.temperature = temperature
        self.timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        url = f"{self.host.rstrip('/')}{path}"
        t0 = now_ms()
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        t1 = now_ms()
        return response.json(), (t1 - t0)

    def chat_text(self, system: str, user: str, model: Optional[str] = None) -> Tuple[str, CallStats]:
        payload = {
            "model": model or self.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        resp, wall_ms = self._post("/api/chat", payload)
        content = resp.get("message", {}).get("content", "")

        stats = CallStats(
            prompt_tokens=int(resp.get("prompt_eval_count", 0) or 0),
            output_tokens=int(resp.get("eval_count", 0) or 0),
            total_duration_ns=int(resp.get("total_duration", 0) or 0),
            load_duration_ns=int(resp.get("load_duration", 0) or 0),
            prompt_eval_duration_ns=int(resp.get("prompt_eval_duration", 0) or 0),
            eval_duration_ns=int(resp.get("eval_duration", 0) or 0),
            wall_ms=wall_ms,
        )
        return content, stats

    def chat_json(self, system: str, user: str, model: Optional[str] = None) -> Tuple[Dict[str, Any], CallStats]:
        payload = {
            "model": model or self.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        resp, wall_ms = self._post("/api/chat", payload)
        content = resp.get("message", {}).get("content", "")
        obj = safe_json_loads(content)

        stats = CallStats(
            prompt_tokens=int(resp.get("prompt_eval_count", 0) or 0),
            output_tokens=int(resp.get("eval_count", 0) or 0),
            total_duration_ns=int(resp.get("total_duration", 0) or 0),
            load_duration_ns=int(resp.get("load_duration", 0) or 0),
            prompt_eval_duration_ns=int(resp.get("prompt_eval_duration", 0) or 0),
            eval_duration_ns=int(resp.get("eval_duration", 0) or 0),
            wall_ms=wall_ms,
        )
        return obj, stats

    def embed(self, texts: List[str], model: Optional[str] = None) -> Tuple[np.ndarray, int, float]:
        vectors: List[List[float]] = []
        total_prompt_tokens = 0
        total_wall_ms = 0.0

        for text in texts:
            payload = {
                "model": model or self.embed_model,
                "input": text,
            }
            resp, wall_ms = self._post("/api/embed", payload)
            total_wall_ms += wall_ms
            total_prompt_tokens += int(resp.get("prompt_eval_count", 0) or 0)

            embeddings = resp.get("embeddings", [])
            if not embeddings:
                raise RuntimeError("Nenhum embedding retornado pelo Ollama.")
            vectors.append(embeddings[0])

        arr = np.array(vectors, dtype=np.float32)
        return arr, total_prompt_tokens, total_wall_ms


# ============================================================
# Document store
# ============================================================

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    title: str
    text: str

    @property
    def cite(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"


class DocStore:
    def __init__(self, docs: Dict[str, Dict[str, Any]]):
        self.docs = docs
        self.chunks: List[Chunk] = []
        self._build_chunks()

    def _build_chunks(self) -> None:
        self.chunks.clear()
        for doc_id, payload in self.docs.items():
            title = payload["title"]
            paragraphs = [p.strip() for p in payload["text"].split("\n\n") if p.strip()]
            for idx, paragraph in enumerate(paragraphs):
                self.chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_id=f"p{idx}",
                        title=title,
                        text=paragraph,
                    )
                )

    def list_docs(self) -> List[Tuple[str, str]]:
        return [(doc_id, payload["title"]) for doc_id, payload in self.docs.items()]

    def get_chunks_by_doc(self, doc_id: str) -> List[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

    def get_chunk(self, cite: str) -> Chunk:
        doc_id, chunk_id = cite.split("#", 1)
        for chunk in self.chunks:
            if chunk.doc_id == doc_id and chunk.chunk_id == chunk_id:
                return chunk
        raise KeyError(f"Chunk não encontrado: {cite}")

    def dump_all_context(self) -> str:
        parts: List[str] = []
        for chunk in self.chunks:
            parts.append(f"[{chunk.cite}] ({chunk.title}) {chunk.text}")
        return "\n\n".join(parts)


# ============================================================
# Proposed non-RAG pipeline
# ============================================================

class Planner:
    def __init__(self, client: OllamaLocalClient):
        self.client = client

    def plan(self, question: str, doc_catalog: List[Tuple[str, str]]) -> Tuple[Dict[str, Any], CallStats]:
        system = (
            "Você é um planejador de QA com documentos locais. "
            "Transforme a pergunta em um plano curto e objetivo em JSON."
        )
        catalog_str = "\n".join([f"- {doc_id}: {title}" for doc_id, title in doc_catalog])

        user = f"""
Pergunta:
{question}

Catálogo de documentos:
{catalog_str}

Retorne JSON no formato:
{{
  "doc_candidates": ["DOC_A", "DOC_B"],
  "reading_budget": {{"max_chunks": 6, "max_evidence_items": 8}},
  "tree_budget": {{"beam_width": 3, "depth": 2}},
  "stop_rules": ["..."]
}}

Regras:
- doc_candidates deve usar doc_ids reais do catálogo.
- max_chunks entre 2 e 8.
- max_evidence_items entre 2 e 10.
- beam_width entre 2 e 4.
- depth entre 1 e 3.
"""
        return self.client.chat_json(system, user)


class DocumentNavigator:
    def __init__(self, client: OllamaLocalClient, store: DocStore):
        self.client = client
        self.store = store

    def pick_entry_chunks(self, question: str, doc_ids: List[str]) -> Tuple[List[str], CallStats]:
        candidates: List[Chunk] = []
        for doc_id in doc_ids:
            candidates.extend(self.store.get_chunks_by_doc(doc_id)[:2])

        if not candidates:
            candidates = self.store.chunks[:2]

        chunks_str = "\n".join(
            [f"- [{c.cite}] ({c.title}) {c.text[:180]}" for c in candidates]
        )

        system = (
            "Você é um navegador de documentos. "
            "Escolha pontos iniciais de leitura que parecem mais promissores."
        )

        user = f"""
Pergunta:
{question}

Possíveis pontos de entrada:
{chunks_str}

Retorne JSON:
{{
  "entry_chunks": ["DOC#p0", "DOC#p1"]
}}

Regras:
- Escolha de 1 a 3 chunks.
- Use apenas chunks existentes na lista.
"""
        obj, stats = self.client.chat_json(system, user)

        entry_chunks = obj.get("entry_chunks", [])
        if not isinstance(entry_chunks, list):
            entry_chunks = []

        valid = {c.cite for c in candidates}
        entry_chunks = [x for x in entry_chunks if isinstance(x, str) and x in valid]

        if not entry_chunks:
            entry_chunks = [candidates[0].cite]

        return entry_chunks[:3], stats


class RecursiveReader:
    def __init__(self, client: OllamaLocalClient, store: DocStore):
        self.client = client
        self.store = store

    def _fallback_next_chunk(self, current_chunk: Chunk) -> List[str]:
        same_doc = self.store.get_chunks_by_doc(current_chunk.doc_id)
        idx = None
        for i, chunk in enumerate(same_doc):
            if chunk.cite == current_chunk.cite:
                idx = i
                break
        if idx is None:
            return []
        if idx + 1 < len(same_doc):
            return [same_doc[idx + 1].cite]
        return []

    def read(
        self,
        question: str,
        entry_chunks: List[str],
        max_chunks: int,
    ) -> Tuple[List[Dict[str, Any]], RunTrace]:
        trace = RunTrace()
        visited = set()
        frontier = list(entry_chunks)
        evidence: List[Dict[str, Any]] = []

        system = (
            "Você está realizando leitura recursiva de documentos. "
            "Leia um trecho, extraia evidências e indique se precisa continuar."
        )

        while frontier and len(visited) < max_chunks:
            current_cite = frontier.pop(0)
            if current_cite in visited:
                continue

            visited.add(current_cite)
            current_chunk = self.store.get_chunk(current_cite)

            user = f"""
Pergunta:
{question}

Trecho atual [{current_chunk.cite}] ({current_chunk.title}):
\"\"\"{current_chunk.text}\"\"\"

Retorne JSON:
{{
  "evidence": [
    {{"claim": "...", "citations": ["{current_chunk.cite}"]}}
  ],
  "need_more": true,
  "next_chunks": ["DOC#p1"]
}}

Regras:
- Cada claim deve estar diretamente apoiada pelo trecho atual.
- Se o trecho já for suficiente, use need_more=false.
- Use apenas citações válidas.
"""

            obj, stats = self.client.chat_json(system, user)
            trace.add("recursive_read", stats)

            ev = obj.get("evidence", [])
            if isinstance(ev, list):
                for item in ev:
                    if (
                        isinstance(item, dict)
                        and isinstance(item.get("claim"), str)
                        and isinstance(item.get("citations"), list)
                    ):
                        evidence.append(item)

            need_more = bool(obj.get("need_more", False))
            next_chunks = obj.get("next_chunks", [])
            if not isinstance(next_chunks, list):
                next_chunks = []

            valid_same_doc = {c.cite for c in self.store.get_chunks_by_doc(current_chunk.doc_id)}
            sanitized_next = [
                x for x in next_chunks
                if isinstance(x, str) and x in valid_same_doc and x not in visited
            ]

            if need_more and not sanitized_next:
                sanitized_next = self._fallback_next_chunk(current_chunk)

            for cite in sanitized_next:
                if cite not in visited:
                    frontier.append(cite)

            frontier = frontier[:8]

        return evidence, trace


class SelectiveMemory:
    def __init__(self, max_items: int = 8):
        self.max_items = max_items

    def _score(self, question: str, claim: str) -> float:
        question_overlap = overlap_score(question, claim)
        length_score = min(len(normalize_text(claim).split()) / 20.0, 1.0)
        return 0.7 * question_overlap + 0.3 * length_score

    def compress(self, question: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in evidence:
            claim = str(item.get("claim", ""))
            scored.append((self._score(question, claim), item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[: self.max_items]]


class TreeReasoner:
    def __init__(self, client: OllamaLocalClient):
        self.client = client

    def reason(
        self,
        question: str,
        memory: List[Dict[str, Any]],
        beam_width: int = 3,
        depth: int = 2,
    ) -> Tuple[str, List[str], RunTrace]:
        trace = RunTrace()

        evidence_text = "\n".join(
            [
                f"- {m.get('claim', '')} | cite: {', '.join(m.get('citations', []))}"
                for m in memory
            ]
        )

        gen_system = (
            "Você é um gerador de respostas baseado apenas em evidências fornecidas. "
            "Não invente fatos e preserve citações."
        )

        judge_system = (
            "Você é um avaliador rigoroso. "
            "Escolha respostas mais corretas, mais fiéis às evidências e com melhores citações."
        )

        best_answer = ""
        best_citations: List[str] = []
        best_score = -1e9

        current_candidates = [""]

        for _ in range(max(1, depth)):
            round_candidates: List[Tuple[str, float, List[str]]] = []

            for seed_answer in current_candidates:
                gen_user = f"""
Pergunta:
{question}

Evidências:
{evidence_text}

Rascunho atual:
{seed_answer}

Retorne JSON:
{{
  "candidates": [
    {{
      "answer": "... [DOC#p0]",
      "citations": ["DOC#p0"]
    }}
  ]
}}

Regras:
- Gere até {beam_width} candidatos.
- Use apenas fatos apoiados pelas evidências.
- Use citações no corpo da resposta.
"""
                obj, stats = self.client.chat_json(gen_system, gen_user)
                trace.add("tree_generate", stats)

                candidates = obj.get("candidates", [])
                if not isinstance(candidates, list):
                    continue

                for cand in candidates[:beam_width]:
                    if not isinstance(cand, dict):
                        continue

                    answer = str(cand.get("answer", "")).strip()
                    citations = cand.get("citations", [])
                    if not isinstance(citations, list):
                        citations = []

                    judge_user = f"""
Pergunta:
{question}

Evidências:
{evidence_text}

Candidato:
{answer}

Retorne JSON:
{{
  "score": 0,
  "reason": "..."
}}

Critérios:
- Recompense respostas corretas e fiéis às evidências.
- Penalize extrapolações.
- Penalize citações irrelevantes.
"""
                    judged, judge_stats = self.client.chat_json(judge_system, judge_user)
                    trace.add("tree_judge", judge_stats)

                    score = float(judged.get("score", 0))
                    round_candidates.append((answer, score, citations))

                    if score > best_score:
                        best_score = score
                        best_answer = answer
                        best_citations = [str(c) for c in citations]

            round_candidates.sort(key=lambda x: x[1], reverse=True)
            current_candidates = [cand[0] for cand in round_candidates[:beam_width]]

        return best_answer, best_citations, trace


class AnswerSynthesizer:
    def __init__(self, client: OllamaLocalClient):
        self.client = client

    def finalize(self, question: str, draft: str) -> Tuple[str, CallStats]:
        system = (
            "Você é um redator final. "
            "Limpe a resposta, preserve apenas afirmações suportadas e mantenha citações."
        )

        user = f"""
Pergunta:
{question}

Rascunho:
{draft}

Retorne apenas a resposta final em português.
"""
        return self.client.chat_text(system, user)


# ============================================================
# Baselines
# ============================================================

class ContextStuffingBaseline:
    def __init__(self, client: OllamaLocalClient, store: DocStore):
        self.client = client
        self.store = store

    def answer(self, question: str) -> Tuple[str, CallStats]:
        context = self.store.dump_all_context()

        system = (
            "Você responderá com base em um grande bloco de contexto. "
            "Use apenas esse contexto e cite trechos no formato [DOC#pN] quando possível."
        )

        user = f"""
Contexto:
{context}

Pergunta:
{question}

Responda em português e inclua citações [DOC#pN].
Se não encontrar evidência suficiente, diga isso explicitamente.
"""
        return self.client.chat_text(system, user)


class SimpleRAGBaseline:
    def __init__(self, client: OllamaLocalClient, store: DocStore, top_k: int = 4):
        self.client = client
        self.store = store
        self.top_k = top_k
        self._embeddings: Optional[np.ndarray] = None
        self._chunk_texts: List[str] = []
        self._chunk_cites: List[str] = []
        self._build_index()

    def _build_index(self) -> None:
        self._chunk_texts = []
        self._chunk_cites = []

        for chunk in self.store.chunks:
            text = f"[{chunk.cite}] ({chunk.title}) {chunk.text}"
            self._chunk_texts.append(text)
            self._chunk_cites.append(chunk.cite)

        try:
            emb, _, _ = self.client.embed(self._chunk_texts)
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            self._embeddings = emb / norms
        except Exception:
            self._embeddings = None

    def _retrieve_by_embeddings(self, question: str) -> List[str]:
        if self._embeddings is None:
            raise RuntimeError("Embeddings não disponíveis.")

        q_emb, _, _ = self.client.embed([question])
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        sims = (self._embeddings @ q_emb.T).reshape(-1)
        idx = np.argsort(-sims)[: self.top_k]
        return [self._chunk_texts[i] for i in idx]

    def _retrieve_lexical_fallback(self, question: str) -> List[str]:
        scored = []
        for chunk in self.store.chunks:
            text = f"[{chunk.cite}] ({chunk.title}) {chunk.text}"
            scored.append((overlap_score(question, text), text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[: self.top_k]]

    def answer(self, question: str) -> Tuple[str, CallStats]:
        try:
            retrieved = self._retrieve_by_embeddings(question)
        except Exception:
            retrieved = self._retrieve_lexical_fallback(question)

        context = "\n\n".join(retrieved)

        system = (
            "Você é um baseline RAG simples. "
            "Use somente o contexto recuperado e cite as evidências."
        )

        user = f"""
Contexto recuperado:
{context}

Pergunta:
{question}

Responda em português.
Use citações [DOC#pN].
Se não houver evidência suficiente, diga isso.
"""
        return self.client.chat_text(system, user)


# ============================================================
# Runner
# ============================================================

def run_nonrag_pipeline(
    client: OllamaLocalClient,
    store: DocStore,
    question: str,
) -> Dict[str, Any]:
    trace = RunTrace()

    planner = Planner(client)
    navigator = DocumentNavigator(client, store)
    synthesizer = AnswerSynthesizer(client)

    plan, stats = planner.plan(question, store.list_docs())
    trace.add("planner", stats)

    valid_doc_ids = {doc_id for doc_id, _ in store.list_docs()}
    doc_candidates = plan.get("doc_candidates", [])
    if not isinstance(doc_candidates, list):
        doc_candidates = []
    doc_candidates = [d for d in doc_candidates if isinstance(d, str) and d in valid_doc_ids]

    if not doc_candidates:
        doc_candidates = [doc_id for doc_id, _ in store.list_docs()[:2]]

    reading_budget = plan.get("reading_budget", {})
    tree_budget = plan.get("tree_budget", {})

    max_chunks = int(reading_budget.get("max_chunks", 6))
    max_evidence_items = int(reading_budget.get("max_evidence_items", 8))
    beam_width = int(tree_budget.get("beam_width", 3))
    depth = int(tree_budget.get("depth", 2))

    max_chunks = max(2, min(max_chunks, 8))
    max_evidence_items = max(2, min(max_evidence_items, 10))
    beam_width = max(2, min(beam_width, 4))
    depth = max(1, min(depth, 3))

    entry_chunks, nav_stats = navigator.pick_entry_chunks(question, doc_candidates)
    trace.add("navigator", nav_stats)

    reader = RecursiveReader(client, store)
    evidence, read_trace = reader.read(question, entry_chunks, max_chunks=max_chunks)
    trace.items.extend(read_trace.items)

    memory = SelectiveMemory(max_items=max_evidence_items).compress(question, evidence)

    reasoner = TreeReasoner(client)
    draft, draft_citations, tree_trace = reasoner.reason(
        question=question,
        memory=memory,
        beam_width=beam_width,
        depth=depth,
    )
    trace.items.extend(tree_trace.items)

    final_answer, fin_stats = synthesizer.finalize(question, draft)
    trace.add("finalize", fin_stats)

    final_citations = parse_citations(final_answer)
    if not final_citations and draft_citations:
        final_citations = draft_citations

    return {
        "question": question,
        "plan": plan,
        "entry_chunks": entry_chunks,
        "memory": memory,
        "draft": draft,
        "answer": final_answer,
        "citations": final_citations,
        "trace": trace,
    }