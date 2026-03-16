from __future__ import annotations

import json
import os
import statistics
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from .demo_data import build_demo_docs_and_qa
from .pipeline import (
    ContextStuffingBaseline,
    DocStore,
    OllamaLocalClient,
    SimpleRAGBaseline,
    citations_pr,
    exact_match_contains,
    parse_citations,
    run_nonrag_pipeline,
    token_f1,
)


def evaluate_one(
    pred_answer: str,
    gold_answer: str,
    pred_citations: List[str],
    gold_citations: List[str],
) -> Dict[str, float]:
    em = exact_match_contains(pred_answer, gold_answer)
    f1 = token_f1(pred_answer, gold_answer)
    cp, cr = citations_pr(pred_citations, gold_citations)

    return {
        "em": em,
        "f1": f1,
        "cite_precision": cp,
        "cite_recall": cr,
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = [
        "em",
        "f1",
        "cite_precision",
        "cite_recall",
        "wall_ms",
        "prompt_tokens",
        "output_tokens",
    ]
    out: Dict[str, float] = {}
    for key in keys:
        values = [float(r[key]) for r in rows]
        out[key] = float(statistics.mean(values)) if values else 0.0
    return out


def plot_component_breakdown(component_rows: Dict[str, Dict[str, float]], out_png: str) -> None:
    names = list(component_rows.keys())
    latencies = [component_rows[name]["wall_ms"] for name in names]

    plt.figure(figsize=(10, 5))
    plt.bar(names, latencies)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Latência (ms)")
    plt.title("Latência por componente - pipeline sem RAG")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def main() -> None:
    os.makedirs("results", exist_ok=True)

    docs, qa = build_demo_docs_and_qa()
    store = DocStore(docs)

    client = OllamaLocalClient(
        llm_model=os.environ.get("OLLAMA_LLM_MODEL", "phi4"),
        embed_model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        temperature=float(os.environ.get("OLLAMA_TEMPERATURE", "0.0")),
    )

    baseline_context = ContextStuffingBaseline(client, store)
    baseline_rag = SimpleRAGBaseline(client, store, top_k=4)

    all_results: List[Dict[str, Any]] = []

    for item in qa:
        qid = item["id"]
        question = item["question"]
        gold_answer = item["answer"]
        gold_citations = item["gold_citations"]

        # --------------------------------------------------------
        # Proposed pipeline
        # --------------------------------------------------------
        proposed = run_nonrag_pipeline(client, store, question)
        proposed_totals = proposed["trace"].totals()
        proposed_metrics = evaluate_one(
            pred_answer=proposed["answer"],
            gold_answer=gold_answer,
            pred_citations=proposed["citations"],
            gold_citations=gold_citations,
        )
        all_results.append(
            {
                "pipeline": "nonrag_recursive_pipeline",
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "gold_citations": gold_citations,
                "answer": proposed["answer"],
                "citations": proposed["citations"],
                **proposed_metrics,
                **proposed_totals,
            }
        )

        # --------------------------------------------------------
        # Baseline context stuffing
        # --------------------------------------------------------
        ctx_answer, ctx_stats = baseline_context.answer(question)
        ctx_citations = parse_citations(ctx_answer)
        ctx_metrics = evaluate_one(
            pred_answer=ctx_answer,
            gold_answer=gold_answer,
            pred_citations=ctx_citations,
            gold_citations=gold_citations,
        )
        all_results.append(
            {
                "pipeline": "baseline_context_stuffing",
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "gold_citations": gold_citations,
                "answer": ctx_answer,
                "citations": ctx_citations,
                **ctx_metrics,
                "wall_ms": ctx_stats.wall_ms,
                "prompt_tokens": ctx_stats.prompt_tokens,
                "output_tokens": ctx_stats.output_tokens,
            }
        )

        # --------------------------------------------------------
        # Baseline simple RAG
        # --------------------------------------------------------
        rag_answer, rag_stats = baseline_rag.answer(question)
        rag_citations = parse_citations(rag_answer)
        rag_metrics = evaluate_one(
            pred_answer=rag_answer,
            gold_answer=gold_answer,
            pred_citations=rag_citations,
            gold_citations=gold_citations,
        )
        all_results.append(
            {
                "pipeline": "baseline_simple_rag",
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "gold_citations": gold_citations,
                "answer": rag_answer,
                "citations": rag_citations,
                **rag_metrics,
                "wall_ms": rag_stats.wall_ms,
                "prompt_tokens": rag_stats.prompt_tokens,
                "output_tokens": rag_stats.output_tokens,
            }
        )

        print(f"\n=== {qid} ===")
        print("Pergunta:", question)
        print("Gold:", gold_answer, gold_citations)
        print("Proposed:", proposed_metrics, proposed_totals)
        print("Context stuffing:", ctx_metrics)
        print("Simple RAG:", rag_metrics)

    by_pipeline: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_results:
        by_pipeline.setdefault(row["pipeline"], []).append(row)

    summary = {pipeline: aggregate(rows) for pipeline, rows in by_pipeline.items()}

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open("results/results.jsonl", "w", encoding="utf-8") as f:
        for row in all_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open("results/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # roda mais uma vez só para extrair breakdown do último exemplo
    breakdown_run = run_nonrag_pipeline(client, store, qa[-1]["question"])
    plot_component_breakdown(
        breakdown_run["trace"].by_component(),
        "results/latency_by_component.png",
    )

    print("\nArquivos gerados:")
    print("- results/results.jsonl")
    print("- results/summary.json")
    print("- results/latency_by_component.png")


if __name__ == "__main__":
    main()