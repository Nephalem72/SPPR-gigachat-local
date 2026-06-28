from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeatable SPPR RAG+LLM API benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=Path("eval/queries.jsonl"))
    parser.add_argument("--profile", choices=("fast", "balanced", "broad"), default="balanced")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    config = requests.get(f"{args.url}/config", timeout=30).json()
    results: list[dict[str, Any]] = []
    for item in load_queries(args.dataset):
        payload = {
            "text": item["text"],
            "question": item["question"],
            "rag_profile": args.profile,
            "return_context": False,
        }
        response = requests.post(f"{args.url}/chat", json=payload, timeout=900)
        response.raise_for_status()
        body = response.json()
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": body["answer"],
                "metrics": body["metrics"],
                "citation_check": body["citation_check"],
                "sources": body["sources"],
            }
        )
        print(f"{item['id']}: {body['metrics']['total_seconds']} sec")

    totals = [item["metrics"]["total_seconds"] for item in results]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "rag_profile": args.profile,
        "summary": {
            "queries": len(results),
            "mean_total_seconds": round(mean(totals), 3) if totals else 0.0,
            "answers_with_citations": sum(item["citation_check"]["has_citations"] for item in results),
            "invalid_citations": sum(len(item["citation_check"]["invalid"]) for item in results),
        },
        "results": results,
    }
    output = args.output or Path("eval/results") / (
        f"{config['llm_model_id'].replace('/', '__')}__{args.profile}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
