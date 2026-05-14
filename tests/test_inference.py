"""Test script for RAG inference evaluation.

This script evaluates the query_documents function by:
1. Picking a random document from the data directory
2. Loading its corresponding QA file
3. Running query_documents for each question
4. Computing multiple similarity metrics between LLM answer and ground truth
5. Reporting accuracy metrics with threshold interpretations
"""

import json
import random
import sys
from pathlib import Path

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import sacrebleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from nexla_mcp.mcp import query_documents
from nexla_mcp.indexer import encode_texts


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = sum(a * a for a in vec1) ** 0.5
    mag2 = sum(b * b for b in vec2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


def compute_bleu_score(reference: str, hypothesis: str) -> float:
    """Compute BLEU score using sacrebleu."""
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
    return bleu.score / 100.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L score (longest common subsequence recall)."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def compute_bertscore(reference: str, hypothesis: str) -> float:
    """Compute BERTScore F1 using sentence-transformers."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([reference, hypothesis], convert_to_tensor=True)
    cos_sim = torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    )
    return cos_sim.item()


def load_qa_file(doc_id: str) -> list[dict]:
    """Load QA file for a given document ID."""
    qa_path = Path(__file__).parent.parent / "data" / doc_id / f"{doc_id}_qa.jsonl"
    qa_data = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_data.append(json.loads(line))
    return qa_data


def get_available_doc_ids() -> list[str]:
    """Get list of available document IDs from data directory."""
    data_dir = Path(__file__).parent.parent / "data"
    doc_ids = []
    for d in data_dir.iterdir():
        if d.is_dir() and (d / f"{d.name}_qa.jsonl").exists():
            doc_ids.append(d.name)
    return doc_ids


def evaluate_answer(expected: str, actual: str) -> dict:
    """Evaluate similarity between expected and actual answers using multiple metrics."""
    # Cosine similarity from embeddings
    embeddings = encode_texts([expected, actual], prompt_name="query")
    cos_sim = cosine_similarity(embeddings[0], embeddings[1])

    # BLEU score
    bleu = compute_bleu_score(expected, actual)

    # ROUGE-L score
    rouge_l = compute_rouge_l(expected, actual)

    # BERTScore F1
    bertscore = compute_bertscore(expected, actual)

    return {
        "cosine_similarity": cos_sim,
        "bleu_score": bleu,
        "rouge_l_score": rouge_l,
        "bertscore": bertscore,
    }


def run_evaluation(doc_id: str, num_questions: int = 10) -> dict:
    """Run evaluation on a specific document."""
    qa_data = load_qa_file(doc_id)

    # Limit to num_questions
    questions_to_eval = qa_data[:num_questions]

    results = []
    for qa in questions_to_eval:
        question = qa["question"]
        expected_answer = qa["answer"]

        # Get LLM response
        response = query_documents(question, top_k=5)
        actual_answer = response["answer"]

        # Compute all metrics
        metrics = evaluate_answer(expected_answer, actual_answer)

        results.append(
            {
                "question": question,
                "expected": expected_answer,
                "actual": actual_answer,
                **metrics,
            }
        )

    return {
        "doc_id": doc_id,
        "total_questions": len(results),
        "results": results,
    }


def interpret_threshold(metric: str, value: float) -> str:
    """Interpret a metric value based on thresholds."""
    thresholds = {
        "cosine_similarity": (0.60, 0.80),
        "bleu_score": (0.30, 0.50),
        "rouge_l_score": (0.30, 0.50),
        "bertscore": (0.70, 0.85),
    }
    low, high = thresholds[metric]
    if value >= high:
        return "Good"
    elif value >= low:
        return "Acceptable"
    else:
        return "Needs Work"


def print_results(eval_result: dict) -> None:
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 80)
    print(f"Document ID: {eval_result['doc_id']}")
    print(f"Questions evaluated: {eval_result['total_questions']}")
    print("=" * 80)

    metrics_list = ["cosine_similarity", "bleu_score", "rouge_l_score", "bertscore"]
    all_metrics = {m: [] for m in metrics_list}

    for i, r in enumerate(eval_result["results"], 1):
        print(f"\n[Question {i}]")
        print(f"Q: {r['question'][:100]}{'...' if len(r['question']) > 100 else ''}")
        print(
            f"Expected: {r['expected'][:150]}{'...' if len(r['expected']) > 150 else ''}"
        )
        print(f"Actual:   {r['actual'][:150]}{'...' if len(r['actual']) > 150 else ''}")
        print(f"Cosine Similarity: {r['cosine_similarity']:.4f}")
        print(f"BLEU Score:       {r['bleu_score']:.4f}")
        print(f"ROUGE-L Score:    {r['rouge_l_score']:.4f}")
        print(f"BERTScore F1:     {r['bertscore']:.4f}")

        for m in metrics_list:
            all_metrics[m].append(r[m])

    print("\n" + "=" * 80)
    print("SUMMARY (Average / Min / Max)")
    print("=" * 80)
    for m in metrics_list:
        values = all_metrics[m]
        avg = sum(values) / len(values) if values else 0
        min_val = min(values) if values else 0
        max_val = max(values) if values else 0
        interpretation = interpret_threshold(m, avg)
        metric_display = m.replace("_", " ").title()
        print(f"{metric_display}:")
        print(
            f"  Avg: {avg:.4f} | Min: {min_val:.4f} | Max: {max_val:.4f} [{interpretation}]"
        )
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG inference")
    parser.add_argument(
        "--doc-id", type=str, default=None, help="Specific document ID to evaluate"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of questions to evaluate (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Get available document IDs
    doc_ids = get_available_doc_ids()
    print(f"Found {len(doc_ids)} documents with QA files")

    if not doc_ids:
        print("No documents found with QA files!")
        return

    # Pick document
    if args.doc_id:
        if args.doc_id not in doc_ids:
            print(f"Document {args.doc_id} not found or has no QA file")
            return
        doc_id = args.doc_id
    else:
        doc_id = random.choice(doc_ids)

    print(f"\nEvaluating document: {doc_id}")
    print(f"Number of questions: {args.n}")

    # Run evaluation
    result = run_evaluation(doc_id, num_questions=args.n)

    # Print results
    print_results(result)


if __name__ == "__main__":
    main()
