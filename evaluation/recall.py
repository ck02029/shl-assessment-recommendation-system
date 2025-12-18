"""
Simple evaluation helpers for Recall@K / Precision@K.
"""

from typing import List, Dict


class RecommendationMetrics:
    @staticmethod
    def recall_at_k(predicted_urls: List[str], relevant_urls: List[str], k: int = 10) -> float:
        if not relevant_urls:
            return 0.0
        hits = len(set(predicted_urls[:k]) & set(relevant_urls))
        return hits / len(relevant_urls)

    @staticmethod
    def precision_at_k(predicted_urls: List[str], relevant_urls: List[str], k: int = 10) -> float:
        denom = min(k, len(predicted_urls))
        if denom == 0:
            return 0.0
        hits = len(set(predicted_urls[:k]) & set(relevant_urls))
        return hits / denom

    @staticmethod
    def calculate_all_metrics(results: List[Dict], k: int = 10) -> Dict[str, float]:
        """Aggregate metrics over all queries."""
        recalls = [r["recall"] for r in results]
        precisions = [r["precision"] for r in results]
        mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
        mean_precision = sum(precisions) / len(precisions) if precisions else 0.0
        return {
            f"mean_recall@{k}": mean_recall,
            f"mean_precision@{k}": mean_precision,
        }

    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        print("\n" + "=" * 80)
        print("AGGREGATE METRICS")
        print("=" * 80)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("=" * 80)
