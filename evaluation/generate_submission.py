"""
Generate submission CSV in Appendix 3 format:

Columns:
    - Query
    - Assessment_url

Each test query is repeated once per recommended assessment URL.
"""

import csv
import logging
from typing import List, Dict, Any

import pandas as pd

from retrieval.retriever import AssessmentRetriever
from retrieval.reranker import AssessmentReranker
from llm.query_parser import QueryParser


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_recommendations_for_query(
    query: str,
    qp: QueryParser,
    retriever: AssessmentRetriever,
    reranker: AssessmentReranker,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Run full pipeline (LLM parse -> retrieve -> rerank) for a single query."""
    analysis = qp.parse_query(query)
    candidates = retriever.retrieve(query, k=50, query_analysis=analysis)
    reranked = reranker.rerank(query, candidates, top_k=top_k, query_analysis=analysis)
    return reranked


def main():
    # Unlabeled test set from SHL; expect a column named "Query"
    test_df = pd.read_csv("data/test_data.csv")
    if "Query" not in test_df.columns:
        raise ValueError("Expected a 'Query' column in data/test_data.csv")

    retriever = AssessmentRetriever()
    reranker = AssessmentReranker()
    qp = QueryParser()

    output_path = "submission.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # EXACT headers required by Appendix 3
        writer.writerow(["Query", "Assessment_url"])

        for _, row in test_df.iterrows():
            query_text = str(row["Query"])
            if not query_text or query_text.strip() == "":
                continue

            logger.info("Generating recommendations for query: %s", query_text)
            recs = get_recommendations_for_query(
                query_text, qp, retriever, reranker, top_k=10
            )

            # One row per recommendation (same Query text, different URLs)
            for r in recs:
                url = r.get("url") or r.get("assessment_url") or ""
                if not url:
                    continue
                writer.writerow([query_text, url])

    logger.info("Saved submission CSV to %s", output_path)


if __name__ == "__main__":
    main()


