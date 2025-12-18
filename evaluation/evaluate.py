"""
Evaluation Script for SHL Recommendation System
Evaluates system performance on train/test data
"""

import pandas as pd
import sys
import os
from typing import List, Dict, Tuple
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.retriever import AssessmentRetriever
from retrieval.reranker import AssessmentReranker
from llm.query_parser import QueryParser
from evaluation.recall import RecommendationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemEvaluator:
    def __init__(self):
        """Initialize evaluator with all components"""
        logger.info("Initializing evaluation components...")
        
        self.retriever = AssessmentRetriever()
        self.reranker = AssessmentReranker()
        self.query_parser = QueryParser()
        
        logger.info("Evaluator initialized successfully")
    
    def load_labeled_data(self, filepath: str) -> pd.DataFrame:
        """
        Load labeled train data
        
        Expected format:
        - Query column with search queries
        - Assessment_url column with relevant URLs (one per row)
        """
        logger.info(f"Loading labeled data from {filepath}")
        
        try:
            # Try reading as CSV
            df = pd.read_csv(filepath)
        except:
            # Try reading as Excel
            df = pd.read_excel(filepath)
        
        logger.info(f"Loaded {len(df)} rows")
        return df
    
    def prepare_evaluation_data(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert dataframe to evaluation format
        Groups URLs by query
        
        Returns:
            List of dicts with 'query' and 'relevant_urls'
        """
        # Normalize column names we accept
        cols = {c.lower(): c for c in df.columns}
        if 'query' not in cols:
            raise KeyError(f"Expected a 'Query' column. Found: {list(df.columns)}")

        # Accept either 'Assessment_url' or a column named 'Assessment' as the label column
        label_col = None
        if 'assessment_url' in cols:
            label_col = cols['assessment_url']
        elif 'assessment' in cols:
            label_col = cols['assessment']
        else:
            raise KeyError(
                "Expected a label column named 'Assessment_url' (preferred) or 'Assessment'. "
                f"Found: {list(df.columns)}"
            )

        # Group by query
        grouped = df.groupby(cols['query'])[label_col].apply(list).reset_index()
        
        eval_data = []
        for _, row in grouped.iterrows():
            eval_data.append({
                'query': row[cols['query']],
                'relevant_urls': row[label_col]
            })
        
        logger.info(f"Prepared {len(eval_data)} unique queries for evaluation")
        return eval_data
    
    def predict(self, query: str, k: int = 10, return_stages: bool = False) -> List[str]:
        """
        Generate predictions for a single query
        
        Args:
            query: Search query
            k: Number of recommendations
            return_stages: If True, return both retrieval and final stages
        
        Returns:
            List of predicted assessment URLs (or dict with stages if return_stages=True)
        """
        # Parse query
        query_analysis = self.query_parser.parse_query(query)
        
        # Retrieve candidates (RETRIEVAL STAGE)
        candidates = self.retriever.retrieve(
            query,
            k=min(30, k * 3),  # Get more candidates for reranking
            query_analysis=query_analysis
        )
        
        # Extract retrieval URLs (before reranking)
        retrieval_urls = [c['url'] for c in candidates[:k]]
        
        # Rerank (FINAL RECOMMENDATION STAGE)
        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=k,
            query_analysis=query_analysis
        )
        
        # Extract final URLs
        final_urls = [result['url'] for result in reranked]
        
        if return_stages:
            return {
                'retrieval': retrieval_urls,
                'final': final_urls
            }
        
        return final_urls
    
    def evaluate_dataset(
        self,
        eval_data: List[Dict],
        k: int = 10
    ) -> Tuple[Dict, List[Dict]]:
        """
        Evaluate system on a dataset
        
        Returns:
            (metrics_dict, detailed_results)
        """
        logger.info(f"Evaluating on {len(eval_data)} queries...")
        
        results = []
        
        for i, item in enumerate(eval_data, 1):
            query = item['query']
            relevant_urls = item['relevant_urls']
            
            logger.info(f"[{i}/{len(eval_data)}] Evaluating: {query}")
            
            # Get predictions with both stages
            predictions = self.predict(query, k=k, return_stages=True)
            retrieval_urls = predictions['retrieval']
            final_urls = predictions['final']
            
            # Calculate RETRIEVAL STAGE metrics
            retrieval_recall = RecommendationMetrics.recall_at_k(
                retrieval_urls,
                relevant_urls,
                k=k
            )
            retrieval_precision = RecommendationMetrics.precision_at_k(
                retrieval_urls,
                relevant_urls,
                k=k
            )
            
            # Calculate FINAL RECOMMENDATION STAGE metrics
            final_recall = RecommendationMetrics.recall_at_k(
                final_urls,
                relevant_urls,
                k=k
            )
            final_precision = RecommendationMetrics.precision_at_k(
                final_urls,
                relevant_urls,
                k=k
            )
            
            result = {
                'query': query,
                'retrieval_predicted': retrieval_urls,
                'final_predicted': final_urls,
                'relevant': relevant_urls,
                # Retrieval stage metrics
                'retrieval_recall': retrieval_recall,
                'retrieval_precision': retrieval_precision,
                'retrieval_hits': len(set(retrieval_urls) & set(relevant_urls)),
                # Final recommendation stage metrics
                'final_recall': final_recall,
                'final_precision': final_precision,
                'final_hits': len(set(final_urls) & set(relevant_urls)),
                # Legacy fields for backward compatibility
                'predicted': final_urls,
                'recall': final_recall,
                'precision': final_precision,
                'num_relevant': len(relevant_urls),
                'num_predicted': len(final_urls),
                'hits': len(set(final_urls) & set(relevant_urls))
            }
            
            results.append(result)
            
            logger.info(f"  RETRIEVAL:   Recall@{k}: {retrieval_recall:.4f}, Precision@{k}: {retrieval_precision:.4f}")
            logger.info(f"  FINAL:       Recall@{k}: {final_recall:.4f}, Precision@{k}: {final_precision:.4f}")
        
        # Calculate aggregate metrics for both stages
        # Retrieval stage
        retrieval_recalls = [r["retrieval_recall"] for r in results]
        retrieval_precisions = [r["retrieval_precision"] for r in results]
        mean_retrieval_recall = sum(retrieval_recalls) / len(retrieval_recalls) if retrieval_recalls else 0.0
        mean_retrieval_precision = sum(retrieval_precisions) / len(retrieval_precisions) if retrieval_precisions else 0.0
        
        # Final recommendation stage
        final_recalls = [r["final_recall"] for r in results]
        final_precisions = [r["final_precision"] for r in results]
        mean_final_recall = sum(final_recalls) / len(final_recalls) if final_recalls else 0.0
        mean_final_precision = sum(final_precisions) / len(final_precisions) if final_precisions else 0.0
        
        metrics = {
            # Retrieval stage metrics
            f"retrieval_mean_recall@{k}": mean_retrieval_recall,
            f"retrieval_mean_precision@{k}": mean_retrieval_precision,
            # Final recommendation stage metrics
            f"final_mean_recall@{k}": mean_final_recall,
            f"final_mean_precision@{k}": mean_final_precision,
            # Legacy fields for backward compatibility
            f"mean_recall@{k}": mean_final_recall,
            f"mean_precision@{k}": mean_final_precision,
        }
        
        return metrics, results
    
    def print_detailed_results(self, results: List[Dict]):
        """Print detailed per-query results"""
        print("\n" + "="*80)
        print("DETAILED RESULTS PER QUERY")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\nQuery {i}: {result['query']}")
            print("-"*80)
            print(f"Relevant assessments: {result['num_relevant']}")
            
            # Retrieval stage
            print(f"\n[RETRIEVAL STAGE]")
            print(f"  Hits in top {len(result['retrieval_predicted'])}: {result['retrieval_hits']}")
            print(f"  Recall@{len(result['retrieval_predicted'])}: {result['retrieval_recall']:.4f}")
            print(f"  Precision@{len(result['retrieval_predicted'])}: {result['retrieval_precision']:.4f}")
            
            # Final recommendation stage
            print(f"\n[FINAL RECOMMENDATION STAGE]")
            print(f"  Hits in top {len(result['final_predicted'])}: {result['final_hits']}")
            print(f"  Recall@{len(result['final_predicted'])}: {result['final_recall']:.4f}")
            print(f"  Precision@{len(result['final_predicted'])}: {result['final_precision']:.4f}")
            
            # Show which relevant items were found
            relevant_set = set(result['relevant'])
            predicted_set = set(result['predicted'])
            
            found = relevant_set & predicted_set
            missed = relevant_set - predicted_set
            
            if found:
                print(f"\n✓ Found relevant assessments ({len(found)}):")
                for url in found:
                    print(f"  - {url}")
            
            if missed:
                print(f"\n✗ Missed relevant assessments ({len(missed)}):")
                for url in missed:
                    print(f"  - {url}")
        
        print("\n" + "="*80)
    
    def save_predictions(
        self,
        results: List[Dict],
        output_file: str = "predictions.csv"
    ):
        """Save predictions to CSV in required format"""
        rows = []
        
        for result in results:
            query = result['query']
            for url in result['predicted']:
                rows.append({
                    'Query': query,
                    'Assessment_url': url
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Predictions saved to {output_file}")


def main():
    """Run evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate SHL Recommendation System')
    parser.add_argument(
        '--train-data',
        default='data/train_data.csv',
        help='Path to labeled train data'
    )
    parser.add_argument(
        '--test-data',
        default='data/test_data.csv',
        help='Path to unlabeled test data'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of recommendations (K for Recall@K)'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'test', 'both'],
        default='train',
        help='Evaluation mode'
    )
    parser.add_argument(
        '--output',
        default='predictions.csv',
        help='Output file for test predictions'
    )
    
    args = parser.parse_args()
    
    evaluator = SystemEvaluator()
    
    # Evaluate on train data
    if args.mode in ['train', 'both']:
        print("\n" + "="*80)
        print("EVALUATING ON TRAIN DATA")
        print("="*80)
        
        train_df = evaluator.load_labeled_data(args.train_data)
        train_data = evaluator.prepare_evaluation_data(train_df)
        
        metrics, results = evaluator.evaluate_dataset(train_data, k=args.k)
        
        RecommendationMetrics.print_metrics(metrics)
        evaluator.print_detailed_results(results)
    
    # Generate predictions on test data
    if args.mode in ['test', 'both']:
        print("\n" + "="*80)
        print("GENERATING PREDICTIONS ON TEST DATA")
        print("="*80)
        
        test_df = evaluator.load_labeled_data(args.test_data)
        
        # Test data has no labels, so we just get unique queries
        test_queries = test_df['Query'].unique()
        
        test_results = []
        for query in test_queries:
            predicted_urls = evaluator.predict(query, k=args.k)
            test_results.append({
                'query': query,
                'predicted': predicted_urls,
                'relevant': []  # No labels for test data
            })
        
        evaluator.save_predictions(test_results, args.output)
        
        print(f"\n✓ Generated predictions for {len(test_queries)} test queries")
        print(f"✓ Saved to {args.output}")


if __name__ == "__main__":
    main()