"""
Process Gen_AI Dataset.xlsx to create training data and assessment catalog
"""

import pandas as pd
import json
import logging
from typing import List, Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_excel_data(filepath: str = "Gen_AI Dataset.xlsx") -> pd.DataFrame:
    """Load data from Excel file"""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_excel(filepath)
    logger.info(f"Loaded {len(df)} rows from Excel")
    return df


def extract_assessments_from_data(df: pd.DataFrame) -> List[Dict]:
    """Extract unique assessments from the dataset"""
    assessments = []

    # Assuming the Excel has columns with assessment names or URLs
    # This is a generic extraction - adjust based on actual Excel structure

    assessment_columns = [col for col in df.columns if 'assessment' in col.lower() or 'test' in col.lower()]

    if assessment_columns:
        # If there are specific assessment columns
        all_assessments = set()
        for col in assessment_columns:
            values = df[col].dropna().unique()
            all_assessments.update(values)

        for assessment_name in all_assessments:
            assessments.append({
                "name": str(assessment_name),
                "url": f"https://www.shl.com/solutions/products/{assessment_name.lower().replace(' ', '-')}/",
                "description": f"Assessment for {assessment_name}",
                "test_type": "K",  # Assuming knowledge test
                "category": "General"
            })
    else:
        # Fallback: extract from text fields
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        all_text = ' '.join(df[text_columns].fillna('').astype(str).sum())

        # Simple extraction of potential assessment names
        # This would need to be customized based on actual data
        potential_assessments = [
            "Java Programming Test", "Python Coding Assessment", "JavaScript Development Test",
            "SQL Database Assessment", "Excel Data Analysis", "Coding Challenge",
            "Verify Numerical Reasoning", "Verify Verbal Reasoning", "Verify Interactive",
            "OPQ32", "Management Judgement", "Situational Judgement Test",
            "Customer Service Assessment", "Leadership Assessment", "Creativity Assessment"
        ]

        for assessment in potential_assessments:
            if assessment.lower() in all_text.lower():
                assessments.append({
                    "name": assessment,
                    "url": f"https://www.shl.com/solutions/products/{assessment.lower().replace(' ', '-')}/",
                    "description": f"Assessment for {assessment}",
                    "test_type": "K",
                    "category": "General"
                })

    logger.info(f"Extracted {len(assessments)} unique assessments")
    return assessments


def create_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create training data with query-assessment pairs"""
    train_data = []

    # Assuming the Excel has a 'Query' column and assessment columns
    query_col = None
    for col in df.columns:
        if 'query' in col.lower() or 'description' in col.lower() or 'job' in col.lower():
            query_col = col
            break

    if query_col is None:
        logger.warning("No query column found, using first text column")
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        query_col = text_cols[0] if text_cols else df.columns[0]

    assessment_cols = [col for col in df.columns if col != query_col and ('assessment' in col.lower() or 'test' in col.lower())]

    for idx, row in df.iterrows():
        query = str(row[query_col]).strip()
        if not query or query.lower() == 'nan':
            continue

        # For each assessment column, create labeled pairs
        for col in assessment_cols:
            assessment = str(row[col]).strip()
            if assessment and assessment.lower() != 'nan':
                train_data.append({
                    'Query': query,
                    'Assessment': assessment,
                    'Label': 1  # Assuming positive label
                })

        # If no specific assessment columns, create pairs with all assessments
        if not assessment_cols:
            # This would need customization based on actual data structure
            pass

    train_df = pd.DataFrame(train_data)
    logger.info(f"Created {len(train_df)} training pairs")
    return train_df


def create_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create test data (unlabeled queries)"""
    # Use a subset of queries for testing
    query_col = None
    for col in df.columns:
        if 'query' in col.lower() or 'description' in col.lower() or 'job' in col.lower():
            query_col = col
            break

    if query_col is None:
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        query_col = text_cols[0] if text_cols else df.columns[0]

    test_queries = df[query_col].dropna().unique()[:10]  # First 10 unique queries

    test_df = pd.DataFrame({'Query': test_queries})
    logger.info(f"Created {len(test_df)} test queries")
    return test_df


def main():
    """Main processing function"""
    try:
        # Load Excel data
        df = load_excel_data()

        # Create assessments catalog
        assessments = extract_assessments_from_data(df)

        # Save assessments to JSON
        with open('data/raw_assessments.json', 'w', encoding='utf-8') as f:
            json.dump(assessments, f, indent=2, ensure_ascii=False)
        logger.info("✓ Saved raw_assessments.json")

        # Create training data
        train_df = create_training_data(df)
        train_df.to_csv('data/train_data.csv', index=False)
        logger.info("✓ Saved train_data.csv")

        # Create test data
        test_df = create_test_data(df)
        test_df.to_csv('data/test_data.csv', index=False)
        logger.info("✓ Saved test_data.csv")

        print("\n" + "="*60)
        print("DATA PROCESSING COMPLETE")
        print("="*60)
        print(f"Files created in data/ folder:")
        print(f"  - raw_assessments.json ({len(assessments)} assessments)")
        print(f"  - train_data.csv ({len(train_df)} training pairs)")
        print(f"  - test_data.csv ({len(test_df)} test queries)")
        print("="*60)

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


if __name__ == "__main__":
    main()