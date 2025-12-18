"""
Data Cleaning and Validation Script
Cleans scraped assessment data and validates train/test datasets
"""

import json
import pandas as pd
import os
from typing import Dict, List
import logging
import re
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        
    def clean_assessments(self, input_file: str = "raw_assessments.json") -> pd.DataFrame:
        """
        Clean raw assessment data
        
        Steps:
        1. Load JSON
        2. Remove duplicates
        3. Validate URLs
        4. Clean text fields
        5. Standardize types
        """
        logger.info(f"Loading assessments from {input_file}")
        
        input_path = os.path.join(self.data_dir, input_file)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            assessments = json.load(f)
        
        logger.info(f"Loaded {len(assessments)} assessments")
        
        # Convert to DataFrame
        df = pd.DataFrame(assessments)
        
        logger.info("Cleaning data...")
        
        # 1. Remove duplicates by URL
        initial_count = len(df)
        df = df.drop_duplicates(subset=['url'], keep='first')
        logger.info(f"Removed {initial_count - len(df)} duplicate URLs")
        
        # 2. Validate URLs
        df = df[df['url'].apply(self._is_valid_url)]
        logger.info(f"Retained {len(df)} assessments with valid URLs")
        
        # 3. Clean text fields
        df['name'] = df['name'].apply(self._clean_text)
        df['description'] = df['description'].apply(self._clean_text)
        
        # 4. Standardize test types
        df['test_type'] = df['test_type'].apply(self._standardize_type)
        
        # 5. Fill missing values
        df['description'] = df['description'].fillna('')
        df['category'] = df['category'].fillna('General')
        
        # 6. Sort by name
        df = df.sort_values('name').reset_index(drop=True)
        
        logger.info(f"Cleaning complete: {len(df)} clean assessments")
        
        return df
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean text field"""
        if pd.isna(text):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', str(text))
        text = text.strip()
        
        # Remove special characters (but keep basic punctuation)
        # text = re.sub(r'[^\w\s\-\.,!?()]', '', text)
        
        return text
    
    def _standardize_type(self, test_type: str) -> str:
        """Standardize test type to K, P, C, or General"""
        if pd.isna(test_type):
            return 'General'
        
        test_type = str(test_type).upper().strip()
        
        if test_type in ['K', 'P', 'C']:
            return test_type
        
        return 'General'
    
    def validate_train_data(self, filepath: str = "train_data.csv") -> bool:
        """Validate training data format"""
        logger.info(f"Validating training data: {filepath}")
        
        try:
            df = pd.read_csv(os.path.join(self.data_dir, filepath))
            
            # Check required columns
            required_cols = ['Query', 'Assessment_url']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing columns: {missing_cols}")
                return False
            
            # Check for null values
            null_queries = df['Query'].isna().sum()
            null_urls = df['Assessment_url'].isna().sum()
            
            if null_queries > 0:
                logger.warning(f"Found {null_queries} null queries")
            if null_urls > 0:
                logger.warning(f"Found {null_urls} null URLs")
            
            # Statistics
            unique_queries = df['Query'].nunique()
            total_rows = len(df)
            
            logger.info(f"✓ Training data valid")
            logger.info(f"  - Unique queries: {unique_queries}")
            logger.info(f"  - Total assessments: {total_rows}")
            logger.info(f"  - Avg assessments per query: {total_rows/unique_queries:.1f}")
            
            # Print sample
            print("\nSample training data:")
            print(df.head(3))
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating training data: {str(e)}")
            return False
    
    def validate_test_data(self, filepath: str = "test_data.csv") -> bool:
        """Validate test data format"""
        logger.info(f"Validating test data: {filepath}")
        
        try:
            df = pd.read_csv(os.path.join(self.data_dir, filepath))
            
            # Check required column
            if 'Query' not in df.columns:
                logger.error("Missing 'Query' column")
                return False
            
            # Check for null values
            null_queries = df['Query'].isna().sum()
            if null_queries > 0:
                logger.warning(f"Found {null_queries} null queries")
            
            # Statistics
            total_queries = len(df)
            
            logger.info(f"✓ Test data valid")
            logger.info(f"  - Total queries: {total_queries}")
            
            # Print sample
            print("\nSample test data:")
            print(df.head(3))
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating test data: {str(e)}")
            return False
    
    def validate_assessment_count(self, df: pd.DataFrame) -> bool:
        """Validate that we have at least 377 assessments"""
        count = len(df)
        
        if count >= 377:
            logger.info(f"✓ Assessment count valid: {count} >= 377")
            return True
        else:
            logger.error(f"✗ Insufficient assessments: {count} < 377")
            return False
    
    def save_clean_data(self, df: pd.DataFrame, output_file: str = "clean_assessments.csv"):
        """Save cleaned data to CSV"""
        output_path = os.path.join(self.data_dir, output_file)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"✓ Saved clean data to {output_path}")
    
    def generate_statistics(self, df: pd.DataFrame):
        """Generate and print data statistics"""
        print("\n" + "="*60)
        print("ASSESSMENT DATA STATISTICS")
        print("="*60)
        
        print(f"\nTotal Assessments: {len(df)}")
        print(f"\nTest Type Distribution:")
        print(df['test_type'].value_counts())
        print(f"\nCategory Distribution:")
        print(df['category'].value_counts().head(10))
        
        print(f"\nSample Assessments:")
        print("-"*60)
        for _, row in df.head(5).iterrows():
            print(f"- {row['name']} ({row['test_type']})")
        
        print("="*60)


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean and validate assessment data')
    parser.add_argument(
        '--data-dir',
        default='.',
        help='Data directory path'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Only validate data without cleaning'
    )
    
    args = parser.parse_args()
    
    cleaner = DataCleaner(data_dir=args.data_dir)
    
    print("\n" + "="*60)
    print("DATA CLEANING AND VALIDATION")
    print("="*60)
    
    if args.validate:
        # Validation mode
        print("\n1. Validating Training Data")
        print("-"*60)
        cleaner.validate_train_data()
        
        print("\n2. Validating Test Data")
        print("-"*60)
        cleaner.validate_test_data()
        
    else:
        # Cleaning mode
        print("\n1. Cleaning Assessment Data")
        print("-"*60)
        
        if not os.path.exists(os.path.join(args.data_dir, 'raw_assessments.json')):
            logger.error("raw_assessments.json not found. Run scraper first:")
            logger.error("  python scraper/scrape_shl.py")
            return
        
        # Clean assessments
        df = cleaner.clean_assessments()
        
        # Validate count
        cleaner.validate_assessment_count(df)
        
        # Save
        cleaner.save_clean_data(df)
        
        # Generate statistics
        cleaner.generate_statistics(df)
        
        print("\n2. Validating Training Data")
        print("-"*60)
        if os.path.exists(os.path.join(args.data_dir, 'train_data.csv')):
            cleaner.validate_train_data()
        else:
            logger.warning("train_data.csv not found")
        
        print("\n3. Validating Test Data")
        print("-"*60)
        if os.path.exists(os.path.join(args.data_dir, 'test_data.csv')):
            cleaner.validate_test_data()
        else:
            logger.warning("test_data.csv not found")
    
    print("\n✓ Data validation complete!")


if __name__ == "__main__":
    main()