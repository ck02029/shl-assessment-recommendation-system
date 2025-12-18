"""
SHL Product Catalog Scraper
Scrapes Individual Test Solutions from SHL website
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHLScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com"
        self.catalog_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.assessments = []

    def scrape_catalog(self) -> List[Dict]:
        """Scrape the main catalog page and extract all Individual Test Solutions"""
        try:
            logger.info(f"Fetching catalog from {self.catalog_url}")
            response = requests.get(self.catalog_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all assessment links - try multiple strategies
            assessment_urls = set()
            
            # Strategy 1: Find all links to product pages
            all_links = soup.find_all('a', href=True)
            for link in all_links:
                href = link.get('href', '')
                # Look for product catalog links
                if '/product-catalog/view/' in href or '/solutions/products/' in href:
                    if href.startswith('http'):
                        url = href
                    elif href.startswith('/'):
                        url = self.base_url + href
                    else:
                        url = self.base_url + '/' + href
                    
                    # Only include Individual Test Solutions (exclude pre-packaged)
                    if '/product-catalog/view/' in url or '/solutions/products/' in url:
                        if not any(exclude in url.lower() for exclude in ['bundle', 'package', 'job-solution', 'pre-packaged']):
                            assessment_urls.add(url)
            
            logger.info(f"Found {len(assessment_urls)} unique assessment URLs")
            
            # Now scrape each individual assessment page for details
            for idx, url in enumerate(assessment_urls, 1):
                try:
                    logger.info(f"Scraping {idx}/{len(assessment_urls)}: {url}")
                    assessment = self.scrape_assessment_page(url)
                    if assessment and self.is_individual_test(assessment):
                        self.assessments.append(assessment)
                    time.sleep(0.5)  # Be respectful to the server
                    
                    # Progress update every 50 items
                    if idx % 50 == 0:
                        logger.info(f"Progress: {idx}/{len(assessment_urls)} assessments processed")
                        
                except Exception as e:
                    logger.warning(f"Error scraping {url}: {str(e)}")
                    continue
            
            logger.info(f"Successfully scraped {len(self.assessments)} Individual Test Solutions")
            return self.assessments
            
        except Exception as e:
            logger.error(f"Error scraping catalog: {str(e)}")
            return []

    def extract_assessment_info(self, element) -> Dict:
        """
        Extract assessment information from HTML element (legacy method)
        This is kept for backward compatibility but scrape_assessment_page is preferred
        """
        try:
            # Try to find link
            link = element if element.name == 'a' else element.find('a')
            if not link or not link.get('href'):
                return None
            
            url = link.get('href')
            if not url.startswith('http'):
                url = self.base_url + url if url.startswith('/') else self.base_url + '/' + url
            
            # Extract name
            name = link.get_text(strip=True)
            if not name or len(name) < 3:
                name_elem = element.find(['h2', 'h3', 'h4', 'span'], class_=re.compile(r'title|name|heading', re.I))
                name = name_elem.get_text(strip=True) if name_elem else None
            
            # If still no name, try to extract from URL
            if not name or len(name) < 3:
                url_parts = url.rstrip('/').split('/')
                name = url_parts[-1].replace('-', ' ').title() if url_parts else "Unknown Assessment"
            
            # Try to get description
            desc_elem = element.find(['p', 'div'], class_=re.compile(r'desc|summary|content', re.I))
            description = desc_elem.get_text(strip=True) if desc_elem else ""
            
            # Extract test type (K = Knowledge/Skills, P = Personality, C = Cognitive)
            test_type = self.infer_test_type(name, description)
            
            # Extract category
            category = self.extract_category(element)
            
            return {
                'name': name,
                'url': url,
                'description': description if description else f"Assessment: {name}",
                'test_type': test_type,
                'category': category
            }
            
        except Exception as e:
            logger.warning(f"Error extracting assessment info: {str(e)}")
            return None

    def is_individual_test(self, assessment: Dict) -> bool:
        """Filter out Pre-packaged Job Solutions"""
        name_lower = assessment['name'].lower()
        desc_lower = assessment['description'].lower()
        
        # Exclude pre-packaged solutions
        exclude_keywords = ['pre-packaged', 'job solution', 'bundle', 'package']
        for keyword in exclude_keywords:
            if keyword in name_lower or keyword in desc_lower:
                return False
        
        return True

    def infer_test_type(self, name: str, description: str) -> str:
        """Infer test type from name and description"""
        text = (name + " " + description).lower()
        
        # Personality/Behavior indicators
        if any(word in text for word in ['personality', 'behavior', 'behaviour', 'motivation', 'opq', 'mq', 'workplace']):
            return 'P'
        
        # Cognitive indicators
        if any(word in text for word in ['cognitive', 'reasoning', 'numerical', 'verbal', 'inductive', 'deductive', 'verify']):
            return 'C'
        
        # Knowledge/Skills indicators
        if any(word in text for word in ['knowledge', 'skill', 'technical', 'programming', 'coding', 'java', 'python', 'excel']):
            return 'K'
        
        return 'General'

    def extract_category(self, element) -> str:
        """Extract category from element"""
        try:
            cat_elem = element.find(['span', 'div'], class_=re.compile(r'category|tag|label', re.I))
            return cat_elem.get_text(strip=True) if cat_elem else "General"
        except:
            return "General"

    def scrape_assessment_page(self, url: str) -> Dict:
        """Scrape detailed information from individual assessment page"""
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract name from page title or h1
            name = None
            title_elem = soup.find('title')
            if title_elem:
                name = title_elem.get_text(strip=True)
                # Clean up title (remove common suffixes)
                name = re.sub(r'\s*[-|]\s*SHL.*$', '', name, flags=re.I)
            
            if not name:
                h1 = soup.find('h1')
                if h1:
                    name = h1.get_text(strip=True)
            
            if not name:
                # Try to extract from URL
                url_parts = url.rstrip('/').split('/')
                name = url_parts[-1].replace('-', ' ').title() if url_parts else "Unknown Assessment"
            
            # Extract description
            description = ""
            
            # Try multiple strategies to find description
            desc_selectors = [
                soup.find(['div', 'section'], class_=re.compile(r'description|content|overview|summary', re.I)),
                soup.find('meta', attrs={'name': re.compile(r'description', re.I)}),
                soup.find('div', id=re.compile(r'description|content|overview', re.I)),
                soup.find('p', class_=re.compile(r'description|summary|intro', re.I))
            ]
            
            for desc_elem in desc_selectors:
                if desc_elem:
                    if desc_elem.name == 'meta':
                        description = desc_elem.get('content', '')
                    else:
                        description = desc_elem.get_text(strip=True)
                    if description and len(description) > 20:  # Only use if meaningful
                        break
            
            # If no description found, try to get first few paragraphs
            if not description or len(description) < 20:
                paragraphs = soup.find_all('p')
                desc_parts = []
                for p in paragraphs[:3]:
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:
                        desc_parts.append(text)
                if desc_parts:
                    description = ' '.join(desc_parts[:2])
            
            # Extract test type
            test_type = self.infer_test_type(name, description)
            
            # Extract category
            category = "General"
            category_elem = soup.find(['span', 'div', 'a'], class_=re.compile(r'category|tag|label|breadcrumb', re.I))
            if category_elem:
                category_text = category_elem.get_text(strip=True)
                if category_text and len(category_text) < 50:
                    category = category_text
            
            # Clean URL (remove duplicates)
            clean_url = url
            if clean_url.endswith('/'):
                clean_url = clean_url[:-1]
            
            return {
                'name': name.strip(),
                'url': clean_url,
                'description': description.strip() if description else f"Assessment: {name}",
                'test_type': test_type,
                'category': category
            }
            
        except Exception as e:
            logger.warning(f"Error scraping page {url}: {str(e)}")
            return None

    def save_to_json(self, filename: str = "data/raw_assessments.json"):
        """Save scraped data to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.assessments, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.assessments)} assessments to {filename}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {str(e)}")

    def load_sample_data(self) -> List[Dict]:
        """
        Load sample data for testing (since actual scraping might be blocked)
        REPLACE THIS WITH ACTUAL SCRAPED DATA
        """
        logger.warning("Using sample data. Replace with actual scraped data!")
        
        sample_assessments = [
            {
                "name": "Java Programming Test",
                "url": "https://www.shl.com/solutions/products/java-programming-test/",
                "description": "Assess Java programming skills including OOP concepts, data structures, and algorithms",
                "test_type": "K",
                "category": "Technical Skills"
            },
            {
                "name": "OPQ32 Personality Assessment",
                "url": "https://www.shl.com/solutions/products/opq32/",
                "description": "Comprehensive personality assessment measuring 32 personality characteristics relevant to workplace behavior",
                "test_type": "P",
                "category": "Personality & Behavior"
            },
            {
                "name": "Python Coding Assessment",
                "url": "https://www.shl.com/solutions/products/python-coding-assessment/",
                "description": "Evaluate Python programming capabilities including scripting, data manipulation, and problem-solving",
                "test_type": "K",
                "category": "Technical Skills"
            },
            {
                "name": "Verify Numerical Reasoning",
                "url": "https://www.shl.com/solutions/products/verify-numerical-reasoning/",
                "description": "Measure numerical reasoning abilities through data interpretation and mathematical problem-solving",
                "test_type": "C",
                "category": "Cognitive Abilities"
            },
            {
                "name": "Teamwork and Collaboration Assessment",
                "url": "https://www.shl.com/solutions/products/teamwork-collaboration/",
                "description": "Evaluate ability to work effectively in teams and collaborate with stakeholders",
                "test_type": "P",
                "category": "Personality & Behavior"
            },
            {
                "name": "SQL Database Assessment",
                "url": "https://www.shl.com/solutions/products/sql-assessment/",
                "description": "Test SQL query writing, database design, and data manipulation skills",
                "test_type": "K",
                "category": "Technical Skills"
            },
            {
                "name": "Verify Verbal Reasoning",
                "url": "https://www.shl.com/solutions/products/verify-verbal-reasoning/",
                "description": "Assess verbal reasoning through comprehension and critical analysis of written information",
                "test_type": "C",
                "category": "Cognitive Abilities"
            },
            {
                "name": "JavaScript Development Test",
                "url": "https://www.shl.com/solutions/products/javascript-test/",
                "description": "Evaluate JavaScript programming skills including ES6+, async programming, and DOM manipulation",
                "test_type": "K",
                "category": "Technical Skills"
            },
            {
                "name": "Leadership Potential Assessment",
                "url": "https://www.shl.com/solutions/products/leadership-potential/",
                "description": "Measure leadership qualities, decision-making abilities, and people management skills",
                "test_type": "P",
                "category": "Personality & Behavior"
            },
            {
                "name": "Data Analysis with Excel",
                "url": "https://www.shl.com/solutions/products/excel-analysis/",
                "description": "Test advanced Excel skills including formulas, pivot tables, and data visualization",
                "test_type": "K",
                "category": "Technical Skills"
            }
        ]
        
        self.assessments = sample_assessments
        return sample_assessments


def main():
    scraper = SHLScraper()
    
    # Try to scrape actual data from SHL website
    logger.info("Starting scrape from official SHL website...")
    assessments = scraper.scrape_catalog()
    
    # If scraping fails or returns too few results, warn but don't auto-fallback
    if len(assessments) < 377:
        logger.warning(f"⚠ Only {len(assessments)} assessments found. Requirement is 377+")
        logger.warning("⚠ The scraper may need adjustment for the current SHL website structure.")
        logger.warning("⚠ You may need to:")
        logger.warning("   1. Check if SHL website structure has changed")
        logger.warning("   2. Update CSS selectors in scrape_catalog() method")
        logger.warning("   3. Manually verify scraping is working")
    else:
        logger.info(f"✓ Successfully scraped {len(assessments)} assessments (meets requirement of 377+)")
    
    # Remove duplicates by URL
    seen_urls = set()
    unique_assessments = []
    for assessment in assessments:
        url = assessment.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_assessments.append(assessment)
    
    if len(unique_assessments) < len(assessments):
        logger.info(f"Removed {len(assessments) - len(unique_assessments)} duplicate assessments")
        scraper.assessments = unique_assessments
    
    # Save to JSON
    scraper.save_to_json()
    
    print(f"\n{'='*60}")
    print(f"Scraping Complete!")
    print(f"Total assessments scraped: {len(unique_assessments)}")
    if unique_assessments:
        print(f"Sample assessment: {unique_assessments[0]['name']}")
        print(f"Sample URL: {unique_assessments[0]['url']}")
    print(f"{'='*60}\n")
    
    # Next steps
    print("Next steps:")
    print("1. Review data/data/raw_assessments.json to verify data quality")
    print("2. Run: python data/clean_data.py to clean and validate")
    print("3. Run: python embeddings/build_embeddings.py to build search index")


if __name__ == "__main__":
    main()