"""
Sample Data Generator
Creates sample data for testing when Gen_AI Dataset.xlsx is not available
"""

import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_train_data():
    """Create sample training data with labeled query-assessment pairs"""
    
    train_data = {
        'Query': [
            # Query 1: Java developer with collaboration
            'I am hiring for Java developers who can also collaborate effectively with my business teams.',
            'I am hiring for Java developers who can also collaborate effectively with my business teams.',
            'I am hiring for Java developers who can also collaborate effectively with my business teams.',
            'I am hiring for Java developers who can also collaborate effectively with my business teams.',
            
            # Query 2: Python, SQL, JavaScript
            'Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.',
            'Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.',
            'Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.',
            'Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.',
            'Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.',
            
            # Query 3: Analyst with cognitive and personality
            'Here is a JD text, can you recommend some assessment that can help me screen applications. I am hiring for an analyst and wants applications to screen using Cognitive and personality tests',
            'Here is a JD text, can you recommend some assessment that can help me screen applications. I am hiring for an analyst and wants applications to screen using Cognitive and personality tests',
            'Here is a JD text, can you recommend some assessment that can help me screen applications. I am hiring for an analyst and wants applications to screen using Cognitive and personality tests',
            'Here is a JD text, can you recommend some assessment that can help me screen applications. I am hiring for an analyst and wants applications to screen using Cognitive and personality tests',
            
            # Query 4: Sales with communication
            'Need sales representatives with excellent communication and interpersonal skills',
            'Need sales representatives with excellent communication and interpersonal skills',
            'Need sales representatives with excellent communication and interpersonal skills',
            
            # Query 5: Leadership
            'Looking for senior leadership candidates with strategic thinking abilities',
            'Looking for senior leadership candidates with strategic thinking abilities',
            'Looking for senior leadership candidates with strategic thinking abilities',
            
            # Query 6: Customer service
            'Customer service representatives who are empathetic and problem-solvers',
            'Customer service representatives who are empathetic and problem-solvers',
            
            # Query 7: Data analyst
            'Need data analysts proficient in Excel, SQL and data visualization',
            'Need data analysts proficient in Excel, SQL and data visualization',
            'Need data analysts proficient in Excel, SQL and data visualization',
            'Need data analysts proficient in Excel, SQL and data visualization',
            
            # Query 8: Software engineer
            'Senior software engineer with full-stack development experience',
            'Senior software engineer with full-stack development experience',
            'Senior software engineer with full-stack development experience',
            
            # Query 9: Project manager
            'Project manager with strong organizational and team management skills',
            'Project manager with strong organizational and team management skills',
            'Project manager with strong organizational and team management skills',
            
            # Query 10: Marketing professional
            'Marketing professional with creativity and analytical mindset',
            'Marketing professional with creativity and analytical mindset',
        ],
        'Assessment_url': [
            # Java + collaboration
            'https://www.shl.com/solutions/products/java-programming-test/',
            'https://www.shl.com/solutions/products/teamwork-collaboration/',
            'https://www.shl.com/solutions/products/verify-interactive/',
            'https://www.shl.com/solutions/products/opq32/',
            
            # Python, SQL, JavaScript
            'https://www.shl.com/solutions/products/python-coding-assessment/',
            'https://www.shl.com/solutions/products/sql-assessment/',
            'https://www.shl.com/solutions/products/javascript-test/',
            'https://www.shl.com/solutions/products/verify-numerical-reasoning/',
            'https://www.shl.com/solutions/products/coding-challenge/',
            
            # Analyst - cognitive + personality
            'https://www.shl.com/solutions/products/verify-numerical-reasoning/',
            'https://www.shl.com/solutions/products/verify-verbal-reasoning/',
            'https://www.shl.com/solutions/products/opq32/',
            'https://www.shl.com/solutions/products/verify-inductive-reasoning/',
            
            # Sales
            'https://www.shl.com/solutions/products/sales-assessment/',
            'https://www.shl.com/solutions/products/customer-service-assessment/',
            'https://www.shl.com/solutions/products/opq32/',
            
            # Leadership
            'https://www.shl.com/solutions/products/leadership-assessment/',
            'https://www.shl.com/solutions/products/management-judgement/',
            'https://www.shl.com/solutions/products/verify-numerical-reasoning/',
            
            # Customer service
            'https://www.shl.com/solutions/products/customer-service-assessment/',
            'https://www.shl.com/solutions/products/situational-judgement-test/',
            
            # Data analyst
            'https://www.shl.com/solutions/products/excel-assessment/',
            'https://www.shl.com/solutions/products/sql-assessment/',
            'https://www.shl.com/solutions/products/verify-numerical-reasoning/',
            'https://www.shl.com/solutions/products/data-analysis-test/',
            
            # Software engineer
            'https://www.shl.com/solutions/products/coding-challenge/',
            'https://www.shl.com/solutions/products/verify-interactive/',
            'https://www.shl.com/solutions/products/technical-skills-assessment/',
            
            # Project manager
            'https://www.shl.com/solutions/products/management-judgement/',
            'https://www.shl.com/solutions/products/leadership-assessment/',
            'https://www.shl.com/solutions/products/opq32/',
            
            # Marketing
            'https://www.shl.com/solutions/products/creativity-assessment/',
            'https://www.shl.com/solutions/products/verify-numerical-reasoning/',
        ]
    }
    
    df = pd.DataFrame(train_data)
    df.to_csv('train_data.csv', index=False)
    
    logger.info(f"✓ Created sample train data: {len(df)} rows, {df['Query'].nunique()} unique queries")
    return df


def create_sample_test_data():
    """Create sample test data (unlabeled queries)"""
    
    test_data = {
        'Query': [
            'Need JavaScript developers with React experience',
            'Looking for HR professionals with recruitment expertise',
            'Data scientist with machine learning and Python skills',
            'Accountant with attention to detail and Excel proficiency',
            'Graphic designer with creative portfolio',
            'Network engineer with cybersecurity knowledge',
            'Business analyst with stakeholder management skills',
            'Content writer with SEO knowledge',
            'Quality assurance tester for software applications'
        ]
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv('test_data.csv', index=False)
    
    logger.info(f"✓ Created sample test data: {len(df)} queries")
    return df


def create_sample_assessments():
    """Create comprehensive sample assessment catalog"""
    
    assessments = [
        # Technical Skills (Type K)
        {
            "name": "Java Programming Test",
            "url": "https://www.shl.com/solutions/products/java-programming-test/",
            "description": "Comprehensive assessment of Java programming skills including OOP concepts, data structures, algorithms, and problem-solving abilities. Tests knowledge of Java syntax, design patterns, and best practices.",
            "test_type": "K",
            "category": "Technical Skills"
        },
        {
            "name": "Python Coding Assessment",
            "url": "https://www.shl.com/solutions/products/python-coding-assessment/",
            "description": "Evaluate Python programming capabilities including scripting, data manipulation, libraries, and algorithmic thinking. Covers Python fundamentals, data structures, and practical coding challenges.",
            "test_type": "K",
            "category": "Technical Skills"
        },
        {
            "name": "JavaScript Development Test",
            "url": "https://www.shl.com/solutions/products/javascript-test/",
            "description": "Assessment of JavaScript skills including ES6+, async programming, DOM manipulation, and modern frameworks. Tests understanding of closures, promises, and event handling.",
            "test_type": "K",
            "category": "Technical Skills"
        },
        {
            "name": "SQL Database Assessment",
            "url": "https://www.shl.com/solutions/products/sql-assessment/",
            "description": "Test SQL query writing, database design, joins, subqueries, and data manipulation. Evaluates knowledge of relational databases and optimization techniques.",
            "test_type": "K",
            "category": "Technical Skills"
        },
        {
            "name": "Excel Data Analysis",
            "url": "https://www.shl.com/solutions/products/excel-assessment/",
            "description": "Advanced Excel skills including formulas, pivot tables, data visualization, macros, and VBA. Tests ability to analyze and present data effectively.",
            "test_type": "K",
            "category": "Technical Skills"
        },
        {
            "name": "Coding Challenge",
            "url": "https://www.shl.com/solutions/products/coding-challenge/",
            "description": "Real-world coding problems to assess algorithmic thinking, code quality, and problem-solving. Language-agnostic evaluation of programming fundamentals.",
            "test_type": "K",
            "category": "Technical Skills"
        },
        {
            "name": "Technical Skills Assessment",
            "url": "https://www.shl.com/solutions/products/technical-skills-assessment/",
            "description": "Comprehensive evaluation of technical competencies across multiple domains. Customizable to specific technical roles and requirements.",
            "test_type": "K",
            "category": "Technical Skills"
        },
        {
            "name": "Data Analysis Test",
            "url": "https://www.shl.com/solutions/products/data-analysis-test/",
            "description": "Assess ability to interpret data, identify patterns, and make data-driven decisions. Covers statistical analysis and data visualization.",
            "test_type": "K",
            "category": "Technical Skills"
        },
        
        # Cognitive Abilities (Type C)
        {
            "name": "Verify Numerical Reasoning",
            "url": "https://www.shl.com/solutions/products/verify-numerical-reasoning/",
            "description": "Measure numerical reasoning abilities through data interpretation, mathematical problem-solving, and quantitative analysis. Industry-standard cognitive assessment.",
            "test_type": "C",
            "category": "Cognitive Abilities"
        },
        {
            "name": "Verify Verbal Reasoning",
            "url": "https://www.shl.com/solutions/products/verify-verbal-reasoning/",
            "description": "Assess verbal reasoning through comprehension and critical analysis of written information. Evaluates ability to understand and evaluate text-based arguments.",
            "test_type": "C",
            "category": "Cognitive Abilities"
        },
        {
            "name": "Verify Inductive Reasoning",
            "url": "https://www.shl.com/solutions/products/verify-inductive-reasoning/",
            "description": "Measure abstract reasoning and pattern recognition abilities. Tests capacity to identify rules and relationships in visual and abstract information.",
            "test_type": "C",
            "category": "Cognitive Abilities"
        },
        {
            "name": "Verify Deductive Reasoning",
            "url": "https://www.shl.com/solutions/products/verify-deductive-reasoning/",
            "description": "Evaluate logical thinking and deductive reasoning skills. Assesses ability to draw conclusions from given premises.",
            "test_type": "C",
            "category": "Cognitive Abilities"
        },
        {
            "name": "Verify Interactive",
            "url": "https://www.shl.com/solutions/products/verify-interactive/",
            "description": "Modern adaptive cognitive assessment combining multiple reasoning types. Engaging, game-based evaluation of cognitive abilities.",
            "test_type": "C",
            "category": "Cognitive Abilities"
        },
        
        # Personality & Behavior (Type P)
        {
            "name": "OPQ32 Personality Assessment",
            "url": "https://www.shl.com/solutions/products/opq32/",
            "description": "Comprehensive personality assessment measuring 32 personality characteristics relevant to workplace behavior. Industry-leading tool for understanding work style and preferences.",
            "test_type": "P",
            "category": "Personality & Behavior"
        },
        {
            "name": "Teamwork and Collaboration Assessment",
            "url": "https://www.shl.com/solutions/products/teamwork-collaboration/",
            "description": "Evaluate ability to work effectively in teams, collaborate with stakeholders, and contribute to group success. Measures interpersonal effectiveness.",
            "test_type": "P",
            "category": "Personality & Behavior"
        },
        {
            "name": "Leadership Assessment",
            "url": "https://www.shl.com/solutions/products/leadership-assessment/",
            "description": "Measure leadership qualities, decision-making abilities, people management skills, and strategic thinking. Identifies leadership potential and development needs.",
            "test_type": "P",
            "category": "Personality & Behavior"
        },
        {
            "name": "Customer Service Assessment",
            "url": "https://www.shl.com/solutions/products/customer-service-assessment/",
            "description": "Evaluate customer service orientation, empathy, problem-solving in customer interactions, and service mindset. Predicts customer satisfaction performance.",
            "test_type": "P",
            "category": "Personality & Behavior"
        },
        {
            "name": "Sales Assessment",
            "url": "https://www.shl.com/solutions/products/sales-assessment/",
            "description": "Assess sales competencies including persuasion, relationship building, resilience, and achievement orientation. Identifies top sales performers.",
            "test_type": "P",
            "category": "Personality & Behavior"
        },
        {
            "name": "Management Judgement",
            "url": "https://www.shl.com/solutions/products/management-judgement/",
            "description": "Evaluate managerial decision-making, judgment in workplace scenarios, and leadership approach. Situational assessment for management roles.",
            "test_type": "P",
            "category": "Personality & Behavior"
        },
        {
            "name": "Situational Judgement Test",
            "url": "https://www.shl.com/solutions/products/situational-judgement-test/",
            "description": "Assess how candidates handle realistic workplace situations. Measures behavioral competencies and decision-making in context.",
            "test_type": "P",
            "category": "Personality & Behavior"
        },
        {
            "name": "Creativity Assessment",
            "url": "https://www.shl.com/solutions/products/creativity-assessment/",
            "description": "Measure creative thinking, innovation, and ability to generate novel solutions. Evaluates divergent thinking and creative problem-solving.",
            "test_type": "P",
            "category": "Personality & Behavior"
        },
    ]
    
    # Save as JSON
    with open('raw_assessments.json', 'w', encoding='utf-8') as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Created sample assessments: {len(assessments)} assessments")
    logger.warning("⚠ This is sample data with only 22 assessments. Real requirement is 377+")
    logger.warning("⚠ Run actual scraper for production: python scraper/scrape_shl.py")
    
    return assessments


def main():
    """Create all sample data files"""
    print("\n" + "="*60)
    print("SAMPLE DATA GENERATOR")
    print("="*60)
    print("\nThis creates sample data for testing purposes.")
    print("For production, use actual data from Gen_AI Dataset.xlsx\n")
    
    # Create train data
    print("1. Creating sample train data...")
    train_df = create_sample_train_data()
    print(f"   ✓ Created: train_data.csv ({len(train_df)} rows)\n")
    
    # Create test data
    print("2. Creating sample test data...")
    test_df = create_sample_test_data()
    print(f"   ✓ Created: test_data.csv ({len(test_df)} rows)\n")
    
    # Create assessments
    print("3. Creating sample assessments...")
    assessments = create_sample_assessments()
    print(f"   ✓ Created: raw_assessments.json ({len(assessments)} assessments)\n")
    
    print("="*60)
    print("SAMPLE DATA CREATED")
    print("="*60)
    print("\nFiles created:")
    print("  - train_data.csv (training queries with labels)")
    print("  - test_data.csv (test queries without labels)")
    print("  - raw_assessments.json (sample assessment catalog)")
    print("\n⚠  Remember: This is SAMPLE data for testing only!")
    print("⚠  For actual submission, use real data:\n")
    print("  1. Place Gen_AI Dataset.xlsx in data/ folder")
    print("  2. Run: python process_data.py")
    print("  3. Run: python scraper/scrape_shl.py")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()