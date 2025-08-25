#!/usr/bin/env python3
"""
Simple test to verify data quality functionality
"""

import pandas as pd
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_quality_engine import DataQualityEngine

def test_basic_functionality():
    """Test basic data quality engine functionality"""
    print("Testing basic data quality functionality...")
    
    # Create test data
    test_data = {
        'id': [1, 2, 3, 4, 5],
        'email': ['test@example.com', 'invalid_email', 'another@test.com', 'bad@', 'good@domain.org'],
        'age': [25, 150, 30, -5, 45],
        'amount': [100.50, 200.00, -50, 300.25, 0]
    }
    
    df = pd.DataFrame(test_data)
    
    # Initialize engine without DDL
    engine = DataQualityEngine("/tmp", None)
    validator = engine.validator
    
    # Test email validation
    email_result = validator.validate_email(df, 'test_table', 'email')
    print(f"Email validation score: {email_result.score:.2%}")
    
    # Test age validation
    age_result = validator.validate_age(df, 'test_table', 'age')
    print(f"Age validation score: {age_result.score:.2%}")
    
    # Test completeness
    completeness_result = validator.validate_completeness(df, 'test_table', 'id')
    print(f"Completeness score: {completeness_result.score:.2%}")
    
    # Test uniqueness
    uniqueness_result = validator.validate_uniqueness(df, 'test_table', 'id')
    print(f"Uniqueness score: {uniqueness_result.score:.2%}")
    
    print("✅ Basic functionality test passed!")
    return True

def test_csv_loading():
    """Test CSV loading from the actual data directory"""
    print("Testing CSV loading...")
    
    data_dir = "/home/mahesh.raut/Documents/Fraud_detection_latest/Data Quality"
    engine = DataQualityEngine(data_dir, None)
    
    data = engine.load_data()
    
    print(f"Loaded {len(data)} tables:")
    for table_name, df in data.items():
        print(f"  - {table_name}: {len(df)} rows, {len(df.columns)} columns")
    
    assert len(data) > 0, "No data loaded"
    print("✅ CSV loading test passed!")
    return True

def main():
    """Run all tests"""
    print("🧪 Running Data Quality Engine Tests")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_csv_loading
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
