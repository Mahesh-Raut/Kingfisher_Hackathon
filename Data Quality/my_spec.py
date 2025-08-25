import re
import pandas as pd
import pytest
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import warnings
from data_quality_engine import DataQualityEngine, DataQualityValidator, DDLParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# -----------------------
# Legacy Validation Functions (for backward compatibility)
# -----------------------

def validate_email(series: pd.Series):
    """Legacy email validation function"""
    pattern = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
    return series.dropna().apply(lambda x: bool(pattern.match(str(x)))).all()

def validate_age(series: pd.Series, min_age=0, max_age=100):
    """Legacy age validation function"""
    return series.dropna().apply(lambda x: min_age <= int(x) <= max_age).all()

def validate_non_null(series: pd.Series):
    """Legacy non-null validation function"""
    return series.notna().all()

def validate_numeric(series: pd.Series):
    """Legacy numeric validation function"""
    return pd.to_numeric(series, errors="coerce").notna().all()

def validate_date(series: pd.Series, date_format=r"^\d{4}-\d{2}-\d{2}$"):
    """Legacy date validation function"""
    pattern = re.compile(date_format)
    return series.dropna().apply(lambda x: bool(pattern.match(str(x)))).all()

def validate_id_unique(series: pd.Series):
    """Legacy unique ID validation function"""
    return series.is_unique

# -----------------------
# Legacy Rule Mapping (for backward compatibility)
# -----------------------

COLUMN_RULES = {
    r".*email.*": [validate_email],
    r".*age.*": [validate_age],
    r".*id.*": [validate_id_unique, validate_non_null],
    r".*date.*": [validate_date],
    r".*amount.*": [validate_numeric],
}

# -----------------------
# Enhanced Pytest Tests using new Data Quality Engine
# -----------------------

@pytest.fixture(scope="session")
def data_quality_engine():
    """Create data quality engine instance for testing"""
    data_dir = "/home/mahesh.raut/Documents/Fraud_detection_latest/Data Quality"
    ddl_file = "/home/mahesh.raut/Documents/Fraud_detection_latest/Data Quality/bank_ddls.sql"
    return DataQualityEngine(data_dir, ddl_file)

@pytest.fixture(scope="session")
def quality_results(data_quality_engine):
    """Run comprehensive data quality assessment"""
    results, summary = data_quality_engine.run_comprehensive_assessment()
    return results, summary

class TestDataQualityComprehensive:
    """Comprehensive data quality test suite"""
    
    def test_overall_quality_score(self, quality_results):
        """Test that overall data quality score meets minimum threshold"""
        results, summary = quality_results
        assert summary.overall_score >= 0.7, f"Overall quality score {summary.overall_score:.2%} below 70% threshold"
    
    def test_completeness_score(self, quality_results):
        """Test that completeness score meets minimum threshold"""
        results, summary = quality_results
        completeness_score = summary.dimension_scores.get("Completeness", 0)
        assert completeness_score >= 0.8, f"Completeness score {completeness_score:.2%} below 80% threshold"
    
    def test_validity_score(self, quality_results):
        """Test that validity score meets minimum threshold"""
        results, summary = quality_results
        validity_score = summary.dimension_scores.get("Validity", 0)
        assert validity_score >= 0.8, f"Validity score {validity_score:.2%} below 80% threshold"
    
    def test_uniqueness_score(self, quality_results):
        """Test that uniqueness score meets minimum threshold"""
        results, summary = quality_results
        uniqueness_score = summary.dimension_scores.get("Uniqueness", 0)
        assert uniqueness_score >= 0.9, f"Uniqueness score {uniqueness_score:.2%} below 90% threshold"
    
    def test_no_critical_failures(self, quality_results):
        """Test that there are no critical data quality failures"""
        results, summary = quality_results
        critical_failures = [r for r in results if r.status == "FAIL" and r.score < 0.5]
        
        if critical_failures:
            failure_details = "\n".join([
                f"- {r.table_name}.{r.column_name}: {r.rule_name} (Score: {r.score:.2%})"
                for r in critical_failures[:5]
            ])
            pytest.fail(f"Critical data quality failures found:\n{failure_details}")
    
    def test_referential_integrity(self, quality_results):
        """Test referential integrity across tables"""
        results, summary = quality_results
        ref_integrity_results = [r for r in results if r.rule_name == "Referential Integrity"]
        
        for result in ref_integrity_results:
            assert result.status == "PASS", \
                f"Referential integrity failure: {result.table_name} - {result.details}"
    
    def test_ddl_compliance(self, quality_results):
        """Test compliance with DDL schema"""
        results, summary = quality_results
        ddl_results = [r for r in results if r.dimension == "Conformity"]
        
        critical_ddl_failures = [r for r in ddl_results if r.status == "FAIL"]
        assert len(critical_ddl_failures) == 0, \
            f"DDL compliance failures: {[r.details for r in critical_ddl_failures]}"

class TestDataQualityByTable:
    """Table-specific data quality tests"""
    
    @pytest.mark.parametrize("table_name", ["users", "accounts", "transactions"])
    def test_table_completeness(self, data_quality_engine, table_name):
        """Test completeness for each table"""
        data = data_quality_engine.load_data()
        if table_name not in data:
            pytest.skip(f"Table {table_name} not found")
        
        df = data[table_name]
        validator = data_quality_engine.validator
        
        # Test primary key completeness (assuming 'id' column exists)
        if 'id' in df.columns:
            result = validator.validate_completeness(df, table_name, 'id')
            assert result.status == "PASS", f"ID column has missing values in {table_name}"
    
    @pytest.mark.parametrize("table_name", ["users", "accounts", "transactions"])
    def test_table_primary_key_uniqueness(self, data_quality_engine, table_name):
        """Test primary key uniqueness for each table"""
        data = data_quality_engine.load_data()
        if table_name not in data:
            pytest.skip(f"Table {table_name} not found")
        
        df = data[table_name]
        validator = data_quality_engine.validator
        
        # Test primary key uniqueness (assuming 'id' column exists)
        if 'id' in df.columns:
            result = validator.validate_uniqueness(df, table_name, 'id')
            assert result.status == "PASS", f"ID column has duplicates in {table_name}"

# -----------------------
# Legacy Pytest Parametrized Tests (for backward compatibility)
# -----------------------

@pytest.mark.parametrize("csv_file", [
    "users.csv",
    "accounts.csv", 
    "transactions.csv"
])
def test_csv_data_quality_legacy(csv_file):
    """Legacy parametrized test for CSV data quality"""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        # Try with full path
        full_path = f"/home/mahesh.raut/Documents/Fraud_detection_latest/Data Quality/{csv_file}"
        df = pd.read_csv(full_path)

    for col in df.columns:
        for pattern, validators in COLUMN_RULES.items():
            if re.match(pattern, col, re.IGNORECASE):
                for validator in validators:
                    assert validator(df[col]), f"Validation failed for column '{col}' with {validator.__name__}"

# -----------------------
# Data Quality Report Generation Tests
# -----------------------

def test_generate_comprehensive_report():
    """Test generation of comprehensive data quality report"""
    data_dir = "/home/mahesh.raut/Documents/Fraud_detection_latest/Data Quality"
    ddl_file = "/home/mahesh.raut/Documents/Fraud_detection_latest/Data Quality/bank_ddls.sql"
    
    engine = DataQualityEngine(data_dir, ddl_file)
    report = engine.generate_report("test_data_quality_report.json")
    
    # Verify report structure
    assert "assessment_timestamp" in report
    assert "summary" in report
    assert "detailed_results" in report
    assert "results_by_dimension" in report
    assert "results_by_table" in report
    
    # Verify summary content
    summary = report["summary"]
    assert "total_tables" in summary
    assert "total_columns" in summary
    assert "total_records" in summary
    assert "overall_score" in summary
    assert "dimension_scores" in summary

if __name__ == "__main__":
    # Run comprehensive data quality assessment when script is executed directly
    print("Running Comprehensive Data Quality Assessment...")
    print("=" * 60)
    
    data_dir = "/home/mahesh.raut/Documents/Fraud_detection_latest/Data Quality"
    ddl_file = "/home/mahesh.raut/Documents/Fraud_detection_latest/Data Quality/bank_ddls.sql"
    
    engine = DataQualityEngine(data_dir, ddl_file)
    engine.print_summary()
    
    # Generate report
    report = engine.generate_report("comprehensive_data_quality_report.json")
    print(f"\nComprehensive report saved to: comprehensive_data_quality_report.json")
