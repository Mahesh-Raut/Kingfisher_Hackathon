"""
Comprehensive Data Quality Assessment Engine

This module provides a comprehensive framework for assessing data quality across multiple dimensions:
- Completeness: Missing data analysis
- Uniqueness: Duplicate detection
- Validity: Format and constraint validation  
- Consistency: Cross-field and cross-table validation
- Accuracy: Business rule validation
- Integrity: Referential integrity checks
- Conformity: DDL schema compliance
"""

import re
import pandas as pd
import pytest
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import sqlparse
from decimal import Decimal
import warnings
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class DataQualityResult:
    """Data structure to hold data quality assessment results"""
    dimension: str
    rule_name: str
    table_name: str
    column_name: str
    status: str  # PASS, FAIL, WARNING
    score: float  # 0.0 to 1.0
    total_records: int
    failed_records: int
    details: str
    timestamp: str

@dataclass
class DataQualitySummary:
    """Summary of overall data quality assessment"""
    total_tables: int
    total_columns: int
    total_records: int
    overall_score: float
    dimension_scores: Dict[str, float]
    critical_issues: List[str]
    recommendations: List[str]

class DDLParser:
    """Parse SQL DDL to extract schema information"""
    
    def __init__(self, ddl_file: str):
        self.ddl_file = ddl_file
        self.schema = {}
        self._parse_ddl()
    
    def _parse_ddl(self):
        """Parse DDL file and extract table schemas"""
        try:
            with open(self.ddl_file, 'r') as f:
                ddl_content = f.read()
            
            # Parse SQL statements
            statements = sqlparse.split(ddl_content)
            
            for statement in statements:
                if 'CREATE TABLE' in statement.upper():
                    self._parse_create_table(statement)
                    
        except Exception as e:
            logger.error(f"Error parsing DDL: {e}")
    
    def _parse_create_table(self, statement: str):
        """Parse CREATE TABLE statement"""
        try:
            # Extract table name
            table_match = re.search(r'CREATE TABLE\s+(\w+\.)?(\w+)', statement, re.IGNORECASE)
            if not table_match:
                return
            
            table_name = table_match.group(2)
            
            # Extract column definitions
            columns = {}
            column_pattern = r'(\w+)\s+([\w\(\),\s]+?)(?:,|\)|$)'
            
            # Get content between parentheses
            paren_match = re.search(r'\((.*)\)', statement, re.DOTALL)
            if paren_match:
                table_def = paren_match.group(1)
                
                # Split by comma but handle nested parentheses
                lines = table_def.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.upper().startswith(('FOREIGN KEY', 'PRIMARY KEY', 'UNIQUE', 'INDEX')):
                        parts = line.split()
                        if len(parts) >= 2:
                            col_name = parts[0].strip(',')
                            col_type = parts[1].strip(',')
                            
                            constraints = []
                            if 'PRIMARY KEY' in line.upper():
                                constraints.append('PRIMARY_KEY')
                            if 'NOT NULL' in line.upper():
                                constraints.append('NOT_NULL')
                            if 'UNIQUE' in line.upper():
                                constraints.append('UNIQUE')
                            
                            columns[col_name] = {
                                'type': col_type,
                                'constraints': constraints,
                                'nullable': 'NOT NULL' not in line.upper()
                            }
            
            self.schema[table_name] = columns
            logger.info(f"Parsed schema for table: {table_name}")
            
        except Exception as e:
            logger.error(f"Error parsing CREATE TABLE statement: {e}")

class DataQualityValidator:
    """Main data quality validation engine"""
    
    def __init__(self, ddl_parser: Optional[DDLParser] = None):
        self.ddl_parser = ddl_parser
        self.results = []
        self.data_cache = {}
    
    # -----------------------
    # COMPLETENESS VALIDATORS
    # -----------------------
    
    def validate_completeness(self, df: pd.DataFrame, table_name: str, column: str) -> DataQualityResult:
        """Check for missing values (null, empty, whitespace)"""
        total_records = len(df)
        
        # Count various types of missing data
        null_count = df[column].isnull().sum()
        empty_count = (df[column] == '').sum() if df[column].dtype == 'object' else 0
        whitespace_count = df[column].str.strip().eq('').sum() if df[column].dtype == 'object' else 0
        
        failed_records = null_count + empty_count + whitespace_count
        score = (total_records - failed_records) / total_records if total_records > 0 else 0
        
        status = "PASS" if score >= 0.95 else "FAIL" if score < 0.8 else "WARNING"
        
        return DataQualityResult(
            dimension="Completeness",
            rule_name="No Missing Values",
            table_name=table_name,
            column_name=column,
            status=status,
            score=score,
            total_records=total_records,
            failed_records=failed_records,
            details=f"Null: {null_count}, Empty: {empty_count}, Whitespace: {whitespace_count}",
            timestamp=datetime.now().isoformat()
        )
    
    # -----------------------
    # UNIQUENESS VALIDATORS
    # -----------------------
    
    def validate_uniqueness(self, df: pd.DataFrame, table_name: str, column: str) -> DataQualityResult:
        """Check for duplicate values"""
        total_records = len(df)
        unique_count = df[column].nunique()
        duplicate_count = total_records - unique_count
        
        score = unique_count / total_records if total_records > 0 else 0
        status = "PASS" if duplicate_count == 0 else "FAIL"
        
        return DataQualityResult(
            dimension="Uniqueness",
            rule_name="No Duplicates",
            table_name=table_name,
            column_name=column,
            status=status,
            score=score,
            total_records=total_records,
            failed_records=duplicate_count,
            details=f"Unique values: {unique_count}, Duplicates: {duplicate_count}",
            timestamp=datetime.now().isoformat()
        )
    
    # -----------------------
    # VALIDITY VALIDATORS
    # -----------------------
    
    def validate_email(self, df: pd.DataFrame, table_name: str, column: str) -> DataQualityResult:
        """Validate email format"""
        pattern = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
        total_records = len(df[df[column].notna()])
        
        if total_records == 0:
            return self._create_no_data_result("Validity", "Email Format", table_name, column)
        
        valid_emails = df[df[column].notna()][column].apply(lambda x: bool(pattern.match(str(x))))
        failed_records = (~valid_emails).sum()
        score = (total_records - failed_records) / total_records
        
        status = "PASS" if score >= 0.95 else "FAIL" if score < 0.8 else "WARNING"
        
        return DataQualityResult(
            dimension="Validity",
            rule_name="Email Format",
            table_name=table_name,
            column_name=column,
            status=status,
            score=score,
            total_records=total_records,
            failed_records=failed_records,
            details=f"Invalid email formats: {failed_records}",
            timestamp=datetime.now().isoformat()
        )
    
    def validate_age(self, df: pd.DataFrame, table_name: str, column: str, min_age: int = 0, max_age: int = 120) -> DataQualityResult:
        """Validate age range"""
        total_records = len(df[df[column].notna()])
        
        if total_records == 0:
            return self._create_no_data_result("Validity", "Age Range", table_name, column)
        
        try:
            ages = pd.to_numeric(df[df[column].notna()][column], errors='coerce')
            valid_ages = (ages >= min_age) & (ages <= max_age) & ages.notna()
            failed_records = (~valid_ages).sum()
            score = (total_records - failed_records) / total_records
            
            status = "PASS" if score >= 0.95 else "FAIL" if score < 0.8 else "WARNING"
            
            return DataQualityResult(
                dimension="Validity",
                rule_name="Age Range",
                table_name=table_name,
                column_name=column,
                status=status,
                score=score,
                total_records=total_records,
                failed_records=failed_records,
                details=f"Ages outside range [{min_age}, {max_age}]: {failed_records}",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            return self._create_error_result("Validity", "Age Range", table_name, column, str(e))
    
    def validate_numeric(self, df: pd.DataFrame, table_name: str, column: str) -> DataQualityResult:
        """Validate numeric format"""
        total_records = len(df[df[column].notna()])
        
        if total_records == 0:
            return self._create_no_data_result("Validity", "Numeric Format", table_name, column)
        
        numeric_values = pd.to_numeric(df[df[column].notna()][column], errors='coerce')
        failed_records = numeric_values.isna().sum()
        score = (total_records - failed_records) / total_records
        
        status = "PASS" if score >= 0.95 else "FAIL" if score < 0.8 else "WARNING"
        
        return DataQualityResult(
            dimension="Validity",
            rule_name="Numeric Format",
            table_name=table_name,
            column_name=column,
            status=status,
            score=score,
            total_records=total_records,
            failed_records=failed_records,
            details=f"Non-numeric values: {failed_records}",
            timestamp=datetime.now().isoformat()
        )
    
    def validate_date_format(self, df: pd.DataFrame, table_name: str, column: str, 
                           date_format: str = r"^\d{4}-\d{2}-\d{2}") -> DataQualityResult:
        """Validate date format"""
        pattern = re.compile(date_format)
        total_records = len(df[df[column].notna()])
        
        if total_records == 0:
            return self._create_no_data_result("Validity", "Date Format", table_name, column)
        
        valid_dates = df[df[column].notna()][column].astype(str).apply(lambda x: bool(pattern.match(x)))
        failed_records = (~valid_dates).sum()
        score = (total_records - failed_records) / total_records
        
        status = "PASS" if score >= 0.95 else "FAIL" if score < 0.8 else "WARNING"
        
        return DataQualityResult(
            dimension="Validity",
            rule_name="Date Format",
            table_name=table_name,
            column_name=column,
            status=status,
            score=score,
            total_records=total_records,
            failed_records=failed_records,
            details=f"Invalid date formats: {failed_records}",
            timestamp=datetime.now().isoformat()
        )
    
    # -----------------------
    # CONSISTENCY VALIDATORS
    # -----------------------
    
    def validate_referential_integrity(self, parent_df: pd.DataFrame, child_df: pd.DataFrame,
                                     parent_table: str, child_table: str,
                                     parent_key: str, foreign_key: str) -> DataQualityResult:
        """Check referential integrity between tables"""
        total_records = len(child_df[child_df[foreign_key].notna()])
        
        if total_records == 0:
            return self._create_no_data_result("Consistency", "Referential Integrity", 
                                             f"{child_table}->{parent_table}", foreign_key)
        
        parent_keys = set(parent_df[parent_key].values)
        orphaned_records = child_df[child_df[foreign_key].notna() & 
                                  (~child_df[foreign_key].isin(parent_keys))]
        
        failed_records = len(orphaned_records)
        score = (total_records - failed_records) / total_records
        
        status = "PASS" if failed_records == 0 else "FAIL"
        
        return DataQualityResult(
            dimension="Consistency",
            rule_name="Referential Integrity",
            table_name=f"{child_table}->{parent_table}",
            column_name=foreign_key,
            status=status,
            score=score,
            total_records=total_records,
            failed_records=failed_records,
            details=f"Orphaned records: {failed_records}",
            timestamp=datetime.now().isoformat()
        )
    
    def validate_data_range_consistency(self, df: pd.DataFrame, table_name: str,
                                      start_col: str, end_col: str) -> DataQualityResult:
        """Validate that start date/value is before end date/value"""
        total_records = len(df[(df[start_col].notna()) & (df[end_col].notna())])
        
        if total_records == 0:
            return self._create_no_data_result("Consistency", "Range Consistency", 
                                             table_name, f"{start_col}-{end_col}")
        
        try:
            start_vals = pd.to_datetime(df[start_col], errors='coerce')
            end_vals = pd.to_datetime(df[end_col], errors='coerce')
            
            invalid_ranges = (start_vals > end_vals) & start_vals.notna() & end_vals.notna()
            failed_records = invalid_ranges.sum()
            score = (total_records - failed_records) / total_records
            
            status = "PASS" if failed_records == 0 else "FAIL"
            
            return DataQualityResult(
                dimension="Consistency",
                rule_name="Range Consistency",
                table_name=table_name,
                column_name=f"{start_col}-{end_col}",
                status=status,
                score=score,
                total_records=total_records,
                failed_records=failed_records,
                details=f"Invalid ranges (start > end): {failed_records}",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            return self._create_error_result("Consistency", "Range Consistency", 
                                           table_name, f"{start_col}-{end_col}", str(e))
    
    # -----------------------
    # CONFORMITY VALIDATORS (DDL Compliance)
    # -----------------------
    
    def validate_ddl_compliance(self, df: pd.DataFrame, table_name: str) -> List[DataQualityResult]:
        """Validate data against DDL schema"""
        results = []
        
        if not self.ddl_parser or table_name not in self.ddl_parser.schema:
            logger.warning(f"No DDL schema found for table: {table_name}")
            return results
        
        schema = self.ddl_parser.schema[table_name]
        
        # Check for missing columns
        missing_cols = set(schema.keys()) - set(df.columns)
        if missing_cols:
            results.append(DataQualityResult(
                dimension="Conformity",
                rule_name="Schema Compliance",
                table_name=table_name,
                column_name=",".join(missing_cols),
                status="FAIL",
                score=0.0,
                total_records=len(df),
                failed_records=len(df),
                details=f"Missing columns: {missing_cols}",
                timestamp=datetime.now().isoformat()
            ))
        
        # Check for extra columns
        extra_cols = set(df.columns) - set(schema.keys())
        if extra_cols:
            results.append(DataQualityResult(
                dimension="Conformity",
                rule_name="Schema Compliance",
                table_name=table_name,
                column_name=",".join(extra_cols),
                status="WARNING",
                score=0.8,
                total_records=len(df),
                failed_records=0,
                details=f"Extra columns: {extra_cols}",
                timestamp=datetime.now().isoformat()
            ))
        
        # Validate individual column constraints
        for col_name, col_schema in schema.items():
            if col_name in df.columns:
                # Check NOT NULL constraints
                if 'NOT_NULL' in col_schema['constraints']:
                    null_count = df[col_name].isnull().sum()
                    if null_count > 0:
                        results.append(DataQualityResult(
                            dimension="Conformity",
                            rule_name="NOT NULL Constraint",
                            table_name=table_name,
                            column_name=col_name,
                            status="FAIL",
                            score=(len(df) - null_count) / len(df),
                            total_records=len(df),
                            failed_records=null_count,
                            details=f"NULL values found in NOT NULL column: {null_count}",
                            timestamp=datetime.now().isoformat()
                        ))
                
                # Check UNIQUE constraints
                if 'UNIQUE' in col_schema['constraints']:
                    dup_count = len(df) - df[col_name].nunique()
                    if dup_count > 0:
                        results.append(DataQualityResult(
                            dimension="Conformity",
                            rule_name="UNIQUE Constraint",
                            table_name=table_name,
                            column_name=col_name,
                            status="FAIL",
                            score=df[col_name].nunique() / len(df),
                            total_records=len(df),
                            failed_records=dup_count,
                            details=f"Duplicate values in UNIQUE column: {dup_count}",
                            timestamp=datetime.now().isoformat()
                        ))
        
        return results
    
    # -----------------------
    # HELPER METHODS
    # -----------------------
    
    def _create_no_data_result(self, dimension: str, rule_name: str, table_name: str, column: str) -> DataQualityResult:
        """Create result for cases with no data to validate"""
        return DataQualityResult(
            dimension=dimension,
            rule_name=rule_name,
            table_name=table_name,
            column_name=column,
            status="WARNING",
            score=0.0,
            total_records=0,
            failed_records=0,
            details="No data to validate",
            timestamp=datetime.now().isoformat()
        )
    
    def _create_error_result(self, dimension: str, rule_name: str, table_name: str, column: str, error: str) -> DataQualityResult:
        """Create result for validation errors"""
        return DataQualityResult(
            dimension=dimension,
            rule_name=rule_name,
            table_name=table_name,
            column_name=column,
            status="FAIL",
            score=0.0,
            total_records=0,
            failed_records=0,
            details=f"Validation error: {error}",
            timestamp=datetime.now().isoformat()
        )

class DataQualityEngine:
    """Main orchestrator for data quality assessment"""
    
    def __init__(self, data_directory: str, ddl_file: Optional[str] = None):
        self.data_directory = Path(data_directory)
        self.ddl_parser = DDLParser(ddl_file) if ddl_file else None
        self.validator = DataQualityValidator(self.ddl_parser)
        self.results = []
        
        # Define column-based validation rules
        self.column_rules = {
            r".*email.*": [self._validate_email],
            r".*age.*": [self._validate_age],
            r".*id.*": [self._validate_uniqueness, self._validate_completeness],
            r".*date.*": [self._validate_date_format],
            r".*amount.*": [self._validate_numeric],
            r".*balance.*": [self._validate_numeric],
            r".*created_at.*": [self._validate_date_format],
            r".*updated_at.*": [self._validate_date_format],
        }
        
        # Define cross-table relationships
        self.relationships = [
            ("users", "id", "accounts", "user_id"),
            ("accounts", "id", "transactions", "from_account_id"),
            ("accounts", "id", "transactions", "to_account_id"),
        ]
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from data directory"""
        data = {}
        csv_files = list(self.data_directory.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                table_name = csv_file.stem
                df = pd.read_csv(csv_file)
                data[table_name] = df
                logger.info(f"Loaded {len(df)} records from {table_name}")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
        
        return data
    
    def run_comprehensive_assessment(self) -> Tuple[List[DataQualityResult], DataQualitySummary]:
        """Run comprehensive data quality assessment"""
        data = self.load_data()
        self.results = []
        
        # Run column-based validations
        for table_name, df in data.items():
            logger.info(f"Assessing data quality for table: {table_name}")
            
            # DDL compliance check
            if self.ddl_parser:
                ddl_results = self.validator.validate_ddl_compliance(df, table_name)
                self.results.extend(ddl_results)
            
            # Column-based validations
            for column in df.columns:
                # Completeness check for all columns
                result = self.validator.validate_completeness(df, table_name, column)
                self.results.append(result)
                
                # Pattern-based validations
                for pattern, validators in self.column_rules.items():
                    if re.match(pattern, column, re.IGNORECASE):
                        for validator_func in validators:
                            result = validator_func(df, table_name, column)
                            if result:
                                self.results.append(result)
        # uncommented
        # Cross-table validations - COMMENTED OUT
        for parent_table, parent_key, child_table, foreign_key in self.relationships:
            if parent_table in data and child_table in data:
                result = self.validator.validate_referential_integrity(
                    data[parent_table], data[child_table],
                    parent_table, child_table, parent_key, foreign_key
                )
                self.results.append(result)
        # uncommented
        # Date range consistency checks - COMMENTED OUT
        for table_name, df in data.items():
            if 'created_at' in df.columns and 'updated_at' in df.columns:
                result = self.validator.validate_data_range_consistency(
                    df, table_name, 'created_at', 'updated_at'
                )
                self.results.append(result)
        
        # Generate summary
        summary = self._generate_summary(data)
        
        return self.results, summary
    
    def _validate_email(self, df: pd.DataFrame, table_name: str, column: str) -> DataQualityResult:
        return self.validator.validate_email(df, table_name, column)
    
    def _validate_age(self, df: pd.DataFrame, table_name: str, column: str) -> DataQualityResult:
        return self.validator.validate_age(df, table_name, column)
    
    def _validate_uniqueness(self, df: pd.DataFrame, table_name: str, column: str) -> DataQualityResult:
        return self.validator.validate_uniqueness(df, table_name, column)
    
    def _validate_completeness(self, df: pd.DataFrame, table_name: str, column: str) -> DataQualityResult:
        return self.validator.validate_completeness(df, table_name, column)
    
    def _validate_numeric(self, df: pd.DataFrame, table_name: str, column: str) -> DataQualityResult:
        return self.validator.validate_numeric(df, table_name, column)
    
    def _validate_date_format(self, df: pd.DataFrame, table_name: str, column: str) -> DataQualityResult:
        return self.validator.validate_date_format(df, table_name, column)
    
    def _generate_summary(self, data: Dict[str, pd.DataFrame]) -> DataQualitySummary:
        """Generate overall data quality summary"""
        total_tables = len(data)
        total_columns = sum(len(df.columns) for df in data.values())
        total_records = sum(len(df) for df in data.values())
        
        # Calculate dimension scores
        dimension_scores = defaultdict(list)
        for result in self.results:
            dimension_scores[result.dimension].append(result.score)
        
        avg_dimension_scores = {
            dim: np.mean(scores) for dim, scores in dimension_scores.items()
        }
        
        overall_score = np.mean([score for scores in dimension_scores.values() for score in scores])
        
        # Identify critical issues
        critical_issues = [
            f"{result.table_name}.{result.column_name}: {result.rule_name}"
            for result in self.results
            if result.status == "FAIL" and result.score < 0.5
        ]
        
        # Generate recommendations
        recommendations = []
        if avg_dimension_scores.get("Completeness", 1.0) < 0.9:
            recommendations.append("Review data collection processes to reduce missing values")
        if avg_dimension_scores.get("Uniqueness", 1.0) < 0.9:
            recommendations.append("Implement data deduplication procedures")
        if avg_dimension_scores.get("Validity", 1.0) < 0.9:
            recommendations.append("Add data validation at point of entry")
        # if avg_dimension_scores.get("Consistency", 1.0) < 0.9:
        #     recommendations.append("Review data consistency across related fields and tables")
        if avg_dimension_scores.get("Conformity", 1.0) < 0.9:
            recommendations.append("Align data structure with schema definitions")
        
        return DataQualitySummary(
            total_tables=total_tables,
            total_columns=total_columns,
            total_records=total_records,
            overall_score=overall_score,
            dimension_scores=avg_dimension_scores,
            critical_issues=critical_issues[:10],  # Top 10 issues
            recommendations=recommendations
        )
    
    def generate_report(self, output_file: str = "data_quality_report.json"):
        """Generate comprehensive data quality report"""
        results, summary = self.run_comprehensive_assessment()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert results to dictionaries with JSON-safe types
        json_results = []
        for result in results:
            result_dict = asdict(result)
            for key, value in result_dict.items():
                result_dict[key] = convert_for_json(value)
            json_results.append(result_dict)
        
        # Convert summary with JSON-safe types
        summary_dict = asdict(summary)
        for key, value in summary_dict.items():
            if isinstance(value, dict):
                summary_dict[key] = {k: convert_for_json(v) for k, v in value.items()}
            elif isinstance(value, list):
                summary_dict[key] = [convert_for_json(item) for item in value]
            else:
                summary_dict[key] = convert_for_json(value)
        
        report = {
            "assessment_timestamp": datetime.now().isoformat(),
            "summary": summary_dict,
            "detailed_results": json_results,
            "results_by_dimension": self._group_results_by_dimension(results),
            "results_by_table": self._group_results_by_table(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=convert_for_json)
        
        logger.info(f"Data quality report generated: {output_file}")
        return report
    
    def _group_results_by_dimension(self, results: List[DataQualityResult]) -> Dict[str, List[Dict]]:
        """Group results by data quality dimension"""
        grouped = defaultdict(list)
        for result in results:
            result_dict = asdict(result)
            # Convert numpy types to native Python types
            for key, value in result_dict.items():
                if isinstance(value, (np.integer, np.floating)):
                    result_dict[key] = float(value) if isinstance(value, np.floating) else int(value)
            grouped[result.dimension].append(result_dict)
        return dict(grouped)
    
    def _group_results_by_table(self, results: List[DataQualityResult]) -> Dict[str, List[Dict]]:
        """Group results by table"""
        grouped = defaultdict(list)
        for result in results:
            result_dict = asdict(result)
            # Convert numpy types to native Python types
            for key, value in result_dict.items():
                if isinstance(value, (np.integer, np.floating)):
                    result_dict[key] = float(value) if isinstance(value, np.floating) else int(value)
            grouped[result.table_name].append(result_dict)
        return dict(grouped)
    
    def print_summary(self):
        """Print a formatted summary of data quality assessment"""
        results, summary = self.run_comprehensive_assessment()
        
        print("\n" + "="*80)
        print("DATA QUALITY ASSESSMENT SUMMARY")
        print("="*80)
        print(f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Tables: {summary.total_tables}")
        print(f"Total Columns: {summary.total_columns}")
        print(f"Total Records: {summary.total_records:,}")
        print(f"Overall Quality Score: {summary.overall_score:.2%}")
        
        print("\nDIMENSION SCORES:")
        print("-" * 40)
        for dimension, score in summary.dimension_scores.items():
            status = "✓" if score >= 0.9 else "⚠" if score >= 0.7 else "✗"
            print(f"{status} {dimension:<15}: {score:.2%}")
        
        if summary.critical_issues:
            print("\nCRITICAL ISSUES:")
            print("-" * 40)
            for issue in summary.critical_issues[:5]:
                print(f"• {issue}")
        
        if summary.recommendations:
            print("\nRECOMMENDations:")
            print("-" * 40)
            for rec in summary.recommendations:
                print(f"• {rec}")
        
        print("\nDETAILED RESULTS:")
        print("-" * 40)
        for result in results:
            status_symbol = "✓" if result.status == "PASS" else "⚠" if result.status == "WARNING" else "✗"
            print(f"{status_symbol} {result.table_name}.{result.column_name:<20} | "
                  f"{result.dimension:<12} | {result.rule_name:<20} | {result.score:.2%}")

if __name__ == "__main__":
    # Initialize the data quality engine
    data_dir = "/home/mahesh.raut/Documents/Fraud_detection_latest/Data Quality"
    ddl_file = "/home/mahesh.raut/Documents/Fraud_detection_latest/Data Quality/bank_ddls.sql"
    
    engine = DataQualityEngine(data_dir, ddl_file)
    
    # Run assessment and print summary
    engine.print_summary()
    
    # Generate detailed report
    report = engine.generate_report("data_quality_report.json")
    print(f"\nDetailed report saved to: data_quality_report.json")
