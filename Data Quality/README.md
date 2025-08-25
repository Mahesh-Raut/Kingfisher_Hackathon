# 📊 Comprehensive Data Quality Assessment Engine

A powerful, extensible data quality assessment framework that validates data across multiple dimensions including completeness, uniqueness, validity, consistency, conformity, and integrity.

## 🎯 Features

### Data Quality Dimensions
- **Completeness**: Detect missing, null, empty, and whitespace-only values
- **Uniqueness**: Identify duplicate records and ensure primary key constraints
- **Validity**: Validate data formats (emails, dates, numbers, phone numbers)
- **Consistency**: Check cross-field and cross-table relationships
- **Conformity**: Validate against DDL schema definitions
- **Integrity**: Verify referential integrity between related tables

### Key Capabilities
- 🔍 **Automated Discovery**: Pattern-based column identification
- 📋 **DDL Compliance**: Validate data against SQL schema definitions
- 🔗 **Relationship Validation**: Check foreign key constraints
- 📊 **Comprehensive Reporting**: JSON, HTML, and console output
- 🧪 **Pytest Integration**: Automated testing framework
- ⚙️ **Configurable Rules**: JSON-based configuration system
- 📈 **Scoring System**: Quantitative quality metrics

## 🚀 Quick Start

### 1. Run Comprehensive Assessment
```bash
# Basic assessment with console output
python data_quality_engine.py

# Full assessment with detailed reporting
python run_dq_tests.py comprehensive --output my_report.json
```

### 2. Quick Validation Check
```bash
python run_dq_tests.py quick
```

### 3. Run Pytest Test Suite
```bash
python run_dq_tests.py pytest --verbose
```

### 4. Generate HTML Report
```bash
python run_dq_tests.py html --output data_quality_report.json
```

## 📁 File Structure

```
Data Quality/
├── data_quality_engine.py      # Core assessment engine
├── my_spec.py                   # Pytest test specifications  
├── run_dq_tests.py             # Enhanced test runner
├── dq_config.json              # Configuration file
├── bank_ddls.sql               # Schema definitions
├── users.csv                   # Sample data
├── accounts.csv                # Sample data
├── transactions.csv            # Sample data
└── README.md                   # This file
```

## 🛠️ Configuration

### Custom Validation Rules
Edit `dq_config.json` to customize:

```json
{
  "data_quality_config": {
    "thresholds": {
      "completeness_minimum": 0.95,
      "validity_minimum": 0.9,
      "uniqueness_minimum": 0.95
    },
    "column_patterns": {
      "email": {
        "regex": ".*email.*",
        "validations": ["email_format", "completeness"]
      }
    }
  }
}
```

### Schema Validation
Provide DDL files for automatic schema compliance checking:

```sql
CREATE TABLE bank.users (
    id BIGINT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL
);
```

## 📊 Sample Output

### Console Summary
```
================================================================================
DATA QUALITY ASSESSMENT SUMMARY
================================================================================
Assessment Date: 2024-08-24 10:30:45
Total Tables: 3
Total Columns: 24
Total Records: 3,006
Overall Quality Score: 87.5%

DIMENSION SCORES:
----------------------------------------
✓ Completeness    : 95.2%
✓ Validity        : 89.8%
✓ Uniqueness     : 92.1%
⚠ Consistency    : 76.3%
✓ Conformity     : 88.9%

CRITICAL ISSUES:
----------------------------------------
• transactions.amount: Negative values found
• users.email: Invalid email formats detected
```

### Detailed JSON Report
```json
{
  "assessment_timestamp": "2024-08-24T10:30:45",
  "summary": {
    "total_tables": 3,
    "total_columns": 24,
    "total_records": 3006,
    "overall_score": 0.875,
    "dimension_scores": {
      "Completeness": 0.952,
      "Validity": 0.898,
      "Uniqueness": 0.921
    }
  },
  "detailed_results": [...]
}
```

## 🧪 Testing Framework

### Pytest Integration
```python
# Run all data quality tests
pytest my_spec.py -v

# Run specific test classes
pytest my_spec.py::TestDataQualityComprehensive -v

# Run tests for specific tables
pytest my_spec.py::TestDataQualityByTable -v
```

### Custom Test Examples
```python
def test_email_validity():
    \"\"\"Test email format validation\"\"\"
    engine = DataQualityEngine(data_dir, ddl_file)
    data = engine.load_data()
    
    result = engine.validator.validate_email(
        data['users'], 'users', 'email'
    )
    assert result.status == "PASS"

def test_referential_integrity():
    \"\"\"Test foreign key relationships\"\"\"
    engine = DataQualityEngine(data_dir, ddl_file)
    data = engine.load_data()
    
    result = engine.validator.validate_referential_integrity(
        data['users'], data['accounts'], 
        'users', 'accounts', 'id', 'user_id'
    )
    assert result.status == "PASS"
```

## 📋 Available Validations

### Completeness
- `validate_completeness()`: Check for missing values
- Detects: NULL, empty strings, whitespace-only

### Uniqueness  
- `validate_uniqueness()`: Detect duplicates
- `validate_id_unique()`: Primary key uniqueness

### Validity
- `validate_email()`: Email format validation
- `validate_age()`: Age range validation
- `validate_numeric()`: Numeric format validation
- `validate_date_format()`: Date format validation
- `validate_phone()`: Phone number format

### Consistency
- `validate_referential_integrity()`: Foreign key validation
- `validate_data_range_consistency()`: Date/value range validation

### Conformity
- `validate_ddl_compliance()`: Schema compliance
- `validate_constraints()`: NOT NULL, UNIQUE constraints

## ⚙️ Command Line Options

```bash
# Comprehensive assessment
python run_dq_tests.py comprehensive \
    --data-dir /path/to/data \
    --ddl-file schema.sql \
    --output report.json

# Quick validation
python run_dq_tests.py quick --data-dir /path/to/data

# Pytest execution
python run_dq_tests.py pytest --verbose

# HTML report generation
python run_dq_tests.py html --output report.json
```

## 🎨 Extending the Framework

### Add Custom Validators
```python
def validate_custom_rule(df: pd.DataFrame, table_name: str, column: str) -> DataQualityResult:
    \"\"\"Custom validation logic\"\"\"
    # Your validation logic here
    return DataQualityResult(...)

# Add to validator
validator.custom_validators['my_rule'] = validate_custom_rule
```

### Add New Patterns
```python
COLUMN_RULES = {
    r".*phone.*": [validate_phone],
    r".*ssn.*": [validate_ssn],
    r".*custom.*": [validate_custom_rule],
}
```

## 🤝 Contributing

1. Add new validation functions to `DataQualityValidator`
2. Update column patterns in configuration
3. Add corresponding pytest tests
4. Update documentation

## 📞 Support

For issues or questions:
1. Check the configuration in `dq_config.json`
2. Review DDL file format and paths
3. Verify CSV file structure and encoding
4. Check logs for detailed error messages

## 🏷️ Version History

- **v1.0**: Initial comprehensive data quality framework
- **v1.1**: Added DDL compliance and HTML reporting
- **v1.2**: Enhanced pytest integration and configurability

---

*Built for robust, scalable data quality assessment across enterprise data pipelines.*
