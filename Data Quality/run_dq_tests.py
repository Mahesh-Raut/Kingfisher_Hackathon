#!/usr/bin/env python3
"""
Enhanced Data Quality Test Runner

This script provides multiple ways to run data quality assessments:
1. Comprehensive assessment with detailed reporting
2. Quick validation checks
3. Pytest-based test suite
4. Custom rule validation
"""

import argparse
import sys
import json
from pathlib import Path
from data_quality_engine import DataQualityEngine
import subprocess

def run_comprehensive_assessment(data_dir: str, ddl_file: str = None, output_file: str = None):
    """Run comprehensive data quality assessment"""
    print("🔍 Running Comprehensive Data Quality Assessment...")
    print("=" * 60)
    
    engine = DataQualityEngine(data_dir, ddl_file)
    
    # Print summary to console
    engine.print_summary()
    
    # Generate detailed report
    if output_file:
        report = engine.generate_report(output_file)
        print(f"\n📊 Detailed report saved to: {output_file}")
    else:
        report = engine.generate_report()
        print(f"\n📊 Detailed report saved to: data_quality_report.json")
    
    return report

def run_pytest_suite(data_dir: str, verbose: bool = False):
    """Run pytest-based data quality test suite"""
    print("🧪 Running Pytest Data Quality Test Suite...")
    print("=" * 60)
    
    # Change to data directory for relative imports
    original_dir = Path.cwd()
    try:
        Path(data_dir).exists() and Path(data_dir).is_dir()
        
        # Build pytest command
        cmd = ["python", "-m", "pytest", "my_spec.py"]
        if verbose:
            cmd.append("-v")
        cmd.extend(["--tb=short"])
        
        # Run pytest
        result = subprocess.run(cmd, cwd=data_dir, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running pytest: {e}")
        return False
    finally:
        # Always change back to original directory
        pass

def run_quick_validation(data_dir: str):
    """Run quick validation checks"""
    print("⚡ Running Quick Data Quality Validation...")
    print("=" * 60)
    
    engine = DataQualityEngine(data_dir)
    data = engine.load_data()
    
    if not data:
        print("❌ No data files found!")
        return False
    
    # Quick checks
    total_records = sum(len(df) for df in data.values())
    total_columns = sum(len(df.columns) for df in data.values())
    
    print(f"📋 Found {len(data)} tables with {total_records:,} total records")
    print(f"📊 Total columns: {total_columns}")
    
    # Check for basic issues
    issues = []
    for table_name, df in data.items():
        # Check for empty tables
        if len(df) == 0:
            issues.append(f"❌ {table_name}: Empty table")
        
        # Check for missing columns
        if len(df.columns) == 0:
            issues.append(f"❌ {table_name}: No columns found")
        
        # Check for high null percentage
        null_percentages = df.isnull().mean()
        high_null_cols = null_percentages[null_percentages > 0.5].index.tolist()
        if high_null_cols:
            issues.append(f"⚠️  {table_name}: High null percentage in {high_null_cols}")
        
        # Check for potential duplicates in ID columns
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        for id_col in id_columns:
            if df[id_col].duplicated().any():
                issues.append(f"❌ {table_name}.{id_col}: Contains duplicates")
    
    if issues:
        print("\n🚨 Issues Found:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
        return False
    else:
        print("\n✅ Quick validation passed - no major issues found")
        return True

def generate_html_report(json_report_file: str, html_output_file: str = None):
    """Generate HTML report from JSON results"""
    if not html_output_file:
        html_output_file = json_report_file.replace('.json', '.html')
    
    try:
        with open(json_report_file, 'r') as f:
            report_data = json.load(f)
        
        html_content = generate_html_template(report_data)
        
        with open(html_output_file, 'w') as f:
            f.write(html_content)
        
        print(f"📄 HTML report generated: {html_output_file}")
        return True
        
    except Exception as e:
        print(f"Error generating HTML report: {e}")
        return False

def generate_html_template(report_data):
    """Generate HTML template for data quality report"""
    summary = report_data.get('summary', {})
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Quality Assessment Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .score {{ font-size: 3em; font-weight: bold; margin: 20px 0; }}
            .score.good {{ color: #27ae60; }}
            .score.warning {{ color: #f39c12; }}
            .score.poor {{ color: #e74c3c; }}
            .dimension {{ display: inline-block; margin: 10px; padding: 15px; border-radius: 8px; min-width: 150px; text-align: center; }}
            .dimension.good {{ background: #d5f4e6; border-left: 4px solid #27ae60; }}
            .dimension.warning {{ background: #fef9e7; border-left: 4px solid #f39c12; }}
            .dimension.poor {{ background: #fadbd8; border-left: 4px solid #e74c3c; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
            .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
            .issues {{ margin: 30px 0; }}
            .issue {{ padding: 10px; margin: 5px 0; border-left: 4px solid #e74c3c; background: #fadbd8; border-radius: 4px; }}
            .recommendations {{ margin: 30px 0; }}
            .recommendation {{ padding: 10px; margin: 5px 0; border-left: 4px solid #3498db; background: #ebf3fd; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Data Quality Assessment Report</h1>
                <p>Generated on {report_data.get('assessment_timestamp', 'Unknown')}</p>
                <div class="score {'good' if summary.get('overall_score', 0) >= 0.8 else 'warning' if summary.get('overall_score', 0) >= 0.6 else 'poor'}">
                    {summary.get('overall_score', 0):.1%}
                </div>
                <p>Overall Data Quality Score</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <h3>{summary.get('total_tables', 0)}</h3>
                    <p>Tables Analyzed</p>
                </div>
                <div class="stat-card">
                    <h3>{summary.get('total_columns', 0)}</h3>
                    <p>Columns Checked</p>
                </div>
                <div class="stat-card">
                    <h3>{summary.get('total_records', 0):,}</h3>
                    <p>Records Processed</p>
                </div>
            </div>
            
            <h2>📈 Quality Dimensions</h2>
            <div style="text-align: center;">
    """
    
    # Add dimension scores
    for dimension, score in summary.get('dimension_scores', {}).items():
        css_class = 'good' if score >= 0.8 else 'warning' if score >= 0.6 else 'poor'
        html += f"""
                <div class="dimension {css_class}">
                    <h4>{dimension}</h4>
                    <div style="font-size: 1.5em; font-weight: bold;">{score:.1%}</div>
                </div>
        """
    
    html += """
            </div>
            
            <h2>🚨 Critical Issues</h2>
            <div class="issues">
    """
    
    # Add critical issues
    issues = summary.get('critical_issues', [])
    if issues:
        for issue in issues:
            html += f'<div class="issue">❌ {issue}</div>'
    else:
        html += '<p>✅ No critical issues found!</p>'
    
    html += """
            </div>
            
            <h2>💡 Recommendations</h2>
            <div class="recommendations">
    """
    
    # Add recommendations
    recommendations = summary.get('recommendations', [])
    if recommendations:
        for rec in recommendations:
            html += f'<div class="recommendation">💡 {rec}</div>'
    else:
        html += '<p>✅ No specific recommendations at this time.</p>'
    
    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def main():
    parser = argparse.ArgumentParser(description="Enhanced Data Quality Test Runner")
    parser.add_argument("command", choices=["comprehensive", "pytest", "quick", "html"], 
                       help="Type of assessment to run")
    parser.add_argument("--data-dir", default="/home/mahesh.raut/Documents/Fraud_detection_latest/Data Quality",
                       help="Directory containing CSV data files")
    parser.add_argument("--ddl-file", default="/home/mahesh.raut/Documents/Fraud_detection_latest/Data Quality/bank_ddls.sql",
                       help="DDL file for schema validation")
    parser.add_argument("--output", help="Output file for reports")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        if args.command == "comprehensive":
            report = run_comprehensive_assessment(args.data_dir, args.ddl_file, args.output)
            
            # Also generate HTML report
            json_file = args.output or "data_quality_report.json"
            generate_html_report(json_file)
            
        elif args.command == "pytest":
            success = run_pytest_suite(args.data_dir, args.verbose)
            sys.exit(0 if success else 1)
            
        elif args.command == "quick":
            success = run_quick_validation(args.data_dir)
            sys.exit(0 if success else 1)
            
        elif args.command == "html":
            if not args.output:
                print("Error: --output required for HTML generation")
                sys.exit(1)
            success = generate_html_report(args.output)
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Assessment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
