#!/usr/bin/env python3
"""
Test runner for AWS Review Cleaner application.

Runs both unit tests and integration tests with proper reporting.
Updated to work with the current functional system architecture.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def run_unit_tests(verbose=False):
    """Run unit tests."""
    print_section("Running Unit Tests")
    
    cmd = [sys.executable, "-m", "pytest", "tests/test_unit.py", "-s"]
    if verbose:
        cmd.append("-v")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        print("Unit Test Output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run unit tests: {e}")
        return False


def run_integration_tests(verbose=False):
    """Run integration tests."""
    print_section("Running Integration Tests")
    
    print("✅ Integration tests updated to work with current system architecture")
    
    cmd = [sys.executable, "-m", "pytest", "tests/test_integration.py", "-s", "--tb=short"]
    if verbose:
        cmd.append("-v")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        print("Integration Test Output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run integration tests: {e}")
        return False


def run_all_tests(verbose=False):
    """Run all tests."""
    print_header("AWS Review Cleaner Test Suite")
    
    start_time = time.time()
    
    # Run unit tests first
    unit_success = run_unit_tests(verbose)
    
    # Run integration tests
    integration_success = run_integration_tests(verbose)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Summary
    print_section("Test Summary")
    print(f"⏱️  Total execution time: {duration:.2f} seconds")
    print(f"📋 Unit Tests: {'✅ PASSED' if unit_success else '❌ FAILED'}")
    print(f"🔗 Integration Tests: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    
    overall_success = unit_success and integration_success
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success


def run_all_tests_together(verbose=False):
    """Run all test files together to ensure no tests are missed."""
    print_section("Running All Tests Together")
    
    cmd = [sys.executable, "-m", "pytest", "tests/", "-s", "--tb=short"]
    if verbose:
        cmd.append("-v")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        print("Combined Test Output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run all tests together: {e}")
        return False


def run_coverage_report():
    """Run tests with coverage reporting."""
    print_section("Running Tests with Coverage")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=src",
        "--cov-report=html:htmlcov",
        "--cov-report=term",
        "tests/"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        if result.returncode == 0:
            print("\n📊 Coverage report generated in htmlcov/ directory")
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run coverage tests: {e}")
        return False


def validate_test_environment():
    """Validate that the test environment is properly set up."""
    print_section("Validating Test Environment")
    
    # Check Python dependencies
    required_packages = ['pytest']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Check source code availability
    src_path = Path(__file__).parent.parent / "src"
    if not src_path.exists():
        print("❌ Source code directory 'src' not found")
        return False
    
    print("✅ Source code directory found")
    
    # Check Lambda functions
    lambda_dir = src_path / "lambdas"
    expected_lambdas = ['preprocess.py', 'profanity_check.py', 'sentiment_analysis.py', 'user_management.py']
    
    for lambda_file in expected_lambdas:
        if (lambda_dir / lambda_file).exists():
            print(f"✅ {lambda_file} found")
        else:
            print(f"❌ {lambda_file} missing")
            return False
    
    # Check shared utilities
    shared_dir = src_path / "shared"
    expected_shared = ['text_utils_simple.py', 'sentiment_utils.py', 'aws_utils.py', 'constants.py']
    
    for shared_file in expected_shared:
        if (shared_dir / shared_file).exists():
            print(f"✅ {shared_file} found")
        else:
            print(f"❌ {shared_file} missing")
            return False
    
    # Check processing results file
    results_file = Path(__file__).parent.parent / "results.json"
    if results_file.exists():
        print("✅ results.json found - dataset validation will work")
    else:
        print("⚠️  results.json not found - dataset validation test will be skipped")
    
    print("\n✅ Test environment validation passed")
    return True


def check_system_functionality():
    """Check if the main system components are functional."""
    print_section("Checking System Functionality")
    
    try:
        # Test basic imports
        from shared.text_utils_simple import preprocess_review
        from shared.sentiment_utils import analyze_sentiment
        from lambdas import preprocess, profanity_check, sentiment_analysis, user_management
        
        print("✅ All main modules can be imported")
        
        # Test basic functionality
        test_summary = "Great product"
        test_review = "This is excellent and works perfectly"
        
        processed = preprocess_review(test_summary, test_review)
        sentiment = analyze_sentiment(processed)
        
        print("✅ Basic text processing and sentiment analysis working")
        
        return True
        
    except Exception as e:
        print(f"❌ System functionality check failed: {e}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="AWS Review Cleaner Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--all", action="store_true", help="Run all tests together")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage reporting")
    parser.add_argument("--validate", action="store_true", help="Validate test environment")
    parser.add_argument("--check", action="store_true", help="Check system functionality")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Default to running all tests together if no specific option provided
    if not any([args.unit, args.integration, args.all, args.coverage, args.validate, args.check]):
        args.all = True
    
    success = True
    
    if args.validate:
        success &= validate_test_environment()
    
    if args.check:
        success &= check_system_functionality()
    
    if args.unit:
        success &= run_unit_tests(args.verbose)
    
    if args.integration:
        success &= run_integration_tests(args.verbose)
    
    if args.all:
        success &= run_all_tests_together(args.verbose)
    
    if args.coverage:
        success &= run_coverage_report()
    
    # Provide summary
    print_section("Final Summary")
    if success:
        print("🎉 All tests completed successfully!")
    else:
        print("❌ Some tests failed. Please review the output above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 