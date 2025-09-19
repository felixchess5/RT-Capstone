#!/usr/bin/env python3
"""
RT-Capstone Test Runner

A comprehensive test runner script for the RT-Capstone project with advanced features:
- Test category selection
- Coverage reporting
- Performance monitoring
- Parallel execution
- Result formatting
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import json


class TestRunner:
    """Advanced test runner for RT-Capstone project."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def run_command(self, command: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run a shell command with error handling."""
        try:
            print(f"🔄 Running: {' '.join(command)}")
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                check=False
            )
            return result
        except Exception as e:
            print(f"❌ Error running command: {e}")
            sys.exit(1)

    def run_tests(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Run tests based on provided arguments."""
        start_time = time.time()

        # Build pytest command
        pytest_cmd = ["pytest"]

        # Add test paths
        if args.unit:
            pytest_cmd.append("tests/unit/")
        elif args.integration:
            pytest_cmd.append("tests/integration/")
        elif args.e2e:
            pytest_cmd.append("tests/e2e/")
        else:
            pytest_cmd.append("tests/")

        # Add verbosity
        if args.verbose:
            pytest_cmd.extend(["-v"])
        elif args.quiet:
            pytest_cmd.extend(["-q"])

        # Add markers
        if args.markers:
            pytest_cmd.extend(["-m", args.markers])

        # Add parallel execution
        if args.parallel:
            pytest_cmd.extend(["-n", "auto"])

        # Add coverage
        if args.coverage:
            pytest_cmd.extend([
                "--cov=src",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing"
            ])

            if args.coverage_xml:
                pytest_cmd.extend(["--cov-report=xml"])

        # Add HTML report
        if args.html:
            pytest_cmd.extend([
                "--html=reports/pytest_report.html",
                "--self-contained-html"
            ])

        # Add JSON report
        if args.json:
            pytest_cmd.extend([
                "--json-report",
                "--json-report-file=reports/pytest_report.json"
            ])

        # Add fail-fast
        if args.fail_fast:
            pytest_cmd.extend(["-x"])

        # Add max failures
        if args.maxfail:
            pytest_cmd.extend([["--maxfail", str(args.maxfail)])

        # Add durations
        if args.durations:
            pytest_cmd.extend(["--durations", str(args.durations)])

        # Add timeout
        if args.timeout:
            pytest_cmd.extend(["--timeout", str(args.timeout)])

        # Add keyword filtering
        if args.keyword:
            pytest_cmd.extend(["-k", args.keyword])

        # Add custom pytest arguments
        if args.pytest_args:
            pytest_cmd.extend(args.pytest_args.split())

        # Set environment variables
        env = os.environ.copy()
        env.update({
            "TESTING": "true",
            "GROQ_API_KEY": "test_key",
            "LANGCHAIN_TRACING_V2": "false"
        })

        # Run pytest
        print("🧪 Starting test execution...")
        print(f"📋 Command: {' '.join(pytest_cmd)}")
        print("=" * 80)

        result = subprocess.run(pytest_cmd, cwd=self.project_root, env=env)

        end_time = time.time()
        duration = end_time - start_time

        # Collect results
        test_results = {
            "exit_code": result.returncode,
            "duration": duration,
            "success": result.returncode == 0,
            "command": ' '.join(pytest_cmd)
        }

        return test_results

    def run_linting(self) -> Dict[str, Any]:
        """Run code quality checks."""
        print("🔍 Running code quality checks...")

        linting_results = {}

        # Black formatting check
        print("🖤 Checking code formatting with Black...")
        black_result = self.run_command(["black", "--check", "--diff", "src", "tests"], capture_output=True)
        linting_results["black"] = {
            "success": black_result.returncode == 0,
            "output": black_result.stdout + black_result.stderr
        }

        # isort import sorting check
        print("📋 Checking import sorting with isort...")
        isort_result = self.run_command(["isort", "--check-only", "--diff", "src", "tests"], capture_output=True)
        linting_results["isort"] = {
            "success": isort_result.returncode == 0,
            "output": isort_result.stdout + isort_result.stderr
        }

        # flake8 linting
        print("🔍 Running flake8 linting...")
        flake8_result = self.run_command([
            "flake8", "src", "tests",
            "--count", "--statistics",
            "--max-line-length=88",
            "--extend-ignore=E203,W503"
        ], capture_output=True)
        linting_results["flake8"] = {
            "success": flake8_result.returncode == 0,
            "output": flake8_result.stdout + flake8_result.stderr
        }

        return linting_results

    def run_security_checks(self) -> Dict[str, Any]:
        """Run security vulnerability checks."""
        print("🔒 Running security checks...")

        security_results = {}

        # Bandit security check
        print("🛡️ Running Bandit security scan...")
        bandit_result = self.run_command([
            "bandit", "-r", "src/",
            "-f", "json",
            "-o", "reports/bandit_report.json"
        ], capture_output=True)
        security_results["bandit"] = {
            "success": bandit_result.returncode == 0,
            "report_file": "reports/bandit_report.json"
        }

        # Safety dependency check
        print("🔍 Checking dependencies with Safety...")
        safety_result = self.run_command([
            "safety", "check",
            "--json",
            "--output", "reports/safety_report.json"
        ], capture_output=True)
        security_results["safety"] = {
            "success": safety_result.returncode == 0,
            "report_file": "reports/safety_report.json"
        }

        return security_results

    def generate_report(self, test_results: Dict[str, Any], lint_results: Dict[str, Any] = None,
                       security_results: Dict[str, Any] = None) -> None:
        """Generate comprehensive test report."""
        print("📊 Generating test report...")

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_results": test_results,
            "linting_results": lint_results,
            "security_results": security_results,
            "summary": {
                "tests_passed": test_results["success"],
                "duration": f"{test_results['duration']:.2f}s",
                "exit_code": test_results["exit_code"]
            }
        }

        # Save JSON report
        report_file = self.reports_dir / "test_summary.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)

        if test_results["success"]:
            print("✅ Tests: PASSED")
        else:
            print("❌ Tests: FAILED")

        print(f"⏱️  Duration: {test_results['duration']:.2f} seconds")

        if lint_results:
            lint_passed = all(result["success"] for result in lint_results.values())
            if lint_passed:
                print("✅ Linting: PASSED")
            else:
                print("❌ Linting: FAILED")

        if security_results:
            security_passed = all(result["success"] for result in security_results.values())
            if security_passed:
                print("✅ Security: PASSED")
            else:
                print("❌ Security: ISSUES FOUND")

        print(f"📁 Reports saved to: {self.reports_dir}")
        print("=" * 80)

    def install_dependencies(self) -> None:
        """Install testing dependencies."""
        print("📦 Installing testing dependencies...")

        dependencies = [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "pytest-asyncio>=0.23.0",
            "pytest-html>=4.1.0",
            "pytest-json-report>=1.5.0",
            "pytest-xdist>=3.5.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "bandit>=1.7.0",
            "safety>=2.3.0"
        ]

        for dep in dependencies:
            result = self.run_command(["pip", "install", dep])
            if result.returncode != 0:
                print(f"❌ Failed to install {dep}")
                sys.exit(1)

        print("✅ All dependencies installed successfully")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="RT-Capstone Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit             # Run only unit tests
  python run_tests.py --coverage         # Run with coverage
  python run_tests.py --lint             # Run linting only
  python run_tests.py --security         # Run security checks
  python run_tests.py --all              # Run everything
  python run_tests.py --install          # Install dependencies
        """
    )

    # Test category selection
    test_group = parser.add_argument_group("Test Categories")
    test_group.add_argument("--unit", action="store_true", help="Run unit tests only")
    test_group.add_argument("--integration", action="store_true", help="Run integration tests only")
    test_group.add_argument("--e2e", action="store_true", help="Run end-to-end tests only")

    # Test execution options
    exec_group = parser.add_argument_group("Execution Options")
    exec_group.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    exec_group.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    exec_group.add_argument("--maxfail", type=int, help="Maximum number of failures")
    exec_group.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    exec_group.add_argument("--durations", type=int, default=10, help="Show slowest N tests")

    # Filtering options
    filter_group = parser.add_argument_group("Filtering Options")
    filter_group.add_argument("--markers", help="Run tests with specific markers")
    filter_group.add_argument("--keyword", help="Run tests matching keyword")
    filter_group.add_argument("--pytest-args", help="Additional pytest arguments")

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--verbose", action="store_true", help="Verbose output")
    output_group.add_argument("--quiet", action="store_true", help="Quiet output")
    output_group.add_argument("--coverage", action="store_true", help="Generate coverage report")
    output_group.add_argument("--coverage-xml", action="store_true", help="Generate XML coverage")
    output_group.add_argument("--html", action="store_true", help="Generate HTML report")
    output_group.add_argument("--json", action="store_true", help="Generate JSON report")

    # Quality checks
    quality_group = parser.add_argument_group("Quality Checks")
    quality_group.add_argument("--lint", action="store_true", help="Run linting checks")
    quality_group.add_argument("--security", action="store_true", help="Run security checks")
    quality_group.add_argument("--all", action="store_true", help="Run tests, linting, and security")

    # Utility options
    util_group = parser.add_argument_group("Utilities")
    util_group.add_argument("--install", action="store_true", help="Install testing dependencies")
    util_group.add_argument("--clean", action="store_true", help="Clean test artifacts")

    args = parser.parse_args()

    runner = TestRunner()

    # Handle utility commands
    if args.install:
        runner.install_dependencies()
        return

    if args.clean:
        print("🧹 Cleaning test artifacts...")
        runner.run_command(["rm", "-rf", ".pytest_cache", "htmlcov", "reports", ".coverage"])
        print("✅ Cleanup complete")
        return

    # Run tests and checks
    test_results = None
    lint_results = None
    security_results = None

    try:
        if args.all or (not args.lint and not args.security):
            test_results = runner.run_tests(args)

        if args.lint or args.all:
            lint_results = runner.run_linting()

        if args.security or args.all:
            security_results = runner.run_security_checks()

        # Generate comprehensive report
        if test_results or lint_results or security_results:
            runner.generate_report(test_results, lint_results, security_results)

        # Exit with appropriate code
        if test_results and not test_results["success"]:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()