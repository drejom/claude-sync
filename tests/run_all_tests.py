#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-asyncio>=0.21.0",
#   "pytest-benchmark>=4.0.0",
#   "psutil>=5.9.0",
#   "cryptography>=41.0.0",
#   "typing-extensions>=4.0.0"
# ]
# ///
"""
Comprehensive Test Runner for Claude-Sync

Automated test execution framework that runs all test categories:
- Unit tests for individual components
- Integration tests for component interactions  
- Performance tests with benchmarking
- Security validation tests
- End-to-end workflow tests
- Error scenario testing
- Cross-platform compatibility tests

Provides detailed reporting, performance analysis, and quality gates for production readiness.
"""

import sys
import os
import time
import json
import platform
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import concurrent.futures

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_framework import TestFramework, TestSession, TestEnvironment
from interfaces import PerformanceTargets

# ============================================================================
# Test Runner Configuration
# ============================================================================

@dataclass
class TestConfiguration:
    """Test execution configuration"""
    test_categories: List[str] = field(default_factory=lambda: ["unit", "integration", "performance"])
    parallel_execution: bool = True
    max_workers: int = 4
    performance_benchmarking: bool = True
    generate_reports: bool = True
    output_directory: Path = field(default_factory=lambda: Path.cwd() / "test_results")
    verbose: bool = False
    quick_mode: bool = False  # Skip slow tests
    
    def __post_init__(self):
        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)

@dataclass
class TestCategoryResult:
    """Results from a test category"""
    category: str
    success_rate: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_time_ms: float
    performance_violations: int
    errors: List[str] = field(default_factory=list)

@dataclass
class ComprehensiveTestReport:
    """Comprehensive test execution report"""
    session_id: str
    start_time: float
    end_time: float
    configuration: TestConfiguration
    system_info: Dict[str, Any]
    category_results: List[TestCategoryResult] = field(default_factory=list)
    overall_success_rate: float = 0.0
    quality_gates_passed: bool = False
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def total_execution_time_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    @property
    def total_tests(self) -> int:
        return sum(result.total_tests for result in self.category_results)
    
    @property
    def total_passed(self) -> int:
        return sum(result.passed_tests for result in self.category_results)
    
    @property
    def total_failed(self) -> int:
        return sum(result.failed_tests for result in self.category_results)

# ============================================================================
# Comprehensive Test Runner
# ============================================================================

class ComprehensiveTestRunner:
    """Main test runner that orchestrates all test categories"""
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.project_root = Path(__file__).parent.parent
        self.test_dir = Path(__file__).parent
        
        # System information
        self.system_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'hostname': platform.node(),
            'architecture': platform.machine()
        }
        
        # Add memory info if psutil available
        try:
            import psutil
            self.system_info.update({
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'disk_free_gb': psutil.disk_usage(str(self.project_root)).free / (1024**3)
            })
        except ImportError:
            pass
    
    def run_all_tests(self) -> ComprehensiveTestReport:
        """Execute all configured test categories"""
        session_id = f"comprehensive_test_{int(time.time())}"
        start_time = time.time()
        
        print("🚀 Claude-Sync Comprehensive Test Suite")
        print("=" * 80)
        print(f"Session ID: {session_id}")
        print(f"Platform: {self.system_info['platform']} {self.system_info['platform_release']}")
        print(f"Python: {self.system_info['python_version']}")
        print(f"Test Categories: {', '.join(self.config.test_categories)}")
        print(f"Parallel Execution: {self.config.parallel_execution}")
        print("=" * 80)
        
        category_results = []
        
        if self.config.parallel_execution:
            category_results = self._run_tests_parallel()
        else:
            category_results = self._run_tests_sequential()
        
        end_time = time.time()
        
        # Create comprehensive report
        report = ComprehensiveTestReport(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            configuration=self.config,
            system_info=self.system_info,
            category_results=category_results
        )
        
        # Calculate overall metrics
        if report.total_tests > 0:
            report.overall_success_rate = report.total_passed / report.total_tests
        
        # Evaluate quality gates
        report.quality_gates_passed = self._evaluate_quality_gates(report)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        # Print and save report
        self._print_comprehensive_report(report)
        
        if self.config.generate_reports:
            self._save_report(report)
        
        return report
    
    def _run_tests_parallel(self) -> List[TestCategoryResult]:
        """Run test categories in parallel"""
        category_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all test categories
            future_to_category = {}
            
            for category in self.config.test_categories:
                future = executor.submit(self._run_test_category, category)
                future_to_category[future] = category
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_category):
                category = future_to_category[future]
                try:
                    result = future.result()
                    if result:
                        category_results.append(result)
                        print(f"✅ {category.title()} tests completed: {result.success_rate:.1%} success rate")
                    else:
                        print(f"❌ {category.title()} tests failed to execute")
                except Exception as e:
                    print(f"❌ {category.title()} tests failed with exception: {e}")
        
        return category_results
    
    def _run_tests_sequential(self) -> List[TestCategoryResult]:
        """Run test categories sequentially"""
        category_results = []
        
        for category in self.config.test_categories:
            print(f"\n📋 Running {category.title()} Tests...")
            print("-" * 50)
            
            result = self._run_test_category(category)
            if result:
                category_results.append(result)
                print(f"✅ {category.title()} tests completed: {result.success_rate:.1%} success rate")
            else:
                print(f"❌ {category.title()} tests failed to execute")
        
        return category_results
    
    def _run_test_category(self, category: str) -> Optional[TestCategoryResult]:
        """Run a specific test category"""
        try:
            category_start_time = time.time()
            
            # Map category to test script
            test_scripts = {
                'unit': 'test_unit_components.py',
                'integration': 'test_integration.py', 
                'performance': 'test_performance.py',
                'security': 'test_security_validation.py',
                'end_to_end': 'test_end_to_end.py',
                'error_scenarios': 'test_error_scenarios.py',
                'cross_platform': 'test_cross_platform.py'
            }
            
            script_name = test_scripts.get(category)
            if not script_name:
                return None
            
            script_path = self.test_dir / script_name
            if not script_path.exists():
                print(f"⚠️ Test script not found: {script_path}")
                return None
            
            # Set up environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            env['CLAUDE_SYNC_TEST_MODE'] = '1'
            
            if self.config.quick_mode:
                env['CLAUDE_SYNC_QUICK_TEST'] = '1'
            
            # Execute test script
            process = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                timeout=300 if not self.config.quick_mode else 60,  # 5 min timeout, 1 min for quick
                env=env
            )
            
            category_end_time = time.time()
            execution_time_ms = (category_end_time - category_start_time) * 1000
            
            # Parse output for test results
            success_rate, total_tests, passed_tests, failed_tests, performance_violations = self._parse_test_output(
                process.stdout.decode(), process.stderr.decode()
            )
            
            errors = []
            if process.returncode != 0:
                errors.append(f"Test script exited with code {process.returncode}")
                errors.append(process.stderr.decode()[-500:])  # Last 500 chars of stderr
            
            return TestCategoryResult(
                category=category,
                success_rate=success_rate,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                execution_time_ms=execution_time_ms,
                performance_violations=performance_violations,
                errors=errors
            )
            
        except subprocess.TimeoutExpired:
            return TestCategoryResult(
                category=category,
                success_rate=0.0,
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                execution_time_ms=300000,  # 5 minutes
                performance_violations=0,
                errors=[f"{category.title()} tests timed out"]
            )
        except Exception as e:
            return TestCategoryResult(
                category=category,
                success_rate=0.0,
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                execution_time_ms=0,
                performance_violations=0,
                errors=[f"{category.title()} tests failed: {str(e)}"]
            )
    
    def _parse_test_output(self, stdout: str, stderr: str) -> Tuple[float, int, int, int, int]:
        """Parse test output to extract metrics"""
        success_rate = 0.0
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        performance_violations = 0
        
        # Look for test result patterns
        lines = stdout.split('\n') + stderr.split('\n')
        
        for line in lines:
            # Look for success rate patterns
            if 'Success Rate:' in line:
                try:
                    rate_str = line.split('Success Rate:')[1].strip().rstrip('%')
                    success_rate = float(rate_str) / 100
                except (IndexError, ValueError):
                    pass
            
            # Look for test count patterns
            if 'tests' in line.lower() and ('passed' in line.lower() or 'failed' in line.lower()):
                # Extract numbers from patterns like "15 tests | 12 passed | 3 failed"
                words = line.split()
                for i, word in enumerate(words):
                    if word.isdigit():
                        num = int(word)
                        if i < len(words) - 1:
                            next_word = words[i + 1].lower()
                            if 'test' in next_word:
                                total_tests = max(total_tests, num)
                            elif 'passed' in next_word:
                                passed_tests = max(passed_tests, num)
                            elif 'failed' in next_word:
                                failed_tests = max(failed_tests, num)
            
            # Look for performance violations
            if 'Performance Violations:' in line or 'performance violations' in line.lower():
                try:
                    # Extract number from line
                    words = line.split()
                    for word in words:
                        if word.isdigit():
                            performance_violations = max(performance_violations, int(word))
                            break
                except ValueError:
                    pass
        
        # Ensure consistency
        if total_tests == 0 and (passed_tests > 0 or failed_tests > 0):
            total_tests = passed_tests + failed_tests
        
        if total_tests > 0 and success_rate == 0.0:
            success_rate = passed_tests / total_tests
        
        return success_rate, total_tests, passed_tests, failed_tests, performance_violations
    
    def _evaluate_quality_gates(self, report: ComprehensiveTestReport) -> bool:
        """Evaluate quality gates for production readiness"""
        # Quality gate criteria
        MIN_OVERALL_SUCCESS_RATE = 0.80  # 80% minimum
        MAX_PERFORMANCE_VIOLATIONS_PERCENT = 0.10  # 10% max
        CRITICAL_CATEGORIES = ['unit', 'integration', 'security']
        MIN_CRITICAL_SUCCESS_RATE = 0.85  # 85% for critical categories
        
        # Check overall success rate
        if report.overall_success_rate < MIN_OVERALL_SUCCESS_RATE:
            return False
        
        # Check performance violations
        total_violations = sum(result.performance_violations for result in report.category_results)
        if report.total_tests > 0:
            violation_rate = total_violations / report.total_tests
            if violation_rate > MAX_PERFORMANCE_VIOLATIONS_PERCENT:
                return False
        
        # Check critical categories
        for result in report.category_results:
            if result.category in CRITICAL_CATEGORIES:
                if result.success_rate < MIN_CRITICAL_SUCCESS_RATE:
                    return False
        
        return True
    
    def _generate_recommendations(self, report: ComprehensiveTestReport) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Overall success rate recommendations
        if report.overall_success_rate < 0.8:
            recommendations.append("🚨 Overall success rate below 80% - investigate failing tests before production deployment")
        
        # Category-specific recommendations
        for result in report.category_results:
            if result.success_rate < 0.7:
                recommendations.append(f"⚠️ {result.category.title()} tests below 70% success - requires immediate attention")
            
            if result.performance_violations > 0:
                recommendations.append(f"⚡ {result.category.title()} has {result.performance_violations} performance violations - optimize before deployment")
            
            if result.errors:
                recommendations.append(f"🐛 {result.category.title()} tests have errors - check logs for details")
        
        # Performance recommendations
        total_violations = sum(result.performance_violations for result in report.category_results)
        if total_violations > 0:
            recommendations.append(f"🎯 {total_violations} total performance violations detected - review PerformanceTargets compliance")
        
        # System-specific recommendations
        if 'memory_gb' in report.system_info and report.system_info['memory_gb'] < 8:
            recommendations.append("💾 System memory below 8GB - some performance tests may be unreliable")
        
        if report.total_execution_time_ms > 300000:  # 5 minutes
            recommendations.append("⏱️ Test execution time over 5 minutes - consider parallel execution or quick mode")
        
        # Success recommendations
        if not recommendations:
            recommendations.append("✅ All tests passed quality gates - system ready for production deployment")
            recommendations.append("🚀 Consider running end-to-end tests in staging environment")
        
        return recommendations
    
    def _print_comprehensive_report(self, report: ComprehensiveTestReport) -> None:
        """Print comprehensive test report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        print(f"Session: {report.session_id}")
        print(f"Duration: {report.total_execution_time_ms/1000:.1f}s")
        print(f"System: {report.system_info['platform']} | Python {report.system_info['python_version']} | {report.system_info.get('cpu_count', 'N/A')} cores")
        
        print(f"\n📈 OVERALL RESULTS:")
        print(f"  • Total Tests: {report.total_tests}")
        print(f"  • Passed: {report.total_passed}")
        print(f"  • Failed: {report.total_failed}")
        print(f"  • Success Rate: {report.overall_success_rate:.1%}")
        print(f"  • Quality Gates: {'✅ PASSED' if report.quality_gates_passed else '❌ FAILED'}")
        
        print(f"\n📋 CATEGORY BREAKDOWN:")
        for result in report.category_results:
            status = "✅" if result.success_rate >= 0.8 else "⚠️" if result.success_rate >= 0.6 else "❌"
            print(f"  {status} {result.category.title()}: {result.success_rate:.1%} ({result.passed_tests}/{result.total_tests}) | {result.execution_time_ms/1000:.1f}s")
            
            if result.performance_violations > 0:
                print(f"    ⚡ {result.performance_violations} performance violations")
            
            if result.errors:
                print(f"    🐛 {len(result.errors)} errors")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, recommendation in enumerate(report.recommendations, 1):
            print(f"  {i}. {recommendation}")
        
        print("\n" + "=" * 80)
        
        # Final verdict
        if report.quality_gates_passed:
            print("🎉 VERDICT: System ready for production deployment!")
        else:
            print("🚨 VERDICT: System requires fixes before production deployment!")
        
        print("=" * 80)
    
    def _save_report(self, report: ComprehensiveTestReport) -> None:
        """Save comprehensive report to file"""
        try:
            # Save JSON report
            json_path = self.config.output_directory / f"test_report_{report.session_id}.json"
            
            # Convert report to serializable format
            report_dict = {
                'session_id': report.session_id,
                'start_time': report.start_time,
                'end_time': report.end_time,
                'total_execution_time_ms': report.total_execution_time_ms,
                'system_info': report.system_info,
                'configuration': {
                    'test_categories': report.configuration.test_categories,
                    'parallel_execution': report.configuration.parallel_execution,
                    'performance_benchmarking': report.configuration.performance_benchmarking,
                    'quick_mode': report.configuration.quick_mode
                },
                'overall_success_rate': report.overall_success_rate,
                'total_tests': report.total_tests,
                'total_passed': report.total_passed,
                'total_failed': report.total_failed,
                'quality_gates_passed': report.quality_gates_passed,
                'category_results': [
                    {
                        'category': result.category,
                        'success_rate': result.success_rate,
                        'total_tests': result.total_tests,
                        'passed_tests': result.passed_tests,
                        'failed_tests': result.failed_tests,
                        'execution_time_ms': result.execution_time_ms,
                        'performance_violations': result.performance_violations,
                        'errors': result.errors
                    }
                    for result in report.category_results
                ],
                'recommendations': report.recommendations
            }
            
            with open(json_path, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            print(f"📄 Test report saved: {json_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save report: {e}")

# ============================================================================
# Command Line Interface
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for Claude-Sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_tests.py                          # Run all tests
  python run_all_tests.py --categories unit perf   # Run only unit and performance tests  
  python run_all_tests.py --quick                  # Quick test mode
  python run_all_tests.py --sequential             # Run tests sequentially
  python run_all_tests.py --output ./results       # Custom output directory
        """
    )
    
    parser.add_argument(
        '--categories',
        nargs='*',
        choices=['unit', 'integration', 'performance', 'security', 'end_to_end', 'error_scenarios', 'cross_platform'],
        default=['unit', 'integration', 'performance'],
        help='Test categories to run (default: unit integration performance)'
    )
    
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run test categories sequentially instead of in parallel'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode - skip slow tests'
    )
    
    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='Do not generate test reports'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path.cwd() / 'test_results',
        help='Output directory for test reports (default: ./test_results)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser

def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = TestConfiguration(
        test_categories=args.categories,
        parallel_execution=not args.sequential,
        max_workers=args.workers,
        generate_reports=not args.no_reports,
        output_directory=args.output,
        verbose=args.verbose,
        quick_mode=args.quick
    )
    
    # Create and run comprehensive test runner
    runner = ComprehensiveTestRunner(config)
    report = runner.run_all_tests()
    
    # Exit with appropriate code
    if report.quality_gates_passed:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())