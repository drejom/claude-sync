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
Claude-Sync Testing Framework

Comprehensive testing framework for validating all integrated components:
- Hook system validation
- Learning infrastructure testing
- Security system verification
- Bootstrap process validation
- Performance benchmarking
- Integration testing
"""

import json
import time
import tempfile
import shutil
import sys
import os
import platform
import subprocess
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
import queue
import psutil

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from interfaces import (
    HookResult, CommandExecutionData, PerformanceTargets, 
    SystemState, ActivationResult, validate_hook_result
)

# ============================================================================
# Test Framework Core
# ============================================================================

@dataclass
class TestResult:
    """Standard test result structure"""
    test_name: str
    passed: bool
    execution_time_ms: float
    memory_used_mb: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def meets_performance_target(self, target_ms: float) -> bool:
        """Check if test meets performance target"""
        return self.execution_time_ms <= target_ms

@dataclass
class TestSuite:
    """Collection of related tests"""
    name: str
    tests: List[Callable] = field(default_factory=list)
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None

@dataclass
class TestSession:
    """Complete test session results"""
    session_id: str
    start_time: float
    end_time: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    performance_violations: int
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate"""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def duration_ms(self) -> float:
        """Total session duration in milliseconds"""
        return (self.end_time - self.start_time) * 1000

class TestFramework:
    """Main testing framework orchestrator"""
    
    def __init__(self, test_dir: Optional[Path] = None):
        self.test_dir = test_dir or Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.temp_dir: Optional[Path] = None
        self.test_suites: Dict[str, TestSuite] = {}
        self.current_session: Optional[TestSession] = None
        
        # System information
        self.system_info = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'disk_space_gb': shutil.disk_usage(str(self.project_root)).free / (1024**3)
        }
    
    def register_test_suite(self, suite: TestSuite) -> None:
        """Register a test suite"""
        self.test_suites[suite.name] = suite
    
    def create_test_environment(self) -> Path:
        """Create isolated test environment"""
        if self.temp_dir:
            return self.temp_dir
        
        self.temp_dir = Path(tempfile.mkdtemp(prefix="claude_sync_test_"))
        
        # Create directory structure
        (self.temp_dir / "learning").mkdir()
        (self.temp_dir / "security").mkdir()
        (self.temp_dir / "hooks").mkdir()
        (self.temp_dir / "backups").mkdir()
        
        return self.temp_dir
    
    def cleanup_test_environment(self) -> None:
        """Clean up test environment"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def run_single_test(self, test_func: Callable, test_name: str) -> TestResult:
        """Execute a single test with performance monitoring"""
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        start_time = time.perf_counter()
        
        try:
            # Execute test
            result = test_func()
            
            # Calculate performance metrics
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_used_mb = max(0, end_memory - start_memory)
            
            # Handle different result types
            if isinstance(result, bool):
                return TestResult(
                    test_name=test_name,
                    passed=result,
                    execution_time_ms=execution_time_ms,
                    memory_used_mb=memory_used_mb,
                    message="Test completed" if result else "Test failed"
                )
            elif isinstance(result, tuple) and len(result) >= 2:
                passed, message = result[0], result[1]
                details = result[2] if len(result) > 2 else {}
                return TestResult(
                    test_name=test_name,
                    passed=passed,
                    execution_time_ms=execution_time_ms,
                    memory_used_mb=memory_used_mb,
                    message=message,
                    details=details
                )
            else:
                return TestResult(
                    test_name=test_name,
                    passed=True,
                    execution_time_ms=execution_time_ms,
                    memory_used_mb=memory_used_mb,
                    message="Test completed successfully"
                )
                
        except Exception as e:
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / (1024 * 1024)
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_used_mb = max(0, end_memory - start_memory)
            
            return TestResult(
                test_name=test_name,
                passed=False,
                execution_time_ms=execution_time_ms,
                memory_used_mb=memory_used_mb,
                message=f"Test failed with exception: {str(e)}",
                details={'exception': str(e), 'traceback': traceback.format_exc()}
            )
    
    def run_test_suite(self, suite_name: str) -> List[TestResult]:
        """Run all tests in a suite"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite '{suite_name}' not found")
        
        suite = self.test_suites[suite_name]
        results = []
        
        print(f"\n🧪 Running test suite: {suite_name}")
        print("=" * 60)
        
        # Run setup if available
        if suite.setup_func:
            try:
                suite.setup_func()
                print(f"✅ Setup completed for {suite_name}")
            except Exception as e:
                print(f"❌ Setup failed for {suite_name}: {e}")
                return results
        
        # Run tests
        for test_func in suite.tests:
            test_name = getattr(test_func, '__name__', str(test_func))
            print(f"🔄 Running {test_name}...", end=" ")
            
            result = self.run_single_test(test_func, test_name)
            results.append(result)
            
            # Print result
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} ({result.execution_time_ms:.1f}ms, {result.memory_used_mb:.1f}MB)")
            
            if result.message and not result.passed:
                print(f"   └─ {result.message}")
        
        # Run teardown if available
        if suite.teardown_func:
            try:
                suite.teardown_func()
                print(f"✅ Teardown completed for {suite_name}")
            except Exception as e:
                print(f"⚠️ Teardown had issues for {suite_name}: {e}")
        
        return results
    
    def run_all_tests(self) -> TestSession:
        """Run all registered test suites"""
        session_id = f"test_session_{int(time.time())}"
        start_time = time.time()
        
        print(f"\n🚀 Starting comprehensive test session: {session_id}")
        print(f"📊 System: {self.system_info['platform']} | Python: {self.system_info['python_version']} | CPU: {self.system_info['cpu_count']} cores | RAM: {self.system_info['memory_gb']:.1f}GB")
        print("=" * 80)
        
        all_results = []
        
        # Create test environment
        try:
            self.create_test_environment()
            print(f"🔧 Test environment created: {self.temp_dir}")
            
            # Run all suites
            for suite_name in self.test_suites:
                suite_results = self.run_test_suite(suite_name)
                all_results.extend(suite_results)
        
        finally:
            # Cleanup
            self.cleanup_test_environment()
            print(f"🧹 Test environment cleaned up")
        
        end_time = time.time()
        
        # Calculate session statistics
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.passed)
        failed_tests = total_tests - passed_tests
        performance_violations = sum(
            1 for r in all_results 
            if hasattr(r, 'meets_performance_target') and not r.meets_performance_target(100)
        )
        
        session = TestSession(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=0,
            performance_violations=performance_violations,
            results=all_results
        )
        
        self.current_session = session
        self.print_session_summary(session)
        
        return session
    
    def print_session_summary(self, session: TestSession) -> None:
        """Print comprehensive session summary"""
        print("\n" + "=" * 80)
        print(f"📋 TEST SESSION SUMMARY: {session.session_id}")
        print("=" * 80)
        
        print(f"⏱️  Duration: {session.duration_ms:.0f}ms")
        print(f"📊 Tests: {session.total_tests} total | {session.passed_tests} passed | {session.failed_tests} failed")
        print(f"✅ Success Rate: {session.success_rate:.1%}")
        print(f"⚡ Performance Violations: {session.performance_violations}")
        
        # Performance breakdown
        if session.results:
            avg_time = sum(r.execution_time_ms for r in session.results) / len(session.results)
            max_time = max(r.execution_time_ms for r in session.results)
            avg_memory = sum(r.memory_used_mb for r in session.results) / len(session.results)
            max_memory = max(r.memory_used_mb for r in session.results)
            
            print(f"📈 Performance: Avg {avg_time:.1f}ms | Max {max_time:.1f}ms | Avg {avg_memory:.1f}MB | Max {max_memory:.1f}MB")
        
        # Failed test details
        failed_tests = [r for r in session.results if not r.passed]
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  • {test.test_name}: {test.message}")
        
        # Performance violations
        slow_tests = [r for r in session.results if r.execution_time_ms > 100]
        if slow_tests:
            print(f"\n⚡ SLOW TESTS (>100ms):")
            for test in slow_tests:
                print(f"  • {test.test_name}: {test.execution_time_ms:.1f}ms")
        
        print("=" * 80)

# ============================================================================
# Performance Benchmarking
# ============================================================================

class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    @staticmethod
    def measure_execution_time(func: Callable, iterations: int = 1) -> Dict[str, float]:
        """Measure function execution time statistics"""
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        times.sort()
        
        return {
            'min_ms': min(times),
            'max_ms': max(times),
            'avg_ms': sum(times) / len(times),
            'median_ms': times[len(times) // 2],
            'p95_ms': times[int(len(times) * 0.95)],
            'p99_ms': times[int(len(times) * 0.99)]
        }
    
    @staticmethod
    def validate_performance_targets(results: List[TestResult]) -> Dict[str, Any]:
        """Validate test results against performance targets"""
        validation_results = {
            'total_tests': len(results),
            'performance_compliant': 0,
            'violations': [],
            'target_compliance': {}
        }
        
        # Define expected targets for different test types
        target_map = {
            'hook_': PerformanceTargets.PRE_TOOL_USE_HOOK_MS,  # Most restrictive
            'learning_': PerformanceTargets.LEARNING_OPERATION_MS,
            'security_': PerformanceTargets.ENCRYPTION_OPERATION_MS,
            'integration_': 200,  # Allow more time for integration tests
            'end_to_end_': 1000   # Allow most time for full workflows
        }
        
        for result in results:
            # Determine target based on test name
            target_ms = 100  # Default target
            for prefix, target in target_map.items():
                if result.test_name.startswith(prefix):
                    target_ms = target
                    break
            
            # Check compliance
            compliant = result.execution_time_ms <= target_ms
            if compliant:
                validation_results['performance_compliant'] += 1
            else:
                validation_results['violations'].append({
                    'test_name': result.test_name,
                    'actual_ms': result.execution_time_ms,
                    'target_ms': target_ms,
                    'violation_ratio': result.execution_time_ms / target_ms
                })
            
            # Track by category
            category = next((prefix for prefix in target_map if result.test_name.startswith(prefix)), 'general_')
            if category not in validation_results['target_compliance']:
                validation_results['target_compliance'][category] = {'compliant': 0, 'total': 0}
            
            validation_results['target_compliance'][category]['total'] += 1
            if compliant:
                validation_results['target_compliance'][category]['compliant'] += 1
        
        return validation_results

# ============================================================================
# Test Environment Setup
# ============================================================================

class TestEnvironment:
    """Isolated test environment manager"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.test_project_dir: Optional[Path] = None
        self.original_env = dict(os.environ)
    
    def setup_isolated_project(self) -> Path:
        """Create isolated claude-sync project structure"""
        if self.test_project_dir:
            return self.test_project_dir
        
        self.test_project_dir = self.base_dir / "test_project"
        self.test_project_dir.mkdir(exist_ok=True)
        
        # Create necessary directories
        (self.test_project_dir / ".claude").mkdir(exist_ok=True)
        (self.test_project_dir / ".claude" / "learning").mkdir(exist_ok=True)
        (self.test_project_dir / ".claude" / "security").mkdir(exist_ok=True)
        (self.test_project_dir / ".claude" / "hooks").mkdir(exist_ok=True)
        
        # Set environment variables
        os.environ["CLAUDE_SYNC_TEST_MODE"] = "1"
        os.environ["CLAUDE_SYNC_DATA_DIR"] = str(self.test_project_dir / ".claude")
        
        return self.test_project_dir
    
    def restore_environment(self) -> None:
        """Restore original environment"""
        os.environ.clear()
        os.environ.update(self.original_env)
        
        if self.test_project_dir and self.test_project_dir.exists():
            shutil.rmtree(self.test_project_dir)
            self.test_project_dir = None

# ============================================================================
# Utility Functions
# ============================================================================

def require_file_exists(file_path: Path, description: str = "") -> bool:
    """Test utility to verify file exists"""
    if not file_path.exists():
        raise AssertionError(f"Required file not found: {file_path} ({description})")
    return True

def require_executable(file_path: Path) -> bool:
    """Test utility to verify file is executable"""
    if not os.access(file_path, os.X_OK):
        raise AssertionError(f"File is not executable: {file_path}")
    return True

def require_json_valid(file_path: Path) -> Dict[str, Any]:
    """Test utility to verify JSON file is valid"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON in {file_path}: {e}")

def require_performance_target(actual_ms: float, target_ms: float, test_name: str = "") -> bool:
    """Test utility to verify performance target is met"""
    if actual_ms > target_ms:
        raise AssertionError(f"Performance target violated: {actual_ms:.1f}ms > {target_ms}ms ({test_name})")
    return True

def simulate_claude_code_environment() -> Dict[str, Any]:
    """Simulate Claude Code execution environment for testing"""
    return {
        'working_directory': os.getcwd(),
        'claude_code_version': '1.0.0',
        'session_id': f"test_session_{int(time.time())}",
        'user_id': 'test_user',
        'timestamp': time.time()
    }

if __name__ == "__main__":
    # Basic framework testing
    framework = TestFramework()
    
    def sample_test():
        time.sleep(0.001)  # Simulate work
        return True, "Sample test passed"
    
    # Create sample suite
    sample_suite = TestSuite(
        name="framework_validation",
        tests=[sample_test]
    )
    
    framework.register_test_suite(sample_suite)
    session = framework.run_all_tests()
    
    print(f"\n🎯 Framework validation complete!")
    print(f"Success rate: {session.success_rate:.1%}")