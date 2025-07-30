#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "psutil>=5.9.0"
# ]
# ///
"""
Demo Test Runner for Claude-Sync Testing Framework

This script demonstrates the testing framework capabilities by running
a subset of tests and showing the reporting features.
"""

import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_framework import TestFramework, TestSuite, TestEnvironment
from tests.mock_data_generators import HookInputGenerator, validate_hook_input

def demo_mock_data_generation():
    """Demonstrate mock data generation capabilities"""
    print("🎭 Demonstrating Mock Data Generation")
    print("-" * 50)
    
    generator = HookInputGenerator(seed=42)
    
    # Generate sample hook inputs
    pretool_input = generator.generate_pretooluse_input("hpc")
    posttool_input = generator.generate_posttooluse_input("r_analysis", success=True)
    prompt_input = generator.generate_userpromptsubmit_input("hpc_help")
    
    print(f"✅ PreToolUse Input: {pretool_input.get('tool_input', {}).get('command', 'N/A')[:60]}...")
    print(f"✅ PostToolUse Input: Exit code {posttool_input.get('tool_output', {}).get('exit_code', 'N/A')}, Duration {posttool_input.get('tool_output', {}).get('duration_ms', 'N/A')}ms")
    print(f"✅ UserPromptSubmit Input: {prompt_input.get('user_prompt', 'N/A')[:60]}...")
    
    # Validate generated data
    validations = [
        validate_hook_input(pretool_input),
        validate_hook_input(posttool_input), 
        validate_hook_input(prompt_input)
    ]
    
    all_valid = all(valid for valid, _ in validations)
    print(f"✅ All generated data valid: {all_valid}")
    
    return all_valid

def demo_test_framework():
    """Demonstrate test framework capabilities"""
    print("\n🧪 Demonstrating Test Framework")
    print("-" * 50)
    
    # Create test framework
    framework = TestFramework()
    test_env = TestEnvironment(framework.test_dir)
    
    # Create demo test functions
    def sample_passing_test():
        time.sleep(0.001)  # Simulate work
        return True, "Sample test passed successfully"
    
    def sample_performance_test():
        start = time.perf_counter()
        time.sleep(0.005)  # 5ms work
        duration = (time.perf_counter() - start) * 1000
        return duration < 10, f"Performance test: {duration:.1f}ms"
    
    def sample_failing_test():
        return False, "This test is designed to fail for demonstration"
    
    # Create test suite
    demo_suite = TestSuite(
        name="demo_test_suite",
        tests=[
            sample_passing_test,
            sample_performance_test,
            sample_failing_test
        ]
    )
    
    try:
        # Setup test environment
        test_env.setup_isolated_project()
        
        # Register and run tests
        framework.register_test_suite(demo_suite)
        session = framework.run_all_tests()
        
        print(f"\n📊 Demo Results:")
        print(f"  • Total Tests: {session.total_tests}")
        print(f"  • Success Rate: {session.success_rate:.1%}")
        print(f"  • Duration: {session.duration_ms:.0f}ms")
        
        return session.success_rate > 0.5  # Expected to have some failures
        
    finally:
        # Cleanup
        test_env.restore_environment()

def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities"""
    print("\n⚡ Demonstrating Performance Monitoring")
    print("-" * 50)
    
    from tests.test_framework import PerformanceBenchmark
    
    def fast_operation():
        return sum(range(100))
    
    def slow_operation():
        time.sleep(0.01)  # 10ms
        return "slow_result"
    
    # Measure execution times
    fast_stats = PerformanceBenchmark.measure_execution_time(fast_operation, iterations=10)
    slow_stats = PerformanceBenchmark.measure_execution_time(slow_operation, iterations=5)
    
    print(f"✅ Fast Operation Stats:")
    print(f"   • Average: {fast_stats['avg_ms']:.3f}ms")
    print(f"   • 95th percentile: {fast_stats['p95_ms']:.3f}ms")
    
    print(f"✅ Slow Operation Stats:")
    print(f"   • Average: {slow_stats['avg_ms']:.1f}ms")  
    print(f"   • 95th percentile: {slow_stats['p95_ms']:.1f}ms")
    
    return True

def demo_interfaces_validation():
    """Demonstrate interfaces validation"""
    print("\n🔗 Demonstrating Interface Validation")
    print("-" * 50)
    
    from interfaces import HookResult, CommandExecutionData, validate_hook_result, PerformanceTargets
    
    # Test HookResult validation
    valid_result = HookResult(block=False, message="Test message", execution_time_ms=5.0)
    invalid_result_data = {"block": "not_boolean", "message": 123}
    
    print(f"✅ Valid HookResult: {validate_hook_result(valid_result)}")
    print(f"✅ Invalid HookResult data: {validate_hook_result(invalid_result_data)}")
    
    # Test CommandExecutionData
    exec_data = CommandExecutionData(
        command="test command",
        exit_code=0,
        duration_ms=100,
        timestamp=time.time(),
        session_id="demo_session",
        working_directory="/test"
    )
    
    print(f"✅ CommandExecutionData created: {exec_data.command}")
    
    # Test performance targets
    print(f"✅ Performance Targets:")
    print(f"   • PreToolUse: {PerformanceTargets.PRE_TOOL_USE_HOOK_MS}ms")
    print(f"   • PostToolUse: {PerformanceTargets.POST_TOOL_USE_HOOK_MS}ms")
    print(f"   • Total Memory: {PerformanceTargets.TOTAL_SYSTEM_MEMORY_MB}MB")
    
    return True

def main():
    """Run comprehensive testing framework demonstration"""
    print("🚀 Claude-Sync Testing Framework Demonstration")
    print("=" * 80)
    
    demos = [
        ("Mock Data Generation", demo_mock_data_generation),
        ("Test Framework Core", demo_test_framework),
        ("Performance Monitoring", demo_performance_monitoring),
        ("Interface Validation", demo_interfaces_validation)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            result = demo_func()
            results.append((demo_name, result, None))
            status = "✅ PASSED" if result else "⚠️ PARTIAL"
            print(f"\n{status} {demo_name}")
        except Exception as e:
            results.append((demo_name, False, str(e)))
            print(f"\n❌ FAILED {demo_name}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 DEMONSTRATION SUMMARY")
    print("=" * 80)
    
    passed_demos = sum(1 for _, result, _ in results if result)
    total_demos = len(results)
    
    print(f"Demonstrations: {passed_demos}/{total_demos} successful")
    
    for demo_name, result, error in results:
        status = "✅" if result else "❌" if error else "⚠️"
        print(f"  {status} {demo_name}")
        if error:
            print(f"      Error: {error}")
    
    print(f"\n🎯 Testing Framework Capabilities:")
    print(f"  • Mock data generation for realistic testing")
    print(f"  • Performance monitoring and benchmarking")
    print(f"  • Test environment isolation")
    print(f"  • Comprehensive result reporting")
    print(f"  • Interface contract validation")
    print(f"  • Quality gate enforcement")
    
    if passed_demos >= total_demos * 0.75:
        print(f"\n✅ Testing framework demonstration successful!")
        print(f"   Ready to validate claude-sync system integration.")
        return 0
    else:
        print(f"\n❌ Testing framework demonstration had issues.")
        print(f"   Check individual demo results above.")
        return 1

if __name__ == "__main__":
    exit(main())