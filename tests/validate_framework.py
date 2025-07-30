#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "psutil>=5.9.0",
#   "typing-extensions>=4.0.0"
# ]
# ///
"""
Quick validation of the end-to-end testing framework

This script performs a lightweight validation to ensure the testing framework
is working correctly before running the full test suite.
"""

import sys
import time
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_framework_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing framework imports...")
    
    try:
        from tests.test_end_to_end import (
            EndToEndTestFramework, EndToEndTestConfig,
            MockClaudeCodeEnvironment, MockLearningSystem, MockSecuritySystem
        )
        from tests.mock_data_generators import HookInputGenerator, RealisticDataSets
        from interfaces import HookResult, CommandExecutionData, validate_hook_result
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_mock_data_generation():
    """Test mock data generation"""
    print("🔍 Testing mock data generation...")
    
    try:
        from tests.mock_data_generators import HookInputGenerator, validate_hook_input, RealisticDataSets
        
        generator = HookInputGenerator(seed=42)
        
        # Test PreToolUse input
        pretool_input = generator.generate_pretooluse_input("hpc")
        valid, errors = validate_hook_input(pretool_input)
        if not valid:
            print(f"❌ PreToolUse validation failed: {errors}")
            return False
        
        # Test PostToolUse input
        posttool_input = generator.generate_posttooluse_input("r_analysis")
        valid, errors = validate_hook_input(posttool_input)
        if not valid:
            print(f"❌ PostToolUse validation failed: {errors}")
            return False
        
        # Test realistic data sets
        bio_workflow = RealisticDataSets.bioinformatics_workflow()
        if len(bio_workflow) == 0:
            print("❌ Bioinformatics workflow generation failed")
            return False
        
        print("✅ Mock data generation working")
        return True
        
    except Exception as e:
        print(f"❌ Mock data generation failed: {e}")
        return False

def test_mock_components():
    """Test mock system components"""
    print("🔍 Testing mock system components...")
    
    try:
        from tests.test_end_to_end import (
            MockClaudeCodeEnvironment, MockLearningSystem, MockSecuritySystem
        )
        from tests.mock_data_generators import HookInputGenerator
        from interfaces import CommandExecutionData
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="test_validate_"))
        
        try:
            # Test MockClaudeCodeEnvironment
            claude_env = MockClaudeCodeEnvironment(temp_dir)
            hook_input = HookInputGenerator().generate_pretooluse_input("hpc")
            hook_result, exec_time = claude_env.execute_hook("test-hook.py", hook_input)
            
            if not isinstance(exec_time, (int, float)) or exec_time < 0:
                print("❌ Hook execution time invalid")
                return False
            
            # Test MockLearningSystem
            learning_system = MockLearningSystem(temp_dir)
            execution_data = CommandExecutionData.from_hook_input(hook_input)
            success = learning_system.store_execution_data(execution_data)
            
            if not success:
                print("❌ Learning system storage failed")
                return False
            
            # Test MockSecuritySystem
            security_system = MockSecuritySystem(temp_dir)
            test_data = {"test": "data"}
            encrypted, encrypt_time = security_system.encrypt_data(test_data)
            decrypted, decrypt_time = security_system.decrypt_data(encrypted)
            
            if decrypted != test_data:
                print("❌ Security system encryption/decryption failed")
                return False
            
            print("✅ Mock components working")
            return True
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"❌ Mock components test failed: {e}")
        return False

def test_performance_measurement():
    """Test performance measurement capabilities"""
    print("🔍 Testing performance measurement...")
    
    try:
        from tests.test_framework import TestFramework, TestResult
        import time
        
        framework = TestFramework()
        
        # Test performance measurement
        def sample_test():
            time.sleep(0.001)  # 1ms
            return True, "Sample test passed"
        
        result = framework.run_single_test(sample_test, "sample_test")
        
        if not isinstance(result, TestResult):
            print("❌ TestResult not returned")
            return False
        
        if result.execution_time_ms <= 0:
            print("❌ Execution time not measured")
            return False
        
        if not result.passed:
            print("❌ Sample test should have passed")
            return False
        
        print("✅ Performance measurement working")
        return True
        
    except Exception as e:
        print(f"❌ Performance measurement test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 Claude-Sync Testing Framework Validation")
    print("=" * 50)
    
    tests = [
        test_framework_imports,
        test_mock_data_generation,
        test_mock_components,
        test_performance_measurement
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Framework validation successful! Ready for end-to-end testing.")
        return True
    else:
        print("❌ Framework validation failed. Please fix issues before running full tests.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)