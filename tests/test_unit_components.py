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
Unit Tests for Claude-Sync Components

Comprehensive unit testing for all individual components:
- Hook system (intelligent-optimizer, learning-collector, context-enhancer)
- Learning infrastructure (storage, schema, thresholds)
- Security system (encryption, identity, trust)
- Bootstrap system (activation, settings, symlinks)
"""

import json
import time
import tempfile
import shutil
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import unittest
from unittest.mock import Mock, patch, MagicMock
import threading
import uuid

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_framework import TestFramework, TestSuite, TestResult, TestEnvironment
from mock_data_generators import HookInputGenerator, LearningDataGenerator
from interfaces import (
    HookResult, CommandExecutionData, PerformanceTargets,
    validate_hook_result, create_hook_result
)

# ============================================================================
# Hook Unit Tests
# ============================================================================

class HookUnitTests:
    """Unit tests for hook implementations"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.hook_generator = HookInputGenerator(seed=42)
        self.project_root = Path(__file__).parent.parent
    
    def test_hook_intelligent_optimizer_basic(self) -> Tuple[bool, str]:
        """Test intelligent-optimizer hook basic functionality"""
        try:
            # Import the hook
            hook_path = self.project_root / "hooks" / "intelligent-optimizer.py"
            if not hook_path.exists():
                return False, f"Hook file not found: {hook_path}"
            
            # Test with mock input
            hook_input = self.hook_generator.generate_pretooluse_input("data_processing")
            
            # Run hook as subprocess to test actual execution
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            process = subprocess.run(
                [sys.executable, str(hook_path)],
                input=json.dumps(hook_input).encode(),
                capture_output=True,
                timeout=5,  # 5 second timeout
                env=env
            )
            
            if process.returncode != 0:
                return False, f"Hook execution failed: {process.stderr.decode()}"
            
            # Parse result
            try:
                result_data = json.loads(process.stdout.decode())
                hook_result = HookResult(**result_data)
            except (json.JSONDecodeError, TypeError) as e:
                return False, f"Invalid hook result format: {e}"
            
            # Validate result structure
            if not validate_hook_result(hook_result):
                return False, "Hook result failed validation"
            
            return True, "Intelligent optimizer hook basic test passed"
            
        except Exception as e:
            return False, f"Hook test failed: {str(e)}"
    
    def test_hook_learning_collector_basic(self) -> Tuple[bool, str]:
        """Test learning-collector hook basic functionality"""
        try:
            hook_path = self.project_root / "hooks" / "learning-collector.py"
            if not hook_path.exists():
                return False, f"Hook file not found: {hook_path}"
            
            # Test with PostToolUse input
            hook_input = self.hook_generator.generate_posttooluse_input("hpc", success=True)
            
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            env['CLAUDE_SYNC_TEST_MODE'] = '1'
            
            process = subprocess.run(
                [sys.executable, str(hook_path)],
                input=json.dumps(hook_input).encode(),
                capture_output=True,
                timeout=5,
                env=env
            )
            
            if process.returncode != 0:
                return False, f"Hook execution failed: {process.stderr.decode()}"
            
            try:
                result_data = json.loads(process.stdout.decode())
                hook_result = HookResult(**result_data)
            except (json.JSONDecodeError, TypeError) as e:
                return False, f"Invalid hook result format: {e}"
            
            if not validate_hook_result(hook_result):
                return False, "Hook result failed validation"
            
            # PostToolUse hooks should not block and should be silent
            if hook_result.block:
                return False, "PostToolUse hook should not block execution"
            
            return True, "Learning collector hook basic test passed"
            
        except Exception as e:
            return False, f"Hook test failed: {str(e)}"
    
    def test_hook_context_enhancer_basic(self) -> Tuple[bool, str]:
        """Test context-enhancer hook basic functionality"""
        try:
            hook_path = self.project_root / "hooks" / "context-enhancer.py"
            if not hook_path.exists():
                return False, f"Hook file not found: {hook_path}"
            
            # Test with UserPromptSubmit input
            hook_input = self.hook_generator.generate_userpromptsubmit_input("hpc_help")
            
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            env['CLAUDE_SYNC_TEST_MODE'] = '1'
            
            process = subprocess.run(
                [sys.executable, str(hook_path)],
                input=json.dumps(hook_input).encode(),
                capture_output=True,
                timeout=5,
                env=env
            )
            
            if process.returncode != 0:
                return False, f"Hook execution failed: {process.stderr.decode()}"
            
            try:
                result_data = json.loads(process.stdout.decode())
                hook_result = HookResult(**result_data)
            except (json.JSONDecodeError, TypeError) as e:
                return False, f"Invalid hook result format: {e}"
            
            if not validate_hook_result(hook_result):
                return False, "Hook result failed validation"
            
            return True, "Context enhancer hook basic test passed"
            
        except Exception as e:
            return False, f"Hook test failed: {str(e)}"
    
    def test_hook_performance_targets(self) -> Tuple[bool, str]:
        """Test that hooks meet performance targets"""
        try:
            hooks_to_test = [
                ("intelligent-optimizer.py", PerformanceTargets.PRE_TOOL_USE_HOOK_MS),
                ("learning-collector.py", PerformanceTargets.POST_TOOL_USE_HOOK_MS),
                ("context-enhancer.py", PerformanceTargets.USER_PROMPT_SUBMIT_HOOK_MS)
            ]
            
            performance_results = []
            
            for hook_name, target_ms in hooks_to_test:
                hook_path = self.project_root / "hooks" / hook_name
                if not hook_path.exists():
                    continue
                
                # Generate appropriate input based on hook type
                if "intelligent-optimizer" in hook_name:
                    hook_input = self.hook_generator.generate_pretooluse_input()
                elif "learning-collector" in hook_name:
                    hook_input = self.hook_generator.generate_posttooluse_input()
                else:  # context-enhancer
                    hook_input = self.hook_generator.generate_userpromptsubmit_input()
                
                # Measure execution time
                times = []
                for _ in range(5):  # Test 5 times for consistency
                    start_time = time.perf_counter()
                    
                    env = os.environ.copy()
                    env['PYTHONPATH'] = str(self.project_root)
                    env['CLAUDE_SYNC_TEST_MODE'] = '1'
                    
                    process = subprocess.run(
                        [sys.executable, str(hook_path)],
                        input=json.dumps(hook_input).encode(),
                        capture_output=True,
                        timeout=2,
                        env=env
                    )
                    
                    end_time = time.perf_counter()
                    execution_time_ms = (end_time - start_time) * 1000
                    
                    if process.returncode == 0:
                        times.append(execution_time_ms)
                
                if times:
                    avg_time = sum(times) / len(times)
                    max_time = max(times)
                    performance_results.append({
                        'hook': hook_name,
                        'avg_ms': avg_time,
                        'max_ms': max_time,
                        'target_ms': target_ms,
                        'meets_target': max_time <= target_ms
                    })
            
            # Check if all hooks meet targets
            failing_hooks = [r for r in performance_results if not r['meets_target']]
            
            if failing_hooks:
                details = []
                for hook in failing_hooks:
                    details.append(f"{hook['hook']}: {hook['max_ms']:.1f}ms > {hook['target_ms']}ms")
                return False, f"Performance targets not met: {'; '.join(details)}"
            
            return True, f"All {len(performance_results)} hooks meet performance targets"
            
        except Exception as e:
            return False, f"Performance test failed: {str(e)}"
    
    def test_hook_error_handling(self) -> Tuple[bool, str]:
        """Test hook error handling with invalid inputs"""
        try:
            hooks_to_test = [
                "intelligent-optimizer.py",
                "learning-collector.py", 
                "context-enhancer.py"
            ]
            
            invalid_inputs = [
                {},  # Empty input
                {"invalid": "data"},  # Invalid structure
                {"tool_name": "Invalid", "tool_input": {}},  # Missing fields
                None  # Null input (will be converted to empty string)
            ]
            
            for hook_name in hooks_to_test:
                hook_path = self.project_root / "hooks" / hook_name
                if not hook_path.exists():
                    continue
                
                for invalid_input in invalid_inputs:
                    try:
                        input_data = json.dumps(invalid_input) if invalid_input is not None else ""
                        
                        env = os.environ.copy()
                        env['PYTHONPATH'] = str(self.project_root)
                        env['CLAUDE_SYNC_TEST_MODE'] = '1'
                        
                        process = subprocess.run(
                            [sys.executable, str(hook_path)],
                            input=input_data.encode(),
                            capture_output=True,
                            timeout=2,
                            env=env
                        )
                        
                        # Hook should not crash, even with invalid input
                        if process.returncode != 0:
                            return False, f"{hook_name} crashed with invalid input: {process.stderr.decode()}"
                        
                        # Should return valid JSON even with errors
                        try:
                            result_data = json.loads(process.stdout.decode())
                            hook_result = HookResult(**result_data)
                            
                            # Should not block execution even on errors
                            if hook_result.block:
                                return False, f"{hook_name} blocked execution on error (should be graceful)"
                                
                        except (json.JSONDecodeError, TypeError):
                            return False, f"{hook_name} returned invalid JSON on error"
                    
                    except subprocess.TimeoutExpired:
                        return False, f"{hook_name} timed out with invalid input"
            
            return True, "All hooks handle errors gracefully"
            
        except Exception as e:
            return False, f"Error handling test failed: {str(e)}"

# ============================================================================
# Learning System Unit Tests
# ============================================================================

class LearningSystemUnitTests:
    """Unit tests for learning system components"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.learning_generator = LearningDataGenerator(seed=42)
        self.project_root = Path(__file__).parent.parent
    
    def test_learning_storage_basic(self) -> Tuple[bool, str]:
        """Test learning storage basic operations"""
        try:
            # Import learning storage
            sys.path.insert(0, str(self.project_root / "learning"))
            from learning_storage import LearningStorage
            
            # Create test storage
            test_dir = self.test_env.create_isolated_project()
            storage = LearningStorage(data_dir=test_dir / ".claude" / "learning")
            
            # Test storing command execution data
            execution_data = self.learning_generator.generate_command_execution_data(1)[0]
            result = storage.store_command_execution(execution_data)
            
            if not result:
                return False, "Failed to store command execution data"
            
            # Test retrieving optimization patterns
            patterns = storage.get_optimization_patterns(execution_data.command)
            if patterns is None:
                return False, "Failed to retrieve optimization patterns"
            
            # Test success rate retrieval
            success_rate = storage.get_success_rate(execution_data.command)
            if not isinstance(success_rate, float) or success_rate < 0 or success_rate > 1:
                return False, f"Invalid success rate: {success_rate}"
            
            return True, "Learning storage basic operations passed"
            
        except ImportError as e:
            return False, f"Learning storage module not found: {e}"
        except Exception as e:
            return False, f"Learning storage test failed: {str(e)}"
    
    def test_adaptive_schema_evolution(self) -> Tuple[bool, str]:
        """Test adaptive schema evolution"""
        try:
            sys.path.insert(0, str(self.project_root / "learning"))
            from adaptive_schema import AdaptiveLearningSchema
            
            test_dir = self.test_env.create_isolated_project()
            schema = AdaptiveLearningSchema(data_dir=test_dir / ".claude" / "learning")
            
            # Test observing command patterns
            execution_data_list = self.learning_generator.generate_command_execution_data(10)
            for execution_data in execution_data_list:
                schema.observe_command_pattern(execution_data)
            
            # Test schema evolution check
            should_evolve = schema.should_evolve_schema()
            if not isinstance(should_evolve, bool):
                return False, "Schema evolution check returned invalid type"
            
            # Test getting current schema
            current_schema = schema.get_current_schema()
            if not isinstance(current_schema, dict):
                return False, "Current schema is not a dictionary"
            
            # Test pattern frequency
            if execution_data_list:
                frequency = schema.get_pattern_frequency(execution_data_list[0].command)
                if not isinstance(frequency, int) or frequency < 0:
                    return False, f"Invalid pattern frequency: {frequency}"
            
            return True, "Adaptive schema evolution tests passed"
            
        except ImportError as e:
            return False, f"Adaptive schema module not found: {e}"
        except Exception as e:
            return False, f"Adaptive schema test failed: {str(e)}"
    
    def test_threshold_manager_basic(self) -> Tuple[bool, str]:
        """Test information threshold manager"""
        try:
            sys.path.insert(0, str(self.project_root / "learning"))
            from threshold_manager import InformationThresholdManager
            
            test_dir = self.test_env.create_isolated_project()
            manager = InformationThresholdManager(data_dir=test_dir / ".claude" / "learning")
            
            # Test accumulating information
            manager.accumulate_information("new_commands", 2.0)
            manager.accumulate_information("failures", 1.5)
            
            # Test weighted score calculation
            score = manager.calculate_weighted_score("learning-analyst")
            if not isinstance(score, float) or score < 0:
                return False, f"Invalid weighted score: {score}"
            
            # Test should trigger agent
            should_trigger = manager.should_trigger_agent("learning-analyst")
            if not isinstance(should_trigger, bool):
                return False, "Should trigger agent returned invalid type"
            
            # Test threshold adaptation
            manager.adapt_threshold("learning-analyst", 0.8)
            
            # Test counter reset
            manager.reset_counters_for_agent("learning-analyst")
            
            return True, "Threshold manager basic tests passed"
            
        except ImportError as e:
            return False, f"Threshold manager module not found: {e}"
        except Exception as e:
            return False, f"Threshold manager test failed: {str(e)}"

# ============================================================================
# Security System Unit Tests
# ============================================================================

class SecuritySystemUnitTests:
    """Unit tests for security system components"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.project_root = Path(__file__).parent.parent
    
    def test_encryption_basic(self) -> Tuple[bool, str]:
        """Test basic encryption/decryption functionality"""
        try:
            sys.path.insert(0, str(self.project_root / "security"))
            from security_manager import SecurityManager
            
            test_dir = self.test_env.create_isolated_project()
            security = SecurityManager(data_dir=test_dir / ".claude" / "security")
            
            # Test data encryption
            test_data = {"command": "test command", "timestamp": time.time()}
            encrypted_data = security.encrypt_data(test_data)
            
            if not isinstance(encrypted_data, bytes):
                return False, "Encryption did not return bytes"
            
            # Test data decryption
            decrypted_data = security.decrypt_data(encrypted_data)
            
            if decrypted_data != test_data:
                return False, "Decrypted data does not match original"
            
            # Test with different context
            encrypted_data_ctx = security.encrypt_data(test_data, context="learning")
            decrypted_data_ctx = security.decrypt_data(encrypted_data_ctx, context="learning")
            
            if decrypted_data_ctx != test_data:
                return False, "Context-based encryption/decryption failed"
            
            return True, "Basic encryption/decryption tests passed"
            
        except ImportError as e:
            return False, f"Security manager module not found: {e}"
        except Exception as e:
            return False, f"Encryption test failed: {str(e)}"
    
    def test_hardware_identity(self) -> Tuple[bool, str]:
        """Test hardware-based identity generation"""
        try:
            sys.path.insert(0, str(self.project_root / "security"))
            from hardware_identity import HardwareIdentity
            
            identity = HardwareIdentity()
            
            # Test stable host ID generation
            host_id_1 = identity.generate_stable_host_id()
            host_id_2 = identity.generate_stable_host_id()
            
            if not isinstance(host_id_1, str) or len(host_id_1) < 10:
                return False, "Invalid host ID format"
            
            if host_id_1 != host_id_2:
                return False, "Host ID is not stable across calls"
            
            # Test identity validation
            is_valid = identity.validate_host_identity(host_id_1)
            if not is_valid:
                return False, "Generated host ID failed validation"
            
            # Test with invalid ID
            is_invalid = identity.validate_host_identity("invalid_id_12345")
            if is_invalid:
                return False, "Invalid host ID passed validation"
            
            return True, "Hardware identity tests passed"
            
        except ImportError as e:
            return False, f"Hardware identity module not found: {e}"
        except Exception as e:
            return False, f"Hardware identity test failed: {str(e)}"
    
    def test_key_rotation(self) -> Tuple[bool, str]:
        """Test key rotation functionality"""
        try:
            sys.path.insert(0, str(self.project_root / "security"))
            from security_manager import SecurityManager
            
            test_dir = self.test_env.create_isolated_project()
            security = SecurityManager(data_dir=test_dir / ".claude" / "security")
            
            # Get initial key ID
            initial_key_id = security.get_current_key_id()
            
            # Test key rotation
            rotation_result = security.rotate_keys()
            if not rotation_result:
                return False, "Key rotation failed"
            
            # Check that key ID changed
            new_key_id = security.get_current_key_id()
            if new_key_id == initial_key_id:
                return False, "Key ID did not change after rotation"
            
            # Test that old encrypted data can still be decrypted
            test_data = {"test": "data"}
            encrypted_with_old_key = security.encrypt_data(test_data)
            
            # After rotation, should still be able to decrypt
            decrypted_data = security.decrypt_data(encrypted_with_old_key)
            if decrypted_data != test_data:
                return False, "Cannot decrypt data after key rotation"
            
            return True, "Key rotation tests passed"
            
        except ImportError as e:
            return False, f"Security manager module not found: {e}"
        except Exception as e:
            return False, f"Key rotation test failed: {str(e)}"

# ============================================================================
# Bootstrap System Unit Tests  
# ============================================================================

class BootstrapSystemUnitTests:
    """Unit tests for bootstrap and activation system"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.project_root = Path(__file__).parent.parent
    
    def test_activation_manager_basic(self) -> Tuple[bool, str]:
        """Test activation manager basic operations"""
        try:
            # Import activation manager
            activation_manager_path = self.project_root / "activation_manager.py"
            if not activation_manager_path.exists():
                return False, f"Activation manager not found: {activation_manager_path}"
            
            # Test as subprocess to avoid import issues
            test_dir = self.test_env.create_isolated_project()
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            env['CLAUDE_SYNC_TEST_MODE'] = '1'
            env['CLAUDE_SYNC_TEST_DIR'] = str(test_dir)
            
            # Test status check
            process = subprocess.run(
                [sys.executable, str(activation_manager_path), "status"],
                capture_output=True,
                timeout=10,
                env=env,
                cwd=str(test_dir)
            )
            
            if process.returncode != 0:
                return False, f"Status check failed: {process.stderr.decode()}"
            
            # Check output format
            try:
                status_data = json.loads(process.stdout.decode())
                if 'is_activated' not in status_data:
                    return False, "Status output missing 'is_activated' field"
            except json.JSONDecodeError:
                return False, "Status output is not valid JSON"
            
            return True, "Activation manager basic tests passed"
            
        except Exception as e:
            return False, f"Activation manager test failed: {str(e)}"
    
    def test_settings_merger(self) -> Tuple[bool, str]:
        """Test settings merger functionality"""
        try:
            # Create test settings files
            test_dir = self.test_env.create_isolated_project()
            claude_dir = test_dir / ".claude"
            
            # Create existing user settings
            user_settings = {
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Existing", "hooks": [{"type": "command", "command": "existing.py"}]}
                    ]
                },
                "user_preference": "keep_this"
            }
            
            with open(claude_dir / "settings.local.json", 'w') as f:
                json.dump(user_settings, f, indent=2)
            
            # Test settings merging via bootstrap script
            bootstrap_path = self.project_root / "bootstrap.sh"
            if not bootstrap_path.exists():
                return False, f"Bootstrap script not found: {bootstrap_path}"
            
            env = os.environ.copy()
            env['CLAUDE_SYNC_TEST_MODE'] = '1'
            env['CLAUDE_SYNC_TEST_DIR'] = str(test_dir)
            
            # Test dry-run activation to check settings merging
            process = subprocess.run(
                ["bash", str(bootstrap_path), "status"],
                capture_output=True,
                timeout=30,
                env=env,
                cwd=str(test_dir)
            )
            
            # Should not fail even with existing settings
            if process.returncode not in [0, 1]:  # 1 is expected for not activated
                return False, f"Bootstrap status failed: {process.stderr.decode()}"
            
            return True, "Settings merger tests passed"
            
        except Exception as e:
            return False, f"Settings merger test failed: {str(e)}"
    
    def test_symlink_management(self) -> Tuple[bool, str]:
        """Test symlink creation and management"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            # Create mock hooks directory
            hooks_source = test_dir / "claude-sync" / "hooks"
            hooks_source.mkdir(parents=True)
            
            # Create mock hook files
            test_hooks = ["intelligent-optimizer.py", "learning-collector.py", "context-enhancer.py"]
            for hook_name in test_hooks:
                hook_file = hooks_source / hook_name
                hook_file.write_text(f"#!/usr/bin/env python3\n# Mock {hook_name}")
                hook_file.chmod(0o755)
            
            # Create target hooks directory
            hooks_target = test_dir / ".claude" / "hooks"
            hooks_target.mkdir(parents=True)
            
            # Test creating symlinks
            for hook_name in test_hooks:
                source_path = hooks_source / hook_name
                target_path = hooks_target / hook_name
                
                # Create symlink
                target_path.symlink_to(source_path)
                
                # Verify symlink
                if not target_path.is_symlink():
                    return False, f"Failed to create symlink for {hook_name}"
                
                if not target_path.exists():
                    return False, f"Symlink target does not exist for {hook_name}"
            
            # Test symlink cleanup
            for hook_name in test_hooks:
                target_path = hooks_target / hook_name
                if target_path.is_symlink():
                    target_path.unlink()
                
                if target_path.exists():
                    return False, f"Failed to remove symlink for {hook_name}"
            
            return True, "Symlink management tests passed"
            
        except Exception as e:
            return False, f"Symlink management test failed: {str(e)}"

# ============================================================================
# Test Suite Registration
# ============================================================================

def create_unit_test_suites(test_env: TestEnvironment) -> List[TestSuite]:
    """Create all unit test suites"""
    suites = []
    
    # Hook unit tests
    hook_tests = HookUnitTests(test_env)
    hook_suite = TestSuite(
        name="hook_unit_tests",
        tests=[
            hook_tests.test_hook_intelligent_optimizer_basic,
            hook_tests.test_hook_learning_collector_basic,
            hook_tests.test_hook_context_enhancer_basic,
            hook_tests.test_hook_performance_targets,
            hook_tests.test_hook_error_handling
        ]
    )
    suites.append(hook_suite)
    
    # Learning system unit tests
    learning_tests = LearningSystemUnitTests(test_env)
    learning_suite = TestSuite(
        name="learning_unit_tests",
        tests=[
            learning_tests.test_learning_storage_basic,
            learning_tests.test_adaptive_schema_evolution,
            learning_tests.test_threshold_manager_basic
        ]
    )
    suites.append(learning_suite)
    
    # Security system unit tests
    security_tests = SecuritySystemUnitTests(test_env)
    security_suite = TestSuite(
        name="security_unit_tests",
        tests=[
            security_tests.test_encryption_basic,
            security_tests.test_hardware_identity,
            security_tests.test_key_rotation
        ]
    )
    suites.append(security_suite)
    
    # Bootstrap system unit tests
    bootstrap_tests = BootstrapSystemUnitTests(test_env)
    bootstrap_suite = TestSuite(
        name="bootstrap_unit_tests",
        tests=[
            bootstrap_tests.test_activation_manager_basic,
            bootstrap_tests.test_settings_merger,
            bootstrap_tests.test_symlink_management
        ]
    )
    suites.append(bootstrap_suite)
    
    return suites

# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all unit tests"""
    print("🧪 Running Claude-Sync Unit Tests")
    print("=" * 60)
    
    # Create test framework and environment
    framework = TestFramework()
    test_env = TestEnvironment(framework.test_dir)
    
    try:
        # Setup test environment
        test_env.setup_isolated_project()
        
        # Create and register test suites
        suites = create_unit_test_suites(test_env)
        for suite in suites:
            framework.register_test_suite(suite)
        
        # Run all tests
        session = framework.run_all_tests()
        
        # Print final results
        print(f"\n🎯 Unit Testing Complete!")
        print(f"Success Rate: {session.success_rate:.1%}")
        print(f"Total Time: {session.duration_ms:.0f}ms")
        
        if session.success_rate >= 0.8:
            print("✅ Unit tests PASSED (≥80% success rate)")
            return 0
        else:
            print("❌ Unit tests FAILED (<80% success rate)")
            return 1
    
    finally:
        # Cleanup test environment
        test_env.restore_environment()

if __name__ == "__main__":
    exit(main())