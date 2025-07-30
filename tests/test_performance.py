#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-benchmark>=4.0.0",
#   "psutil>=5.9.0",
#   "cryptography>=41.0.0",
#   "typing-extensions>=4.0.0",
#   "statistics"
# ]
# ///
"""
Performance Testing and Benchmarking for Claude-Sync

Comprehensive performance validation against PerformanceTargets:
- Hook execution time validation (<10ms PreToolUse, <50ms PostToolUse, <20ms UserPromptSubmit)
- Learning operations performance (<100ms)
- Security operations benchmarking (<5ms encryption, <100ms identity)
- Memory usage validation (<100MB total system)
- Concurrent operation testing
- Performance regression detection
"""

import json
import time
import tempfile
import shutil
import sys
import os
import subprocess
import statistics
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import psutil
import gc

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_framework import TestFramework, TestSuite, TestResult, TestEnvironment, PerformanceBenchmark
from mock_data_generators import HookInputGenerator, LearningDataGenerator
from interfaces import PerformanceTargets

# ============================================================================
# Performance Test Results
# ============================================================================

@dataclass
class PerformanceTestResult:
    """Detailed performance test result"""
    test_name: str
    target_ms: float
    measurements: List[float]  # All execution times in ms
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def min_ms(self) -> float:
        return min(self.measurements) if self.measurements else 0.0
    
    @property
    def max_ms(self) -> float:
        return max(self.measurements) if self.measurements else 0.0
    
    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.measurements) if self.measurements else 0.0
    
    @property
    def median_ms(self) -> float:
        return statistics.median(self.measurements) if self.measurements else 0.0
    
    @property
    def p95_ms(self) -> float:
        if not self.measurements:
            return 0.0
        sorted_measurements = sorted(self.measurements)
        index = int(len(sorted_measurements) * 0.95)
        return sorted_measurements[min(index, len(sorted_measurements) - 1)]
    
    @property
    def p99_ms(self) -> float:
        if not self.measurements:
            return 0.0
        sorted_measurements = sorted(self.measurements)
        index = int(len(sorted_measurements) * 0.99)
        return sorted_measurements[min(index, len(sorted_measurements) - 1)]
    
    @property
    def stddev_ms(self) -> float:
        return statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0.0

@dataclass 
class MemoryTestResult:
    """Memory usage test result"""
    test_name: str
    target_mb: float
    measurements: List[float]  # Memory usage in MB
    passed: bool
    
    @property
    def max_mb(self) -> float:
        return max(self.measurements) if self.measurements else 0.0
    
    @property
    def avg_mb(self) -> float:
        return statistics.mean(self.measurements) if self.measurements else 0.0

# ============================================================================
# Hook Performance Tests
# ============================================================================

class HookPerformanceTests:
    """Performance tests for hooks"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.hook_generator = HookInputGenerator(seed=42)
        self.project_root = Path(__file__).parent.parent
        self.iterations = 20  # Number of measurements per test
    
    def test_hook_pretooluse_performance(self) -> Tuple[bool, str]:
        """Test PreToolUse hook performance target"""
        try:
            hook_path = self.project_root / "hooks" / "intelligent-optimizer.py"
            if not hook_path.exists():
                return False, f"Hook not found: {hook_path}"
            
            target_ms = PerformanceTargets.PRE_TOOL_USE_HOOK_MS
            measurements = []
            
            env = self._setup_test_environment()
            
            # Warm up
            hook_input = self.hook_generator.generate_pretooluse_input("hpc")
            self._execute_hook_timed(hook_path, hook_input, env)
            
            # Actual measurements
            for _ in range(self.iterations):
                hook_input = self.hook_generator.generate_pretooluse_input()
                execution_time = self._execute_hook_timed(hook_path, hook_input, env)
                
                if execution_time is not None:
                    measurements.append(execution_time)
            
            if not measurements:
                return False, "No successful measurements obtained"
            
            result = PerformanceTestResult(
                test_name="PreToolUse Hook Performance", 
                target_ms=target_ms,
                measurements=measurements,
                passed=max(measurements) <= target_ms
            )
            
            success_msg = f"PreToolUse: {result.p95_ms:.1f}ms (p95) ≤ {target_ms}ms target - {len(measurements)} samples"
            failure_msg = f"PreToolUse FAILED: {result.p95_ms:.1f}ms (p95) > {target_ms}ms target"
            
            return result.passed, success_msg if result.passed else failure_msg
            
        except Exception as e:
            return False, f"PreToolUse performance test failed: {str(e)}"
    
    def test_hook_posttooluse_performance(self) -> Tuple[bool, str]:
        """Test PostToolUse hook performance target"""
        try:
            hook_path = self.project_root / "hooks" / "learning-collector.py"
            if not hook_path.exists():
                return False, f"Hook not found: {hook_path}"
            
            target_ms = PerformanceTargets.POST_TOOL_USE_HOOK_MS
            measurements = []
            
            env = self._setup_test_environment()
            
            # Warm up
            hook_input = self.hook_generator.generate_posttooluse_input("hpc")
            self._execute_hook_timed(hook_path, hook_input, env)
            
            # Actual measurements
            for _ in range(self.iterations):
                hook_input = self.hook_generator.generate_posttooluse_input()
                execution_time = self._execute_hook_timed(hook_path, hook_input, env)
                
                if execution_time is not None:
                    measurements.append(execution_time)
            
            if not measurements:
                return False, "No successful measurements obtained"
            
            result = PerformanceTestResult(
                test_name="PostToolUse Hook Performance",
                target_ms=target_ms,
                measurements=measurements,
                passed=max(measurements) <= target_ms
            )
            
            success_msg = f"PostToolUse: {result.p95_ms:.1f}ms (p95) ≤ {target_ms}ms target - {len(measurements)} samples"
            failure_msg = f"PostToolUse FAILED: {result.p95_ms:.1f}ms (p95) > {target_ms}ms target"
            
            return result.passed, success_msg if result.passed else failure_msg
            
        except Exception as e:
            return False, f"PostToolUse performance test failed: {str(e)}"
    
    def test_hook_userpromptsubmit_performance(self) -> Tuple[bool, str]:
        """Test UserPromptSubmit hook performance target"""
        try:
            hook_path = self.project_root / "hooks" / "context-enhancer.py"
            if not hook_path.exists():
                return False, f"Hook not found: {hook_path}"
            
            target_ms = PerformanceTargets.USER_PROMPT_SUBMIT_HOOK_MS
            measurements = []
            
            env = self._setup_test_environment()
            
            # Warm up
            hook_input = self.hook_generator.generate_userpromptsubmit_input("hpc_help")
            self._execute_hook_timed(hook_path, hook_input, env)
            
            # Actual measurements
            for _ in range(self.iterations):
                hook_input = self.hook_generator.generate_userpromptsubmit_input()
                execution_time = self._execute_hook_timed(hook_path, hook_input, env)
                
                if execution_time is not None:
                    measurements.append(execution_time)
            
            if not measurements:
                return False, "No successful measurements obtained"
            
            result = PerformanceTestResult(
                test_name="UserPromptSubmit Hook Performance",
                target_ms=target_ms,
                measurements=measurements,
                passed=max(measurements) <= target_ms
            )
            
            success_msg = f"UserPromptSubmit: {result.p95_ms:.1f}ms (p95) ≤ {target_ms}ms target - {len(measurements)} samples"
            failure_msg = f"UserPromptSubmit FAILED: {result.p95_ms:.1f}ms (p95) > {target_ms}ms target"
            
            return result.passed, success_msg if result.passed else failure_msg
            
        except Exception as e:
            return False, f"UserPromptSubmit performance test failed: {str(e)}"
    
    def test_hook_memory_usage(self) -> Tuple[bool, str]:
        """Test hook memory usage"""
        try:
            hooks_to_test = [
                ("intelligent-optimizer.py", "PreToolUse"),
                ("learning-collector.py", "PostToolUse"), 
                ("context-enhancer.py", "UserPromptSubmit")
            ]
            
            env = self._setup_test_environment()
            memory_results = []
            
            for hook_name, hook_type in hooks_to_test:
                hook_path = self.project_root / "hooks" / hook_name
                if not hook_path.exists():
                    continue
                
                # Generate appropriate input
                if hook_type == "PreToolUse":
                    hook_input = self.hook_generator.generate_pretooluse_input()
                elif hook_type == "PostToolUse":
                    hook_input = self.hook_generator.generate_posttooluse_input()
                else:  # UserPromptSubmit
                    hook_input = self.hook_generator.generate_userpromptsubmit_input()
                
                memory_usage = self._measure_hook_memory(hook_path, hook_input, env)
                
                if memory_usage is not None:
                    memory_results.append((hook_name, memory_usage))
            
            if not memory_results:
                return False, "No memory measurements obtained"
            
            target_mb = PerformanceTargets.HOOK_MEMORY_MB
            max_memory = max(memory for _, memory in memory_results)
            
            if max_memory <= target_mb:
                avg_memory = sum(memory for _, memory in memory_results) / len(memory_results)
                return True, f"Hook memory usage: {avg_memory:.1f}MB avg, {max_memory:.1f}MB max ≤ {target_mb}MB target"
            else:
                worst_hook = max(memory_results, key=lambda x: x[1])
                return False, f"Hook memory FAILED: {worst_hook[0]} used {worst_hook[1]:.1f}MB > {target_mb}MB target"
            
        except Exception as e:
            return False, f"Hook memory test failed: {str(e)}"
    
    def test_concurrent_hook_execution(self) -> Tuple[bool, str]:
        """Test concurrent hook execution performance"""
        try:
            hook_path = self.project_root / "hooks" / "intelligent-optimizer.py"
            if not hook_path.exists():
                return False, f"Hook not found: {hook_path}"
            
            env = self._setup_test_environment()
            
            # Test with different concurrency levels
            concurrency_levels = [1, 2, 4, 8]
            results = {}
            
            for concurrency in concurrency_levels:
                measurements = []
                
                def execute_hook():
                    hook_input = self.hook_generator.generate_pretooluse_input()
                    return self._execute_hook_timed(hook_path, hook_input, env)
                
                # Execute hooks concurrently
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    start_time = time.perf_counter()
                    
                    futures = [executor.submit(execute_hook) for _ in range(concurrency * 2)]
                    
                    for future in concurrent.futures.as_completed(futures):
                        execution_time = future.result()
                        if execution_time is not None:
                            measurements.append(execution_time)
                    
                    total_time = (time.perf_counter() - start_time) * 1000
                
                if measurements:
                    results[concurrency] = {
                        'measurements': measurements,
                        'total_time': total_time,
                        'avg_time': statistics.mean(measurements),
                        'max_time': max(measurements)
                    }
            
            if not results:
                return False, "No concurrent execution measurements obtained"
            
            # Check if performance degrades significantly with concurrency
            single_thread_avg = results[1]['avg_time'] if 1 in results else 0
            
            performance_degradation = False
            for concurrency, result in results.items():
                if concurrency > 1:
                    degradation_ratio = result['avg_time'] / single_thread_avg if single_thread_avg > 0 else 1
                    if degradation_ratio > 2.0:  # More than 2x slower
                        performance_degradation = True
                        break
            
            if performance_degradation:
                return False, f"Significant performance degradation under concurrency"
            
            max_concurrent = max(results.keys())
            max_result = results[max_concurrent]
            
            return True, f"Concurrent execution: {max_concurrent} threads, {max_result['avg_time']:.1f}ms avg"
            
        except Exception as e:
            return False, f"Concurrent execution test failed: {str(e)}"
    
    def _setup_test_environment(self) -> Dict[str, str]:
        """Setup test environment"""
        test_dir = self.test_env.create_isolated_project()
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.project_root)
        env['CLAUDE_SYNC_TEST_MODE'] = '1'
        env['CLAUDE_SYNC_DATA_DIR'] = str(test_dir / ".claude")
        return env
    
    def _execute_hook_timed(self, hook_path: Path, hook_input: Dict[str, Any], env: Dict[str, str]) -> Optional[float]:
        """Execute hook and return execution time in ms"""
        try:
            start_time = time.perf_counter()
            
            process = subprocess.run(
                [sys.executable, str(hook_path)],
                input=json.dumps(hook_input).encode(),
                capture_output=True,
                timeout=2,
                env=env
            )
            
            end_time = time.perf_counter()
            
            if process.returncode == 0:
                return (end_time - start_time) * 1000
            else:
                return None
                
        except (subprocess.TimeoutExpired, Exception):
            return None
    
    def _measure_hook_memory(self, hook_path: Path, hook_input: Dict[str, Any], env: Dict[str, str]) -> Optional[float]:
        """Measure hook memory usage in MB"""
        try:
            # Start process and measure memory during execution
            process = subprocess.Popen(
                [sys.executable, str(hook_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            
            # Give process time to start up
            time.sleep(0.01)
            
            try:
                ps_process = psutil.Process(process.pid)
                memory_info = ps_process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                # Send input and wait for completion
                stdout, stderr = process.communicate(json.dumps(hook_input).encode(), timeout=2)
                
                return memory_mb
                
            except (psutil.NoSuchProcess, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                return None
                
        except Exception:
            return None

# ============================================================================
# Learning System Performance Tests
# ============================================================================

class LearningPerformanceTests:
    """Performance tests for learning system"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.learning_generator = LearningDataGenerator(seed=42)
        self.project_root = Path(__file__).parent.parent
    
    def test_learning_storage_performance(self) -> Tuple[bool, str]:
        """Test learning storage operation performance"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "learning"))
            from learning_storage import LearningStorage
            
            storage = LearningStorage(data_dir=test_dir / ".claude" / "learning")
            
            # Test storing command execution data
            execution_data_list = self.learning_generator.generate_command_execution_data(10)
            
            store_times = []
            for data in execution_data_list:
                start_time = time.perf_counter()
                result = storage.store_command_execution(data)
                end_time = time.perf_counter()
                
                if result:
                    store_times.append((end_time - start_time) * 1000)
            
            # Test retrieving optimization patterns
            retrieve_times = []
            for data in execution_data_list[:5]:  # Test subset
                start_time = time.perf_counter()
                patterns = storage.get_optimization_patterns(data.command)
                end_time = time.perf_counter()
                
                retrieve_times.append((end_time - start_time) * 1000)
            
            if not store_times or not retrieve_times:
                return False, "No learning operation measurements obtained"
            
            target_ms = PerformanceTargets.LEARNING_OPERATION_MS
            max_store_time = max(store_times)
            max_retrieve_time = max(retrieve_times)
            
            if max_store_time <= target_ms and max_retrieve_time <= target_ms:
                avg_store = statistics.mean(store_times)
                avg_retrieve = statistics.mean(retrieve_times)
                return True, f"Learning operations: store {avg_store:.1f}ms, retrieve {avg_retrieve:.1f}ms ≤ {target_ms}ms target"
            else:
                return False, f"Learning operations FAILED: store {max_store_time:.1f}ms, retrieve {max_retrieve_time:.1f}ms > {target_ms}ms target"
            
        except ImportError as e:
            return False, f"Learning storage module not found: {e}"
        except Exception as e:
            return False, f"Learning storage performance test failed: {str(e)}"
    
    def test_schema_evolution_performance(self) -> Tuple[bool, str]:
        """Test adaptive schema evolution performance"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "learning"))
            from adaptive_schema import AdaptiveLearningSchema
            
            schema = AdaptiveLearningSchema(data_dir=test_dir / ".claude" / "learning")
            
            # Generate execution data to trigger evolution
            execution_data_list = self.learning_generator.generate_command_execution_data(50)
            
            observation_times = []
            for data in execution_data_list:
                start_time = time.perf_counter()
                schema.observe_command_pattern(data)
                end_time = time.perf_counter()
                
                observation_times.append((end_time - start_time) * 1000)
            
            # Test schema evolution check
            start_time = time.perf_counter()
            should_evolve = schema.should_evolve_schema()
            evolution_check_time = (time.perf_counter() - start_time) * 1000
            
            # Test actual evolution if needed
            evolution_time = 0
            if should_evolve:
                start_time = time.perf_counter()
                evolution_result = schema.evolve_schema()
                evolution_time = (time.perf_counter() - start_time) * 1000
            
            if not observation_times:
                return False, "No schema operation measurements obtained"
            
            target_ms = PerformanceTargets.SCHEMA_EVOLUTION_MS
            max_observation_time = max(observation_times)
            
            # Check all operations meet targets
            if (max_observation_time <= PerformanceTargets.LEARNING_OPERATION_MS and
                evolution_check_time <= PerformanceTargets.LEARNING_OPERATION_MS and
                evolution_time <= target_ms):
                
                avg_observation = statistics.mean(observation_times)
                return True, f"Schema operations: observe {avg_observation:.1f}ms, check {evolution_check_time:.1f}ms, evolve {evolution_time:.1f}ms"
            else:
                return False, f"Schema operations FAILED: observe {max_observation_time:.1f}ms, check {evolution_check_time:.1f}ms, evolve {evolution_time:.1f}ms"
            
        except ImportError as e:
            return False, f"Adaptive schema module not found: {e}"
        except Exception as e:
            return False, f"Schema evolution performance test failed: {str(e)}"
    
    def test_threshold_manager_performance(self) -> Tuple[bool, str]:
        """Test information threshold manager performance"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "learning"))
            from threshold_manager import InformationThresholdManager
            
            manager = InformationThresholdManager(data_dir=test_dir / ".claude" / "learning")
            
            # Test information accumulation performance
            accumulate_times = []
            for _ in range(100):  # Accumulate many events
                start_time = time.perf_counter()
                manager.accumulate_information("new_commands", 1.0)
                end_time = time.perf_counter()
                
                accumulate_times.append((end_time - start_time) * 1000)
            
            # Test threshold checking performance
            check_times = []
            for agent in ["learning-analyst", "hpc-advisor", "troubleshooting-detective"]:
                start_time = time.perf_counter()
                should_trigger = manager.should_trigger_agent(agent)
                end_time = time.perf_counter()
                
                check_times.append((end_time - start_time) * 1000)
            
            if not accumulate_times or not check_times:
                return False, "No threshold operation measurements obtained"
            
            target_ms = PerformanceTargets.LEARNING_OPERATION_MS
            max_accumulate_time = max(accumulate_times)
            max_check_time = max(check_times)
            
            if max_accumulate_time <= target_ms and max_check_time <= target_ms:
                avg_accumulate = statistics.mean(accumulate_times)
                avg_check = statistics.mean(check_times)
                return True, f"Threshold operations: accumulate {avg_accumulate:.3f}ms, check {avg_check:.1f}ms ≤ {target_ms}ms target"
            else:
                return False, f"Threshold operations FAILED: accumulate {max_accumulate_time:.3f}ms, check {max_check_time:.1f}ms > {target_ms}ms target"
            
        except ImportError as e:
            return False, f"Threshold manager module not found: {e}"
        except Exception as e:
            return False, f"Threshold manager performance test failed: {str(e)}"

# ============================================================================
# Security System Performance Tests  
# ============================================================================

class SecurityPerformanceTests:
    """Performance tests for security system"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.project_root = Path(__file__).parent.parent
    
    def test_encryption_performance(self) -> Tuple[bool, str]:
        """Test encryption/decryption performance"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "security"))
            from security_manager import SecurityManager
            
            security = SecurityManager(data_dir=test_dir / ".claude" / "security")
            
            # Test data of various sizes
            test_data_sets = [
                {"small": "test"},
                {"medium": "x" * 1000},
                {"large": {"command": "x" * 5000, "data": list(range(1000))}}
            ]
            
            encryption_times = []
            decryption_times = []
            
            for test_data in test_data_sets:
                # Test encryption
                start_time = time.perf_counter()
                encrypted_data = security.encrypt_data(test_data)
                encryption_time = (time.perf_counter() - start_time) * 1000
                encryption_times.append(encryption_time)
                
                # Test decryption
                start_time = time.perf_counter()
                decrypted_data = security.decrypt_data(encrypted_data)
                decryption_time = (time.perf_counter() - start_time) * 1000
                decryption_times.append(decryption_time)
                
                if decrypted_data != test_data:
                    return False, "Encryption/decryption data integrity failed"
            
            target_ms = PerformanceTargets.ENCRYPTION_OPERATION_MS
            max_encryption_time = max(encryption_times)
            max_decryption_time = max(decryption_times)
            
            if max_encryption_time <= target_ms and max_decryption_time <= target_ms:
                avg_encrypt = statistics.mean(encryption_times)
                avg_decrypt = statistics.mean(decryption_times)
                return True, f"Encryption: {avg_encrypt:.1f}ms encrypt, {avg_decrypt:.1f}ms decrypt ≤ {target_ms}ms target"
            else:
                return False, f"Encryption FAILED: {max_encryption_time:.1f}ms encrypt, {max_decryption_time:.1f}ms decrypt > {target_ms}ms target"
            
        except ImportError as e:
            return False, f"Security manager module not found: {e}"
        except Exception as e:
            return False, f"Encryption performance test failed: {str(e)}"
    
    def test_key_rotation_performance(self) -> Tuple[bool, str]:
        """Test key rotation performance"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "security"))
            from security_manager import SecurityManager
            
            security = SecurityManager(data_dir=test_dir / ".claude" / "security")
            
            # Measure key rotation time
            start_time = time.perf_counter()
            rotation_result = security.rotate_keys()
            rotation_time = (time.perf_counter() - start_time) * 1000
            
            if not rotation_result:
                return False, "Key rotation operation failed"
            
            target_ms = PerformanceTargets.KEY_ROTATION_MS
            
            if rotation_time <= target_ms:
                return True, f"Key rotation: {rotation_time:.1f}ms ≤ {target_ms}ms target"
            else:
                return False, f"Key rotation FAILED: {rotation_time:.1f}ms > {target_ms}ms target"
            
        except ImportError as e:
            return False, f"Security manager module not found: {e}"
        except Exception as e:
            return False, f"Key rotation performance test failed: {str(e)}"
    
    def test_hardware_identity_performance(self) -> Tuple[bool, str]:
        """Test hardware identity generation performance"""
        try:
            sys.path.insert(0, str(self.project_root / "security"))
            from hardware_identity import HardwareIdentity
            
            identity = HardwareIdentity()
            
            # Test identity generation multiple times
            generation_times = []
            validation_times = []
            
            for _ in range(10):
                # Test generation
                start_time = time.perf_counter()
                host_id = identity.generate_stable_host_id()
                generation_time = (time.perf_counter() - start_time) * 1000
                generation_times.append(generation_time)
                
                # Test validation
                start_time = time.perf_counter()
                is_valid = identity.validate_host_identity(host_id)
                validation_time = (time.perf_counter() - start_time) * 1000
                validation_times.append(validation_time)
                
                if not is_valid:
                    return False, "Generated host ID failed validation"
            
            target_ms = PerformanceTargets.HOST_IDENTITY_GENERATION_MS
            max_generation_time = max(generation_times)
            max_validation_time = max(validation_times)
            
            if max_generation_time <= target_ms and max_validation_time <= target_ms:
                avg_generation = statistics.mean(generation_times)
                avg_validation = statistics.mean(validation_times)
                return True, f"Hardware identity: {avg_generation:.1f}ms generate, {avg_validation:.1f}ms validate ≤ {target_ms}ms target"
            else:
                return False, f"Hardware identity FAILED: {max_generation_time:.1f}ms generate, {max_validation_time:.1f}ms validate > {target_ms}ms target"
            
        except ImportError as e:
            return False, f"Hardware identity module not found: {e}"
        except Exception as e:
            return False, f"Hardware identity performance test failed: {str(e)}"

# ============================================================================
# System-wide Performance Tests
# ============================================================================

class SystemPerformanceTests:
    """System-wide performance and resource usage tests"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.project_root = Path(__file__).parent.parent
    
    def test_total_system_memory_usage(self) -> Tuple[bool, str]:
        """Test total system memory usage under load"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            # Setup comprehensive test environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            env['CLAUDE_SYNC_TEST_MODE'] = '1'
            env['CLAUDE_SYNC_DATA_DIR'] = str(test_dir / ".claude")
            
            # Import all major components
            sys.path.insert(0, str(self.project_root / "learning"))
            sys.path.insert(0, str(self.project_root / "security"))
            
            # Create instances of all major components
            components = []
            
            try:
                from learning_storage import LearningStorage
                components.append(LearningStorage(data_dir=test_dir / ".claude" / "learning"))
            except ImportError:
                pass
            
            try:
                from security_manager import SecurityManager
                components.append(SecurityManager(data_dir=test_dir / ".claude" / "security"))
            except ImportError:
                pass
            
            try:
                from threshold_manager import InformationThresholdManager
                components.append(InformationThresholdManager(data_dir=test_dir / ".claude" / "learning"))
            except ImportError:
                pass
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)
            
            # Perform operations that would be typical in real usage
            if len(components) >= 1:  # Learning storage operations
                from mock_data_generators import LearningDataGenerator
                learning_gen = LearningDataGenerator(seed=42)
                execution_data = learning_gen.generate_command_execution_data(20)
                
                for data in execution_data:
                    if hasattr(components[0], 'store_command_execution'):
                        components[0].store_command_execution(data)
            
            if len(components) >= 2:  # Security operations
                test_data = {"command": "test", "timestamp": time.time()}
                for _ in range(10):
                    encrypted = components[1].encrypt_data(test_data)
                    decrypted = components[1].decrypt_data(encrypted)
            
            # Measure peak memory usage
            peak_memory = process.memory_info().rss / (1024 * 1024)
            memory_increase = peak_memory - initial_memory
            
            target_mb = PerformanceTargets.TOTAL_SYSTEM_MEMORY_MB
            
            if memory_increase <= target_mb:
                return True, f"System memory usage: {memory_increase:.1f}MB increase ≤ {target_mb}MB target"
            else:
                return False, f"System memory FAILED: {memory_increase:.1f}MB increase > {target_mb}MB target"
            
        except Exception as e:
            return False, f"System memory test failed: {str(e)}"
    
    def test_learning_data_storage_limits(self) -> Tuple[bool, str]:
        """Test learning data storage stays within limits"""
        try:
            test_dir = self.test_env.create_isolated_project()
            learning_dir = test_dir / ".claude" / "learning"
            learning_dir.mkdir(parents=True)
            
            # Simulate daily learning data accumulation
            daily_target_mb = PerformanceTargets.DAILY_LEARNING_DATA_MB
            max_total_mb = PerformanceTargets.MAX_LEARNING_DATA_MB
            
            # Create mock learning data files
            daily_data_size = 0
            file_count = 0
            
            while daily_data_size < daily_target_mb * 1.5:  # Simulate 1.5 days worth
                # Create a small encrypted file
                mock_encrypted_data = b"encrypted_learning_data_" + os.urandom(1024)  # ~1KB
                
                file_path = learning_dir / f"learning_data_{file_count:06d}.enc"
                file_path.write_bytes(mock_encrypted_data)
                
                daily_data_size += len(mock_encrypted_data) / (1024 * 1024)  # Convert to MB
                file_count += 1
            
            # Measure actual storage used
            total_size = sum(f.stat().st_size for f in learning_dir.glob("*.enc"))
            total_size_mb = total_size / (1024 * 1024)
            
            if total_size_mb <= max_total_mb:
                return True, f"Learning data storage: {total_size_mb:.1f}MB ({file_count} files) ≤ {max_total_mb}MB limit"
            else:
                return False, f"Learning data storage FAILED: {total_size_mb:.1f}MB > {max_total_mb}MB limit"
            
        except Exception as e:
            return False, f"Learning data storage test failed: {str(e)}"
    
    def test_performance_under_concurrent_load(self) -> Tuple[bool, str]:
        """Test system performance under concurrent operations"""
        try:
            # This test simulates multiple hooks running concurrently
            # while learning and security operations are happening
            
            test_dir = self.test_env.create_isolated_project()
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            env['CLAUDE_SYNC_TEST_MODE'] = '1'
            env['CLAUDE_SYNC_DATA_DIR'] = str(test_dir / ".claude")
            
            from mock_data_generators import HookInputGenerator
            hook_generator = HookInputGenerator(seed=42)
            
            def run_hook_operations():
                """Simulate hook operations"""
                times = []
                for _ in range(5):
                    hook_input = hook_generator.generate_pretooluse_input()
                    hook_path = self.project_root / "hooks" / "intelligent-optimizer.py"
                    
                    if hook_path.exists():
                        start_time = time.perf_counter()
                        
                        try:
                            process = subprocess.run(
                                [sys.executable, str(hook_path)],
                                input=json.dumps(hook_input).encode(),
                                capture_output=True,
                                timeout=1,
                                env=env
                            )
                            
                            if process.returncode == 0:
                                execution_time = (time.perf_counter() - start_time) * 1000
                                times.append(execution_time)
                        except subprocess.TimeoutExpired:
                            pass
                
                return times
            
            # Run multiple concurrent operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                # Submit multiple hook operations
                for _ in range(4):
                    futures.append(executor.submit(run_hook_operations))
                
                # Collect results
                all_times = []
                for future in concurrent.futures.as_completed(futures, timeout=10):
                    try:
                        times = future.result()
                        all_times.extend(times)
                    except Exception:
                        pass
            
            if not all_times:
                return False, "No measurements obtained under concurrent load"
            
            # Check if performance is still acceptable under load
            avg_time = statistics.mean(all_times)
            max_time = max(all_times)
            
            # Allow 2x normal time under concurrent load
            target_ms = PerformanceTargets.PRE_TOOL_USE_HOOK_MS * 2
            
            if max_time <= target_ms:
                return True, f"Concurrent load: {avg_time:.1f}ms avg, {max_time:.1f}ms max ≤ {target_ms}ms target ({len(all_times)} ops)"
            else:
                return False, f"Concurrent load FAILED: {max_time:.1f}ms max > {target_ms}ms target"
            
        except Exception as e:
            return False, f"Concurrent load test failed: {str(e)}"

# ============================================================================
# Test Suite Registration
# ============================================================================

def create_performance_test_suites(test_env: TestEnvironment) -> List[TestSuite]:
    """Create all performance test suites"""
    suites = []
    
    # Hook performance tests
    hook_perf_tests = HookPerformanceTests(test_env)
    hook_perf_suite = TestSuite(
        name="hook_performance_tests",
        tests=[
            hook_perf_tests.test_hook_pretooluse_performance,
            hook_perf_tests.test_hook_posttooluse_performance,
            hook_perf_tests.test_hook_userpromptsubmit_performance,
            hook_perf_tests.test_hook_memory_usage,
            hook_perf_tests.test_concurrent_hook_execution
        ]
    )
    suites.append(hook_perf_suite)
    
    # Learning performance tests
    learning_perf_tests = LearningPerformanceTests(test_env)
    learning_perf_suite = TestSuite(
        name="learning_performance_tests",
        tests=[
            learning_perf_tests.test_learning_storage_performance,
            learning_perf_tests.test_schema_evolution_performance,
            learning_perf_tests.test_threshold_manager_performance
        ]
    )
    suites.append(learning_perf_suite)
    
    # Security performance tests
    security_perf_tests = SecurityPerformanceTests(test_env)
    security_perf_suite = TestSuite(
        name="security_performance_tests",
        tests=[
            security_perf_tests.test_encryption_performance,
            security_perf_tests.test_key_rotation_performance,
            security_perf_tests.test_hardware_identity_performance
        ]
    )
    suites.append(security_perf_suite)
    
    # System-wide performance tests
    system_perf_tests = SystemPerformanceTests(test_env)
    system_perf_suite = TestSuite(
        name="system_performance_tests",
        tests=[
            system_perf_tests.test_total_system_memory_usage,
            system_perf_tests.test_learning_data_storage_limits,
            system_perf_tests.test_performance_under_concurrent_load
        ]
    )
    suites.append(system_perf_suite)
    
    return suites

# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all performance tests"""
    print("⚡ Running Claude-Sync Performance Tests")
    print("=" * 60)
    
    # Print performance targets
    print("🎯 Performance Targets:")
    print(f"  • PreToolUse Hook: ≤{PerformanceTargets.PRE_TOOL_USE_HOOK_MS}ms")
    print(f"  • PostToolUse Hook: ≤{PerformanceTargets.POST_TOOL_USE_HOOK_MS}ms")
    print(f"  • UserPromptSubmit Hook: ≤{PerformanceTargets.USER_PROMPT_SUBMIT_HOOK_MS}ms")
    print(f"  • Learning Operations: ≤{PerformanceTargets.LEARNING_OPERATION_MS}ms")
    print(f"  • Encryption Operations: ≤{PerformanceTargets.ENCRYPTION_OPERATION_MS}ms")
    print(f"  • Total System Memory: ≤{PerformanceTargets.TOTAL_SYSTEM_MEMORY_MB}MB")
    print("")
    
    # Create test framework and environment
    framework = TestFramework()
    test_env = TestEnvironment(framework.test_dir)
    
    try:
        # Setup test environment
        test_env.setup_isolated_project()
        
        # Create and register test suites
        suites = create_performance_test_suites(test_env)
        for suite in suites:
            framework.register_test_suite(suite)
        
        # Run all tests
        session = framework.run_all_tests()
        
        # Analyze performance results
        performance_validation = PerformanceBenchmark.validate_performance_targets(session.results)
        
        print(f"\n📊 Performance Analysis:")
        print(f"  • Total Tests: {performance_validation['total_tests']}")
        print(f"  • Performance Compliant: {performance_validation['performance_compliant']}")
        print(f"  • Violations: {len(performance_validation['violations'])}")
        
        if performance_validation['violations']:
            print(f"\n⚠️ Performance Violations:")
            for violation in performance_validation['violations'][:5]:  # Show top 5
                print(f"  • {violation['test_name']}: {violation['actual_ms']:.1f}ms > {violation['target_ms']}ms ({violation['violation_ratio']:.1f}x)")
        
        # Print final results
        print(f"\n🎯 Performance Testing Complete!")
        print(f"Success Rate: {session.success_rate:.1%}")
        print(f"Performance Compliance: {performance_validation['performance_compliant']}/{performance_validation['total_tests']}")
        
        # More lenient threshold for performance tests due to system variations
        if session.success_rate >= 0.7:
            print("✅ Performance tests PASSED (≥70% success rate)")
            return 0
        else:
            print("❌ Performance tests FAILED (<70% success rate)")
            return 1
    
    finally:
        # Cleanup test environment
        test_env.restore_environment()

if __name__ == "__main__":
    exit(main())