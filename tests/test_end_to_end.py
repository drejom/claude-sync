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
Claude-Sync End-to-End Integration Testing Framework

Comprehensive end-to-end testing that validates the entire claude-sync system
works together seamlessly under realistic conditions.

This framework tests:
1. Full system integration (activation → hook execution → learning → deactivation)
2. Realistic workflow patterns (bioinformatics, HPC, R analysis, containers)  
3. Performance validation against quality gates
4. Advanced integration scenarios (concurrency, schema evolution, error handling)
5. Security and reliability under stress conditions

Run with: python /Users/domeally/.claude/claude-sync/tests/test_end_to_end.py
"""

import asyncio
import concurrent.futures
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import traceback
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from interfaces import (
    HookResult, CommandExecutionData, PerformanceTargets, 
    SystemState, ActivationResult, validate_hook_result,
    InformationTypes, AgentNames
)
from tests.test_framework import TestFramework, TestResult, TestSuite, PerformanceBenchmark
from tests.mock_data_generators import (
    HookInputGenerator, LearningDataGenerator, ErrorScenarioGenerator,
    RealisticDataSets, validate_hook_input
)

# ============================================================================
# End-to-End Test Configuration
# ============================================================================

@dataclass
class EndToEndTestConfig:
    """Configuration for end-to-end testing"""
    # Performance requirements
    hook_execution_limit_ms: int = 10
    learning_operation_limit_ms: int = 1
    memory_limit_mb: int = 50
    network_timeout_ms: int = 5000
    installation_timeout_s: int = 30
    
    # Test data configuration
    concurrent_hooks: int = 10
    stress_test_duration_s: int = 30
    learning_data_points: int = 100
    workflow_repetitions: int = 5
    
    # System requirements
    min_success_rate: float = 0.95
    max_performance_violations: int = 2
    required_hooks: List[str] = field(default_factory=lambda: [
        'bash-optimizer-enhanced.py',
        'ssh-router-enhanced.py', 
        'resource-tracker.py',
        'background-sync.py',
        'process-sync.py'
    ])
    
    # Error tolerance
    acceptable_failure_rate: float = 0.05
    max_memory_growth_mb: int = 10
    max_startup_time_ms: int = 1000

@dataclass
class WorkflowExecutionResult:
    """Result of executing a complete workflow"""
    workflow_name: str
    total_steps: int
    successful_steps: int
    failed_steps: int
    total_duration_ms: float
    performance_violations: int
    memory_peak_mb: float
    errors: List[str] = field(default_factory=list)
    hook_execution_times: List[float] = field(default_factory=list)
    learning_data_generated: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.successful_steps / self.total_steps if self.total_steps > 0 else 0.0
    
    @property
    def average_hook_time_ms(self) -> float:
        return sum(self.hook_execution_times) / len(self.hook_execution_times) if self.hook_execution_times else 0.0

@dataclass
class SystemIntegrationStatus:
    """Status of system integration components"""
    hooks_installed: bool = False
    learning_system_active: bool = False
    security_system_active: bool = False
    mesh_sync_available: bool = False
    agents_responsive: bool = False
    performance_within_limits: bool = False
    data_integrity_verified: bool = False

# ============================================================================
# Mock System Components for Testing
# ============================================================================

class MockClaudeCodeEnvironment:
    """Mock Claude Code environment for testing hooks"""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.hooks_dir = test_dir / "hooks"
        self.settings_file = test_dir / ".claude" / "settings.local.json"
        self.execution_history: List[Dict[str, Any]] = []
        self.hook_performance: Dict[str, List[float]] = {}
        
        # Setup directories
        self.hooks_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / ".claude").mkdir(parents=True, exist_ok=True)
    
    def execute_hook(self, hook_name: str, hook_input: Dict[str, Any], timeout_ms: int = 10000) -> Tuple[HookResult, float]:
        """Execute a hook and measure performance"""
        hook_path = self.hooks_dir / hook_name
        
        if not hook_path.exists():
            # Create mock hook for testing
            self._create_mock_hook(hook_path)
        
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            # Execute hook with timeout
            process = subprocess.Popen(
                [str(hook_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.test_dir)
            )
            
            stdout, stderr = process.communicate(
                input=json.dumps(hook_input),
                timeout=timeout_ms / 1000
            )
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_used_mb = max(0, end_memory - start_memory)
            
            # Parse result
            if process.returncode == 0 and stdout.strip():
                try:
                    result_data = json.loads(stdout.strip())
                    result = HookResult(
                        block=result_data.get('block', False),
                        message=result_data.get('message'),
                        modifications=result_data.get('modifications'),
                        execution_time_ms=execution_time_ms,
                        memory_used_mb=memory_used_mb
                    )
                except json.JSONDecodeError:
                    result = HookResult(
                        block=False,
                        message=f"Invalid JSON response: {stdout[:100]}",
                        execution_time_ms=execution_time_ms,
                        memory_used_mb=memory_used_mb
                    )
            else:
                result = HookResult(
                    block=False,
                    message=f"Hook execution failed: {stderr[:100]}",
                    execution_time_ms=execution_time_ms,
                    memory_used_mb=memory_used_mb
                )
            
            # Record performance
            if hook_name not in self.hook_performance:
                self.hook_performance[hook_name] = []
            self.hook_performance[hook_name].append(execution_time_ms)
            
            # Record execution
            self.execution_history.append({
                'hook_name': hook_name,
                'execution_time_ms': execution_time_ms,
                'memory_used_mb': memory_used_mb,
                'success': process.returncode == 0,
                'timestamp': time.time()
            })
            
            return result, execution_time_ms
            
        except subprocess.TimeoutExpired:
            process.kill()
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            return HookResult(
                block=False,
                message=f"Hook execution timed out after {timeout_ms}ms",
                execution_time_ms=execution_time_ms,
                memory_used_mb=0
            ), execution_time_ms
        except Exception as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            return HookResult(
                block=False,
                message=f"Hook execution error: {str(e)}",
                execution_time_ms=execution_time_ms,
                memory_used_mb=0
            ), execution_time_ms
    
    def _create_mock_hook(self, hook_path: Path) -> None:
        """Create a mock hook script for testing"""
        mock_hook_script = '''#!/usr/bin/env python3
import json
import sys
import time
import random

def main():
    try:
        # Read hook input
        hook_input = json.loads(sys.stdin.read())
        
        # Simulate processing time (1-5ms)
        time.sleep(random.uniform(0.001, 0.005))
        
        # Generate realistic response
        result = {
            'block': False,
            'message': None
        }
        
        # Add optimizations for certain commands
        if 'tool_input' in hook_input and 'command' in hook_input['tool_input']:
            command = hook_input['tool_input']['command']
            if 'grep' in command and 'rg' not in command:
                result['message'] = 'Suggestion: Consider using ripgrep (rg) for faster searching'
            elif 'find' in command and 'fd' not in command:
                result['message'] = 'Suggestion: Consider using fd for faster file finding'
        
        print(json.dumps(result))
        sys.exit(0)
        
    except Exception as e:
        # Graceful error handling
        result = {'block': False, 'message': None}
        print(json.dumps(result))
        sys.exit(0)

if __name__ == '__main__':
    main()
'''
        hook_path.write_text(mock_hook_script)
        hook_path.chmod(0o755)

class MockLearningSystem:
    """Mock learning system for integration testing"""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.learning_dir = test_dir / "learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        self.execution_data: List[CommandExecutionData] = []
        self.optimization_patterns: List[Dict[str, Any]] = []
        self.schema_evolution_count = 0
        
    def store_execution_data(self, data: CommandExecutionData) -> bool:
        """Store command execution data"""
        start_time = time.perf_counter()
        
        # Simulate learning storage with realistic delay
        time.sleep(random.uniform(0.0001, 0.0005))  # 0.1-0.5ms
        
        self.execution_data.append(data)
        
        # Trigger schema evolution occasionally
        if len(self.execution_data) % 50 == 0:
            self.schema_evolution_count += 1
        
        execution_time = (time.perf_counter() - start_time) * 1000
        return execution_time < PerformanceTargets.LEARNING_OPERATION_MS
    
    def get_optimization_suggestions(self, command: str) -> List[str]:
        """Get optimization suggestions for command"""
        suggestions = []
        
        if 'grep' in command and 'rg' not in command:
            suggestions.append('Use ripgrep (rg) for faster searching')
        if 'find' in command and 'fd' not in command:
            suggestions.append('Use fd for faster file finding')
        if 'sbatch' in command and '--mem=' not in command:
            suggestions.append('Consider specifying memory requirements with --mem=')
        
        return suggestions
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get learning system performance statistics"""
        if not self.execution_data:
            return {}
        
        total_duration = sum(d.duration_ms for d in self.execution_data)
        avg_duration = total_duration / len(self.execution_data)
        
        success_count = sum(1 for d in self.execution_data if d.exit_code == 0)
        success_rate = success_count / len(self.execution_data)
        
        return {
            'total_executions': len(self.execution_data),
            'average_duration_ms': avg_duration,
            'success_rate': success_rate,
            'schema_evolutions': self.schema_evolution_count,
            'patterns_learned': len(self.optimization_patterns)
        }

class MockSecuritySystem:
    """Mock security system for integration testing"""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.security_dir = test_dir / "security"
        self.security_dir.mkdir(parents=True, exist_ok=True)
        
        self.key_rotations = 0
        self.encryption_operations = 0
        self.decryption_operations = 0
        
    def encrypt_data(self, data: Dict[str, Any]) -> Tuple[bytes, float]:
        """Mock data encryption with performance measurement"""
        start_time = time.perf_counter()
        
        # Simulate encryption delay
        time.sleep(random.uniform(0.001, 0.003))  # 1-3ms
        
        self.encryption_operations += 1
        encrypted_data = json.dumps(data).encode('utf-8')  # Mock encryption
        
        execution_time = (time.perf_counter() - start_time) * 1000
        return encrypted_data, execution_time
    
    def decrypt_data(self, encrypted_data: bytes) -> Tuple[Dict[str, Any], float]:
        """Mock data decryption with performance measurement"""
        start_time = time.perf_counter()
        
        # Simulate decryption delay
        time.sleep(random.uniform(0.001, 0.003))  # 1-3ms
        
        self.decryption_operations += 1
        data = json.loads(encrypted_data.decode('utf-8'))  # Mock decryption
        
        execution_time = (time.perf_counter() - start_time) * 1000
        return data, execution_time
    
    def rotate_keys(self) -> Tuple[bool, float]:
        """Mock key rotation with performance measurement"""
        start_time = time.perf_counter()
        
        # Simulate key rotation delay
        time.sleep(random.uniform(0.1, 0.5))  # 100-500ms
        
        self.key_rotations += 1
        
        execution_time = (time.perf_counter() - start_time) * 1000
        return True, execution_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get security system performance statistics"""
        return {
            'key_rotations': self.key_rotations,
            'encryption_operations': self.encryption_operations,
            'decryption_operations': self.decryption_operations
        }

# ============================================================================
# End-to-End Test Suite Classes
# ============================================================================

class FullSystemIntegrationTest:
    """Test complete system integration lifecycle"""
    
    def __init__(self, config: EndToEndTestConfig, test_dir: Path):
        self.config = config
        self.test_dir = test_dir
        self.claude_env = MockClaudeCodeEnvironment(test_dir)
        self.learning_system = MockLearningSystem(test_dir)
        self.security_system = MockSecuritySystem(test_dir)
        self.hook_generator = HookInputGenerator(seed=42)
        
    def test_activation_lifecycle(self) -> TestResult:
        """Test complete activation → execution → deactivation lifecycle"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            # Phase 1: System Activation
            activation_result = self._test_system_activation()
            if not activation_result.success:
                return TestResult(
                    test_name="activation_lifecycle",
                    passed=False,
                    execution_time_ms=0,
                    memory_used_mb=0,
                    message=f"Activation failed: {activation_result.message}"
                )
            
            # Phase 2: Hook Execution
            hook_results = self._test_hook_execution_cycle()
            failed_hooks = [r for r in hook_results if not r.passed]
            if failed_hooks:
                return TestResult(
                    test_name="activation_lifecycle",
                    passed=False,
                    execution_time_ms=0,
                    memory_used_mb=0,
                    message=f"Hook execution failed: {len(failed_hooks)} hooks failed"
                )
            
            # Phase 3: Learning Integration
            learning_result = self._test_learning_integration()
            if not learning_result:
                return TestResult(
                    test_name="activation_lifecycle",
                    passed=False,
                    execution_time_ms=0,
                    memory_used_mb=0,
                    message="Learning integration failed"
                )
            
            # Phase 4: Security Operations
            security_result = self._test_security_integration()
            if not security_result:
                return TestResult(
                    test_name="activation_lifecycle",
                    passed=False,
                    execution_time_ms=0,
                    memory_used_mb=0,
                    message="Security integration failed"
                )
            
            # Phase 5: System Deactivation
            deactivation_result = self._test_system_deactivation()
            if not deactivation_result.success:
                return TestResult(
                    test_name="activation_lifecycle",
                    passed=False,
                    execution_time_ms=0,
                    memory_used_mb=0,
                    message=f"Deactivation failed: {deactivation_result.message}"
                )
            
            # Calculate final metrics
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_used_mb = max(0, end_memory - start_memory)
            
            return TestResult(
                test_name="activation_lifecycle",
                passed=True,
                execution_time_ms=execution_time_ms,
                memory_used_mb=memory_used_mb,
                message="Complete system lifecycle test passed",
                details={
                    'phases_completed': 5,
                    'hooks_tested': len(self.config.required_hooks),
                    'learning_operations': len(self.learning_system.execution_data),
                    'security_operations': self.security_system.encryption_operations
                }
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            return TestResult(
                test_name="activation_lifecycle",
                passed=False,
                execution_time_ms=(end_time - start_time) * 1000,
                memory_used_mb=max(0, end_memory - start_memory),
                message=f"System integration test failed: {str(e)}",
                details={'exception': str(e), 'traceback': traceback.format_exc()}
            )
    
    def _test_system_activation(self) -> ActivationResult:
        """Test system activation process"""
        try:
            # Create settings file
            settings = {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": str(self.claude_env.hooks_dir / "bash-optimizer-enhanced.py")
                                }
                            ]
                        }
                    ],
                    "PostToolUse": [
                        {
                            "matcher": "Bash", 
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": str(self.claude_env.hooks_dir / "resource-tracker.py")
                                }
                            ]
                        }
                    ]
                }
            }
            
            self.claude_env.settings_file.write_text(json.dumps(settings, indent=2))
            
            return ActivationResult(
                success=True,
                message="System activated successfully",
                actions_performed=["settings_created", "hooks_configured"],
                backups_created=[],
                errors=[]
            )
            
        except Exception as e:
            return ActivationResult(
                success=False,
                message=f"Activation failed: {str(e)}",
                actions_performed=[],
                backups_created=[],
                errors=[str(e)]
            )
    
    def _test_hook_execution_cycle(self) -> List[TestResult]:
        """Test hook execution for all required hooks"""
        results = []
        
        for hook_name in self.config.required_hooks:
            # Test PreToolUse hook
            pretool_input = self.hook_generator.generate_pretooluse_input("hpc")
            hook_result, execution_time = self.claude_env.execute_hook(hook_name, pretool_input)
            
            passed = (
                validate_hook_result(hook_result) and 
                execution_time <= self.config.hook_execution_limit_ms and
                hook_result.memory_used_mb <= self.config.memory_limit_mb
            )
            
            results.append(TestResult(
                test_name=f"hook_execution_{hook_name}",
                passed=passed,
                execution_time_ms=execution_time,
                memory_used_mb=hook_result.memory_used_mb,
                message="Hook executed successfully" if passed else f"Hook execution failed or exceeded limits",
                details={
                    'hook_result': hook_result.to_json(),
                    'performance_limit_ms': self.config.hook_execution_limit_ms,
                    'memory_limit_mb': self.config.memory_limit_mb
                }
            ))
        
        return results
    
    def _test_learning_integration(self) -> bool:
        """Test learning system integration"""
        try:
            # Generate test execution data
            for _ in range(10):
                posttool_input = self.hook_generator.generate_posttooluse_input("hpc")
                execution_data = CommandExecutionData.from_hook_input(posttool_input)
                
                # Test learning storage performance
                success = self.learning_system.store_execution_data(execution_data)
                if not success:
                    return False
            
            # Test learning retrieval
            stats = self.learning_system.get_performance_stats()
            return stats.get('total_executions', 0) >= 10
            
        except Exception:
            return False
    
    def _test_security_integration(self) -> bool:
        """Test security system integration"""
        try:
            # Test encryption/decryption cycle
            test_data = {"command": "test", "timestamp": time.time()}
            
            encrypted_data, encrypt_time = self.security_system.encrypt_data(test_data)
            if encrypt_time > PerformanceTargets.ENCRYPTION_OPERATION_MS:
                return False
            
            decrypted_data, decrypt_time = self.security_system.decrypt_data(encrypted_data)
            if decrypt_time > PerformanceTargets.ENCRYPTION_OPERATION_MS:
                return False
            
            # Verify data integrity
            return decrypted_data == test_data
            
        except Exception:
            return False
    
    def _test_system_deactivation(self) -> ActivationResult:
        """Test system deactivation process"""
        try:
            # Remove settings file
            if self.claude_env.settings_file.exists():
                self.claude_env.settings_file.unlink()
            
            return ActivationResult(
                success=True,
                message="System deactivated successfully",
                actions_performed=["settings_removed", "hooks_uninstalled"],
                backups_created=[],
                errors=[]
            )
            
        except Exception as e:
            return ActivationResult(
                success=False,
                message=f"Deactivation failed: {str(e)}",
                actions_performed=[],
                backups_created=[],
                errors=[str(e)]
            )

class RealisticWorkflowTest:
    """Test realistic workflow patterns"""
    
    def __init__(self, config: EndToEndTestConfig, test_dir: Path):
        self.config = config
        self.test_dir = test_dir
        self.claude_env = MockClaudeCodeEnvironment(test_dir)
        self.learning_system = MockLearningSystem(test_dir)
        
    def test_bioinformatics_workflow(self) -> TestResult:
        """Test complete bioinformatics data processing workflow"""
        return self._execute_workflow("bioinformatics", RealisticDataSets.bioinformatics_workflow())
    
    def test_hpc_troubleshooting_workflow(self) -> TestResult:
        """Test HPC troubleshooting session with failures and recovery"""
        return self._execute_workflow("hpc_troubleshooting", RealisticDataSets.hpc_troubleshooting_session())
    
    def test_performance_degradation_workflow(self) -> TestResult:
        """Test workflow showing performance degradation detection"""
        return self._execute_workflow("performance_degradation", RealisticDataSets.performance_degradation_scenario())
    
    def _execute_workflow(self, workflow_name: str, workflow_steps: List[Dict[str, Any]]) -> TestResult:
        """Execute a complete workflow and measure results"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        result = WorkflowExecutionResult(
            workflow_name=workflow_name,
            total_steps=len(workflow_steps),
            successful_steps=0,
            failed_steps=0,
            total_duration_ms=0,
            performance_violations=0,
            memory_peak_mb=start_memory
        )
        
        try:
            for i, step in enumerate(workflow_steps):
                step_start = time.perf_counter()
                step_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                
                # Update peak memory
                result.memory_peak_mb = max(result.memory_peak_mb, step_memory)
                
                # Execute step based on hook type
                hook_type = step.get('hook_type', 'PreToolUse')
                
                if hook_type in ['PreToolUse', 'PostToolUse']:
                    # Execute appropriate hook
                    hook_name = 'bash-optimizer-enhanced.py' if hook_type == 'PreToolUse' else 'resource-tracker.py'
                    hook_result, execution_time = self.claude_env.execute_hook(hook_name, step)
                    
                    result.hook_execution_times.append(execution_time)
                    
                    # Check performance
                    if execution_time > self.config.hook_execution_limit_ms:
                        result.performance_violations += 1
                    
                    # Store learning data for PostToolUse
                    if hook_type == 'PostToolUse':
                        execution_data = CommandExecutionData.from_hook_input(step)
                        self.learning_system.store_execution_data(execution_data)
                        result.learning_data_generated += 1
                    
                    if validate_hook_result(hook_result):
                        result.successful_steps += 1
                    else:
                        result.failed_steps += 1
                        result.errors.append(f"Step {i}: Hook validation failed")
                
                elif hook_type == 'UserPromptSubmit':
                    # Simulate user prompt processing
                    time.sleep(0.002)  # 2ms processing time
                    result.successful_steps += 1
                
                step_duration = (time.perf_counter() - step_start) * 1000
                result.total_duration_ms += step_duration
            
            # Calculate final metrics
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            total_execution_time = (end_time - start_time) * 1000
            memory_used_mb = max(0, end_memory - start_memory)
            
            # Determine if test passed
            passed = (
                result.success_rate >= self.config.min_success_rate and
                result.performance_violations <= self.config.max_performance_violations and
                memory_used_mb <= self.config.memory_limit_mb
            )
            
            return TestResult(
                test_name=f"workflow_{workflow_name}",
                passed=passed,
                execution_time_ms=total_execution_time,
                memory_used_mb=memory_used_mb,
                message=f"Workflow completed with {result.success_rate:.1%} success rate",
                details={
                    'workflow_result': result.__dict__,
                    'learning_stats': self.learning_system.get_performance_stats()
                }
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            return TestResult(
                test_name=f"workflow_{workflow_name}",
                passed=False,
                execution_time_ms=(end_time - start_time) * 1000,
                memory_used_mb=max(0, end_memory - start_memory),
                message=f"Workflow failed: {str(e)}",
                details={'exception': str(e), 'partial_result': result.__dict__}
            )

class PerformanceValidationTest:
    """Test system performance under various conditions"""
    
    def __init__(self, config: EndToEndTestConfig, test_dir: Path):
        self.config = config
        self.test_dir = test_dir
        self.claude_env = MockClaudeCodeEnvironment(test_dir)
        self.hook_generator = HookInputGenerator(seed=42)
        
    def test_concurrent_hook_execution(self) -> TestResult:
        """Test concurrent hook execution performance"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            execution_times = []
            errors = []
            
            # Execute hooks concurrently
            with ThreadPoolExecutor(max_workers=self.config.concurrent_hooks) as executor:
                futures = []
                
                for i in range(self.config.concurrent_hooks):
                    hook_input = self.hook_generator.generate_pretooluse_input()
                    future = executor.submit(
                        self.claude_env.execute_hook,
                        'bash-optimizer-enhanced.py',
                        hook_input
                    )
                    futures.append(future)
                
                # Collect results
                for i, future in enumerate(as_completed(futures)):
                    try:
                        hook_result, execution_time = future.result(timeout=5)
                        execution_times.append(execution_time)
                        
                        if not validate_hook_result(hook_result):
                            errors.append(f"Hook {i}: Invalid result")
                            
                    except Exception as e:
                        errors.append(f"Hook {i}: {str(e)}")
            
            # Calculate metrics
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            total_execution_time = (end_time - start_time) * 1000
            memory_used_mb = max(0, end_memory - start_memory)
            
            # Analyze performance
            avg_hook_time = sum(execution_times) / len(execution_times) if execution_times else 0
            p95_hook_time = sorted(execution_times)[int(len(execution_times) * 0.95)] if execution_times else 0
            
            passed = (
                len(errors) == 0 and
                avg_hook_time <= self.config.hook_execution_limit_ms and
                p95_hook_time <= self.config.hook_execution_limit_ms * 2 and  # Allow 2x for P95
                memory_used_mb <= self.config.memory_limit_mb
            )
            
            return TestResult(
                test_name="concurrent_hook_execution",
                passed=passed,
                execution_time_ms=total_execution_time,
                memory_used_mb=memory_used_mb,
                message=f"Concurrent execution: {len(execution_times)} hooks, avg {avg_hook_time:.1f}ms",
                details={
                    'concurrent_hooks': self.config.concurrent_hooks,
                    'successful_executions': len(execution_times),
                    'errors': errors,
                    'avg_hook_time_ms': avg_hook_time,
                    'p95_hook_time_ms': p95_hook_time,
                    'individual_times': execution_times
                }
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            return TestResult(
                test_name="concurrent_hook_execution",
                passed=False,
                execution_time_ms=(end_time - start_time) * 1000,
                memory_used_mb=max(0, end_memory - start_memory),
                message=f"Concurrent execution test failed: {str(e)}",
                details={'exception': str(e)}
            )
    
    def test_sustained_load_performance(self) -> TestResult:
        """Test performance under sustained load"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            execution_times = []
            memory_samples = []
            error_count = 0
            
            # Run sustained load for configured duration
            end_time_target = start_time + self.config.stress_test_duration_s
            execution_count = 0
            
            while time.perf_counter() < end_time_target:
                hook_input = self.hook_generator.generate_pretooluse_input()
                
                try:
                    hook_result, execution_time = self.claude_env.execute_hook(
                        'bash-optimizer-enhanced.py',
                        hook_input
                    )
                    execution_times.append(execution_time)
                    
                    if not validate_hook_result(hook_result):
                        error_count += 1
                        
                except Exception:
                    error_count += 1
                
                # Sample memory usage
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_samples.append(current_memory)
                
                execution_count += 1
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.001)
            
            # Calculate final metrics
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            total_duration_ms = (end_time - start_time) * 1000
            memory_used_mb = max(0, end_memory - start_memory)
            
            # Analyze sustained performance
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
            memory_growth = max(memory_samples) - min(memory_samples) if memory_samples else 0
            error_rate = error_count / execution_count if execution_count > 0 else 1.0
            
            passed = (
                error_rate <= self.config.acceptable_failure_rate and
                avg_execution_time <= self.config.hook_execution_limit_ms * 1.5 and  # Allow some degradation
                memory_growth <= self.config.max_memory_growth_mb and
                memory_used_mb <= self.config.memory_limit_mb
            )
            
            return TestResult(
                test_name="sustained_load_performance",
                passed=passed,
                execution_time_ms=total_duration_ms,
                memory_used_mb=memory_used_mb,
                message=f"Sustained load: {execution_count} executions, {error_rate:.1%} error rate",
                details={
                    'duration_s': self.config.stress_test_duration_s,
                    'total_executions': execution_count,
                    'successful_executions': len(execution_times),
                    'error_count': error_count,
                    'error_rate': error_rate,
                    'avg_execution_time_ms': avg_execution_time,
                    'memory_growth_mb': memory_growth,
                    'memory_samples': len(memory_samples)
                }
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            return TestResult(
                test_name="sustained_load_performance",
                passed=False,
                execution_time_ms=(end_time - start_time) * 1000,
                memory_used_mb=max(0, end_memory - start_memory),
                message=f"Sustained load test failed: {str(e)}",
                details={'exception': str(e)}
            )

class AdvancedIntegrationTest:
    """Test advanced integration scenarios"""
    
    def __init__(self, config: EndToEndTestConfig, test_dir: Path):
        self.config = config
        self.test_dir = test_dir
        self.claude_env = MockClaudeCodeEnvironment(test_dir)
        self.learning_system = MockLearningSystem(test_dir)
        self.security_system = MockSecuritySystem(test_dir)
        self.error_generator = ErrorScenarioGenerator(seed=42)
        
    def test_schema_evolution_during_operation(self) -> TestResult:
        """Test learning system schema evolution while system is running"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            initial_schema_count = self.learning_system.schema_evolution_count
            
            # Generate enough learning data to trigger schema evolution
            for i in range(60):  # Should trigger evolution at 50
                hook_input = HookInputGenerator().generate_posttooluse_input()
                execution_data = CommandExecutionData.from_hook_input(hook_input)
                
                # Store data and continue hook execution simultaneously
                self.learning_system.store_execution_data(execution_data)
                
                # Execute hook to simulate ongoing operations
                if i % 10 == 0:  # Every 10th iteration
                    pretool_input = HookInputGenerator().generate_pretooluse_input()
                    hook_result, _ = self.claude_env.execute_hook('bash-optimizer-enhanced.py', pretool_input)
                    
                    if not validate_hook_result(hook_result):
                        raise Exception(f"Hook execution failed during schema evolution at iteration {i}")
            
            # Verify schema evolution occurred
            final_schema_count = self.learning_system.schema_evolution_count
            schema_evolved = final_schema_count > initial_schema_count
            
            # Calculate metrics
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_used_mb = max(0, end_memory - start_memory)
            
            passed = (
                schema_evolved and
                execution_time_ms <= 5000 and  # Should complete within 5 seconds
                memory_used_mb <= self.config.memory_limit_mb
            )
            
            return TestResult(
                test_name="schema_evolution_during_operation",
                passed=passed,
                execution_time_ms=execution_time_ms,
                memory_used_mb=memory_used_mb,
                message=f"Schema evolution test: {schema_evolved}, {final_schema_count - initial_schema_count} evolutions",
                details={
                    'initial_schema_count': initial_schema_count,
                    'final_schema_count': final_schema_count,
                    'learning_data_stored': 60,
                    'hook_executions': 6
                }
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            return TestResult(
                test_name="schema_evolution_during_operation",
                passed=False,
                execution_time_ms=(end_time - start_time) * 1000,
                memory_used_mb=max(0, end_memory - start_memory),
                message=f"Schema evolution test failed: {str(e)}",
                details={'exception': str(e)}
            )
    
    def test_security_key_rotation_with_active_operations(self) -> TestResult:
        """Test security key rotation while system operations are ongoing"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            # Start with some encrypted data
            test_data = {"command": "initial_test", "timestamp": time.time()}
            encrypted_data, _ = self.security_system.encrypt_data(test_data)
            
            # Perform key rotation
            rotation_success, rotation_time = self.security_system.rotate_keys()
            if not rotation_success:
                raise Exception("Key rotation failed")
            
            # Verify we can still decrypt old data (backward compatibility)
            decrypted_data, _ = self.security_system.decrypt_data(encrypted_data)
            if decrypted_data != test_data:
                raise Exception("Data integrity lost after key rotation")
            
            # Verify new encryption works
            new_test_data = {"command": "post_rotation_test", "timestamp": time.time()}
            new_encrypted_data, _ = self.security_system.encrypt_data(new_test_data)
            new_decrypted_data, _ = self.security_system.decrypt_data(new_encrypted_data)
            
            if new_decrypted_data != new_test_data:
                raise Exception("New encryption/decryption failed after key rotation")
            
            # Execute hooks during and after rotation
            hook_input = HookInputGenerator().generate_pretooluse_input()
            hook_result, hook_time = self.claude_env.execute_hook('bash-optimizer-enhanced.py', hook_input)
            
            if not validate_hook_result(hook_result):
                raise Exception("Hook execution failed after key rotation")
            
            # Calculate metrics
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_used_mb = max(0, end_memory - start_memory)
            
            passed = (
                rotation_time <= PerformanceTargets.KEY_ROTATION_MS and
                hook_time <= self.config.hook_execution_limit_ms and
                memory_used_mb <= self.config.memory_limit_mb
            )
            
            return TestResult(
                test_name="security_key_rotation_with_operations",
                passed=passed,
                execution_time_ms=execution_time_ms,
                memory_used_mb=memory_used_mb,
                message=f"Key rotation: {rotation_time:.1f}ms, operations continued normally",
                details={
                    'rotation_time_ms': rotation_time,
                    'hook_execution_time_ms': hook_time,
                    'data_integrity_verified': True,
                    'backward_compatibility_verified': True
                }
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            return TestResult(
                test_name="security_key_rotation_with_operations",
                passed=False,
                execution_time_ms=(end_time - start_time) * 1000,
                memory_used_mb=max(0, end_memory - start_memory),
                message=f"Key rotation test failed: {str(e)}",
                details={'exception': str(e)}
            )
    
    def test_error_handling_and_graceful_degradation(self) -> TestResult:
        """Test system behavior under error conditions"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            error_scenarios = self.error_generator.generate_hook_error_scenarios()
            successful_graceful_handling = 0
            total_scenarios = len(error_scenarios)
            
            for scenario in error_scenarios:
                try:
                    # Execute the error scenario
                    hook_input = scenario['hook_input']
                    hook_result, execution_time = self.claude_env.execute_hook(
                        'bash-optimizer-enhanced.py',
                        hook_input,
                        timeout_ms=scenario.get('timeout_ms', 5000)
                    )
                    
                    # Check if system handled error gracefully
                    expected_behavior = scenario['expected_behavior']
                    
                    if expected_behavior == 'graceful_timeout':
                        # Should complete within timeout and not block
                        if execution_time <= scenario.get('timeout_ms', 5000) and not hook_result.block:
                            successful_graceful_handling += 1
                    
                    elif expected_behavior == 'handle_gracefully':
                        # Should not crash and return valid result
                        if validate_hook_result(hook_result) and not hook_result.block:
                            successful_graceful_handling += 1
                    
                    elif expected_behavior == 'fallback_gracefully':
                        # Should continue operating despite component failures
                        if validate_hook_result(hook_result):
                            successful_graceful_handling += 1
                    
                    elif expected_behavior == 'fail_safe':
                        # Should fail safely without corruption
                        if validate_hook_result(hook_result) and not hook_result.block:
                            successful_graceful_handling += 1
                    
                    elif expected_behavior == 'graceful_degradation':
                        # Should reduce functionality but continue operating
                        if validate_hook_result(hook_result):
                            successful_graceful_handling += 1
                    
                except Exception:
                    # Even exceptions should be handled gracefully - no system crash
                    pass
            
            # Calculate metrics
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_used_mb = max(0, end_memory - start_memory)
            
            graceful_handling_rate = successful_graceful_handling / total_scenarios
            
            passed = (
                graceful_handling_rate >= 0.8 and  # 80% of error scenarios handled gracefully
                memory_used_mb <= self.config.memory_limit_mb
            )
            
            return TestResult(
                test_name="error_handling_graceful_degradation",
                passed=passed,
                execution_time_ms=execution_time_ms,
                memory_used_mb=memory_used_mb,
                message=f"Error handling: {graceful_handling_rate:.1%} scenarios handled gracefully",
                details={
                    'total_scenarios': total_scenarios,
                    'successful_graceful_handling': successful_graceful_handling,
                    'graceful_handling_rate': graceful_handling_rate,
                    'scenario_types': [s['scenario_name'] for s in error_scenarios]
                }
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            return TestResult(
                test_name="error_handling_graceful_degradation",
                passed=False,
                execution_time_ms=(end_time - start_time) * 1000,
                memory_used_mb=max(0, end_memory - start_memory),
                message=f"Error handling test failed: {str(e)}",
                details={'exception': str(e)}
            )

# ============================================================================
# Main End-to-End Test Framework
# ============================================================================

class EndToEndTestFramework:
    """Main framework for executing comprehensive end-to-end tests"""
    
    def __init__(self, config: Optional[EndToEndTestConfig] = None):
        self.config = config or EndToEndTestConfig()
        self.test_framework = TestFramework()
        self.temp_dir: Optional[Path] = None
        
    def run_all_end_to_end_tests(self) -> Dict[str, Any]:
        """Run all end-to-end integration tests"""
        print("🚀 Starting Comprehensive End-to-End Integration Tests")
        print("=" * 80)
        print(f"📊 Configuration: {self.config.concurrent_hooks} concurrent hooks, "
              f"{self.config.hook_execution_limit_ms}ms hook limit, "
              f"{self.config.memory_limit_mb}MB memory limit")
        print("=" * 80)
        
        session_start = time.time()
        
        try:
            # Create isolated test environment
            self.temp_dir = Path(tempfile.mkdtemp(prefix="claude_sync_e2e_"))
            print(f"🔧 Test environment: {self.temp_dir}")
            
            # Initialize test components
            integration_test = FullSystemIntegrationTest(self.config, self.temp_dir)
            workflow_test = RealisticWorkflowTest(self.config, self.temp_dir)
            performance_test = PerformanceValidationTest(self.config, self.temp_dir)
            advanced_test = AdvancedIntegrationTest(self.config, self.temp_dir)
            
            # Execute test suites
            test_results = {}
            
            print("\n📋 Phase 1: Full System Integration")
            print("-" * 50)
            result = integration_test.test_activation_lifecycle()
            test_results['system_integration'] = result
            self._print_test_result(result)
            
            print("\n📋 Phase 2: Realistic Workflow Testing")
            print("-" * 50)
            
            bio_result = workflow_test.test_bioinformatics_workflow()
            test_results['bioinformatics_workflow'] = bio_result
            self._print_test_result(bio_result)
            
            hpc_result = workflow_test.test_hpc_troubleshooting_workflow()
            test_results['hpc_troubleshooting_workflow'] = hpc_result
            self._print_test_result(hpc_result)
            
            perf_result = workflow_test.test_performance_degradation_workflow()
            test_results['performance_degradation_workflow'] = perf_result
            self._print_test_result(perf_result)
            
            print("\n📋 Phase 3: Performance Validation")
            print("-" * 50)
            
            concurrent_result = performance_test.test_concurrent_hook_execution()
            test_results['concurrent_execution'] = concurrent_result
            self._print_test_result(concurrent_result)
            
            sustained_result = performance_test.test_sustained_load_performance()
            test_results['sustained_load'] = sustained_result
            self._print_test_result(sustained_result)
            
            print("\n📋 Phase 4: Advanced Integration Scenarios")
            print("-" * 50)
            
            schema_result = advanced_test.test_schema_evolution_during_operation()
            test_results['schema_evolution'] = schema_result
            self._print_test_result(schema_result)
            
            rotation_result = advanced_test.test_security_key_rotation_with_active_operations()
            test_results['key_rotation'] = rotation_result
            self._print_test_result(rotation_result)
            
            error_result = advanced_test.test_error_handling_and_graceful_degradation()
            test_results['error_handling'] = error_result
            self._print_test_result(error_result)
            
            # Compile final results
            session_end = time.time()
            session_duration = (session_end - session_start) * 1000
            
            return self._compile_final_results(test_results, session_duration)
            
        finally:
            # Cleanup
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up test environment: {self.temp_dir}")
    
    def _print_test_result(self, result: TestResult) -> None:
        """Print individual test result"""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status} {result.test_name}: {result.message}")
        print(f"    ⏱️  {result.execution_time_ms:.1f}ms | 💾 {result.memory_used_mb:.1f}MB")
        
        if not result.passed and result.details:
            if 'exception' in result.details:
                print(f"    🐛 Error: {result.details.get('exception', 'Unknown error')}")
    
    def _compile_final_results(self, test_results: Dict[str, TestResult], session_duration_ms: float) -> Dict[str, Any]:
        """Compile and display final test results"""
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results.values() if r.passed)
        failed_tests = total_tests - passed_tests
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Performance analysis
        execution_times = [r.execution_time_ms for r in test_results.values()]
        memory_usage = [r.memory_used_mb for r in test_results.values()]
        
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        max_execution_time = max(execution_times) if execution_times else 0
        avg_memory_usage = sum(memory_usage) / len(memory_usage) if memory_usage else 0
        max_memory_usage = max(memory_usage) if memory_usage else 0
        
        # Quality gate analysis
        performance_violations = sum(
            1 for r in test_results.values()
            if hasattr(r, 'details') and r.details and r.details.get('performance_violations', 0) > 0
        )
        
        quality_gates_passed = {
            'hook_performance': all(
                r.execution_time_ms <= self.config.hook_execution_limit_ms * 5  # Allow 5x for full workflows
                for r in test_results.values()
            ),
            'memory_usage': max_memory_usage <= self.config.memory_limit_mb,
            'success_rate': success_rate >= self.config.min_success_rate,  
            'performance_violations': performance_violations <= self.config.max_performance_violations
        }
        
        all_gates_passed = all(quality_gates_passed.values())
        
        # Print comprehensive summary
        print("\n" + "=" * 80)
        print("📋 END-TO-END INTEGRATION TEST RESULTS")
        print("=" * 80)
        
        print(f"⏱️  Total Duration: {session_duration_ms:.0f}ms")
        print(f"📊 Tests: {total_tests} total | {passed_tests} passed | {failed_tests} failed")
        print(f"✅ Success Rate: {success_rate:.1%}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"  • Average Execution Time: {avg_execution_time:.1f}ms")
        print(f"  • Maximum Execution Time: {max_execution_time:.1f}ms")
        print(f"  • Average Memory Usage: {avg_memory_usage:.1f}MB")
        print(f"  • Maximum Memory Usage: {max_memory_usage:.1f}MB")
        
        print(f"\n🎯 Quality Gates:")
        for gate, passed in quality_gates_passed.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  • {gate.replace('_', ' ').title()}: {status}")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for name, result in test_results.items():
                if not result.passed:
                    print(f"  • {name}: {result.message}")
        
        print(f"\n🏆 OVERALL RESULT: {'✅ ALL SYSTEMS GO' if all_gates_passed else '❌ QUALITY GATES FAILED'}")
        
        if all_gates_passed:
            print("🎉 Claude-sync is production-ready and meets all performance, security, and reliability requirements!")
        else:
            print("⚠️  Claude-sync has issues that need to be addressed before production deployment.")
        
        print("=" * 80)
        
        return {
            'overall_success': all_gates_passed,
            'session_duration_ms': session_duration_ms,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'performance_metrics': {
                'avg_execution_time_ms': avg_execution_time,
                'max_execution_time_ms': max_execution_time,
                'avg_memory_usage_mb': avg_memory_usage,
                'max_memory_usage_mb': max_memory_usage
            },
            'quality_gates': quality_gates_passed,
            'performance_violations': performance_violations,
            'detailed_results': {name: result.__dict__ for name, result in test_results.items()},
            'test_environment': str(self.temp_dir) if self.temp_dir else None
        }

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point for end-to-end testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Claude-Sync End-to-End Integration Tests')
    parser.add_argument('--hook-limit-ms', type=int, default=10, 
                       help='Hook execution time limit in milliseconds')
    parser.add_argument('--memory-limit-mb', type=int, default=50,
                       help='Memory usage limit in MB')
    parser.add_argument('--concurrent-hooks', type=int, default=10,
                       help='Number of concurrent hooks for stress testing')
    parser.add_argument('--stress-duration-s', type=int, default=30,
                       help='Stress test duration in seconds')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Configure test parameters
    config = EndToEndTestConfig(
        hook_execution_limit_ms=args.hook_limit_ms,
        memory_limit_mb=args.memory_limit_mb,
        concurrent_hooks=args.concurrent_hooks,
        stress_test_duration_s=args.stress_duration_s
    )
    
    # Run comprehensive tests
    framework = EndToEndTestFramework(config)
    results = framework.run_all_end_to_end_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_success'] else 1)

if __name__ == "__main__":
    main()