#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0",
#   "psutil>=5.9.0"
# ]
# ///
"""
Hook Performance Test Framework

Comprehensive testing framework for claude-sync hooks with performance
monitoring, stress testing, and integration validation.

Performance Targets:
- intelligent-optimizer.py (PreToolUse): <10ms 
- learning-collector.py (PostToolUse): <50ms
- context-enhancer.py (UserPromptSubmit): <100ms

Tests include:
- Performance benchmarking
- Memory usage monitoring
- Stress testing with concurrent execution
- Learning system integration validation
- Error handling and graceful degradation
"""

import json
import sys
import time
import subprocess
import statistics
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Performance targets from interfaces
PERFORMANCE_TARGETS = {
    'intelligent-optimizer.py': 10,     # PreToolUse: <10ms
    'learning-collector.py': 50,       # PostToolUse: <50ms  
    'context-enhancer.py': 100,        # UserPromptSubmit: <100ms
}

@dataclass
class HookTestResult:
    """Result of a single hook test execution"""
    hook_name: str
    execution_time_ms: float
    memory_used_mb: float
    success: bool
    output: Optional[Dict[str, Any]]
    error_message: Optional[str]
    
    def meets_target(self) -> bool:
        """Check if execution meets performance target"""
        target = PERFORMANCE_TARGETS.get(self.hook_name, 1000)
        return self.execution_time_ms <= target

@dataclass
class HookBenchmarkResult:
    """Result of benchmark testing for a hook"""
    hook_name: str
    test_count: int
    avg_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    avg_memory_mb: float
    success_rate: float
    target_met_rate: float
    errors: List[str]

class HookPerformanceTester:
    """Performance testing framework for claude-sync hooks"""
    
    def __init__(self, hooks_dir: Path = None):
        self.hooks_dir = hooks_dir or Path(__file__).parent
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate test data for different hook types"""
        return {
            'intelligent-optimizer.py': [
                # Basic bash commands
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'grep pattern file.txt'},
                    'context': {'working_directory': '/home/user'}
                },
                {
                    'tool_name': 'Bash', 
                    'tool_input': {'command': 'find /data -name "*.txt"'},
                    'context': {'working_directory': '/data'}
                },
                # SLURM commands
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'sbatch job.sh'},
                    'context': {'working_directory': '/scratch/user'}
                },
                # Container commands
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'singularity exec container.sif python script.py'},
                    'context': {'working_directory': '/home'}
                },
                # R commands
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'Rscript analysis.R'},
                    'context': {'working_directory': '/analysis'}
                },
                # Safety test cases
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'rm -rf /important/data'},
                    'context': {'working_directory': '/tmp'}
                },
                # Non-Bash tools (should exit quickly)
                {
                    'tool_name': 'Read',
                    'tool_input': {'file_path': '/some/file.txt'},
                    'context': {}
                }
            ],
            
            'learning-collector.py': [
                # Successful command execution
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'ls -la'},
                    'tool_output': {'exit_code': 0, 'duration_ms': 150},
                    'context': {'working_directory': '/home/user', 'timestamp': time.time()}
                },
                # Failed command execution
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'cat nonexistent.txt'},
                    'tool_output': {'exit_code': 1, 'duration_ms': 50},
                    'context': {'working_directory': '/tmp'}
                },
                # Long-running command
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'sleep 60'},
                    'tool_output': {'exit_code': 0, 'duration_ms': 60000},
                    'context': {'working_directory': '/home'}
                },
                # Complex command
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'find /data -name "*.fastq" | head -100 | xargs -I {} wc -l {}'},
                    'tool_output': {'exit_code': 0, 'duration_ms': 5000},
                    'context': {'working_directory': '/data/genomics'}
                },
                # SLURM job completion
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'sbatch --mem=32G --time=4:00:00 job.sh'},
                    'tool_output': {'exit_code': 0, 'duration_ms': 100},
                    'context': {'working_directory': '/scratch/jobs'}
                }
            ],
            
            'context-enhancer.py': [
                # SLURM-related prompts
                {
                    'user_prompt': 'How do I submit a SLURM job for my genomics analysis?'
                },
                {
                    'user_prompt': 'My SLURM job keeps getting killed, what should I do?'
                },
                # Container-related prompts
                {
                    'user_prompt': 'How do I run my analysis in a Singularity container?'
                },
                {
                    'user_prompt': 'The container can\'t access my data files'
                },
                # R-related prompts
                {
                    'user_prompt': 'My R script is running out of memory, how to fix?'
                },
                {
                    'user_prompt': 'Help me optimize my R data analysis workflow'
                },
                # Performance-related prompts
                {
                    'user_prompt': 'This command is too slow, how can I make it faster?'
                },
                # Error-related prompts
                {
                    'user_prompt': 'I\'m getting permission denied errors'
                },
                # Network-related prompts
                {
                    'user_prompt': 'SSH connection keeps timing out to the cluster'
                },
                # Short prompt (should exit quickly)
                {
                    'user_prompt': 'Hi'
                },
                # Multi-context prompt
                {
                    'user_prompt': 'I need to run my R analysis on the SLURM cluster using a container but it\'s slow and failing'
                }
            ]
        }
    
    def test_single_hook(self, hook_name: str, test_input: Dict[str, Any]) -> HookTestResult:
        """Test a single hook execution"""
        hook_path = self.hooks_dir / hook_name
        
        if not hook_path.exists():
            return HookTestResult(
                hook_name=hook_name,
                execution_time_ms=0,
                memory_used_mb=0,
                success=False,
                output=None,
                error_message=f"Hook file not found: {hook_path}"
            )
        
        # Measure memory before execution
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Execute hook
        start_time = time.perf_counter()
        try:
            result = subprocess.run(
                [str(hook_path)],
                input=json.dumps(test_input),
                capture_output=True,
                text=True,
                timeout=5.0  # 5 second timeout
            )
            
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = max(0, memory_after - memory_before)
            
            # Parse output
            output = None
            if result.stdout.strip():
                try:
                    output = json.loads(result.stdout.strip())
                except json.JSONDecodeError:
                    output = {'raw_output': result.stdout}
            
            return HookTestResult(
                hook_name=hook_name,
                execution_time_ms=execution_time_ms,
                memory_used_mb=memory_used,
                success=result.returncode == 0,
                output=output,
                error_message=result.stderr if result.stderr else None
            )
            
        except subprocess.TimeoutExpired:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return HookTestResult(
                hook_name=hook_name,
                execution_time_ms=execution_time_ms,
                memory_used_mb=0,
                success=False,
                output=None,
                error_message="Hook execution timed out"
            )
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return HookTestResult(
                hook_name=hook_name,
                execution_time_ms=execution_time_ms,
                memory_used_mb=0,
                success=False,
                output=None,
                error_message=str(e)
            )
    
    def benchmark_hook(self, hook_name: str, iterations: int = 50) -> HookBenchmarkResult:
        """Benchmark a hook with multiple test cases"""
        test_cases = self.test_data.get(hook_name, [])
        if not test_cases:
            raise ValueError(f"No test cases defined for {hook_name}")
        
        all_results = []
        errors = []
        
        print(f"Benchmarking {hook_name} with {iterations} iterations...")
        
        # Run tests
        for i in range(iterations):
            # Cycle through test cases
            test_input = test_cases[i % len(test_cases)]
            result = self.test_single_hook(hook_name, test_input)
            all_results.append(result)
            
            if not result.success:
                errors.append(f"Iteration {i}: {result.error_message}")
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{iterations} iterations")
        
        # Calculate statistics
        times = [r.execution_time_ms for r in all_results if r.success]
        memories = [r.memory_used_mb for r in all_results if r.success]
        successes = [r for r in all_results if r.success]
        target_met = [r for r in successes if r.meets_target()]
        
        if not times:
            raise RuntimeError(f"No successful executions for {hook_name}")
        
        return HookBenchmarkResult(
            hook_name=hook_name,
            test_count=iterations,
            avg_time_ms=statistics.mean(times),
            median_time_ms=statistics.median(times),
            p95_time_ms=statistics.quantiles(times, n=20)[18],  # 95th percentile
            p99_time_ms=statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
            avg_memory_mb=statistics.mean(memories) if memories else 0,
            success_rate=len(successes) / len(all_results),
            target_met_rate=len(target_met) / len(successes) if successes else 0,
            errors=errors[:10]  # Limit error list
        )
    
    def stress_test_hook(self, hook_name: str, concurrent_executions: int = 10) -> Dict[str, Any]:
        """Stress test a hook with concurrent executions"""
        test_cases = self.test_data.get(hook_name, [])
        if not test_cases:
            raise ValueError(f"No test cases defined for {hook_name}")
        
        print(f"Stress testing {hook_name} with {concurrent_executions} concurrent executions...")
        
        results = []
        start_time = time.perf_counter()
        
        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=concurrent_executions) as executor:
            # Submit all tasks
            futures = []
            for i in range(concurrent_executions):
                test_input = test_cases[i % len(test_cases)]
                future = executor.submit(self.test_single_hook, hook_name, test_input)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(HookTestResult(
                        hook_name=hook_name,
                        execution_time_ms=0,
                        memory_used_mb=0,
                        success=False,
                        output=None,
                        error_message=str(e)
                    ))
        
        total_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        times = [r.execution_time_ms for r in successful_results]
        
        return {
            'hook_name': hook_name,
            'concurrent_executions': concurrent_executions,
            'total_time_ms': total_time_ms,
            'success_count': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'avg_time_ms': statistics.mean(times) if times else 0,
            'max_time_ms': max(times) if times else 0,
            'errors': [r.error_message for r in results if not r.success][:5]
        }
    
    def test_learning_integration(self) -> Dict[str, Any]:
        """Test integration with learning system"""
        print("Testing learning system integration...")
        
        integration_results = {
            'learning_directory_exists': False,
            'learning_engine_available': False,
            'hooks_can_access_learning': {},
            'errors': []
        }
        
        try:
            # Check learning directory
            learning_dir = Path.home() / '.claude' / 'learning'
            integration_results['learning_directory_exists'] = learning_dir.exists()
            
            # Try to import learning engine
            try:
                sys.path.append(str(Path(__file__).parent.parent))
                from learning import create_learning_engine
                engine = create_learning_engine()
                integration_results['learning_engine_available'] = engine is not None
            except Exception as e:
                integration_results['errors'].append(f"Learning engine import failed: {e}")
            
            # Test each hook's ability to access learning system
            for hook_name in PERFORMANCE_TARGETS.keys():
                hook_path = self.hooks_dir / hook_name
                if hook_path.exists():
                    # Test with a simple input that should trigger learning access
                    test_input = self.test_data[hook_name][0] if hook_name in self.test_data else {}
                    result = self.test_single_hook(hook_name, test_input)
                    
                    integration_results['hooks_can_access_learning'][hook_name] = {
                        'success': result.success,
                        'execution_time_ms': result.execution_time_ms,
                        'has_output': result.output is not None,
                        'error': result.error_message
                    }
                else:
                    integration_results['hooks_can_access_learning'][hook_name] = {
                        'success': False,
                        'error': 'Hook file not found'
                    }
        
        except Exception as e:
            integration_results['errors'].append(f"Integration test failed: {e}")
        
        return integration_results
    
    def run_full_test_suite(self, iterations: int = 30) -> Dict[str, Any]:
        """Run the complete test suite"""
        print("🧪 Starting Claude-Sync Hook Performance Test Suite")
        print("=" * 60)
        
        suite_results = {
            'timestamp': time.time(),
            'performance_targets': PERFORMANCE_TARGETS,
            'benchmark_results': {},
            'stress_test_results': {},
            'learning_integration': {},
            'summary': {}
        }
        
        # Run benchmark tests for each hook
        print("\n📊 Running Performance Benchmarks...")
        for hook_name in PERFORMANCE_TARGETS.keys():
            try:
                benchmark = self.benchmark_hook(hook_name, iterations)
                suite_results['benchmark_results'][hook_name] = benchmark
                print(f"✅ {hook_name}: {benchmark.avg_time_ms:.1f}ms avg (target: {PERFORMANCE_TARGETS[hook_name]}ms)")
            except Exception as e:
                print(f"❌ {hook_name}: Benchmark failed - {e}")
                suite_results['benchmark_results'][hook_name] = {'error': str(e)}
        
        # Run stress tests
        print("\n🔥 Running Stress Tests...")
        for hook_name in PERFORMANCE_TARGETS.keys():
            try:
                stress_result = self.stress_test_hook(hook_name, concurrent_executions=5)
                suite_results['stress_test_results'][hook_name] = stress_result
                print(f"✅ {hook_name}: {stress_result['success_rate']:.0%} success rate under load")
            except Exception as e:
                print(f"❌ {hook_name}: Stress test failed - {e}")
                suite_results['stress_test_results'][hook_name] = {'error': str(e)}
        
        # Test learning integration
        print("\n🧠 Testing Learning System Integration...")
        suite_results['learning_integration'] = self.test_learning_integration()
        
        # Generate summary
        suite_results['summary'] = self._generate_summary(suite_results)
        
        print("\n" + "=" * 60)
        print("🎯 Test Suite Summary:")
        summary = suite_results['summary']
        print(f"   Performance Targets Met: {summary['targets_met']}/{summary['total_hooks']}")
        print(f"   Average Success Rate: {summary['avg_success_rate']:.1%}")
        print(f"   Learning Integration: {'✅ Working' if summary['learning_working'] else '❌ Issues'}")
        
        return suite_results
    
    def _generate_summary(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test suite summary"""
        benchmarks = suite_results['benchmark_results']
        
        targets_met = 0
        total_hooks = 0
        success_rates = []
        
        for hook_name, result in benchmarks.items():
            if isinstance(result, dict) and 'error' not in result:
                total_hooks += 1
                target = PERFORMANCE_TARGETS[hook_name]
                if hasattr(result, 'avg_time_ms') and result.avg_time_ms <= target:
                    targets_met += 1
                if hasattr(result, 'success_rate'):
                    success_rates.append(result.success_rate)
        
        learning_integration = suite_results['learning_integration']
        learning_working = (
            learning_integration.get('learning_directory_exists', False) and
            learning_integration.get('learning_engine_available', False) and
            len(learning_integration.get('errors', [])) == 0
        )
        
        return {
            'targets_met': targets_met,
            'total_hooks': total_hooks,
            'avg_success_rate': statistics.mean(success_rates) if success_rates else 0,
            'learning_working': learning_working,
            'test_timestamp': time.time()
        }

def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Claude-Sync Hook Performance Tester')
    parser.add_argument('--hook', help='Test specific hook only')
    parser.add_argument('--iterations', type=int, default=30, help='Number of test iterations')
    parser.add_argument('--stress', action='store_true', help='Run stress tests only')
    parser.add_argument('--integration', action='store_true', help='Test learning integration only')
    parser.add_argument('--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    tester = HookPerformanceTester()
    
    if args.hook:
        # Test specific hook
        if args.hook not in PERFORMANCE_TARGETS:
            print(f"❌ Unknown hook: {args.hook}")
            print(f"Available hooks: {', '.join(PERFORMANCE_TARGETS.keys())}")
            sys.exit(1)
        
        benchmark = tester.benchmark_hook(args.hook, args.iterations)
        print(f"\n📊 Benchmark Results for {args.hook}:")
        print(f"   Average: {benchmark.avg_time_ms:.1f}ms")
        print(f"   Target: {PERFORMANCE_TARGETS[args.hook]}ms")
        print(f"   Success Rate: {benchmark.success_rate:.1%}")
        print(f"   Target Met Rate: {benchmark.target_met_rate:.1%}")
        
    elif args.stress:
        # Run stress tests only
        for hook_name in PERFORMANCE_TARGETS.keys():
            result = tester.stress_test_hook(hook_name)
            print(f"\n🔥 Stress Test Results for {hook_name}:")
            print(f"   Success Rate: {result['success_rate']:.1%}")
            print(f"   Average Time: {result['avg_time_ms']:.1f}ms")
            
    elif args.integration:
        # Test integration only
        result = tester.test_learning_integration()
        print("\n🧠 Learning Integration Results:")
        print(f"   Learning Directory: {'✅' if result['learning_directory_exists'] else '❌'}")
        print(f"   Learning Engine: {'✅' if result['learning_engine_available'] else '❌'}")
        for hook, status in result['hooks_can_access_learning'].items():
            print(f"   {hook}: {'✅' if status['success'] else '❌'}")
    
    else:
        # Run full test suite
        results = tester.run_full_test_suite(args.iterations)
        
        if args.output:
            # Save results to file
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.output}")

if __name__ == '__main__':
    main()