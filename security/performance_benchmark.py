#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0",
#   "psutil>=5.9.0",
# ]
# ///
"""
Security Performance Benchmark Suite

Comprehensive performance testing for all security components to ensure
they meet the performance targets defined in interfaces.py.

Performance Targets (from PerformanceTargets class):
- Encryption Operations: <5ms
- Key Rotation: <1000ms  
- Host Identity Generation: <100ms
- Hook Memory Usage: <10MB
- Learning Cache: <50MB
- Total System Memory: <100MB

Test Categories:
- Hardware identity generation speed and consistency
- Encryption/decryption throughput and latency
- Key rotation and cleanup performance
- Trust management operation speed
- Audit logging performance impact
- Memory usage under load
- Concurrent operation handling
"""

import time
import gc
import psutil
import statistics
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json
import threading
import concurrent.futures

from hardware_identity import HardwareIdentity
from host_trust import SimpleHostTrust
from key_manager import SimpleKeyManager
from secure_storage import SecureLearningStorage
from security_manager import SecurityManager
from audit_logger import SecurityAuditLogger

# Import performance targets
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from interfaces import PerformanceTargets

class PerformanceBenchmark:
    """Comprehensive security performance benchmarking"""
    
    def __init__(self):
        self.results = {}
        self.baseline_memory = None
        self.process = psutil.Process()
        
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def time_operation(self, operation_func, *args, **kwargs) -> Tuple[Any, float]:
        """Time an operation and return result with duration in ms"""
        start_time = time.perf_counter()
        result = operation_func(*args, **kwargs)
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        return result, duration_ms

    def run_multiple_times(self, operation_func, iterations: int = 10, *args, **kwargs) -> Dict[str, float]:
        """Run operation multiple times and return statistics"""
        durations = []
        results = []
        
        for _ in range(iterations):
            result, duration = self.time_operation(operation_func, *args, **kwargs)
            durations.append(duration)
            results.append(result)
        
        return {
            'min_ms': min(durations),
            'max_ms': max(durations),
            'avg_ms': statistics.mean(durations),
            'median_ms': statistics.median(durations),
            'p95_ms': statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations),
            'std_dev_ms': statistics.stdev(durations) if len(durations) > 1 else 0,
            'iterations': iterations,
            'results': results
        }

    def benchmark_hardware_identity(self) -> Dict[str, Any]:
        """Benchmark hardware identity generation"""
        print("🔍 Benchmarking Hardware Identity Generation...")
        
        identity = HardwareIdentity()
        
        # Test host ID generation
        print("  Testing host ID generation...")
        host_id_stats = self.run_multiple_times(identity.generate_stable_host_id, iterations=20)
        
        # Test individual hardware sources
        print("  Testing individual hardware sources...")
        cpu_stats = self.run_multiple_times(identity.get_cpu_serial, iterations=10)
        mb_stats = self.run_multiple_times(identity.get_motherboard_uuid, iterations=10)
        mac_stats = self.run_multiple_times(identity.get_network_mac_primary, iterations=10)
        
        # Test identity validation
        print("  Testing identity validation...")
        test_host_id = identity.generate_stable_host_id()
        validation_stats = self.run_multiple_times(identity.validate_host_identity, 5, test_host_id)
        
        # Consistency test
        print("  Testing consistency across calls...")
        host_ids = [identity.generate_stable_host_id() for _ in range(10)]
        all_same = len(set(host_ids)) == 1
        
        results = {
            'host_id_generation': host_id_stats,
            'cpu_serial_lookup': cpu_stats,
            'motherboard_uuid_lookup': mb_stats,
            'mac_address_lookup': mac_stats,
            'identity_validation': validation_stats,
            'consistency_test': {
                'all_ids_identical': all_same,
                'unique_ids': len(set(host_ids)),
                'total_calls': len(host_ids)
            },
            'target_ms': PerformanceTargets.HOST_IDENTITY_GENERATION_MS,
            'meets_target': host_id_stats['p95_ms'] <= PerformanceTargets.HOST_IDENTITY_GENERATION_MS
        }
        
        return results

    def benchmark_encryption(self) -> Dict[str, Any]:
        """Benchmark encryption and decryption operations"""
        print("🔐 Benchmarking Encryption Operations...")
        
        key_manager = SimpleKeyManager()
        security_manager = SecurityManager()
        
        # Test data of various sizes
        test_data_sets = {
            'small': {'command': 'ls', 'success': True},
            'medium': {
                'command_pattern': 'ssh_with_tunneling',
                'optimization_applied': True,
                'performance_metrics': {
                    'connection_time_ms': 150,
                    'throughput_mbps': 100,
                    'cpu_usage_percent': 15.5
                },
                'host_type': 'compute-server-a1b2',
                'timestamp': time.time()
            },
            'large': {
                'detailed_metrics': {f'metric_{i}': f'value_{i}' * 10 for i in range(100)},
                'command_history': [f'command_{i}' for i in range(50)],
                'performance_data': {f'perf_{i}': i * 1.5 for i in range(200)}
            }
        }
        
        results = {}
        
        for size_name, test_data in test_data_sets.items():
            print(f"  Testing {size_name} data encryption...")
            
            # Test raw key manager encryption
            json_data = json.dumps(test_data).encode('utf-8')
            encrypt_stats = self.run_multiple_times(key_manager.encrypt_data, 20, json_data)
            
            # Test decryption
            encrypted_sample = key_manager.encrypt_data(json_data)
            decrypt_stats = self.run_multiple_times(key_manager.decrypt_data, 20, encrypted_sample)
            
            # Test security manager encryption (higher level)
            sm_encrypt_stats = self.run_multiple_times(security_manager.encrypt_data, 20, test_data, 'benchmark')
            
            # Test security manager decryption
            sm_encrypted_sample = security_manager.encrypt_data(test_data, 'benchmark')
            sm_decrypt_stats = self.run_multiple_times(security_manager.decrypt_data, 20, sm_encrypted_sample, 'benchmark')
            
            results[f'{size_name}_data'] = {
                'data_size_bytes': len(json_data),
                'raw_encryption': encrypt_stats,
                'raw_decryption': decrypt_stats,
                'security_manager_encryption': sm_encrypt_stats,
                'security_manager_decryption': sm_decrypt_stats,
                'target_ms': PerformanceTargets.ENCRYPTION_OPERATION_MS,
                'encryption_meets_target': encrypt_stats['p95_ms'] <= PerformanceTargets.ENCRYPTION_OPERATION_MS,
                'decryption_meets_target': decrypt_stats['p95_ms'] <= PerformanceTargets.ENCRYPTION_OPERATION_MS
            }
        
        return results

    def benchmark_key_operations(self) -> Dict[str, Any]:
        """Benchmark key management operations"""
        print("🔑 Benchmarking Key Management Operations...")
        
        key_manager = SimpleKeyManager()
        
        # Test key generation
        print("  Testing key generation...")
        key_gen_stats = self.run_multiple_times(key_manager.get_current_key, iterations=10)
        
        # Test key rotation
        print("  Testing key rotation...")
        rotation_stats = self.run_multiple_times(key_manager.rotate_keys, iterations=5)
        
        # Test key cleanup
        print("  Testing key cleanup...")
        cleanup_stats = self.run_multiple_times(key_manager.cleanup_old_keys, iterations=5)
        
        # Test key metadata operations
        print("  Testing key statistics...")
        stats_gen_stats = self.run_multiple_times(key_manager.get_key_statistics, iterations=10)
        
        results = {
            'key_generation': key_gen_stats,
            'key_rotation': rotation_stats,
            'key_cleanup': cleanup_stats,
            'statistics_generation': stats_gen_stats,
            'key_rotation_target_ms': PerformanceTargets.KEY_ROTATION_MS,
            'meets_rotation_target': rotation_stats['p95_ms'] <= PerformanceTargets.KEY_ROTATION_MS
        }
        
        return results

    def benchmark_trust_operations(self) -> Dict[str, Any]:
        """Benchmark trust management operations"""
        print("🤝 Benchmarking Trust Management Operations...")
        
        host_trust = SimpleHostTrust()
        
        # Create test host IDs
        test_host_ids = [f"test{i:012d}" for i in range(10)]
        
        # Test authorization
        print("  Testing host authorization...")
        auth_durations = []
        for host_id in test_host_ids[:5]:
            _, duration = self.time_operation(host_trust.authorize_host, host_id, f"Test host {host_id}")
            auth_durations.append(duration)
        
        auth_stats = {
            'min_ms': min(auth_durations),
            'max_ms': max(auth_durations),
            'avg_ms': statistics.mean(auth_durations),
            'iterations': len(auth_durations)
        }
        
        # Test trust checking
        print("  Testing trust verification...")
        trust_check_stats = self.run_multiple_times(host_trust.is_trusted_host, 20, test_host_ids[0])
        
        # Test listing hosts
        print("  Testing host listing...")
        list_stats = self.run_multiple_times(host_trust.list_trusted_hosts, iterations=10)
        
        # Test revocation
        print("  Testing host revocation...")
        revoke_durations = []
        for host_id in test_host_ids[:3]:
            _, duration = self.time_operation(host_trust.revoke_host, host_id)
            revoke_durations.append(duration)
        
        revoke_stats = {
            'min_ms': min(revoke_durations),
            'max_ms': max(revoke_durations),
            'avg_ms': statistics.mean(revoke_durations),
            'iterations': len(revoke_durations)
        }
        
        # Test statistics generation
        print("  Testing trust statistics...")
        trust_stats_gen = self.run_multiple_times(host_trust.get_trust_statistics, iterations=10)
        
        results = {
            'host_authorization': auth_stats,
            'trust_verification': trust_check_stats,
            'host_listing': list_stats,
            'host_revocation': revoke_stats,
            'statistics_generation': trust_stats_gen
        }
        
        return results

    def benchmark_audit_logging(self) -> Dict[str, Any]:
        """Benchmark audit logging performance"""
        print("📋 Benchmarking Audit Logging Performance...")
        
        audit_logger = SecurityAuditLogger()
        
        # Test event logging
        print("  Testing event logging...")
        test_event_data = {
            'test_metric': 'benchmark_value',
            'timestamp': time.time(),
            'operation_id': 'bench_001'
        }
        
        log_stats = self.run_multiple_times(
            audit_logger.log_event, 50, 
            'benchmark_test', test_event_data, 'info', 'benchmark'
        )
        
        # Test event retrieval
        print("  Testing event retrieval...")
        retrieval_stats = self.run_multiple_times(audit_logger.get_recent_events, 10, 20)
        
        # Test audit summary generation
        print("  Testing audit summary...")
        summary_stats = self.run_multiple_times(audit_logger.get_audit_summary, 5, 7)
        
        # Test log integrity verification
        print("  Testing integrity verification...")
        integrity_stats = self.run_multiple_times(audit_logger.verify_log_integrity, 3, 1)
        
        results = {
            'event_logging': log_stats,
            'event_retrieval': retrieval_stats,
            'summary_generation': summary_stats,
            'integrity_verification': integrity_stats
        }
        
        return results

    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage under various loads"""
        print("💾 Benchmarking Memory Usage...")
        
        # Get baseline memory
        gc.collect()  # Force garbage collection
        baseline_memory = self.get_memory_usage_mb()
        
        results = {
            'baseline_memory_mb': baseline_memory,
            'target_limits': {
                'hook_memory_mb': PerformanceTargets.HOOK_MEMORY_MB,
                'learning_cache_mb': PerformanceTargets.LEARNING_CACHE_MB,
                'total_system_mb': PerformanceTargets.TOTAL_SYSTEM_MEMORY_MB
            }
        }
        
        # Test individual component memory usage
        components = [
            ('hardware_identity', HardwareIdentity),
            ('host_trust', SimpleHostTrust),
            ('key_manager', SimpleKeyManager),
            ('secure_storage', SecureLearningStorage),
            ('audit_logger', SecurityAuditLogger)
        ]
        
        for name, component_class in components:
            print(f"  Testing {name} memory usage...")
            
            gc.collect()
            before_memory = self.get_memory_usage_mb()
            
            # Create multiple instances
            instances = [component_class() for _ in range(5)]
            
            # Use them briefly
            for instance in instances:
                if hasattr(instance, 'get_host_identity'):
                    instance.get_host_identity()
                elif hasattr(instance, 'get_current_key'):
                    instance.get_current_key()
                elif hasattr(instance, 'log_event'):
                    instance.log_event('memory_test', {'test': True})
            
            after_memory = self.get_memory_usage_mb()
            component_memory = after_memory - before_memory
            
            results[f'{name}_memory_mb'] = component_memory
            results[f'{name}_meets_target'] = component_memory <= PerformanceTargets.HOOK_MEMORY_MB
            
            # Clean up
            del instances
            gc.collect()
        
        # Test integrated system memory usage
        print("  Testing integrated system memory...")
        gc.collect()
        before_integrated = self.get_memory_usage_mb()
        
        security_manager = SecurityManager()
        
        # Perform various operations
        test_data = {'benchmark': True, 'memory_test': time.time()}
        for i in range(10):
            security_manager.encrypt_data(test_data, f'context_{i}')
            security_manager.get_comprehensive_status()
        
        after_integrated = self.get_memory_usage_mb()
        integrated_memory = after_integrated - before_integrated
        
        results['integrated_system_memory_mb'] = integrated_memory
        results['meets_total_target'] = integrated_memory <= PerformanceTargets.TOTAL_SYSTEM_MEMORY_MB
        
        return results

    def benchmark_concurrent_operations(self) -> Dict[str, Any]:
        """Benchmark concurrent operation performance"""
        print("⚡ Benchmarking Concurrent Operations...")
        
        security_manager = SecurityManager()
        
        def encryption_worker(data, context):
            """Worker function for concurrent encryption"""
            return security_manager.encrypt_data(data, context)
        
        def decryption_worker(encrypted_data, context):
            """Worker function for concurrent decryption"""
            return security_manager.decrypt_data(encrypted_data, context)
        
        # Test concurrent encryption
        print("  Testing concurrent encryption...")
        test_data = {'concurrent_test': True, 'worker_id': 0, 'timestamp': time.time()}
        
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit 20 encryption tasks
            encryption_futures = []
            for i in range(20):
                data = test_data.copy()
                data['worker_id'] = i
                future = executor.submit(encryption_worker, data, f'concurrent_{i}')
                encryption_futures.append(future)
            
            # Wait for all to complete
            encrypted_results = []
            for future in concurrent.futures.as_completed(encryption_futures):
                try:
                    result = future.result(timeout=10)
                    encrypted_results.append(result)
                except Exception as e:
                    print(f"    Encryption task failed: {e}")
        
        encryption_duration = (time.perf_counter() - start_time) * 1000
        
        # Test concurrent decryption
        print("  Testing concurrent decryption...")
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit decryption tasks
            decryption_futures = []
            for i, encrypted_data in enumerate(encrypted_results[:10]):  # Use first 10
                future = executor.submit(decryption_worker, encrypted_data, f'concurrent_{i}')
                decryption_futures.append(future)
            
            # Wait for all to complete
            decrypted_results = []
            for future in concurrent.futures.as_completed(decryption_futures):
                try:
                    result = future.result(timeout=10)
                    if result:
                        decrypted_results.append(result)
                except Exception as e:
                    print(f"    Decryption task failed: {e}")
        
        decryption_duration = (time.perf_counter() - start_time) * 1000
        
        results = {
            'concurrent_encryption': {
                'tasks': len(encryption_futures),
                'successful': len(encrypted_results),
                'total_duration_ms': encryption_duration,
                'avg_per_task_ms': encryption_duration / len(encrypted_results) if encrypted_results else float('inf')
            },
            'concurrent_decryption': {
                'tasks': len(decryption_futures),
                'successful': len(decrypted_results),
                'total_duration_ms': decryption_duration,
                'avg_per_task_ms': decryption_duration / len(decrypted_results) if decrypted_results else float('inf')
            }
        }
        
        return results

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("🚀 Claude-Sync Security Performance Benchmark")
        print("=" * 60)
        print(f"Started at: {datetime.now().isoformat()}")
        print()
        
        # Get baseline system info
        self.baseline_memory = self.get_memory_usage_mb()
        
        # Run all benchmarks
        self.results['hardware_identity'] = self.benchmark_hardware_identity()
        print()
        
        self.results['encryption'] = self.benchmark_encryption()
        print()
        
        self.results['key_operations'] = self.benchmark_key_operations()
        print()
        
        self.results['trust_operations'] = self.benchmark_trust_operations()
        print()
        
        self.results['audit_logging'] = self.benchmark_audit_logging()
        print()
        
        self.results['memory_usage'] = self.benchmark_memory_usage()
        print()
        
        self.results['concurrent_operations'] = self.benchmark_concurrent_operations()
        print()
        
        # Generate summary
        self.results['benchmark_summary'] = self._generate_summary()
        
        print("✅ Benchmark Complete!")
        print("=" * 60)
        
        return self.results

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary with pass/fail analysis"""
        summary = {
            'overall_performance': 'PASS',
            'performance_issues': [],
            'benchmark_completed_at': datetime.now().isoformat(),
            'target_compliance': {}
        }
        
        # Check hardware identity performance
        hw_results = self.results.get('hardware_identity', {})
        if not hw_results.get('meets_target', True):
            summary['performance_issues'].append(f"Host identity generation: {hw_results.get('host_id_generation', {}).get('p95_ms', 0):.1f}ms > {PerformanceTargets.HOST_IDENTITY_GENERATION_MS}ms target")
            summary['overall_performance'] = 'FAIL'
        
        summary['target_compliance']['host_identity_generation'] = hw_results.get('meets_target', False)
        
        # Check encryption performance
        enc_results = self.results.get('encryption', {})
        for size in ['small', 'medium', 'large']:
            size_results = enc_results.get(f'{size}_data', {})
            if not size_results.get('encryption_meets_target', True):
                summary['performance_issues'].append(f"Encryption ({size}): {size_results.get('raw_encryption', {}).get('p95_ms', 0):.1f}ms > {PerformanceTargets.ENCRYPTION_OPERATION_MS}ms target")
                summary['overall_performance'] = 'FAIL'
            
            if not size_results.get('decryption_meets_target', True):
                summary['performance_issues'].append(f"Decryption ({size}): {size_results.get('raw_decryption', {}).get('p95_ms', 0):.1f}ms > {PerformanceTargets.ENCRYPTION_OPERATION_MS}ms target")
                summary['overall_performance'] = 'FAIL'
        
        summary['target_compliance']['encryption_operations'] = all(
            enc_results.get(f'{size}_data', {}).get('encryption_meets_target', False) and
            enc_results.get(f'{size}_data', {}).get('decryption_meets_target', False)
            for size in ['small', 'medium', 'large']
        )
        
        # Check key rotation performance
        key_results = self.results.get('key_operations', {})
        if not key_results.get('meets_rotation_target', True):
            summary['performance_issues'].append(f"Key rotation: {key_results.get('key_rotation', {}).get('p95_ms', 0):.1f}ms > {PerformanceTargets.KEY_ROTATION_MS}ms target")
            summary['overall_performance'] = 'FAIL'
        
        summary['target_compliance']['key_rotation'] = key_results.get('meets_rotation_target', False)
        
        # Check memory usage
        mem_results = self.results.get('memory_usage', {})
        if not mem_results.get('meets_total_target', True):
            summary['performance_issues'].append(f"Total memory: {mem_results.get('integrated_system_memory_mb', 0):.1f}MB > {PerformanceTargets.TOTAL_SYSTEM_MEMORY_MB}MB target")
            summary['overall_performance'] = 'FAIL'
        
        summary['target_compliance']['memory_usage'] = mem_results.get('meets_total_target', False)
        
        # Overall compliance rate
        compliant_targets = sum(summary['target_compliance'].values())
        total_targets = len(summary['target_compliance'])
        summary['compliance_rate'] = compliant_targets / total_targets if total_targets > 0 else 0
        
        return summary

    def print_summary(self):
        """Print human-readable benchmark summary"""
        if 'benchmark_summary' not in self.results:
            print("No benchmark summary available")
            return
        
        summary = self.results['benchmark_summary']
        
        print("\n📊 BENCHMARK SUMMARY")
        print("=" * 40)
        print(f"Overall Performance: {summary['overall_performance']}")
        print(f"Compliance Rate: {summary['compliance_rate']:.1%}")
        print()
        
        if summary['performance_issues']:
            print("❌ Performance Issues Found:")
            for issue in summary['performance_issues']:
                print(f"  • {issue}")
            print()
        
        print("📋 Target Compliance:")
        compliance = summary['target_compliance']
        for target, meets in compliance.items():
            status = "✅" if meets else "❌"
            print(f"  {status} {target.replace('_', ' ').title()}")
        
        print()
        print("💡 Key Metrics:")
        
        # Hardware identity
        hw_stats = self.results.get('hardware_identity', {}).get('host_id_generation', {})
        print(f"  Host Identity Generation: {hw_stats.get('avg_ms', 0):.1f}ms (target: <{PerformanceTargets.HOST_IDENTITY_GENERATION_MS}ms)")
        
        # Encryption
        enc_small = self.results.get('encryption', {}).get('small_data', {}).get('raw_encryption', {})
        print(f"  Small Data Encryption: {enc_small.get('avg_ms', 0):.1f}ms (target: <{PerformanceTargets.ENCRYPTION_OPERATION_MS}ms)")
        
        # Key rotation
        key_rotation = self.results.get('key_operations', {}).get('key_rotation', {})
        print(f"  Key Rotation: {key_rotation.get('avg_ms', 0):.1f}ms (target: <{PerformanceTargets.KEY_ROTATION_MS}ms)")
        
        # Memory
        total_mem = self.results.get('memory_usage', {}).get('integrated_system_memory_mb', 0)
        print(f"  System Memory Usage: {total_mem:.1f}MB (target: <{PerformanceTargets.TOTAL_SYSTEM_MEMORY_MB}MB)")
        
        print()

def main():
    """Run security performance benchmark"""
    benchmark = PerformanceBenchmark()
    
    try:
        results = benchmark.run_full_benchmark()
        benchmark.print_summary()
        
        # Optionally save detailed results
        results_file = Path.home() / '.claude' / 'security_benchmark_results.json'
        try:
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save results file: {e}")
        
        # Return appropriate exit code
        summary = results.get('benchmark_summary', {})
        return 0 if summary.get('overall_performance') == 'PASS' else 1
        
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Benchmark failed with error: {e}")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())