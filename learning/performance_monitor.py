#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "psutil>=5.9.0"
# ]
# ///
"""
Performance Monitor - Real-time monitoring and optimization for learning operations

This implements comprehensive performance monitoring for the learning system,
ensuring all operations meet the strict performance targets required for
real-time hook execution. Key features:

- Real-time performance tracking with <1ms overhead
- Automatic performance target enforcement  
- Memory usage monitoring and alerting
- Learning operation profiling and optimization
- Performance degradation detection and reporting
- Integration with hook execution timing

Based on REFACTOR_PLAN.md performance requirements and PerformanceTargets
"""

import time
import threading
import psutil
import json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import logging
import statistics

# Import our interfaces
import sys
sys.path.append(str(Path(__file__).parent.parent))
from interfaces import (
    PerformanceMonitorInterface,
    PerformanceTargets
)

@dataclass
class PerformanceMetric:
    """Individual performance measurement"""
    timestamp: float
    operation_type: str
    duration_ms: float
    memory_used_mb: float
    cpu_percent: float
    success: bool
    context: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceAlert:
    """Performance alert when targets are exceeded"""
    timestamp: float
    alert_type: str
    operation_type: str
    measured_value: float
    target_value: float
    severity: str  # 'warning', 'critical'
    context: Optional[Dict[str, Any]] = None

class PerformanceMonitor(PerformanceMonitorInterface):
    """
    Real-time performance monitoring system for learning operations.
    
    This class provides comprehensive performance monitoring with minimal
    overhead, ensuring the learning system maintains optimal performance
    for real-time hook execution.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir or Path.home() / '.claude' / 'learning')
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance data storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: deque = deque(maxlen=100)
        
        # Real-time monitoring
        self.active_operations: Dict[str, float] = {}  # operation_id -> start_time
        self.operation_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Performance targets (from interfaces)
        self.targets = {
            'hook_pre_tool_use': PerformanceTargets.PRE_TOOL_USE_HOOK_MS,
            'hook_post_tool_use': PerformanceTargets.POST_TOOL_USE_HOOK_MS,
            'hook_user_prompt_submit': PerformanceTargets.USER_PROMPT_SUBMIT_HOOK_MS,
            'learning_operation': PerformanceTargets.LEARNING_OPERATION_MS,
            'pattern_lookup': PerformanceTargets.PATTERN_LOOKUP_MS,
            'schema_evolution': PerformanceTargets.SCHEMA_EVOLUTION_MS,
            'encryption_operation': PerformanceTargets.ENCRYPTION_OPERATION_MS,
            'key_rotation': PerformanceTargets.KEY_ROTATION_MS
        }
        
        # Memory monitoring
        self.memory_targets = {
            'hook_memory': PerformanceTargets.HOOK_MEMORY_MB,
            'learning_cache': PerformanceTargets.LEARNING_CACHE_MB,
            'total_system': PerformanceTargets.TOTAL_SYSTEM_MEMORY_MB
        }
        
        # Statistics tracking
        self.stats = {
            'total_operations': 0,
            'target_violations': 0,
            'average_performance': {},
            'peak_memory_usage': 0.0,
            'system_health_score': 100.0
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self._monitor_thread.start()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def record_hook_execution(self, hook_name: str, duration_ms: int) -> None:
        """Record hook execution time and check performance targets"""
        try:
            with self._lock:
                # Determine hook type for target checking
                hook_type = self._classify_hook_type(hook_name)
                target_ms = self.targets.get(hook_type, 100)  # Default 100ms
                
                # Record metric
                metric = PerformanceMetric(
                    timestamp=time.time(),
                    operation_type=hook_type,
                    duration_ms=float(duration_ms),
                    memory_used_mb=self._get_current_memory_usage(),
                    cpu_percent=psutil.cpu_percent(),
                    success=True,
                    context={'hook_name': hook_name}
                )
                
                self.metrics[hook_type].append(metric)
                self.stats['total_operations'] += 1
                
                # Check performance target
                if duration_ms > target_ms:
                    self._create_performance_alert(
                        'duration_exceeded',
                        hook_type,
                        float(duration_ms),
                        float(target_ms),
                        'critical' if duration_ms > target_ms * 2 else 'warning',
                        {'hook_name': hook_name}
                    )
                    self.stats['target_violations'] += 1
                
                # Update average performance
                self._update_average_performance(hook_type)
                
        except Exception as e:
            self.logger.error(f"Error recording hook execution: {e}")
    
    def record_learning_operation(self, operation: str, duration_ms: int) -> None:
        """Record learning system operation time"""
        try:
            with self._lock:
                target_ms = self.targets.get(operation, PerformanceTargets.LEARNING_OPERATION_MS)
                
                # Record metric
                metric = PerformanceMetric(
                    timestamp=time.time(),
                    operation_type=operation,
                    duration_ms=float(duration_ms),
                    memory_used_mb=self._get_current_memory_usage(),
                    cpu_percent=psutil.cpu_percent(),
                    success=True,
                    context={'operation': operation}
                )
                
                self.metrics[operation].append(metric)
                self.stats['total_operations'] += 1
                
                # Check performance target
                if duration_ms > target_ms:
                    self._create_performance_alert(
                        'learning_operation_slow',
                        operation,
                        float(duration_ms),
                        float(target_ms),
                        'critical' if duration_ms > target_ms * 2 else 'warning',
                        {'operation': operation}
                    )
                    self.stats['target_violations'] += 1
                
                # Update average performance
                self._update_average_performance(operation)
                
        except Exception as e:
            self.logger.error(f"Error recording learning operation: {e}")
    
    def start_operation_monitoring(self, operation_id: str, operation_type: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Start monitoring a long-running operation"""
        with self._lock:
            self.active_operations[operation_id] = time.time()
            if context:
                self.operation_contexts[operation_id] = context
    
    def end_operation_monitoring(self, operation_id: str, success: bool = True) -> float:
        """End monitoring and return duration"""
        with self._lock:
            if operation_id not in self.active_operations:
                return 0.0
            
            start_time = self.active_operations.pop(operation_id)
            duration_ms = (time.time() - start_time) * 1000
            
            context = self.operation_contexts.pop(operation_id, {})
            operation_type = context.get('operation_type', 'unknown')
            
            # Record the operation
            self.record_learning_operation(operation_type, int(duration_ms))
            
            return duration_ms
    
    def get_performance_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics for recent period"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (hours * 3600)
            
            stats = {
                'time_period_hours': hours,
                'total_operations': 0,
                'target_violations': 0,
                'operation_stats': {},
                'memory_stats': self._get_memory_stats(),
                'system_health': self._calculate_system_health(),
                'recent_alerts': self._get_recent_alerts(hours)
            }
            
            # Analyze metrics by operation type
            for operation_type, metrics in self.metrics.items():
                recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
                
                if recent_metrics:
                    durations = [m.duration_ms for m in recent_metrics]
                    memory_usage = [m.memory_used_mb for m in recent_metrics]
                    
                    operation_stats = {
                        'count': len(recent_metrics),
                        'avg_duration_ms': statistics.mean(durations),
                        'median_duration_ms': statistics.median(durations),
                        'p95_duration_ms': self._percentile(durations, 95),
                        'max_duration_ms': max(durations),
                        'avg_memory_mb': statistics.mean(memory_usage),
                        'max_memory_mb': max(memory_usage),
                        'target_ms': self.targets.get(operation_type, 100),
                        'violations': len([d for d in durations if d > self.targets.get(operation_type, 100)]),
                        'success_rate': len([m for m in recent_metrics if m.success]) / len(recent_metrics)
                    }
                    
                    stats['operation_stats'][operation_type] = operation_stats
                    stats['total_operations'] += len(recent_metrics)
                    stats['target_violations'] += operation_stats['violations']
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {'error': str(e)}
    
    def check_performance_targets(self) -> Dict[str, bool]:
        """Check if system meets performance targets"""
        try:
            results = {}
            
            # Check recent performance for each operation type
            current_time = time.time()
            recent_cutoff = current_time - 3600  # Last hour
            
            for operation_type, target_ms in self.targets.items():
                recent_metrics = [
                    m for m in self.metrics[operation_type] 
                    if m.timestamp >= recent_cutoff
                ]
                
                if recent_metrics:
                    # Check if 95th percentile meets target
                    durations = [m.duration_ms for m in recent_metrics]
                    p95_duration = self._percentile(durations, 95)
                    results[operation_type] = p95_duration <= target_ms
                else:
                    results[operation_type] = True  # No recent data, assume OK
            
            # Check memory targets
            current_memory = self._get_current_memory_usage()
            results['memory_usage'] = current_memory <= self.memory_targets['total_system']
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error checking performance targets: {e}")
            return {}
    
    def register_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Register callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time performance status"""
        return {
            'active_operations': len(self.active_operations),
            'current_memory_mb': self._get_current_memory_usage(),
            'cpu_percent': psutil.cpu_percent(),
            'system_health_score': self._calculate_system_health(),
            'recent_violations': len([a for a in self.alerts if time.time() - a.timestamp < 300]),  # Last 5 minutes
            'performance_targets_met': all(self.check_performance_targets().values())
        }
    
    def _classify_hook_type(self, hook_name: str) -> str:
        """Classify hook type for performance target selection"""
        if 'pre_tool_use' in hook_name.lower() or 'pretooluse' in hook_name.lower():
            return 'hook_pre_tool_use'
        elif 'post_tool_use' in hook_name.lower() or 'posttooluse' in hook_name.lower():
            return 'hook_post_tool_use'
        elif 'user_prompt_submit' in hook_name.lower() or 'userpromptsubmit' in hook_name.lower():
            return 'hook_user_prompt_submit'
        else:
            return 'hook_unknown'
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def _create_performance_alert(self, alert_type: str, operation_type: str, 
                                measured_value: float, target_value: float, 
                                severity: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Create and process performance alert"""
        alert = PerformanceAlert(
            timestamp=time.time(),
            alert_type=alert_type,
            operation_type=operation_type,
            measured_value=measured_value,
            target_value=target_value,
            severity=severity,
            context=context
        )
        
        self.alerts.append(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        # Log the alert
        if severity == 'critical':
            self.logger.critical(f"Performance alert: {alert_type} - {operation_type} took {measured_value:.1f}ms (target: {target_value:.1f}ms)")
        else:
            self.logger.warning(f"Performance warning: {alert_type} - {operation_type} took {measured_value:.1f}ms (target: {target_value:.1f}ms)")
    
    def _update_average_performance(self, operation_type: str) -> None:
        """Update rolling average performance for operation type"""
        recent_metrics = list(self.metrics[operation_type])[-100:]  # Last 100 operations
        if recent_metrics:
            avg_duration = statistics.mean([m.duration_ms for m in recent_metrics])
            self.stats['average_performance'][operation_type] = avg_duration
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'current_mb': memory_info.rss / (1024 * 1024),
                'peak_mb': self.stats['peak_memory_usage'],
                'target_mb': self.memory_targets['total_system'],
                'system_available_mb': psutil.virtual_memory().available / (1024 * 1024),
                'system_total_mb': psutil.virtual_memory().total / (1024 * 1024)
            }
        except:
            return {'error': 'Unable to get memory stats'}
    
    def _get_recent_alerts(self, hours: int) -> List[Dict[str, Any]]:
        """Get recent performance alerts"""
        cutoff_time = time.time() - (hours * 3600)
        recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]
        return [asdict(alert) for alert in recent_alerts]
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            health_score = 100.0
            
            # Penalize based on target violations
            if self.stats['total_operations'] > 0:
                violation_rate = self.stats['target_violations'] / self.stats['total_operations']
                health_score -= violation_rate * 50  # Up to 50 point penalty
            
            # Penalize based on memory usage
            current_memory = self._get_current_memory_usage()
            memory_target = self.memory_targets['total_system']
            if current_memory > memory_target:
                memory_penalty = min(30, (current_memory - memory_target) / memory_target * 30)
                health_score -= memory_penalty
            
            # Penalize based on recent alerts
            recent_critical_alerts = len([
                a for a in self.alerts 
                if a.severity == 'critical' and time.time() - a.timestamp < 3600
            ])
            health_score -= recent_critical_alerts * 10  # 10 points per critical alert
            
            return max(0.0, health_score)
            
        except:
            return 50.0  # Default middle score if calculation fails
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index % 1)
    
    def _background_monitor(self) -> None:
        """Background monitoring thread"""
        while self._monitoring_active:
            try:
                # Update peak memory usage
                current_memory = self._get_current_memory_usage()
                if current_memory > self.stats['peak_memory_usage']:
                    self.stats['peak_memory_usage'] = current_memory
                
                # Check for long-running operations
                current_time = time.time()
                for op_id, start_time in list(self.active_operations.items()):
                    duration_ms = (current_time - start_time) * 1000
                    if duration_ms > 10000:  # 10 seconds
                        self.logger.warning(f"Long-running operation detected: {op_id} ({duration_ms:.0f}ms)")
                
                # Sleep for monitoring interval
                time.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in background monitor: {e}")
                time.sleep(5)  # Longer sleep on error
    
    def shutdown(self) -> None:
        """Shutdown monitoring system"""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        # Save final statistics
        self._save_performance_data()
    
    def _save_performance_data(self) -> None:
        """Save performance data to storage"""
        try:
            # Prepare data for saving (convert deques to lists)
            save_data = {
                'stats': self.stats,
                'recent_metrics': {
                    op_type: [asdict(m) for m in list(metrics)[-100:]]  # Save last 100 metrics per type
                    for op_type, metrics in self.metrics.items()
                },
                'recent_alerts': [asdict(a) for a in list(self.alerts)[-50:]],  # Save last 50 alerts
                'timestamp': time.time()
            }
            
            perf_file = self.storage_dir / 'performance_data.json'
            with open(perf_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except:
            pass

if __name__ == "__main__":
    # Example usage and testing
    monitor = PerformanceMonitor()
    
    # Register alert callback
    def alert_handler(alert: PerformanceAlert):
        print(f"Performance Alert: {alert.alert_type} - {alert.operation_type} ({alert.severity})")
    
    monitor.register_alert_callback(alert_handler)
    
    # Simulate some operations
    monitor.record_hook_execution('pre_tool_use_hook', 15)  # Should be OK
    monitor.record_hook_execution('pre_tool_use_hook', 25)  # Should trigger warning
    monitor.record_learning_operation('pattern_lookup', 5)  # Should be OK
    monitor.record_learning_operation('schema_evolution', 250)  # Should trigger warning
    
    # Get performance stats
    stats = monitor.get_performance_stats(1)  # Last hour
    print(json.dumps(stats, indent=2, default=str))
    
    # Check targets
    targets_met = monitor.check_performance_targets()
    print(f"Performance targets met: {targets_met}")
    
    # Get real-time status
    status = monitor.get_real_time_status()
    print(f"Real-time status: {status}")
    
    # Cleanup
    monitor.shutdown()