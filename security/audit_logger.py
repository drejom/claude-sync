#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "psutil>=5.9.0",
# ]
# ///
"""
Security Audit Logger

Comprehensive audit trail for all security-related operations in claude-sync.
Provides tamper-evident logging with structured event recording and analysis.

Key Features:
- Structured audit event logging with JSON format
- Tamper-evident log entries with checksums
- Automatic log rotation and retention management
- Performance monitoring and security metrics
- Event correlation and analysis capabilities
- Secure file permissions and atomic operations

Security Model:
- All security operations are logged
- Log entries include context and metadata
- Checksums prevent tampering detection
- Log files have secure permissions (600)
- Automatic cleanup prevents disk exhaustion
- No sensitive data logged (only abstractions)
"""

import json
import time
import hashlib
import os
import fcntl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import gzip

class SecurityAuditLogger:
    """Comprehensive security audit logging system"""
    
    def __init__(self, retention_days: int = 90):
        self.retention_days = retention_days
        
        # Log storage
        self.claude_dir = Path.home() / '.claude'
        self.audit_dir = self.claude_dir / 'audit'
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Current log file (rotates daily)
        self.current_log_file = self._get_log_file_path()
        
        # Metrics storage
        self.metrics_file = self.audit_dir / 'security_metrics.json'
        
        # Set secure permissions
        self._ensure_secure_permissions()
        
        # Initialize metrics if needed
        self._initialize_metrics()

    def _ensure_secure_permissions(self):
        """Ensure audit directory and files have secure permissions"""
        try:
            # Set directory permissions: owner read/write/execute only
            os.chmod(self.audit_dir, 0o700)
            
            # Set permissions on existing files
            for file_path in self.audit_dir.glob("*"):
                if file_path.is_file():
                    os.chmod(file_path, 0o600)
                    
        except Exception:
            pass  # Permission errors shouldn't break logging

    def _get_log_file_path(self, date_str: str = None) -> Path:
        """Get log file path for specific date"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        return self.audit_dir / f"security_audit_{date_str}.log"

    def _calculate_entry_checksum(self, entry: Dict[str, Any]) -> str:
        """Calculate checksum for audit entry to detect tampering"""
        # Create deterministic string representation
        entry_str = json.dumps(entry, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(entry_str.encode('utf-8')).hexdigest()[:16]

    def _initialize_metrics(self):
        """Initialize metrics file if it doesn't exist"""
        if not self.metrics_file.exists():
            initial_metrics = {
                'version': '1.0',
                'created_at': time.time(),
                'total_events': 0,
                'event_types': {},
                'daily_stats': {},
                'performance_metrics': {
                    'avg_log_write_time_ms': 0,
                    'total_log_writes': 0,
                    'log_write_errors': 0
                },
                'security_alerts': {
                    'total_alerts': 0,
                    'alert_types': {},
                    'last_alert': None
                }
            }
            self._save_metrics(initial_metrics)

    def _load_metrics(self) -> Dict[str, Any]:
        """Load current metrics"""
        try:
            if not self.metrics_file.exists():
                self._initialize_metrics()
                
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except Exception:
            # Return empty metrics if loading fails
            return self._create_empty_metrics()

    def _create_empty_metrics(self) -> Dict[str, Any]:
        """Create empty metrics structure"""
        return {
            'version': '1.0',
            'created_at': time.time(),
            'total_events': 0,
            'event_types': {},
            'daily_stats': {},
            'performance_metrics': {
                'avg_log_write_time_ms': 0,
                'total_log_writes': 0,
                'log_write_errors': 0
            },
            'security_alerts': {
                'total_alerts': 0,
                'alert_types': {},
                'last_alert': None
            }
        }

    def _save_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Save metrics with atomic operation"""
        try:
            # Write to temporary file first
            temp_file = self.metrics_file.with_suffix('.tmp')
            
            with open(temp_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(metrics, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            
            # Set secure permissions
            os.chmod(temp_file, 0o600)
            
            # Atomic rename
            temp_file.replace(self.metrics_file)
            
            return True
            
        except Exception:
            # Clean up temp file if it exists
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
            return False

    def _update_metrics(self, event_type: str, write_time_ms: float):
        """Update performance and event metrics"""
        try:
            metrics = self._load_metrics()
            
            # Update total counters
            metrics['total_events'] += 1
            
            # Update event type counters
            event_types = metrics.get('event_types', {})
            event_types[event_type] = event_types.get(event_type, 0) + 1
            metrics['event_types'] = event_types
            
            # Update daily stats
            today = datetime.now().strftime('%Y-%m-%d')
            daily_stats = metrics.get('daily_stats', {})
            if today not in daily_stats:
                daily_stats[today] = {'events': 0, 'event_types': {}}
            
            daily_stats[today]['events'] += 1
            daily_event_types = daily_stats[today].get('event_types', {})
            daily_event_types[event_type] = daily_event_types.get(event_type, 0) + 1
            daily_stats[today]['event_types'] = daily_event_types
            metrics['daily_stats'] = daily_stats
            
            # Update performance metrics
            perf_metrics = metrics.get('performance_metrics', {})
            total_writes = perf_metrics.get('total_log_writes', 0)
            avg_time = perf_metrics.get('avg_log_write_time_ms', 0)
            
            # Calculate new running average
            new_avg = (avg_time * total_writes + write_time_ms) / (total_writes + 1)
            
            perf_metrics['avg_log_write_time_ms'] = new_avg
            perf_metrics['total_log_writes'] = total_writes + 1
            metrics['performance_metrics'] = perf_metrics
            
            # Save updated metrics
            self._save_metrics(metrics)
            
        except Exception:
            pass  # Metrics failure shouldn't break logging

    def log_event(self, event_type: str, event_data: Dict[str, Any], 
                  severity: str = 'info', source_component: str = 'unknown') -> bool:
        """Log a security event with full audit trail"""
        start_time = time.time()
        
        try:
            # Create audit entry
            entry = {
                'timestamp': datetime.now().isoformat(),
                'timestamp_unix': time.time(),
                'event_type': event_type,
                'severity': severity,
                'source_component': source_component,
                'event_data': event_data,
                'process_id': os.getpid(),
                'audit_version': '1.0'
            }
            
            # Add checksum for tamper detection
            entry['checksum'] = self._calculate_entry_checksum(entry)
            
            # Write to log file with file locking
            success = self._write_log_entry(entry)
            
            # Update metrics
            write_time_ms = (time.time() - start_time) * 1000
            self._update_metrics(event_type, write_time_ms)
            
            # Check for security alerts
            self._check_security_alerts(event_type, event_data, severity)
            
            return success
            
        except Exception:
            # Update error metrics
            try:
                metrics = self._load_metrics()
                perf_metrics = metrics.get('performance_metrics', {})
                perf_metrics['log_write_errors'] = perf_metrics.get('log_write_errors', 0) + 1
                metrics['performance_metrics'] = perf_metrics
                self._save_metrics(metrics)
            except:
                pass
            
            return False

    def _write_log_entry(self, entry: Dict[str, Any]) -> bool:
        """Write log entry to file with atomic operation"""
        try:
            log_file = self.current_log_file
            
            # Ensure log file has correct permissions if it's new
            if not log_file.exists():
                log_file.touch()
                os.chmod(log_file, 0o600)
            
            # Write entry with file locking
            with open(log_file, 'a') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(entry, separators=(',', ':')) + '\n')
                f.flush()
                os.fsync(f.fileno())
            
            return True
            
        except Exception:
            return False

    def _check_security_alerts(self, event_type: str, event_data: Dict[str, Any], severity: str):
        """Check for security alert conditions"""
        try:
            alert_conditions = []
            
            # Check for suspicious patterns
            if event_type == 'authorization_failed':
                reason = event_data.get('reason', '')
                if 'invalid_format' in reason or 'multiple_failures' in reason:
                    alert_conditions.append('suspicious_authorization_attempts')
            
            elif event_type == 'decryption_failed':
                if event_data.get('consecutive_failures', 0) > 5:
                    alert_conditions.append('excessive_decryption_failures')
            
            elif event_type == 'permission_denied':
                alert_conditions.append('permission_security_issue')
            
            elif severity == 'critical':
                alert_conditions.append('critical_security_event')
            
            # Log alerts
            if alert_conditions:
                self._record_security_alert(alert_conditions, event_type, event_data)
                
        except Exception:
            pass  # Alert checking shouldn't break main logging

    def _record_security_alert(self, alert_types: List[str], 
                             trigger_event: str, trigger_data: Dict[str, Any]):
        """Record security alert in metrics"""
        try:
            metrics = self._load_metrics()
            security_alerts = metrics.get('security_alerts', {})
            
            # Update alert counters
            security_alerts['total_alerts'] = security_alerts.get('total_alerts', 0) + 1
            security_alerts['last_alert'] = time.time()
            
            alert_type_counts = security_alerts.get('alert_types', {})
            for alert_type in alert_types:
                alert_type_counts[alert_type] = alert_type_counts.get(alert_type, 0) + 1
            
            security_alerts['alert_types'] = alert_type_counts
            metrics['security_alerts'] = security_alerts
            
            # Save updated metrics
            self._save_metrics(metrics)
            
            # Log the alert event itself
            alert_entry = {
                'alert_types': alert_types,
                'trigger_event': trigger_event,
                'trigger_data': trigger_data,
                'alert_time': time.time()
            }
            
            self.log_event('security_alert', alert_entry, severity='warning', 
                          source_component='audit_logger')
            
        except Exception:
            pass

    def get_recent_events(self, limit: int = 100, 
                         event_type: str = None, 
                         severity: str = None,
                         days: int = 7) -> List[Dict[str, Any]]:
        """Get recent audit events with filtering"""
        events = []
        cutoff_time = time.time() - (days * 86400)
        
        try:
            # Get log files to check
            log_files = []
            for i in range(days):
                date_str = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                log_file = self._get_log_file_path(date_str)
                if log_file.exists():
                    log_files.append(log_file)
            
            # Read events from log files
            for log_file in sorted(log_files, reverse=True):  # Newest first
                try:
                    with open(log_file, 'r') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        lines = f.readlines()
                    
                    # Process lines in reverse order (newest first)
                    for line in reversed(lines):
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            event = json.loads(line)
                            
                            # Check time filter
                            event_time = event.get('timestamp_unix', 0)
                            if event_time < cutoff_time:
                                continue
                            
                            # Check type filter
                            if event_type and event.get('event_type') != event_type:
                                continue
                            
                            # Check severity filter
                            if severity and event.get('severity') != severity:
                                continue
                            
                            events.append(event)
                            
                            # Check limit
                            if len(events) >= limit:
                                return events
                                
                        except json.JSONDecodeError:
                            continue  # Skip malformed entries
                            
                except Exception:
                    continue  # Skip problematic files
            
        except Exception:
            pass
        
        return events

    def get_audit_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get audit summary and statistics"""
        try:
            metrics = self._load_metrics()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Collect daily stats for the period
            daily_stats = metrics.get('daily_stats', {})
            period_events = 0
            period_event_types = {}
            
            for i in range(days):
                date_str = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
                day_stats = daily_stats.get(date_str, {})
                
                period_events += day_stats.get('events', 0)
                
                day_event_types = day_stats.get('event_types', {})
                for event_type, count in day_event_types.items():
                    period_event_types[event_type] = period_event_types.get(event_type, 0) + count
            
            # Security alerts
            security_alerts = metrics.get('security_alerts', {})
            
            # Performance metrics
            perf_metrics = metrics.get('performance_metrics', {})
            
            summary = {
                'period_days': days,
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'total_events': period_events,
                'event_types': period_event_types,
                'security_alerts': {
                    'total': security_alerts.get('total_alerts', 0),
                    'by_type': security_alerts.get('alert_types', {}),
                    'last_alert': security_alerts.get('last_alert')
                },
                'performance': {
                    'avg_write_time_ms': perf_metrics.get('avg_log_write_time_ms', 0),
                    'total_writes': perf_metrics.get('total_log_writes', 0),
                    'write_errors': perf_metrics.get('log_write_errors', 0)
                },
                'storage': {
                    'audit_directory': str(self.audit_dir),
                    'retention_days': self.retention_days,
                    'total_log_files': len(list(self.audit_dir.glob('security_audit_*.log')))
                }
            }
            
            return summary
            
        except Exception:
            return {'error': 'Failed to generate audit summary'}

    def cleanup_old_logs(self, retention_days: int = None) -> int:
        """Clean up old audit logs and compress recent ones"""
        if retention_days is None:
            retention_days = self.retention_days
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        
        removed_count = 0
        compressed_count = 0
        
        try:
            # Process log files
            for log_file in self.audit_dir.glob('security_audit_*.log'):
                try:
                    # Extract date from filename
                    filename = log_file.stem
                    date_part = filename.split('_')[-1]  # Get YYYY-MM-DD part
                    
                    if date_part < cutoff_str:
                        # Remove old log
                        log_file.unlink()
                        removed_count += 1
                    elif date_part < datetime.now().strftime('%Y-%m-%d'):
                        # Compress recent but not current logs
                        compressed_file = log_file.with_suffix('.log.gz')
                        if not compressed_file.exists():
                            with open(log_file, 'rb') as f_in:
                                with gzip.open(compressed_file, 'wb') as f_out:
                                    f_out.writelines(f_in)
                            
                            # Set secure permissions on compressed file
                            os.chmod(compressed_file, 0o600)
                            
                            # Remove original
                            log_file.unlink()
                            compressed_count += 1
                            
                except Exception:
                    continue  # Skip problematic files
            
            # Log cleanup event
            self.log_event('log_cleanup', {
                'removed_files': removed_count,
                'compressed_files': compressed_count,
                'retention_days': retention_days
            }, severity='info', source_component='audit_logger')
            
        except Exception:
            pass
        
        return removed_count

    def verify_log_integrity(self, days: int = 7) -> Dict[str, Any]:
        """Verify integrity of recent audit logs"""
        results = {
            'total_entries': 0,
            'valid_entries': 0,
            'invalid_checksums': 0,
            'malformed_entries': 0,
            'files_checked': 0,
            'integrity_score': 0.0,
            'issues': []
        }
        
        try:
            # Check recent log files
            for i in range(days):
                date_str = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                log_file = self._get_log_file_path(date_str)
                
                if not log_file.exists():
                    continue
                
                results['files_checked'] += 1
                
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        results['total_entries'] += 1
                        
                        try:
                            entry = json.loads(line)
                            
                            # Verify checksum if present
                            if 'checksum' in entry:
                                stored_checksum = entry.pop('checksum')
                                calculated_checksum = self._calculate_entry_checksum(entry)
                                
                                if stored_checksum == calculated_checksum:
                                    results['valid_entries'] += 1
                                else:
                                    results['invalid_checksums'] += 1
                                    results['issues'].append(f"{log_file.name}:{line_num} - Invalid checksum")
                            else:
                                results['valid_entries'] += 1  # Old entries without checksums
                                
                        except json.JSONDecodeError:
                            results['malformed_entries'] += 1
                            results['issues'].append(f"{log_file.name}:{line_num} - Malformed JSON")
                            
                except Exception as e:
                    results['issues'].append(f"{log_file.name} - Read error: {str(e)}")
            
            # Calculate integrity score
            if results['total_entries'] > 0:
                results['integrity_score'] = results['valid_entries'] / results['total_entries']
            
        except Exception as e:
            results['issues'].append(f"Integrity check error: {str(e)}")
        
        return results

def main():
    """Test audit logger functionality"""
    logger = SecurityAuditLogger(retention_days=30)
    
    print("Security Audit Logger Test")
    print("=" * 50)
    
    # Test various event types
    test_events = [
        ('host_authorized', {'host_id': 'abc123def456', 'description': 'Test host'}, 'info'),
        ('key_rotation', {'key_id': 'key_2024-01-15_abc123', 'rotation_type': 'automatic'}, 'info'),
        ('decryption_failed', {'error': 'invalid_key', 'file': 'learning_data_123.enc'}, 'warning'),
        ('authorization_failed', {'host_id': 'invalid123', 'reason': 'invalid_format'}, 'warning'),
        ('security_violation', {'event': 'suspicious_activity', 'details': 'multiple_failed_attempts'}, 'critical')
    ]
    
    print("Logging test events...")
    for event_type, event_data, severity in test_events:
        success = logger.log_event(event_type, event_data, severity, 'test_component')
        print(f"  {event_type}: {'✓' if success else '✗'}")
    
    print()
    
    # Get recent events
    recent_events = logger.get_recent_events(limit=5)
    print(f"Recent Events ({len(recent_events)}):")
    for event in recent_events[-3:]:  # Show last 3
        timestamp = event.get('timestamp', 'Unknown')
        event_type = event.get('event_type', 'unknown')
        severity = event.get('severity', 'info')
        print(f"  [{timestamp}] {event_type.upper()} ({severity})")
    
    print()
    
    # Show audit summary
    summary = logger.get_audit_summary(days=1)
    print("Audit Summary (Last 24 hours):")
    print(f"  Total Events: {summary.get('total_events', 0)}")
    print(f"  Event Types: {len(summary.get('event_types', {}))}")
    print(f"  Security Alerts: {summary.get('security_alerts', {}).get('total', 0)}")
    
    perf = summary.get('performance', {})
    print(f"  Avg Write Time: {perf.get('avg_write_time_ms', 0):.2f}ms")
    print(f"  Write Errors: {perf.get('write_errors', 0)}")
    
    print()
    
    # Test integrity verification
    print("Testing log integrity...")
    integrity = logger.verify_log_integrity(days=1)
    print(f"  Total Entries: {integrity['total_entries']}")
    print(f"  Valid Entries: {integrity['valid_entries']}")
    print(f"  Integrity Score: {integrity['integrity_score']:.2%}")
    
    if integrity['issues']:
        print(f"  Issues Found: {len(integrity['issues'])}")

if __name__ == '__main__':
    main()