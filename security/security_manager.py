#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0",
#   "psutil>=5.9.0",
# ]
# ///
"""
Security Manager - Unified Security System Interface

Integrates all security components into a cohesive system that implements
the SecurityInterface from interfaces.py. Provides seamless encrypted storage
for the learning system while maintaining military-grade security.

Key Features:
- Unified interface for all security operations
- Hardware-based host identity and trust management
- Daily automatic key rotation with PBKDF2/Fernet encryption
- Encrypted learning data storage with expiration
- Comprehensive audit logging for all security events
- Performance monitoring and health checks

Integration Points:
- Implements SecurityInterface, HostAuthorizationInterface contracts
- Provides factory methods for learning system integration
- Handles automatic key rotation and trust verification
- Manages encrypted storage with transparent encryption/decryption
- Logs all security events for audit and compliance
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib

from hardware_identity import HardwareIdentity
from host_trust import SimpleHostTrust
from key_manager import SimpleKeyManager
from secure_storage import SecureLearningStorage
from audit_logger import SecurityAuditLogger

# Import interfaces for type checking
import sys
sys.path.append(str(Path(__file__).parent.parent))
from interfaces import (
    SecurityInterface, 
    HostAuthorizationInterface, 
    HardwareIdentityInterface,
    PerformanceTargets
)

class SecurityManager(SecurityInterface, HostAuthorizationInterface):
    """Unified security system manager"""
    
    def __init__(self, retention_days: int = 30, audit_retention_days: int = 90):
        # Initialize all security components
        self.hardware_identity = HardwareIdentity()
        self.host_trust = SimpleHostTrust()
        self.key_manager = SimpleKeyManager()
        self.secure_storage = SecureLearningStorage(retention_days)
        self.audit_logger = SecurityAuditLogger(audit_retention_days)
        
        # Cache frequently accessed values
        self._host_id = None
        self._last_key_rotation_check = 0
        self._performance_metrics = {
            'total_encryptions': 0,
            'total_decryptions': 0,
            'avg_encryption_time_ms': 0,
            'avg_decryption_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Log system initialization
        self.audit_logger.log_event('security_system_init', {
            'components': ['hardware_identity', 'host_trust', 'key_manager', 'secure_storage', 'audit_logger'],
            'retention_days': retention_days,
            'audit_retention_days': audit_retention_days,
            'host_id': self.get_host_identity()
        }, severity='info', source_component='security_manager')

    # SecurityInterface implementation
    def encrypt_data(self, data: Dict[str, Any], context: str = "default") -> bytes:
        """Encrypt data with current key"""
        start_time = time.time()
        
        try:
            # Ensure key rotation is current
            self._check_key_rotation()
            
            # Use secure storage for encryption
            encrypted_data = self.secure_storage.encrypt_data(data, context)
            
            # Update performance metrics
            duration_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics('encrypt', duration_ms)
            
            # Log encryption event (without sensitive data)
            self.audit_logger.log_event('data_encrypted', {
                'context': context,
                'data_size_bytes': len(str(data)),
                'encryption_time_ms': duration_ms,
                'key_id': self.key_manager.get_current_key_id()
            }, severity='info', source_component='security_manager')
            
            return encrypted_data
            
        except Exception as e:
            # Log encryption failure
            self.audit_logger.log_event('encryption_failed', {
                'context': context,
                'error': str(e),
                'data_size_bytes': len(str(data)) if data else 0
            }, severity='error', source_component='security_manager')
            
            raise RuntimeError(f"Data encryption failed: {str(e)}")

    def decrypt_data(self, encrypted_data: bytes, context: str = "default") -> Optional[Dict[str, Any]]:
        """Decrypt data, return None if failed"""
        start_time = time.time()
        
        try:
            # Use secure storage for decryption
            decrypted_data = self.secure_storage.decrypt_data(encrypted_data, context)
            
            # Update performance metrics
            duration_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics('decrypt', duration_ms)
            
            if decrypted_data is not None:
                # Log successful decryption
                self.audit_logger.log_event('data_decrypted', {
                    'context': context,
                    'encrypted_size_bytes': len(encrypted_data),
                    'decryption_time_ms': duration_ms,
                    'success': True
                }, severity='info', source_component='security_manager')
                
                return decrypted_data
            else:
                # Log decryption failure
                self.audit_logger.log_event('decryption_failed', {
                    'context': context,
                    'encrypted_size_bytes': len(encrypted_data),
                    'decryption_time_ms': duration_ms,
                    'error': 'decryption_returned_none'
                }, severity='warning', source_component='security_manager')
                
                return None
                
        except Exception as e:
            # Log decryption error
            self.audit_logger.log_event('decryption_error', {
                'context': context,
                'encrypted_size_bytes': len(encrypted_data),
                'error': str(e)
            }, severity='error', source_component='security_manager')
            
            return None

    def rotate_keys(self) -> bool:
        """Perform daily key rotation"""
        try:
            # Perform key rotation
            success = self.key_manager.rotate_keys()
            
            # Log rotation event
            self.audit_logger.log_event('key_rotation', {
                'success': success,
                'new_key_id': self.key_manager.get_current_key_id() if success else None,
                'rotation_type': 'manual'
            }, severity='info' if success else 'error', source_component='security_manager')
            
            # Update last rotation check
            self._last_key_rotation_check = time.time()
            
            return success
            
        except Exception as e:
            self.audit_logger.log_event('key_rotation_error', {
                'error': str(e),
                'rotation_type': 'manual'
            }, severity='error', source_component='security_manager')
            
            return False

    def get_current_key_id(self) -> str:
        """Get current encryption key identifier"""
        return self.key_manager.get_current_key_id()

    def cleanup_old_keys(self, retention_days: int = 7) -> int:
        """Remove old encryption keys, return count removed"""
        try:
            removed_count = self.key_manager.cleanup_old_keys(retention_days)
            
            # Log cleanup event
            self.audit_logger.log_event('key_cleanup', {
                'removed_keys': removed_count,
                'retention_days': retention_days
            }, severity='info', source_component='security_manager')
            
            return removed_count
            
        except Exception as e:
            self.audit_logger.log_event('key_cleanup_error', {
                'error': str(e),
                'retention_days': retention_days
            }, severity='error', source_component='security_manager')
            
            return 0

    # HostAuthorizationInterface implementation
    def get_host_identity(self) -> str:
        """Get stable hardware-based host identity"""
        if self._host_id is None:
            self._host_id = self.host_trust.get_host_identity()
        return self._host_id

    def is_trusted_host(self, host_id: str) -> bool:
        """Check if host is in trust list"""
        return self.host_trust.is_trusted_host(host_id)

    def authorize_host(self, host_id: str, host_description: str = "") -> bool:
        """Add host to trust list"""
        try:
            # Validate host ID format
            if not host_id or len(host_id) != 16:
                self.audit_logger.log_event('authorization_failed', {
                    'host_id': host_id,
                    'reason': 'invalid_host_id_format',
                    'description': host_description
                }, severity='warning', source_component='security_manager')
                return False
            
            # Check if already trusted
            if self.host_trust.is_trusted_host(host_id):
                self.audit_logger.log_event('authorization_duplicate', {
                    'host_id': host_id,
                    'reason': 'already_trusted',
                    'description': host_description
                }, severity='info', source_component='security_manager')
                return True
            
            # Authorize the host
            success = self.host_trust.authorize_host(host_id, host_description)
            
            if success:
                # Trigger key rotation for security
                self.rotate_keys()
                
                self.audit_logger.log_event('host_authorized', {
                    'host_id': host_id,
                    'description': host_description,
                    'authorized_by': self.get_host_identity()
                }, severity='info', source_component='security_manager')
            
            return success
            
        except Exception as e:
            self.audit_logger.log_event('authorization_error', {
                'host_id': host_id,
                'error': str(e),
                'description': host_description
            }, severity='error', source_component='security_manager')
            
            return False

    def revoke_host(self, host_id: str) -> bool:
        """Remove host from trust list"""
        try:
            # Check if host is trusted
            if not self.host_trust.is_trusted_host(host_id):
                self.audit_logger.log_event('revocation_failed', {
                    'host_id': host_id,
                    'reason': 'host_not_trusted'
                }, severity='warning', source_component='security_manager')
                return False
            
            # Revoke the host
            success = self.host_trust.revoke_host(host_id)
            
            if success:
                # Trigger key rotation for security
                self.rotate_keys()
                
                self.audit_logger.log_event('host_revoked', {
                    'host_id': host_id,
                    'revoked_by': self.get_host_identity()
                }, severity='warning', source_component='security_manager')
            
            return success
            
        except Exception as e:
            self.audit_logger.log_event('revocation_error', {
                'host_id': host_id,
                'error': str(e)
            }, severity='error', source_component='security_manager')
            
            return False

    def list_trusted_hosts(self) -> List[Dict[str, str]]:
        """Get list of trusted hosts with metadata"""
        return self.host_trust.list_trusted_hosts()

    # Additional integration methods
    def store_encrypted_learning_data(self, data_type: str, data: Dict[str, Any]) -> bool:
        """Store learning data with encryption"""
        try:
            success = self.secure_storage.store_learning_data(data_type, data)
            
            if success:
                self.audit_logger.log_event('learning_data_stored', {
                    'data_type': data_type,
                    'data_size_bytes': len(str(data)),
                    'storage_success': True
                }, severity='info', source_component='security_manager')
            
            return success
            
        except Exception as e:
            self.audit_logger.log_event('learning_data_store_error', {
                'data_type': data_type,
                'error': str(e)
            }, severity='error', source_component='security_manager')
            
            return False

    def load_encrypted_learning_data(self, data_type: str, max_age_days: int = None) -> List[Dict[str, Any]]:
        """Load encrypted learning data"""
        try:
            data = self.secure_storage.load_learning_data(data_type, max_age_days)
            
            self.audit_logger.log_event('learning_data_loaded', {
                'data_type': data_type,
                'records_loaded': len(data),
                'max_age_days': max_age_days
            }, severity='info', source_component='security_manager')
            
            return data
            
        except Exception as e:
            self.audit_logger.log_event('learning_data_load_error', {
                'data_type': data_type,
                'error': str(e)
            }, severity='error', source_component='security_manager')
            
            return []

    def cleanup_expired_data(self, retention_days: int = None) -> int:
        """Remove expired learning data"""
        try:
            removed_count = self.secure_storage.cleanup_expired_data(retention_days)
            
            self.audit_logger.log_event('data_cleanup', {
                'removed_files': removed_count,
                'retention_days': retention_days or self.secure_storage.retention_days
            }, severity='info', source_component='security_manager')
            
            return removed_count
            
        except Exception as e:
            self.audit_logger.log_event('data_cleanup_error', {
                'error': str(e),
                'retention_days': retention_days
            }, severity='error', source_component='security_manager')
            
            return 0

    def _check_key_rotation(self):
        """Check if key rotation is needed (daily check)"""
        now = time.time()
        if now - self._last_key_rotation_check > 3600:  # Check every hour
            # Get current key to trigger rotation if needed
            self.key_manager.get_current_key()
            self._last_key_rotation_check = now

    def _update_performance_metrics(self, operation: str, duration_ms: float):
        """Update internal performance metrics"""
        if operation == 'encrypt':
            count = self._performance_metrics['total_encryptions']
            avg = self._performance_metrics['avg_encryption_time_ms']
            
            # Running average
            new_avg = (avg * count + duration_ms) / (count + 1)
            
            self._performance_metrics['total_encryptions'] = count + 1
            self._performance_metrics['avg_encryption_time_ms'] = new_avg
            
        elif operation == 'decrypt':
            count = self._performance_metrics['total_decryptions']
            avg = self._performance_metrics['avg_decryption_time_ms']
            
            # Running average
            new_avg = (avg * count + duration_ms) / (count + 1)
            
            self._performance_metrics['total_decryptions'] = count + 1
            self._performance_metrics['avg_decryption_time_ms'] = new_avg

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive security system status"""
        try:
            # Collect status from all components
            hardware_info = self.hardware_identity.get_identity_info()
            trust_stats = self.host_trust.get_trust_statistics()
            key_stats = self.key_manager.get_key_statistics()
            storage_stats = self.secure_storage.get_storage_statistics()
            audit_summary = self.audit_logger.get_audit_summary(days=7)
            
            # Performance analysis
            performance_issues = []
            
            # Check encryption performance
            if self._performance_metrics['avg_encryption_time_ms'] > PerformanceTargets.ENCRYPTION_OPERATION_MS:
                performance_issues.append(f"Encryption time {self._performance_metrics['avg_encryption_time_ms']:.1f}ms exceeds target {PerformanceTargets.ENCRYPTION_OPERATION_MS}ms")
            
            if self._performance_metrics['avg_decryption_time_ms'] > PerformanceTargets.ENCRYPTION_OPERATION_MS:
                performance_issues.append(f"Decryption time {self._performance_metrics['avg_decryption_time_ms']:.1f}ms exceeds target {PerformanceTargets.ENCRYPTION_OPERATION_MS}ms")
            
            # Check key generation performance from key stats
            if key_stats.get('keys_generated_today', 0) > 0:
                # This would need to be tracked in key_manager if we want precise timing
                pass
            
            status = {
                'system_info': {
                    'local_host_id': self.get_host_identity(),
                    'system_initialized': True,
                    'components_active': ['hardware_identity', 'host_trust', 'key_manager', 'secure_storage', 'audit_logger']
                },
                'trust_management': {
                    'trusted_hosts': trust_stats['total_trusted_hosts'],
                    'recently_active_hosts': trust_stats['recently_active_hosts'],
                    'trust_file_size_bytes': trust_stats['trust_file_size_bytes']
                },
                'encryption_system': {
                    'current_key_id': key_stats['current_key_id'],
                    'active_keys': key_stats['active_keys_count'],
                    'key_retention_days': key_stats['retention_days'],
                    'pbkdf2_iterations': key_stats['pbkdf2_iterations']
                },
                'data_storage': {
                    'total_files': storage_stats['total_files'],
                    'total_size_bytes': storage_stats['total_size_bytes'],
                    'data_types': storage_stats['data_types'],
                    'retention_days': storage_stats['retention_days']
                },
                'audit_system': {
                    'total_events_7d': audit_summary.get('total_events', 0),
                    'security_alerts': audit_summary.get('security_alerts', {}),
                    'avg_write_time_ms': audit_summary.get('performance', {}).get('avg_write_time_ms', 0)
                },
                'performance_metrics': self._performance_metrics.copy(),
                'performance_issues': performance_issues,
                'hardware_sources': hardware_info['hardware_sources'],
                'status_generated_at': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.audit_logger.log_event('status_generation_error', {
                'error': str(e)
            }, severity='error', source_component='security_manager')
            
            return {
                'error': f'Failed to generate status: {str(e)}',
                'local_host_id': self.get_host_identity(),
                'status_generated_at': datetime.now().isoformat()
            }

    def run_security_health_check(self) -> Dict[str, Any]:
        """Run comprehensive security health check"""
        try:
            health_results = {
                'overall_health': 'healthy',
                'issues': [],
                'warnings': [],
                'recommendations': [],
                'component_health': {}
            }
            
            # Check storage integrity
            storage_integrity = self.secure_storage.verify_integrity()
            health_results['component_health']['storage'] = {
                'status': 'healthy' if storage_integrity['corrupted_files'] == 0 else 'degraded',
                'total_files': storage_integrity['total_files'],
                'readable_files': storage_integrity['readable_files'],
                'corrupted_files': storage_integrity['corrupted_files']
            }
            
            if storage_integrity['corrupted_files'] > 0:
                health_results['issues'].append(f"{storage_integrity['corrupted_files']} corrupted storage files found")
                health_results['overall_health'] = 'degraded'
            
            # Check audit log integrity
            audit_integrity = self.audit_logger.verify_log_integrity(days=7)
            health_results['component_health']['audit'] = {
                'status': 'healthy' if audit_integrity['integrity_score'] > 0.95 else 'degraded',
                'integrity_score': audit_integrity['integrity_score'],
                'total_entries': audit_integrity['total_entries'],
                'invalid_checksums': audit_integrity['invalid_checksums']
            }
            
            if audit_integrity['integrity_score'] < 0.95:
                health_results['warnings'].append(f"Audit log integrity score {audit_integrity['integrity_score']:.1%} below 95%")
            
            # Check performance metrics
            if self._performance_metrics['avg_encryption_time_ms'] > PerformanceTargets.ENCRYPTION_OPERATION_MS:
                health_results['warnings'].append("Encryption performance below target")
            
            # Check key system
            key_stats = self.key_manager.get_key_statistics()
            health_results['component_health']['keys'] = {
                'status': 'healthy',
                'active_keys': key_stats['active_keys_count'],
                'current_key_id': key_stats['current_key_id']
            }
            
            if key_stats['active_keys_count'] == 0:
                health_results['issues'].append("No active encryption keys found")
                health_results['overall_health'] = 'critical'
            
            # Security recommendations
            trust_stats = self.host_trust.get_trust_statistics()
            if trust_stats['total_trusted_hosts'] == 0:
                health_results['recommendations'].append("Consider authorizing trusted hosts for mesh learning")
            
            if len(health_results['issues']) > 0:
                health_results['overall_health'] = 'critical'
            elif len(health_results['warnings']) > 0 and health_results['overall_health'] == 'healthy':
                health_results['overall_health'] = 'degraded'
            
            # Log health check
            self.audit_logger.log_event('security_health_check', {
                'overall_health': health_results['overall_health'],
                'issues_count': len(health_results['issues']),
                'warnings_count': len(health_results['warnings']),
                'components_checked': list(health_results['component_health'].keys())
            }, severity='info', source_component='security_manager')
            
            return health_results
            
        except Exception as e:
            self.audit_logger.log_event('health_check_error', {
                'error': str(e)
            }, severity='error', source_component='security_manager')
            
            return {
                'overall_health': 'error',
                'error': f'Health check failed: {str(e)}',
                'issues': ['Health check system failure']
            }

def main():
    """Test security manager integration"""
    print("Security Manager Integration Test")
    print("=" * 50)
    
    # Initialize security manager
    security = SecurityManager(retention_days=7, audit_retention_days=30)
    
    # Test basic operations
    print(f"Local Host ID: {security.get_host_identity()}")
    print(f"Current Key ID: {security.get_current_key_id()}")
    print()
    
    # Test encryption/decryption
    test_data = {
        'command_pattern': 'ssh_optimization',
        'success_rate': 0.95,
        'performance_gain': 25.5,
        'timestamp': time.time()
    }
    
    print("Testing encryption/decryption...")
    encrypted = security.encrypt_data(test_data, 'test_context')
    print(f"Encrypted size: {len(encrypted)} bytes")
    
    decrypted = security.decrypt_data(encrypted, 'test_context')
    data_match = (decrypted and 
                 decrypted.get('command_pattern') == test_data['command_pattern'] and
                 decrypted.get('success_rate') == test_data['success_rate'])
    print(f"Decryption success: {'✓' if data_match else '✗'}")
    print()
    
    # Test learning data storage
    print("Testing learning data storage...")
    store_success = security.store_encrypted_learning_data('test_patterns', test_data)
    print(f"Storage success: {'✓' if store_success else '✗'}")
    
    loaded_data = security.load_encrypted_learning_data('test_patterns', max_age_days=1)
    print(f"Loaded {len(loaded_data)} records")
    print()
    
    # Show comprehensive status
    print("Security System Status:")
    status = security.get_comprehensive_status()
    
    # Display key status information
    system_info = status.get('system_info', {})
    print(f"  Components Active: {len(system_info.get('components_active', []))}")
    
    trust_info = status.get('trust_management', {})
    print(f"  Trusted Hosts: {trust_info.get('trusted_hosts', 0)}")
    
    encryption_info = status.get('encryption_system', {})
    print(f"  Active Keys: {encryption_info.get('active_keys', 0)}")
    
    storage_info = status.get('data_storage', {})
    print(f"  Stored Files: {storage_info.get('total_files', 0)}")
    
    perf_metrics = status.get('performance_metrics', {})
    print(f"  Avg Encryption Time: {perf_metrics.get('avg_encryption_time_ms', 0):.2f}ms")
    
    issues = status.get('performance_issues', [])
    if issues:
        print(f"  Performance Issues: {len(issues)}")
    
    print()
    
    # Run health check
    print("Running security health check...")
    health = security.run_security_health_check()
    print(f"Overall Health: {health.get('overall_health', 'unknown').upper()}")
    print(f"Issues: {len(health.get('issues', []))}")
    print(f"Warnings: {len(health.get('warnings', []))}")
    print(f"Components Checked: {len(health.get('component_health', {}))}")

if __name__ == '__main__':
    main()