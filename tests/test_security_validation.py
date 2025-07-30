#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-asyncio>=0.21.0",
#   "cryptography>=41.0.0",
#   "psutil>=5.9.0",
#   "typing-extensions>=4.0.0"
# ]
# ///
"""
Security Validation Tests for Claude-Sync

Comprehensive security testing including:
- Encryption/decryption cycle validation
- Key rotation security testing
- Hardware identity verification
- Host authorization security
- Data abstraction security
- Attack scenario simulation
- Audit trail validation
"""

import json
import time
import tempfile
import shutil
import sys
import os
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import threading
import concurrent.futures

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_framework import TestFramework, TestSuite, TestResult, TestEnvironment
from mock_data_generators import LearningDataGenerator
from interfaces import PerformanceTargets

# ============================================================================
# Encryption Security Tests
# ============================================================================

class EncryptionSecurityTests:
    """Comprehensive encryption security validation"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.project_root = Path(__file__).parent.parent
        self.learning_generator = LearningDataGenerator(seed=42)
    
    def test_encryption_decryption_cycle_integrity(self) -> Tuple[bool, str]:
        """Test that encryption/decryption preserves data integrity"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "security"))
            from security_manager import SecurityManager
            
            security = SecurityManager(data_dir=test_dir / ".claude" / "security")
            
            # Test with various data types and sizes
            test_datasets = [
                {"simple": "test"},
                {"unicode": "测试数据 with émojis 🔒"},
                {"numbers": [1, 2, 3, 4, 5, 42, -1, 0]},
                {"nested": {"commands": ["grep", "find"], "metadata": {"version": 1.0}}},
                {"large_text": "x" * 10000},  # 10KB text
                {"binary_like": bytes(range(256)).hex()}
            ]
            
            integrity_failures = []
            
            for i, test_data in enumerate(test_datasets):
                # Encrypt data
                encrypted = security.encrypt_data(test_data)
                
                if not isinstance(encrypted, bytes):
                    integrity_failures.append(f"Dataset {i}: encryption didn't return bytes")
                    continue
                
                # Decrypt data
                decrypted = security.decrypt_data(encrypted)
                
                if decrypted != test_data:
                    integrity_failures.append(f"Dataset {i}: decrypted data doesn't match original")
                    continue
                
                # Verify encrypted data is not readable as plain text
                try:
                    plain_attempt = json.loads(encrypted.decode())
                    integrity_failures.append(f"Dataset {i}: encrypted data is readable as JSON")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Good - encrypted data is not readable
                    pass
            
            if integrity_failures:
                return False, f"Integrity failures: {'; '.join(integrity_failures)}"
            
            return True, f"All {len(test_datasets)} datasets passed encryption/decryption integrity tests"
            
        except ImportError as e:
            return False, f"Security manager module not found: {e}"
        except Exception as e:
            return False, f"Encryption integrity test failed: {str(e)}"
    
    def test_encryption_uniqueness(self) -> Tuple[bool, str]:
        """Test that identical data encrypts to different ciphertext each time"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "security"))
            from security_manager import SecurityManager
            
            security = SecurityManager(data_dir=test_dir / ".claude" / "security")
            
            test_data = {"command": "test command", "timestamp": 12345}
            
            # Encrypt same data multiple times
            ciphertexts = []
            for _ in range(5):
                encrypted = security.encrypt_data(test_data)
                ciphertexts.append(encrypted)
                
                # Verify it still decrypts correctly
                decrypted = security.decrypt_data(encrypted)
                if decrypted != test_data:
                    return False, "Encrypted data doesn't decrypt correctly"
            
            # Verify all ciphertexts are different (due to random IV/nonce)
            unique_ciphertexts = set(ciphertexts)
            if len(unique_ciphertexts) != len(ciphertexts):
                return False, "Identical data produced identical ciphertext (no randomization)"
            
            return True, f"Encryption produces unique ciphertext: {len(unique_ciphertexts)} unique from {len(ciphertexts)} attempts"
            
        except ImportError as e:
            return False, f"Security manager module not found: {e}"
        except Exception as e:
            return False, f"Encryption uniqueness test failed: {str(e)}"
    
    def test_encryption_context_isolation(self) -> Tuple[bool, str]:
        """Test that different contexts produce different encryption results"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "security"))
            from security_manager import SecurityManager
            
            security = SecurityManager(data_dir=test_dir / ".claude" / "security")
            
            test_data = {"sensitive": "data"}
            contexts = ["default", "learning", "hooks", "mesh_sync"]
            
            # Encrypt same data with different contexts
            context_results = {}
            for context in contexts:
                encrypted = security.encrypt_data(test_data, context=context)
                context_results[context] = encrypted
                
                # Verify it decrypts correctly with same context
                decrypted = security.decrypt_data(encrypted, context=context)
                if decrypted != test_data:
                    return False, f"Data encrypted with context '{context}' doesn't decrypt correctly"
            
            # Verify different contexts produce different ciphertext
            all_ciphertexts = list(context_results.values())
            unique_ciphertexts = set(all_ciphertexts)
            
            if len(unique_ciphertexts) != len(contexts):
                return False, "Different contexts produced identical ciphertext"
            
            # Verify cross-context decryption fails or produces different results
            cross_context_failures = 0
            for ctx1, ciphertext in context_results.items():
                for ctx2 in contexts:
                    if ctx1 != ctx2:
                        try:
                            decrypted = security.decrypt_data(ciphertext, context=ctx2)
                            if decrypted == test_data:
                                return False, f"Data encrypted with '{ctx1}' decrypted correctly with '{ctx2}' context"
                        except Exception:
                            # Expected - different contexts should not be interchangeable
                            cross_context_failures += 1
            
            return True, f"Context isolation working: {len(contexts)} contexts, {cross_context_failures} cross-context failures"
            
        except ImportError as e:
            return False, f"Security manager module not found: {e}"
        except Exception as e:
            return False, f"Context isolation test failed: {str(e)}"
    
    def test_key_rotation_security(self) -> Tuple[bool, str]:
        """Test security aspects of key rotation"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "security"))
            from security_manager import SecurityManager
            
            security = SecurityManager(data_dir=test_dir / ".claude" / "security")
            
            # Store data with initial key
            test_data = {"command": "sensitive data", "timestamp": time.time()}
            encrypted_with_old_key = security.encrypt_data(test_data)
            old_key_id = security.get_current_key_id()
            
            # Rotate keys
            rotation_success = security.rotate_keys()
            if not rotation_success:
                return False, "Key rotation failed"
            
            new_key_id = security.get_current_key_id()
            
            # Verify key ID changed
            if new_key_id == old_key_id:
                return False, "Key ID didn't change after rotation"
            
            # Verify old data can still be decrypted
            decrypted_old_data = security.decrypt_data(encrypted_with_old_key)
            if decrypted_old_data != test_data:
                return False, "Cannot decrypt old data after key rotation"
            
            # Verify new data is encrypted with new key
            encrypted_with_new_key = security.encrypt_data(test_data)
            
            # New encryption should be different from old (different key)
            if encrypted_with_new_key == encrypted_with_old_key:
                return False, "New encryption identical to old encryption (key not rotated)"
            
            # Verify new data decrypts correctly
            decrypted_new_data = security.decrypt_data(encrypted_with_new_key)
            if decrypted_new_data != test_data:
                return False, "Cannot decrypt new data with new key"
            
            # Test key cleanup (if implemented)
            keys_cleaned = security.cleanup_old_keys(retention_days=0)  # Aggressive cleanup
            
            return True, f"Key rotation secure: old key {old_key_id[:8]}..., new key {new_key_id[:8]}..., {keys_cleaned} keys cleaned"
            
        except ImportError as e:
            return False, f"Security manager module not found: {e}"
        except Exception as e:
            return False, f"Key rotation security test failed: {str(e)}"

# ============================================================================
# Hardware Identity Security Tests  
# ============================================================================

class HardwareIdentitySecurityTests:
    """Hardware-based identity security validation"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.project_root = Path(__file__).parent.parent
    
    def test_hardware_identity_stability(self) -> Tuple[bool, str]:
        """Test hardware identity is stable across multiple calls"""
        try:
            sys.path.insert(0, str(self.project_root / "security"))
            from hardware_identity import HardwareIdentity
            
            identity = HardwareIdentity()
            
            # Generate identity multiple times
            identities = []
            for _ in range(10):
                host_id = identity.generate_stable_host_id()
                identities.append(host_id)
                
                # Verify identity format
                if not isinstance(host_id, str) or len(host_id) < 16:
                    return False, f"Invalid host ID format: {host_id}"
            
            # All identities should be identical
            unique_identities = set(identities)
            if len(unique_identities) != 1:
                return False, f"Hardware identity not stable: {len(unique_identities)} different IDs generated"
            
            # Test validation
            stable_id = identities[0]
            if not identity.validate_host_identity(stable_id):
                return False, "Generated host ID fails validation"
            
            # Test invalid ID rejection
            fake_ids = [
                "invalid_id_12345",
                stable_id + "modified",
                stable_id[:-5] + "00000",
                ""
            ]
            
            for fake_id in fake_ids:
                if identity.validate_host_identity(fake_id):
                    return False, f"Fake ID passed validation: {fake_id}"
            
            return True, f"Hardware identity stable and secure: {stable_id[:16]}... (validated {len(fake_ids)} fake IDs rejected)"
            
        except ImportError as e:
            return False, f"Hardware identity module not found: {e}"
        except Exception as e:
            return False, f"Hardware identity security test failed: {str(e)}"
    
    def test_hardware_identity_uniqueness(self) -> Tuple[bool, str]:
        """Test hardware identity components for uniqueness indicators"""
        try:
            sys.path.insert(0, str(self.project_root / "security"))
            from hardware_identity import HardwareIdentity
            
            identity = HardwareIdentity()
            
            # Test individual components
            components = {
                'cpu_serial': identity.get_cpu_serial(),
                'motherboard_uuid': identity.get_motherboard_uuid(),
                'primary_mac': identity.get_network_mac_primary()
            }
            
            # At least one component should be available
            available_components = {k: v for k, v in components.items() if v is not None}
            
            if not available_components:
                return False, "No hardware identity components available"
            
            # Components should have reasonable length and format
            for component_name, component_value in available_components.items():
                if len(component_value) < 8:
                    return False, f"{component_name} too short: {component_value}"
                
                # Should not be obviously fake
                if component_value in ["unknown", "not_found", "error", "none"]:
                    return False, f"{component_name} has placeholder value: {component_value}"
            
            # Generate final ID and verify it incorporates components
            host_id = identity.generate_stable_host_id()
            
            # ID should be deterministic based on available components
            host_id_2 = identity.generate_stable_host_id()
            if host_id != host_id_2:
                return False, "Host ID generation not deterministic"
            
            return True, f"Hardware identity uses {len(available_components)} components: {list(available_components.keys())}"
            
        except ImportError as e:
            return False, f"Hardware identity module not found: {e}"
        except Exception as e:
            return False, f"Hardware identity uniqueness test failed: {str(e)}"

# ============================================================================
# Host Authorization Security Tests
# ============================================================================

class HostAuthorizationSecurityTests:
    """Host trust and authorization security validation"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.project_root = Path(__file__).parent.parent
    
    def test_host_trust_management_security(self) -> Tuple[bool, str]:
        """Test host trust management security"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "security"))
            from host_trust import HostTrustManager
            from hardware_identity import HardwareIdentity
            
            trust_manager = HostTrustManager(data_dir=test_dir / ".claude" / "security")
            identity = HardwareIdentity()
            
            current_host = identity.generate_stable_host_id()
            
            # Test unauthorized host rejection
            fake_host = "fake_host_12345"
            if trust_manager.is_trusted_host(fake_host):
                return False, "Fake host is trusted by default"
            
            # Test host authorization process
            auth_result = trust_manager.authorize_host(fake_host, "Test host authorization")
            if not auth_result:
                return False, "Failed to authorize test host"
            
            # Verify host is now trusted
            if not trust_manager.is_trusted_host(fake_host):
                return False, "Host not trusted after authorization"
            
            # Test host revocation
            revoke_result = trust_manager.revoke_host(fake_host)
            if not revoke_result:
                return False, "Failed to revoke test host"
            
            # Verify host is no longer trusted
            if trust_manager.is_trusted_host(fake_host):
                return False, "Host still trusted after revocation"
            
            # Test trust list integrity
            trusted_hosts_before = trust_manager.list_trusted_hosts()
            
            # Add multiple hosts
            test_hosts = [f"test_host_{i}" for i in range(5)]
            for host in test_hosts:
                trust_manager.authorize_host(host, f"Test host {host}")
            
            trusted_hosts_after = trust_manager.list_trusted_hosts()
            
            if len(trusted_hosts_after) != len(trusted_hosts_before) + len(test_hosts):
                return False, "Trust list count inconsistent after adding hosts"
            
            # Verify all test hosts are in the list
            trusted_host_ids = [host['host_id'] for host in trusted_hosts_after]
            for host in test_hosts:
                if host not in trusted_host_ids:
                    return False, f"Authorized host {host} not found in trust list"
            
            # Test bulk revocation
            for host in test_hosts:
                trust_manager.revoke_host(host)
            
            final_trusted_hosts = trust_manager.list_trusted_hosts()
            if len(final_trusted_hosts) != len(trusted_hosts_before):
                return False, "Trust list count inconsistent after revoking hosts"
            
            return True, f"Host trust management secure: authorized {len(test_hosts)} hosts, revoked all successfully"
            
        except ImportError as e:
            return False, f"Host trust module not found: {e}"
        except Exception as e:
            return False, f"Host trust security test failed: {str(e)}"
    
    def test_host_authorization_persistence(self) -> Tuple[bool, str]:
        """Test that host authorizations persist across restarts"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "security"))
            from host_trust import HostTrustManager
            
            # First instance - authorize hosts
            trust_manager_1 = HostTrustManager(data_dir=test_dir / ".claude" / "security")
            
            test_hosts = ["persistent_host_1", "persistent_host_2", "persistent_host_3"]
            for host in test_hosts:
                trust_manager_1.authorize_host(host, f"Persistent test host {host}")
            
            trusted_before = trust_manager_1.list_trusted_hosts()
            
            # Second instance - simulate restart
            trust_manager_2 = HostTrustManager(data_dir=test_dir / ".claude" / "security")
            
            # Verify hosts are still trusted
            for host in test_hosts:
                if not trust_manager_2.is_trusted_host(host):
                    return False, f"Host {host} not trusted after restart"
            
            trusted_after = trust_manager_2.list_trusted_hosts()
            
            # Trust lists should be identical
            if len(trusted_before) != len(trusted_after):
                return False, f"Trust list size changed: {len(trusted_before)} -> {len(trusted_after)}"
            
            before_ids = {host['host_id'] for host in trusted_before}
            after_ids = {host['host_id'] for host in trusted_after}
            
            if before_ids != after_ids:
                return False, "Trust list contents changed after restart"
            
            return True, f"Host authorization persistence verified: {len(test_hosts)} hosts persisted across restart"
            
        except ImportError as e:
            return False, f"Host trust module not found: {e}"
        except Exception as e:
            return False, f"Host authorization persistence test failed: {str(e)}"

# ============================================================================
# Data Abstraction Security Tests
# ============================================================================

class DataAbstractionSecurityTests:
    """Data abstraction and privacy security validation"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.project_root = Path(__file__).parent.parent
    
    def test_command_abstraction_security(self) -> Tuple[bool, str]:
        """Test that command abstraction removes sensitive information"""
        try:
            sys.path.insert(0, str(self.project_root / "learning"))
            from abstraction import DataAbstractor
            
            abstractor = DataAbstractor()
            
            # Test commands with sensitive information
            sensitive_commands = [
                "ssh user@private-server.company.com 'ls /secret/path'",
                "scp sensitive_file.txt user@192.168.1.100:/home/user/",
                "rsync -avz /home/john/personal_data/ backup:/confidential/",
                "sbatch --job-name=project_alpha_secret analysis.sh",
                "mysql -u admin -p'password123' -h database.internal",
                "curl -H 'Authorization: Bearer secret_token_12345' https://api.private.com/data"
            ]
            
            abstraction_failures = []
            
            for command in sensitive_commands:
                abstracted = abstractor.abstract_command(command)
                
                if not isinstance(abstracted, dict):
                    abstraction_failures.append(f"Command abstraction didn't return dict: {command}")
                    continue
                
                # Check if sensitive information is still present
                abstracted_str = json.dumps(abstracted).lower()
                
                # These should not appear in abstracted data
                sensitive_patterns = [
                    'password123', 'secret_token', 'private-server.company.com',
                    '192.168.1.100', '/home/john/', 'project_alpha_secret',
                    'database.internal', '/secret/path', '/confidential/',
                    'admin', 'bearer'
                ]
                
                for pattern in sensitive_patterns:
                    if pattern.lower() in abstracted_str:
                        abstraction_failures.append(f"Sensitive pattern '{pattern}' found in abstracted data")
                
                # Check that abstracted data contains useful learning information
                if not any(key in abstracted for key in ['command_type', 'pattern', 'category']):
                    abstraction_failures.append(f"Abstracted data lacks learning information: {abstracted}")
            
            if abstraction_failures:
                return False, f"Abstraction failures: {'; '.join(abstraction_failures[:3])}..."  # Show first 3
            
            return True, f"Command abstraction secure: {len(sensitive_commands)} sensitive commands properly abstracted"
            
        except ImportError as e:
            return False, f"Data abstraction module not found: {e}"
        except Exception as e:
            return False, f"Command abstraction security test failed: {str(e)}"
    
    def test_hostname_abstraction_security(self) -> Tuple[bool, str]:
        """Test that hostname abstraction preserves utility while removing identity"""
        try:
            sys.path.insert(0, str(self.project_root / "learning"))
            from abstraction import DataAbstractor
            
            abstractor = DataAbstractor()
            
            # Test hostnames that should be abstracted
            sensitive_hostnames = [
                "hpc-cluster-gpu01.university.edu",
                "workstation-lab-205.department.org",
                "johns-macbook-pro.local",
                "server01.confidential-company.com",
                "192.168.1.50",
                "10.0.0.100"
            ]
            
            abstraction_results = []
            
            for hostname in sensitive_hostnames:
                abstracted = abstractor.abstract_hostname(hostname)
                
                if not isinstance(abstracted, str):
                    return False, f"Hostname abstraction didn't return string: {hostname}"
                
                # Should not contain original hostname
                if hostname.lower() in abstracted.lower():
                    return False, f"Original hostname found in abstraction: {hostname} -> {abstracted}"
                
                # Should not contain obviously sensitive parts
                sensitive_parts = ['university.edu', 'department.org', 'johns-macbook', 'confidential-company']
                for part in sensitive_parts:
                    if part.lower() in abstracted.lower():
                        return False, f"Sensitive hostname part found in abstraction: {part} in {abstracted}"
                
                # Should provide semantic categorization
                if not any(category in abstracted.lower() for category in ['hpc', 'gpu', 'workstation', 'server', 'compute', 'host']):
                    return False, f"Abstracted hostname lacks semantic information: {abstracted}"
                
                abstraction_results.append((hostname, abstracted))
            
            # Test consistency - same hostname should always get same abstraction
            for hostname, first_abstraction in abstraction_results:
                second_abstraction = abstractor.abstract_hostname(hostname)
                if first_abstraction != second_abstraction:
                    return False, f"Hostname abstraction not consistent: {hostname} -> {first_abstraction} vs {second_abstraction}"
            
            return True, f"Hostname abstraction secure: {len(sensitive_hostnames)} hostnames properly abstracted"
            
        except ImportError as e:
            return False, f"Data abstraction module not found: {e}"
        except Exception as e:
            return False, f"Hostname abstraction security test failed: {str(e)}"
    
    def test_path_abstraction_security(self) -> Tuple[bool, str]:
        """Test that path abstraction removes sensitive directory information"""
        try:
            sys.path.insert(0, str(self.project_root / "learning"))
            from abstraction import DataAbstractor
            
            abstractor = DataAbstractor()
            
            # Test paths with sensitive information
            sensitive_paths = [
                "/home/john.doe/personal_research/confidential_data.txt",
                "/Users/researcher/Documents/Project_Alpha/secret_results.csv",
                "/data/genomics/patient_samples/sample_12345.fastq",
                "/scratch/lab_members/jane_smith/private_analysis/",
                "/project/department_budget/financial_data_2024.xlsx",
                "/mnt/shared_drive/HR_documents/employee_records.db"
            ]
            
            abstraction_failures = []
            
            for path in sensitive_paths:
                abstracted = abstractor.abstract_path(path)
                
                if not isinstance(abstracted, str):
                    abstraction_failures.append(f"Path abstraction didn't return string: {path}")
                    continue
                
                # Should not contain personal names or sensitive identifiers
                sensitive_identifiers = ['john.doe', 'jane_smith', 'patient_samples', 'sample_12345', 
                                       'project_alpha', 'department_budget', 'hr_documents', 
                                       'employee_records', 'financial_data', 'personal_research']
                
                for identifier in sensitive_identifiers:
                    if identifier.lower() in abstracted.lower():
                        abstraction_failures.append(f"Sensitive identifier '{identifier}' in abstracted path: {abstracted}")
                
                # Should provide useful categorization
                if not any(category in abstracted.lower() for category in ['data', 'genomics', 'analysis', 
                                                                          'project', 'results', 'home', 'documents']):
                    abstraction_failures.append(f"Abstracted path lacks useful categorization: {abstracted}")
            
            if abstraction_failures:
                return False, f"Path abstraction failures: {'; '.join(abstraction_failures[:3])}..."
            
            return True, f"Path abstraction secure: {len(sensitive_paths)} paths properly abstracted"
            
        except ImportError as e:
            return False, f"Data abstraction module not found: {e}"
        except Exception as e:
            return False, f"Path abstraction security test failed: {str(e)}"

# ============================================================================
# Attack Scenario Simulation Tests
# ============================================================================

class AttackScenarioTests:
    """Simulate security attack scenarios to test defenses"""
    
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
        self.project_root = Path(__file__).parent.parent
    
    def test_encrypted_data_tampering_resistance(self) -> Tuple[bool, str]:
        """Test resistance to encrypted data tampering"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "security"))
            from security_manager import SecurityManager
            
            security = SecurityManager(data_dir=test_dir / ".claude" / "security")
            
            original_data = {"command": "sensitive_command", "user": "admin"}
            encrypted_data = security.encrypt_data(original_data)
            
            # Test various tampering attempts
            tampering_attempts = [
                # Flip random bits
                encrypted_data[:-5] + bytes([encrypted_data[-1] ^ 0xFF]),
                # Truncate data
                encrypted_data[:-10],
                # Prepend garbage
                b"garbage" + encrypted_data,
                # Append garbage  
                encrypted_data + b"garbage",
                # Replace with different valid-looking data
                b"A" * len(encrypted_data),
                # Swap bytes
                encrypted_data[1:] + encrypted_data[:1]
            ]
            
            successful_tampering = 0
            
            for i, tampered_data in enumerate(tampering_attempts):
                try:
                    decrypted = security.decrypt_data(tampered_data)
                    
                    if decrypted == original_data:
                        successful_tampering += 1
                        return False, f"Tampering attempt {i} succeeded - data integrity compromised"
                    elif decrypted is not None:
                        # Decryption succeeded but returned different data
                        return False, f"Tampering attempt {i} returned different data instead of failing"
                
                except Exception:
                    # Expected - tampering should cause decryption to fail
                    pass
            
            return True, f"Tamper resistance successful: {len(tampering_attempts)} tampering attempts failed as expected"
            
        except ImportError as e:
            return False, f"Security manager module not found: {e}"
        except Exception as e:
            return False, f"Tamper resistance test failed: {str(e)}"
    
    def test_timing_attack_resistance(self) -> Tuple[bool, str]:
        """Test resistance to timing-based attacks"""
        try:
            test_dir = self.test_env.create_isolated_project()
            
            sys.path.insert(0, str(self.project_root / "security"))
            from security_manager import SecurityManager
            
            security = SecurityManager(data_dir=test_dir / ".claude" / "security")
            
            # Test data of different sizes
            test_datasets = [
                {"small": "x"},
                {"medium": "x" * 1000},
                {"large": "x" * 10000}
            ]
            
            timing_measurements = {}
            
            for size_name, data in test_datasets:
                # Measure encryption times
                encrypt_times = []
                for _ in range(10):
                    start = time.perf_counter()
                    encrypted = security.encrypt_data(data)
                    end = time.perf_counter()
                    encrypt_times.append((end - start) * 1000)  # ms
                
                # Measure decryption times
                decrypt_times = []
                for _ in range(10):
                    start = time.perf_counter()
                    decrypted = security.decrypt_data(encrypted)
                    end = time.perf_counter()
                    decrypt_times.append((end - start) * 1000)  # ms
                
                timing_measurements[size_name] = {
                    'encrypt_avg': sum(encrypt_times) / len(encrypt_times),
                    'decrypt_avg': sum(decrypt_times) / len(decrypt_times),
                    'encrypt_var': max(encrypt_times) - min(encrypt_times),
                    'decrypt_var': max(decrypt_times) - min(decrypt_times)
                }
            
            # Check for obvious timing correlations with data size
            # (This is a simplified check - real timing attacks are more sophisticated)
            
            small_encrypt = timing_measurements['small']['encrypt_avg']
            large_encrypt = timing_measurements['large']['encrypt_avg']
            
            # If large data takes significantly longer, it might be vulnerable to timing attacks
            timing_ratio = large_encrypt / small_encrypt if small_encrypt > 0 else 1
            
            if timing_ratio > 10:  # More than 10x difference suggests size correlation
                return False, f"Potential timing attack vulnerability: {timing_ratio:.1f}x timing difference between small and large data"
            
            return True, f"Timing attack resistance: max {timing_ratio:.1f}x timing variation across data sizes"
            
        except ImportError as e:
            return False, f"Security manager module not found: {e}"
        except Exception as e:
            return False, f"Timing attack resistance test failed: {str(e)}"
    
    def test_host_identity_spoofing_resistance(self) -> Tuple[bool, str]:
        """Test resistance to host identity spoofing"""
        try:
            sys.path.insert(0, str(self.project_root / "security"))
            from hardware_identity import HardwareIdentity
            
            identity = HardwareIdentity()
            legitimate_id = identity.generate_stable_host_id()
            
            # Generate spoofing attempts
            spoofing_attempts = [
                legitimate_id + "x",  # Append char
                legitimate_id[:-1] + "x",  # Replace last char
                legitimate_id[::-1],  # Reverse
                legitimate_id.upper(),  # Case change
                legitimate_id.lower(),  # Case change
                legitimate_id.replace('a', 'b'),  # Simple substitution
                hashlib.sha256(legitimate_id.encode()).hexdigest(),  # Hash of legitimate ID
                "host_" + legitimate_id,  # Prefix
                legitimate_id[:len(legitimate_id)//2],  # Truncate
                legitimate_id + legitimate_id  # Duplicate
            ]
            
            successful_spoofs = 0
            
            for i, spoofed_id in enumerate(spoofing_attempts):
                if identity.validate_host_identity(spoofed_id):
                    successful_spoofs += 1
                    return False, f"Spoofing attempt {i} succeeded: {spoofed_id}"
            
            # Test that legitimate ID still validates
            if not identity.validate_host_identity(legitimate_id):
                return False, "Legitimate host ID fails validation"
            
            return True, f"Host identity spoofing resistance: {len(spoofing_attempts)} spoofing attempts failed, legitimate ID validated"
            
        except ImportError as e:
            return False, f"Hardware identity module not found: {e}"
        except Exception as e:
            return False, f"Host identity spoofing resistance test failed: {str(e)}"

# ============================================================================
# Test Suite Registration
# ============================================================================

def create_security_test_suites(test_env: TestEnvironment) -> List[TestSuite]:
    """Create all security validation test suites"""
    suites = []
    
    # Encryption security tests
    encryption_tests = EncryptionSecurityTests(test_env)
    encryption_suite = TestSuite(
        name="encryption_security_tests",
        tests=[
            encryption_tests.test_encryption_decryption_cycle_integrity,
            encryption_tests.test_encryption_uniqueness,
            encryption_tests.test_encryption_context_isolation,
            encryption_tests.test_key_rotation_security
        ]
    )
    suites.append(encryption_suite)
    
    # Hardware identity security tests
    identity_tests = HardwareIdentitySecurityTests(test_env)
    identity_suite = TestSuite(
        name="hardware_identity_security_tests",
        tests=[
            identity_tests.test_hardware_identity_stability,
            identity_tests.test_hardware_identity_uniqueness
        ]
    )
    suites.append(identity_suite)
    
    # Host authorization security tests
    auth_tests = HostAuthorizationSecurityTests(test_env)
    auth_suite = TestSuite(
        name="host_authorization_security_tests",
        tests=[
            auth_tests.test_host_trust_management_security,
            auth_tests.test_host_authorization_persistence
        ]
    )
    suites.append(auth_suite)
    
    # Data abstraction security tests
    abstraction_tests = DataAbstractionSecurityTests(test_env)
    abstraction_suite = TestSuite(
        name="data_abstraction_security_tests",
        tests=[
            abstraction_tests.test_command_abstraction_security,
            abstraction_tests.test_hostname_abstraction_security,
            abstraction_tests.test_path_abstraction_security
        ]
    )
    suites.append(abstraction_suite)
    
    # Attack scenario simulation tests
    attack_tests = AttackScenarioTests(test_env)
    attack_suite = TestSuite(
        name="attack_scenario_tests",
        tests=[
            attack_tests.test_encrypted_data_tampering_resistance,
            attack_tests.test_timing_attack_resistance,
            attack_tests.test_host_identity_spoofing_resistance
        ]
    )
    suites.append(attack_suite)
    
    return suites

# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all security validation tests"""
    print("🔒 Running Claude-Sync Security Validation Tests")
    print("=" * 60)
    
    # Create test framework and environment
    framework = TestFramework()
    test_env = TestEnvironment(framework.test_dir)
    
    try:
        # Setup test environment
        test_env.setup_isolated_project()
        
        # Create and register test suites
        suites = create_security_test_suites(test_env)
        for suite in suites:
            framework.register_test_suite(suite)
        
        # Run all tests
        session = framework.run_all_tests()
        
        # Print final results
        print(f"\n🎯 Security Validation Complete!")
        print(f"Success Rate: {session.success_rate:.1%}")
        print(f"Total Time: {session.duration_ms:.0f}ms")
        
        if session.success_rate >= 0.9:  # High bar for security tests
            print("✅ Security validation PASSED (≥90% success rate)")
            return 0
        else:
            print("❌ Security validation FAILED (<90% success rate)")
            return 1
    
    finally:
        # Cleanup test environment
        test_env.restore_environment()

if __name__ == "__main__":
    exit(main())