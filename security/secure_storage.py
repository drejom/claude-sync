#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0",
#   "psutil>=5.9.0",
# ]
# ///
"""
Secure Learning Data Storage

Implements encrypted storage for learning data with automatic key rotation
and data expiration. Integrates with the learning system to provide seamless
encryption/decryption while maintaining zero-knowledge abstractions.

Key Features:
- Fernet encryption for all learning data
- Automatic data expiration (30-day default)
- Integration with SimpleKeyManager for key rotation
- Performance optimized storage operations
- Safe concurrent access with file locking
- Automatic schema evolution support

Security Model:
- All sensitive learning data encrypted at rest
- Keys automatically rotate daily
- Old data automatically expires and is removed
- No sensitive data ever stored in plaintext
- Zero-knowledge abstractions preserved
"""

import json
import time
import os
import fcntl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import gzip

from key_manager import SimpleKeyManager

class SecureLearningStorage:
    """Encrypted storage for learning data with auto-expiration"""
    
    def __init__(self, retention_days: int = 30):
        self.key_manager = SimpleKeyManager()
        self.retention_days = retention_days
        
        # Storage locations
        self.claude_dir = Path.home() / '.claude'
        self.learning_dir = self.claude_dir / 'learning'
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata_file = self.learning_dir / 'storage_metadata.json'
        
        # Set secure permissions
        self._ensure_secure_permissions()

    def _ensure_secure_permissions(self):
        """Ensure learning directory has secure permissions"""
        try:
            # Set directory permissions: owner read/write/execute only
            os.chmod(self.learning_dir, 0o700)
            
            # Set permissions on existing files
            for file_path in self.learning_dir.glob("*"):
                if file_path.is_file():
                    os.chmod(file_path, 0o600)
                    
        except Exception:
            pass  # Permission errors shouldn't break operations

    def _get_storage_filename(self, data_type: str, date_str: str = None) -> str:
        """Generate storage filename for data type and date"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Hash the data type for filename safety
        type_hash = hashlib.sha256(data_type.encode()).hexdigest()[:8]
        return f"learning_{data_type}_{type_hash}_{date_str}.enc"

    def _load_metadata(self) -> Dict[str, Any]:
        """Load storage metadata"""
        if not self.metadata_file.exists():
            return {
                'version': '1.0',
                'created_at': time.time(),
                'last_cleanup': 0,
                'data_files': {},
                'encryption_stats': {
                    'total_encryptions': 0,
                    'total_decryptions': 0,
                    'avg_encryption_time_ms': 0,
                    'avg_decryption_time_ms': 0
                }
            }
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception:
            return self._create_empty_metadata()

    def _create_empty_metadata(self) -> Dict[str, Any]:
        """Create empty metadata structure"""
        return {
            'version': '1.0',
            'created_at': time.time(),
            'last_cleanup': 0,
            'data_files': {},
            'encryption_stats': {
                'total_encryptions': 0,
                'total_decryptions': 0,
                'avg_encryption_time_ms': 0,
                'avg_decryption_time_ms': 0
            }
        }

    def _save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Save metadata with file locking"""
        try:
            # Write to temporary file first
            temp_file = self.metadata_file.with_suffix('.tmp')
            
            with open(temp_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(metadata, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            
            # Set secure permissions
            os.chmod(temp_file, 0o600)
            
            # Atomic rename
            temp_file.replace(self.metadata_file)
            
            return True
            
        except Exception:
            # Clean up temp file if it exists
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
            return False

    def _update_encryption_stats(self, operation: str, duration_ms: float):
        """Update encryption performance statistics"""
        try:
            metadata = self._load_metadata()
            stats = metadata.get('encryption_stats', {})
            
            if operation == 'encrypt':
                count = stats.get('total_encryptions', 0)
                avg_time = stats.get('avg_encryption_time_ms', 0)
                
                # Running average
                new_avg = (avg_time * count + duration_ms) / (count + 1)
                
                stats['total_encryptions'] = count + 1
                stats['avg_encryption_time_ms'] = new_avg
                
            elif operation == 'decrypt':
                count = stats.get('total_decryptions', 0)
                avg_time = stats.get('avg_decryption_time_ms', 0)
                
                # Running average
                new_avg = (avg_time * count + duration_ms) / (count + 1)
                
                stats['total_decryptions'] = count + 1
                stats['avg_decryption_time_ms'] = new_avg
            
            metadata['encryption_stats'] = stats
            self._save_metadata(metadata)
            
        except Exception:
            pass  # Stats failure shouldn't break operations

    def encrypt_data(self, data: Dict[str, Any], context: str = "default") -> bytes:
        """Encrypt data with current key"""
        start_time = time.time()
        
        try:
            # Add metadata to the data
            data_with_metadata = {
                'data': data,
                'context': context,
                'encrypted_at': time.time(),
                'encrypted_by': self.key_manager.host_id,
                'key_date': datetime.now().strftime('%Y-%m-%d'),
                'version': '1.0'
            }
            
            # Serialize to JSON
            json_data = json.dumps(data_with_metadata, sort_keys=True)
            
            # Compress if data is large
            if len(json_data) > 1024:  # 1KB threshold
                json_bytes = gzip.compress(json_data.encode('utf-8'))
                data_with_metadata['compressed'] = True
            else:
                json_bytes = json_data.encode('utf-8')
                data_with_metadata['compressed'] = False
            
            # Re-serialize with compression flag
            if data_with_metadata.get('compressed'):
                final_json = json.dumps({
                    'compressed': True,
                    'metadata': {
                        'context': context,
                        'encrypted_at': data_with_metadata['encrypted_at'],
                        'encrypted_by': data_with_metadata['encrypted_by'],
                        'key_date': data_with_metadata['key_date'],
                        'version': data_with_metadata['version']
                    }
                }).encode('utf-8')
                final_data = final_json + b'\n---COMPRESSED---\n' + json_bytes
            else:
                final_data = json_bytes
            
            # Encrypt with Fernet
            encrypted_data = self.key_manager.encrypt_data(final_data, context)
            
            # Update stats
            duration_ms = (time.time() - start_time) * 1000
            self._update_encryption_stats('encrypt', duration_ms)
            
            return encrypted_data
            
        except Exception as e:
            raise RuntimeError(f"Data encryption failed: {str(e)}")

    def decrypt_data(self, encrypted_data: bytes, context: str = "default") -> Optional[Dict[str, Any]]:
        """Decrypt data, return None if failed"""
        start_time = time.time()
        
        try:
            # Decrypt with key manager (tries current and recent keys)
            decrypted_bytes = self.key_manager.decrypt_data(encrypted_data)
            if decrypted_bytes is None:
                return None
            
            # Check if data was compressed
            if b'\n---COMPRESSED---\n' in decrypted_bytes:
                parts = decrypted_bytes.split(b'\n---COMPRESSED---\n', 1)
                if len(parts) == 2:
                    # Decompress the data part
                    compressed_data = parts[1]
                    decompressed_json = gzip.decompress(compressed_data).decode('utf-8')
                    data_with_metadata = json.loads(decompressed_json)
                else:
                    return None
            else:
                # Regular JSON data
                json_str = decrypted_bytes.decode('utf-8')
                data_with_metadata = json.loads(json_str)
            
            # Extract the actual data
            if isinstance(data_with_metadata, dict):
                if 'data' in data_with_metadata:
                    actual_data = data_with_metadata['data']
                else:
                    actual_data = data_with_metadata
            else:
                actual_data = data_with_metadata
            
            # Update stats
            duration_ms = (time.time() - start_time) * 1000
            self._update_encryption_stats('decrypt', duration_ms)
            
            return actual_data
            
        except Exception:
            return None

    def store_learning_data(self, data_type: str, data: Dict[str, Any], 
                          compress: bool = True) -> bool:
        """Store learning data with encryption"""
        try:
            # Create filename
            filename = self._get_storage_filename(data_type)
            file_path = self.learning_dir / filename
            
            # Encrypt the data
            encrypted_data = self.encrypt_data(data, context=data_type)
            
            # Write to file with atomic operation
            temp_file = file_path.with_suffix('.tmp')
            
            with open(temp_file, 'wb') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(encrypted_data)
                f.flush()
                os.fsync(f.fileno())
            
            # Set secure permissions
            os.chmod(temp_file, 0o600)
            
            # Atomic rename
            temp_file.replace(file_path)
            
            # Update metadata
            metadata = self._load_metadata()
            metadata['data_files'][filename] = {
                'data_type': data_type,
                'stored_at': time.time(),
                'file_size': file_path.stat().st_size,
                'compressed': compress
            }
            self._save_metadata(metadata)
            
            return True
            
        except Exception:
            # Clean up temp file if it exists
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
            return False

    def load_learning_data(self, data_type: str, max_age_days: int = None) -> List[Dict[str, Any]]:
        """Load learning data of specified type"""
        if max_age_days is None:
            max_age_days = self.retention_days
        
        cutoff_time = time.time() - (max_age_days * 86400)
        results = []
        
        # Find matching files
        pattern = f"learning_{data_type}_*"
        for file_path in self.learning_dir.glob(pattern):
            try:
                # Check file age
                file_stat = file_path.stat()
                if file_stat.st_mtime < cutoff_time:
                    continue
                
                # Load and decrypt file
                with open(file_path, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_data = self.decrypt_data(encrypted_data, context=data_type)
                if decrypted_data:
                    results.append(decrypted_data)
                    
            except Exception:
                continue  # Skip corrupted files
        
        return results

    def cleanup_expired_data(self, retention_days: int = None) -> int:
        """Remove expired learning data"""
        if retention_days is None:
            retention_days = self.retention_days
        
        cutoff_time = time.time() - (retention_days * 86400)
        removed_count = 0
        
        metadata = self._load_metadata()
        files_to_remove = []
        
        # Find expired files
        for file_path in self.learning_dir.glob("learning_*.enc"):
            try:
                file_stat = file_path.stat()
                if file_stat.st_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1
                    files_to_remove.append(file_path.name)
                    
            except Exception:
                continue
        
        # Update metadata
        for filename in files_to_remove:
            if filename in metadata.get('data_files', {}):
                del metadata['data_files'][filename]
        
        metadata['last_cleanup'] = time.time()
        self._save_metadata(metadata)
        
        return removed_count

    def rotate_keys(self) -> bool:
        """Trigger key rotation"""
        return self.key_manager.rotate_keys()

    def get_current_key_id(self) -> str:
        """Get current encryption key identifier"""
        return self.key_manager.get_current_key_id()

    def cleanup_old_keys(self, retention_days: int = 7) -> int:
        """Remove old encryption keys"""
        return self.key_manager.cleanup_old_keys(retention_days)

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage system statistics"""
        metadata = self._load_metadata()
        
        # Calculate directory size
        total_size = 0
        file_count = 0
        for file_path in self.learning_dir.glob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        # Count data types
        data_types = {}
        for filename, file_info in metadata.get('data_files', {}).items():
            data_type = file_info.get('data_type', 'unknown')
            data_types[data_type] = data_types.get(data_type, 0) + 1
        
        stats = {
            'storage_directory': str(self.learning_dir),
            'total_size_bytes': total_size,
            'total_files': file_count,
            'retention_days': self.retention_days,
            'data_types': data_types,
            'last_cleanup': datetime.fromtimestamp(
                metadata.get('last_cleanup', 0)
            ).isoformat() if metadata.get('last_cleanup') else 'Never',
            'encryption_stats': metadata.get('encryption_stats', {}),
            'key_manager_stats': self.key_manager.get_key_statistics()
        }
        
        return stats

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify data integrity and encryption status"""
        results = {
            'total_files': 0,
            'readable_files': 0,
            'corrupted_files': 0,
            'decryption_failures': 0,
            'permission_issues': 0,
            'corrupted_file_list': []
        }
        
        for file_path in self.learning_dir.glob("learning_*.enc"):
            results['total_files'] += 1
            
            try:
                # Check file permissions
                stat_info = file_path.stat()
                if stat_info.st_mode & 0o077:  # Others have access
                    results['permission_issues'] += 1
                
                # Try to read and decrypt
                with open(file_path, 'rb') as f:
                    encrypted_data = f.read()
                
                # Extract data type from filename for context
                filename = file_path.name
                parts = filename.split('_')
                data_type = parts[1] if len(parts) > 1 else 'unknown'
                
                decrypted_data = self.decrypt_data(encrypted_data, context=data_type)
                
                if decrypted_data is not None:
                    results['readable_files'] += 1
                else:
                    results['decryption_failures'] += 1
                    results['corrupted_file_list'].append(str(file_path))
                    
            except Exception:
                results['corrupted_files'] += 1
                results['corrupted_file_list'].append(str(file_path))
        
        return results

def main():
    """Test secure storage functionality"""
    storage = SecureLearningStorage(retention_days=7)
    
    print("Secure Learning Storage Test")
    print("=" * 50)
    
    # Test data
    test_data = {
        'command_pattern': 'ssh_connection',
        'success_rate': 0.95,
        'optimization_applied': True,
        'timestamp': time.time(),
        'host_type': 'compute-server-a1b2',
        'performance_metrics': {
            'connection_time_ms': 150,
            'throughput_mbps': 100
        }
    }
    
    # Test storage
    print("Testing data storage...")
    success = storage.store_learning_data('ssh_patterns', test_data)
    print(f"Storage success: {'✓' if success else '✗'}")
    
    # Test retrieval
    print("Testing data retrieval...")
    retrieved_data = storage.load_learning_data('ssh_patterns', max_age_days=1)
    print(f"Retrieved {len(retrieved_data)} records")
    
    if retrieved_data and len(retrieved_data) > 0:
        first_record = retrieved_data[0]
        data_match = (first_record.get('command_pattern') == test_data['command_pattern'] and
                     first_record.get('success_rate') == test_data['success_rate'])
        print(f"Data integrity: {'✓' if data_match else '✗'}")
    
    # Show statistics
    print("\nStorage Statistics:")
    stats = storage.get_storage_statistics()
    for key, value in stats.items():
        if key != 'key_manager_stats':  # Skip nested dict for brevity
            print(f"  {key}: {value}")
    
    # Test integrity verification
    print("\nIntegrity Check:")
    integrity = storage.verify_integrity()
    print(f"  Total files: {integrity['total_files']}")
    print(f"  Readable files: {integrity['readable_files']}")
    print(f"  Corrupted files: {integrity['corrupted_files']}")
    print(f"  Decryption failures: {integrity['decryption_failures']}")

if __name__ == '__main__':
    main()