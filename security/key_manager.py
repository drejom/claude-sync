#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0",
#   "psutil>=5.9.0",
# ]
# ///
"""
Simple Key Management with Auto-Rotation

Implements daily automatic key rotation for encrypted learning data storage.
Uses PBKDF2 for key derivation and Fernet for symmetric encryption.

Key Features:
- Daily automatic key rotation based on date
- PBKDF2HMAC with 100,000 iterations for key derivation  
- Host-specific entropy ensures unique keys per host
- Automatic cleanup of old keys (7-day retention)
- Performance optimized (<5ms encryption operations)
- Deterministic key generation for consistency

Security Model:
- Keys derived from host ID + date + constant salt
- Each host generates unique keys independently
- Keys automatically rotate daily without manual intervention
- Old keys retained for 7 days to handle clock skew
- No network communication required for key generation
"""

import hashlib
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from hardware_identity import HardwareIdentity

class SimpleKeyManager:
    """Simple key rotation without complex synchronization"""
    
    def __init__(self):
        self.hardware_identity = HardwareIdentity()
        self.host_id = self.hardware_identity.generate_stable_host_id()
        
        # Key storage
        self.claude_dir = Path.home() / '.claude'
        self.key_dir = self.claude_dir / 'keys'
        self.key_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.key_metadata_file = self.key_dir / 'key_metadata.json'
        
        # Security constants
        self.PBKDF2_ITERATIONS = 100000
        self.KEY_RETENTION_DAYS = 7
        self.FERNET_KEY_SIZE = 32  # 256 bits
        
        # Salt for key derivation (constant per installation)
        self.CLAUDE_SALT = b'claude-learning-encryption-v1'
        
        # Set secure permissions
        self._ensure_secure_permissions()
        
        # Cache for current key
        self._current_key_cache = None
        self._current_key_date = None

    def _ensure_secure_permissions(self):
        """Ensure key directories have secure permissions"""
        try:
            # Set directory permissions: owner read/write/execute only
            os.chmod(self.key_dir, 0o700)
            
            # Set permissions on existing key files
            for key_file in self.key_dir.glob("key_*.enc"):
                os.chmod(key_file, 0o600)
                
            if self.key_metadata_file.exists():
                os.chmod(self.key_metadata_file, 0o600)
                
        except Exception:
            pass  # Permission errors shouldn't break key operations

    def _get_date_string(self, offset_days: int = 0) -> str:
        """Get date string for key generation"""
        target_date = datetime.now() + timedelta(days=offset_days)
        return target_date.strftime('%Y-%m-%d')

    def _generate_key(self, date_str: str) -> bytes:
        """Generate deterministic key from host ID + date"""
        # Combine host ID, date, and version info for seed material
        seed_material = f"{self.host_id}:{date_str}:claude-sync-v1"
        
        # Use PBKDF2 for key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.FERNET_KEY_SIZE,
            salt=self.CLAUDE_SALT,
            iterations=self.PBKDF2_ITERATIONS,
        )
        
        # Derive key
        derived_key = kdf.derive(seed_material.encode('utf-8'))
        
        # Return Fernet-compatible key (base64 encoded)
        return base64.urlsafe_b64encode(derived_key)

    def _load_key(self, key_file: Path) -> Optional[bytes]:
        """Load key from encrypted file"""
        try:
            if not key_file.exists():
                return None
                
            with open(key_file, 'rb') as f:
                encrypted_key = f.read()
            
            # For simplicity, we'll store the key directly (already derived)
            # In production, you might want additional encryption layers
            return encrypted_key
            
        except Exception:
            return None

    def _save_key(self, key_file: Path, key: bytes) -> bool:
        """Save key to encrypted file"""
        try:
            # Write to temporary file first for atomic operation
            temp_file = key_file.with_suffix('.tmp')
            
            with open(temp_file, 'wb') as f:
                f.write(key)
                f.flush()
                os.fsync(f.fileno())
            
            # Set secure permissions on temp file
            os.chmod(temp_file, 0o600)
            
            # Atomic rename
            temp_file.replace(key_file)
            
            # Update metadata
            self._update_key_metadata(key_file.stem, time.time())
            
            return True
            
        except Exception:
            # Clean up temp file if it exists
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
            return False

    def _update_key_metadata(self, key_name: str, created_at: float):
        """Update key metadata for tracking"""
        try:
            metadata = self._load_key_metadata()
            metadata['keys'][key_name] = {
                'created_at': created_at,
                'host_id': self.host_id,
                'key_version': '1.0'
            }
            metadata['last_updated'] = time.time()
            
            # Save metadata
            with open(self.key_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            os.chmod(self.key_metadata_file, 0o600)
            
        except Exception:
            pass  # Metadata failure shouldn't break key operations

    def _load_key_metadata(self) -> Dict[str, Any]:
        """Load key metadata"""
        if not self.key_metadata_file.exists():
            return {
                'version': '1.0',
                'keys': {},
                'host_id': self.host_id,
                'created_at': time.time(),
                'last_updated': time.time()
            }
        
        try:
            with open(self.key_metadata_file, 'r') as f:
                return json.load(f)
        except Exception:
            return self._create_empty_metadata()

    def _create_empty_metadata(self) -> Dict[str, Any]:
        """Create empty metadata structure"""
        return {
            'version': '1.0', 
            'keys': {},
            'host_id': self.host_id,
            'created_at': time.time(),
            'last_updated': time.time()
        }

    def get_current_key(self) -> bytes:
        """Get today's encryption key - auto-generates if needed"""
        today = self._get_date_string()
        
        # Check cache first
        if (self._current_key_cache and 
            self._current_key_date == today):
            return self._current_key_cache
        
        key_file = self.key_dir / f"key_{today}.enc"
        
        # Try to load existing key
        key = self._load_key(key_file)
        
        if key is None:
            # Generate new key for today
            start_time = time.time()
            key = self._generate_key(today)
            generation_time = (time.time() - start_time) * 1000
            
            # Check performance target
            if generation_time > 50:  # Allow some overhead beyond the 5ms target
                print(f"Warning: Key generation took {generation_time:.1f}ms")
            
            # Save the key
            if not self._save_key(key_file, key):
                raise RuntimeError(f"Failed to save encryption key for {today}")
            
            # Clean up old keys
            self._cleanup_old_keys()
        
        # Update cache
        self._current_key_cache = key
        self._current_key_date = today
        
        return key

    def _cleanup_old_keys(self) -> int:
        """Remove keys older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.KEY_RETENTION_DAYS)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        
        removed_count = 0
        
        for key_file in self.key_dir.glob("key_*.enc"):
            try:
                # Extract date from filename
                filename = key_file.stem  # Remove .enc extension
                date_part = filename.split('_', 1)[1]  # Remove 'key_' prefix
                
                # Compare dates (string comparison works for YYYY-MM-DD format)
                if date_part < cutoff_str:
                    key_file.unlink()
                    removed_count += 1
                    
                    # Remove from metadata
                    metadata = self._load_key_metadata()
                    if filename in metadata.get('keys', {}):
                        del metadata['keys'][filename]
                        with open(self.key_metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                    
            except Exception:
                # Skip malformed files
                continue
        
        return removed_count

    def rotate_keys(self) -> bool:
        """Perform key rotation (mainly cleanup since keys auto-generate)"""
        try:
            # Ensure we have today's key
            self.get_current_key()
            
            # Clean up old keys
            removed = self._cleanup_old_keys()
            
            # Clear cache to force regeneration
            self._current_key_cache = None
            self._current_key_date = None
            
            return True
            
        except Exception:
            return False

    def get_current_key_id(self) -> str:
        """Get current encryption key identifier"""
        today = self._get_date_string()
        return f"key_{today}_{self.host_id[:8]}"

    def cleanup_old_keys(self, retention_days: int = None) -> int:
        """Remove old encryption keys"""
        if retention_days is not None:
            old_retention = self.KEY_RETENTION_DAYS
            self.KEY_RETENTION_DAYS = retention_days
            
        try:
            return self._cleanup_old_keys()
        finally:
            if retention_days is not None:
                self.KEY_RETENTION_DAYS = old_retention

    def get_key_for_date(self, date_str: str) -> Optional[bytes]:
        """Get key for specific date (for decrypting old data)"""
        key_file = self.key_dir / f"key_{date_str}.enc"
        
        # Try to load existing key
        key = self._load_key(key_file)
        
        if key is None:
            # Check if date is within retention period
            try:
                key_date = datetime.strptime(date_str, '%Y-%m-%d')
                now = datetime.now()
                days_old = (now - key_date).days
                
                if days_old <= self.KEY_RETENTION_DAYS:
                    # Generate key for that date
                    key = self._generate_key(date_str)
                    # Don't save historical keys to avoid proliferation
                
            except ValueError:
                pass
        
        return key

    def create_fernet_cipher(self, key: Optional[bytes] = None) -> Fernet:
        """Create Fernet cipher with current or specified key"""
        if key is None:
            key = self.get_current_key()
        return Fernet(key)

    def encrypt_data(self, data: bytes, context: str = "default") -> bytes:
        """Encrypt data with current key"""
        start_time = time.time()
        
        try:
            fernet = self.create_fernet_cipher()
            encrypted = fernet.encrypt(data)
            
            # Check performance target
            duration_ms = (time.time() - start_time) * 1000
            if duration_ms > 5:  # 5ms target from interfaces.py
                print(f"Warning: Encryption took {duration_ms:.1f}ms (target: <5ms)")
            
            return encrypted
            
        except Exception as e:
            raise RuntimeError(f"Encryption failed: {str(e)}")

    def decrypt_data(self, encrypted_data: bytes, date_hint: str = None) -> Optional[bytes]:
        """Decrypt data, trying current key first, then recent keys"""
        # Try current key first
        try:
            fernet = self.create_fernet_cipher()
            return fernet.decrypt(encrypted_data)
        except Exception:
            pass
        
        # Try keys from recent days
        for days_back in range(1, self.KEY_RETENTION_DAYS + 1):
            try:
                date_str = self._get_date_string(-days_back)
                key = self.get_key_for_date(date_str)
                if key:
                    fernet = Fernet(key)
                    return fernet.decrypt(encrypted_data)
            except Exception:
                continue
        
        # If date hint provided, try that specific key
        if date_hint:
            try:
                key = self.get_key_for_date(date_hint)
                if key:
                    fernet = Fernet(key)
                    return fernet.decrypt(encrypted_data)
            except Exception:
                pass
        
        return None

    def get_key_statistics(self) -> Dict[str, Any]:
        """Get key management statistics"""
        metadata = self._load_key_metadata()
        
        # Count active keys
        active_keys = len(list(self.key_dir.glob("key_*.enc")))
        
        # Calculate total key directory size
        total_size = sum(f.stat().st_size for f in self.key_dir.glob("*") if f.is_file())
        
        stats = {
            'host_id': self.host_id,
            'current_key_id': self.get_current_key_id(),
            'active_keys_count': active_keys,
            'retention_days': self.KEY_RETENTION_DAYS,
            'key_directory_size_bytes': total_size,
            'last_cleanup': metadata.get('last_cleanup', 'Never'),
            'key_directory_path': str(self.key_dir),
            'pbkdf2_iterations': self.PBKDF2_ITERATIONS
        }
        
        # Recent activity
        recent_keys = 0
        now = time.time()
        for key_info in metadata.get('keys', {}).values():
            created_at = key_info.get('created_at', 0)
            if (now - created_at) < 86400:  # Last 24 hours
                recent_keys += 1
        
        stats['keys_generated_today'] = recent_keys
        
        return stats

def main():
    """Test key management functionality"""
    key_manager = SimpleKeyManager()
    
    print("Key Management System Test")
    print("=" * 50)
    
    # Test key generation performance
    start_time = time.time()
    current_key = key_manager.get_current_key()
    generation_time = (time.time() - start_time) * 1000
    
    print(f"Current Key ID: {key_manager.get_current_key_id()}")
    print(f"Key Generation Time: {generation_time:.1f}ms")
    print()
    
    # Test encryption/decryption performance
    test_data = b"This is test learning data that needs to be encrypted safely"
    
    start_time = time.time()
    encrypted = key_manager.encrypt_data(test_data)
    encryption_time = (time.time() - start_time) * 1000
    
    start_time = time.time()
    decrypted = key_manager.decrypt_data(encrypted)
    decryption_time = (time.time() - start_time) * 1000
    
    print(f"Encryption Time: {encryption_time:.1f}ms")
    print(f"Decryption Time: {decryption_time:.1f}ms")
    print(f"Data Integrity: {'✓' if decrypted == test_data else '✗'}")
    print()
    
    # Show statistics
    stats = key_manager.get_key_statistics()
    print("Key Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Test cleanup
    print("Testing key cleanup...")
    removed = key_manager.cleanup_old_keys()
    print(f"Old keys removed: {removed}")

if __name__ == '__main__':
    main()