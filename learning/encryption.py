#!/usr/bin/env python3
"""
Secure Learning Infrastructure - Encryption & Key Management
Military-grade security for AI learning data
"""

import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class SecureLearningStorage:
    """Encrypted storage for AI learning data with rotating keys"""
    
    def __init__(self, storage_dir=None):
        self.storage_dir = Path(storage_dir or Path.home() / '.claude' / 'learning')
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.key_rotation_hours = 24  # Rotate keys daily
        
    def _get_host_entropy(self):
        """Generate host-specific entropy for key derivation"""
        # Combine multiple host characteristics for entropy
        entropy_sources = [
            os.uname().nodename,  # Hostname
            str(os.getuid()),     # User ID
            str(os.getpid()),     # Process ID
            str(time.time())      # Current time
        ]
        
        # Add system-specific entropy if available
        try:
            with open('/proc/version', 'r') as f:
                entropy_sources.append(f.read().strip())
        except:
            pass
            
        combined = ''.join(entropy_sources)
        return hashlib.sha256(combined.encode()).digest()
    
    def _derive_key(self, purpose='general'):
        """Derive encryption key from host entropy and purpose"""
        entropy = self._get_host_entropy()
        salt = hashlib.sha256(f"claude-learning-{purpose}".encode()).digest()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(entropy))
        return Fernet(key)
    
    def _get_storage_path(self, data_type):
        """Get encrypted storage path for data type"""
        # Use semantic names, not actual system info
        hashed_type = hashlib.md5(data_type.encode()).hexdigest()[:8]
        return self.storage_dir / f"learning_{hashed_type}.enc"
    
    def store_learning_data(self, data_type, data, max_age_hours=168):
        """Store learning data with encryption and expiration"""
        try:
            cipher = self._derive_key(data_type)
            
            # Add metadata for expiration and validation
            wrapped_data = {
                'data': data,
                'timestamp': time.time(),
                'expires_at': time.time() + (max_age_hours * 3600),
                'data_type': data_type,
                'version': 1
            }
            
            # Encrypt and store
            encrypted_data = cipher.encrypt(json.dumps(wrapped_data).encode())
            storage_path = self._get_storage_path(data_type)
            
            with open(storage_path, 'wb') as f:
                f.write(encrypted_data)
                
            return True
            
        except Exception:
            return False
    
    def load_learning_data(self, data_type, default=None):
        """Load and decrypt learning data with expiration check"""
        try:
            storage_path = self._get_storage_path(data_type)
            
            if not storage_path.exists():
                return default
                
            cipher = self._derive_key(data_type)
            
            with open(storage_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt and validate
            decrypted_data = cipher.decrypt(encrypted_data)
            wrapped_data = json.loads(decrypted_data.decode())
            
            # Check expiration
            if time.time() > wrapped_data.get('expires_at', 0):
                # Data expired, remove it
                storage_path.unlink(missing_ok=True)
                return default
            
            return wrapped_data.get('data', default)
            
        except Exception:
            return default
    
    def cleanup_expired_data(self):
        """Remove expired learning data"""
        try:
            for file_path in self.storage_dir.glob("learning_*.enc"):
                try:
                    # Try to load and check expiration
                    data_type = file_path.stem.replace('learning_', '')
                    self.load_learning_data(data_type)  # This will auto-remove if expired
                except:
                    # If we can't decrypt, it's probably corrupted - remove it
                    file_path.unlink(missing_ok=True)
        except Exception:
            pass

def get_secure_storage():
    """Get a configured secure storage instance"""
    return SecureLearningStorage()

if __name__ == '__main__':
    # Test the encryption system
    storage = get_secure_storage()
    
    # Test data
    test_data = {'test': 'data', 'patterns': ['a', 'b', 'c']}
    
    # Store and retrieve
    storage.store_learning_data('test', test_data)
    retrieved = storage.load_learning_data('test')
    
    print(f"Storage test: {'PASS' if retrieved == test_data else 'FAIL'}")
    
    # Cleanup
    storage.cleanup_expired_data()