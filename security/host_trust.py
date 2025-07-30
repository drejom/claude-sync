#!/usr/bin/env -S uv run
# /// script  
# requires-python = ">=3.10"
# dependencies = [
#   "psutil>=5.9.0",
# ]
# ///
"""
Simple Host Trust Management

Implements binary trust model: hosts are either trusted or not trusted.
Uses hardware-based host identity for stable identification across OS reinstalls.

Key Features:
- Hardware-based host identification
- Simple trust list management (add/remove/list)
- Audit trail for authorization events
- Atomic operations for trust modifications
- Safe concurrent access to trust data

Security Model:
- Trust decisions are binary (trusted/not trusted)
- Host identity tied to stable hardware characteristics
- All trust changes are logged for audit
- Trust file protected with appropriate permissions
- No complex permission hierarchies or capabilities
"""

import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import fcntl
import os

from hardware_identity import HardwareIdentity

class SimpleHostTrust:
    """Dead simple host trust based on hardware fingerprint"""
    
    def __init__(self):
        self.hardware_identity = HardwareIdentity()
        self.host_id = self.hardware_identity.generate_stable_host_id()
        
        # Trust storage locations
        self.claude_dir = Path.home() / '.claude'
        self.trust_file = self.claude_dir / 'trusted_hosts.json'
        self.audit_file = self.claude_dir / 'security_audit.log'
        
        # Ensure directories exist
        self.claude_dir.mkdir(parents=True, exist_ok=True)
        
        # Set secure permissions on trust file
        self._ensure_secure_permissions()
    
    def _ensure_secure_permissions(self):
        """Ensure trust files have secure permissions"""
        try:
            # Set directory permissions: owner read/write/execute only
            os.chmod(self.claude_dir, 0o700)
            
            # Set file permissions if they exist
            for file_path in [self.trust_file, self.audit_file]:
                if file_path.exists():
                    os.chmod(file_path, 0o600)  # owner read/write only
                    
        except Exception as e:
            self._log_audit_event('security_warning', {
                'event': 'permission_set_failed',
                'error': str(e)
            })

    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log security audit event"""
        try:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'timestamp_unix': time.time(),
                'event_type': event_type,
                'local_host_id': self.host_id,
                'details': details
            }
            
            # Append to audit log with file locking
            with open(self.audit_file, 'a') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(audit_entry) + '\n')
                f.flush()
                os.fsync(f.fileno())
                
            # Set secure permissions on audit file
            os.chmod(self.audit_file, 0o600)
            
        except Exception:
            # Audit failure shouldn't break operations
            pass

    def _load_trust_data(self) -> Dict[str, Any]:
        """Load trust data with file locking"""
        if not self.trust_file.exists():
            return {
                'version': '1.0',
                'trusted_hosts': {},
                'created_at': time.time(),
                'last_modified': time.time()
            }
        
        try:
            with open(self.trust_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                data = json.load(f)
                
            # Validate data structure
            if not isinstance(data, dict) or 'trusted_hosts' not in data:
                self._log_audit_event('data_corruption', {
                    'event': 'invalid_trust_file_structure',
                    'action': 'recreating_trust_file'
                })
                return self._create_empty_trust_data()
                
            return data
            
        except (json.JSONDecodeError, IOError) as e:
            self._log_audit_event('data_corruption', {
                'event': 'trust_file_load_failed',
                'error': str(e),
                'action': 'recreating_trust_file'
            })
            return self._create_empty_trust_data()

    def _create_empty_trust_data(self) -> Dict[str, Any]:
        """Create empty trust data structure"""
        return {
            'version': '1.0',
            'trusted_hosts': {},
            'created_at': time.time(),
            'last_modified': time.time()
        }

    def _save_trust_data(self, data: Dict[str, Any]) -> bool:
        """Save trust data with atomic operations and file locking"""
        try:
            data['last_modified'] = time.time()
            
            # Write to temporary file first (atomic operation)
            temp_file = self.trust_file.with_suffix('.tmp')
            
            with open(temp_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                json.dump(data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            
            # Set secure permissions on temp file
            os.chmod(temp_file, 0o600)
            
            # Atomic rename
            temp_file.replace(self.trust_file)
            
            return True
            
        except Exception as e:
            self._log_audit_event('save_error', {
                'event': 'trust_file_save_failed',
                'error': str(e)
            })
            
            # Clean up temp file if it exists
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
                
            return False

    def get_host_identity(self) -> str:
        """Get stable hardware-based host identity"""
        return self.host_id

    def is_trusted_host(self, host_id: str) -> bool:
        """Check if host is in trust list"""
        if not host_id or len(host_id) != 16:  # Basic validation
            return False
            
        trust_data = self._load_trust_data()
        trusted_hosts = trust_data.get('trusted_hosts', {})
        
        return host_id in trusted_hosts

    def authorize_host(self, host_id: str, host_description: str = "") -> bool:
        """Add host to trust list"""
        if not host_id or len(host_id) != 16:
            self._log_audit_event('authorization_failed', {
                'event': 'invalid_host_id',
                'provided_host_id': host_id,
                'reason': 'invalid_format'
            })
            return False
        
        # Don't allow adding self (shouldn't be necessary)
        if host_id == self.host_id:
            self._log_audit_event('authorization_failed', {
                'event': 'self_authorization_attempted',
                'host_id': host_id,
                'reason': 'cannot_trust_self'
            })
            return False
        
        trust_data = self._load_trust_data()
        trusted_hosts = trust_data.get('trusted_hosts', {})
        
        # Check if already trusted
        if host_id in trusted_hosts:
            self._log_audit_event('authorization_duplicate', {
                'event': 'host_already_trusted',
                'host_id': host_id,
                'existing_entry': trusted_hosts[host_id]
            })
            return True  # Already trusted, return success
        
        # Add to trust list
        trusted_hosts[host_id] = {
            'host_id': host_id,
            'description': host_description,
            'authorized_at': time.time(),
            'authorized_by_host': self.host_id,
            'last_seen': None,
            'authorization_count': 1
        }
        
        trust_data['trusted_hosts'] = trusted_hosts
        
        # Save changes
        success = self._save_trust_data(trust_data)
        
        if success:
            self._log_audit_event('host_authorized', {
                'event': 'host_added_to_trust_list',
                'host_id': host_id,
                'description': host_description,
                'authorized_by': self.host_id
            })
        else:
            self._log_audit_event('authorization_failed', {
                'event': 'trust_file_save_failed',
                'host_id': host_id
            })
        
        return success

    def revoke_host(self, host_id: str) -> bool:
        """Remove host from trust list"""
        if not host_id:
            return False
        
        trust_data = self._load_trust_data()
        trusted_hosts = trust_data.get('trusted_hosts', {})
        
        # Check if host is trusted
        if host_id not in trusted_hosts:
            self._log_audit_event('revocation_failed', {
                'event': 'host_not_in_trust_list',
                'host_id': host_id
            })
            return False
        
        # Store revoked host info for audit
        revoked_host_info = trusted_hosts[host_id].copy()
        
        # Remove from trust list
        del trusted_hosts[host_id]
        trust_data['trusted_hosts'] = trusted_hosts
        
        # Save changes
        success = self._save_trust_data(trust_data)
        
        if success:
            self._log_audit_event('host_revoked', {
                'event': 'host_removed_from_trust_list',
                'host_id': host_id,
                'revoked_by': self.host_id,
                'revoked_host_info': revoked_host_info
            })
        else:
            self._log_audit_event('revocation_failed', {
                'event': 'trust_file_save_failed',
                'host_id': host_id
            })
        
        return success

    def list_trusted_hosts(self) -> List[Dict[str, str]]:
        """Get list of trusted hosts with metadata"""
        trust_data = self._load_trust_data()
        trusted_hosts = trust_data.get('trusted_hosts', {})
        
        # Convert to list format expected by interface
        host_list = []
        for host_id, host_info in trusted_hosts.items():
            entry = {
                'host_id': host_id,
                'description': host_info.get('description', ''),
                'authorized_at': datetime.fromtimestamp(
                    host_info.get('authorized_at', 0)
                ).isoformat() if host_info.get('authorized_at') else '',
                'authorized_by_host': host_info.get('authorized_by_host', ''),
                'last_seen': datetime.fromtimestamp(
                    host_info.get('last_seen', 0)
                ).isoformat() if host_info.get('last_seen') else 'Never'
            }
            host_list.append(entry)
        
        # Sort by authorization date (newest first)
        host_list.sort(key=lambda x: x.get('authorized_at', ''), reverse=True)
        
        return host_list

    def update_host_activity(self, peer_host_id: str) -> None:
        """Update last seen time for peer host (for mesh sync)"""
        if not self.is_trusted_host(peer_host_id):
            return
        
        trust_data = self._load_trust_data()
        trusted_hosts = trust_data.get('trusted_hosts', {})
        
        if peer_host_id in trusted_hosts:
            trusted_hosts[peer_host_id]['last_seen'] = time.time()
            trust_data['trusted_hosts'] = trusted_hosts
            self._save_trust_data(trust_data)

    def get_trust_statistics(self) -> Dict[str, Any]:
        """Get trust system statistics"""
        trust_data = self._load_trust_data()
        trusted_hosts = trust_data.get('trusted_hosts', {})
        
        stats = {
            'total_trusted_hosts': len(trusted_hosts),
            'trust_file_created': datetime.fromtimestamp(
                trust_data.get('created_at', 0)
            ).isoformat() if trust_data.get('created_at') else 'Unknown',
            'last_modified': datetime.fromtimestamp(
                trust_data.get('last_modified', 0)
            ).isoformat() if trust_data.get('last_modified') else 'Unknown',
            'local_host_id': self.host_id,
            'trust_file_exists': self.trust_file.exists(),
            'trust_file_size_bytes': self.trust_file.stat().st_size if self.trust_file.exists() else 0
        }
        
        # Recent activity statistics
        now = time.time()
        recently_seen = 0
        for host_info in trusted_hosts.values():
            last_seen = host_info.get('last_seen')
            if last_seen and (now - last_seen) < 86400:  # 24 hours
                recently_seen += 1
        
        stats['recently_active_hosts'] = recently_seen
        
        return stats

    def get_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit events"""
        if not self.audit_file.exists():
            return []
        
        events = []
        try:
            with open(self.audit_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                lines = f.readlines()
            
            # Get last N lines
            recent_lines = lines[-limit:] if len(lines) > limit else lines
            
            for line in recent_lines:
                line = line.strip()
                if line:
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
                        
        except Exception:
            pass
        
        return events

def main():
    """Test host trust functionality"""
    trust = SimpleHostTrust()
    
    print("Host Trust System Test")
    print("=" * 50)
    
    # Show current host identity
    print(f"Local Host ID: {trust.get_host_identity()}")
    print()
    
    # Show trust statistics
    stats = trust.get_trust_statistics()
    print("Trust Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # List currently trusted hosts
    trusted = trust.list_trusted_hosts()
    print(f"Currently Trusted Hosts ({len(trusted)}):")
    if trusted:
        for host in trusted:
            print(f"  {host['host_id'][:8]}... - {host.get('description', 'No description')}")
            print(f"    Authorized: {host.get('authorized_at', 'Unknown')}")
            print(f"    Last seen: {host.get('last_seen', 'Never')}")
    else:
        print("  No trusted hosts")
    print()
    
    # Show recent audit events
    events = trust.get_audit_events(5)
    print(f"Recent Audit Events ({len(events)}):")
    for event in events[-3:]:  # Show last 3
        print(f"  {event.get('timestamp', 'Unknown')}: {event.get('event_type')}")
        if event.get('details', {}).get('event'):
            print(f"    {event['details']['event']}")

if __name__ == '__main__':
    main()