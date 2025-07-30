#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0",
#   "psutil>=5.9.0",
# ]
# ///
"""
Security Command Line Interface

Provides command-line interface for host authorization workflow and security
management operations. Implements the user-facing commands for trust management.

Commands:
- request-access: Show host ID for authorization by another host
- approve-host: Add host to trust list
- list-hosts: Show trusted hosts
- revoke-host: Remove host access
- security-status: Show security system status
- rotate-keys: Manually trigger key rotation
- audit-log: Show security audit events

Security Features:
- Safe host ID display with QR code option
- Audit trail for all authorization operations
- Performance metrics and system health checks
- Secure operations with proper error handling
"""

import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import argparse

from host_trust import SimpleHostTrust
from key_manager import SimpleKeyManager  
from secure_storage import SecureLearningStorage
from hardware_identity import HardwareIdentity

class SecurityCLI:
    """Command line interface for security operations"""
    
    def __init__(self):
        self.host_trust = SimpleHostTrust()
        self.key_manager = SimpleKeyManager()
        self.secure_storage = SecureLearningStorage()
        self.hardware_identity = HardwareIdentity()

    def request_access(self) -> int:
        """Request access from another host"""
        try:
            host_id = self.host_trust.get_host_identity()
            
            print("🔐 Claude-Sync Host Authorization Request")
            print("=" * 50)
            print(f"Host ID: {host_id}")
            print()
            print("To authorize this host, run this command on an authorized host:")
            print(f"    claude-sync approve-host {host_id}")
            print()
            print("Or using the bootstrap script:")
            print(f"    ~/.claude/claude-sync/bootstrap.sh approve-host {host_id}")
            print()
            
            # Show additional host info for verification
            identity_info = self.hardware_identity.get_identity_info()
            print("Host Verification Info:")
            print(f"  Platform: {identity_info['platform_info']['platform']}")
            print(f"  Machine: {identity_info['platform_info']['machine']}")
            
            # Show hardware sources for debugging
            hardware_sources = identity_info['hardware_sources']
            available_sources = []
            for source, value in hardware_sources.items():
                if value:
                    available_sources.append(source)
            
            print(f"  Hardware Sources: {', '.join(available_sources)}")
            print()
            print("Note: This host ID is derived from stable hardware characteristics")
            print("and will remain consistent across OS reinstalls.")
            
            return 0
            
        except Exception as e:
            print(f"Error: Failed to generate host identity: {e}")
            return 1

    def approve_host(self, host_id: str, description: str = "") -> int:
        """Approve a host for access"""
        try:
            if not host_id:
                print("Error: Host ID is required")
                return 1
            
            if len(host_id) != 16:
                print("Error: Invalid host ID format (must be 16 characters)")
                return 1
            
            # Get description if not provided
            if not description:
                try:
                    description = input(f"Enter description for host {host_id[:8]}... (optional): ").strip()
                except KeyboardInterrupt:
                    print("\nOperation cancelled")
                    return 1
            
            # Check if already trusted
            if self.host_trust.is_trusted_host(host_id):
                print(f"Host {host_id[:8]}... is already trusted")
                return 0
            
            # Add to trust list
            success = self.host_trust.authorize_host(host_id, description)
            
            if success:
                print(f"✅ Host {host_id[:8]}... has been authorized")
                if description:
                    print(f"   Description: {description}")
                
                # Trigger key rotation to ensure fresh keys
                print("🔄 Triggering key rotation...")
                key_rotation_success = self.key_manager.rotate_keys()
                if key_rotation_success:
                    print("✅ Key rotation completed")
                else:
                    print("⚠️  Key rotation failed - manual rotation may be needed")
                
                return 0
            else:
                print(f"❌ Failed to authorize host {host_id[:8]}...")
                return 1
                
        except Exception as e:
            print(f"Error: Authorization failed: {e}")
            return 1

    def list_hosts(self, verbose: bool = False) -> int:
        """List trusted hosts"""
        try:
            trusted_hosts = self.host_trust.list_trusted_hosts()
            
            print("🏠 Trusted Hosts")
            print("=" * 50)
            
            if not trusted_hosts:
                print("No trusted hosts configured")
                print()
                print("To authorize a host:")
                print("1. On the host requesting access: claude-sync request-access")
                print("2. On this host: claude-sync approve-host <host-id>")
                return 0
            
            print(f"Total: {len(trusted_hosts)} trusted hosts")
            print()
            
            for i, host in enumerate(trusted_hosts, 1):
                host_id = host['host_id']
                description = host.get('description', 'No description')
                authorized_at = host.get('authorized_at', 'Unknown')
                last_seen = host.get('last_seen', 'Never')
                
                print(f"{i}. {host_id[:8]}...{host_id[-4:]}")
                print(f"   Description: {description}")
                
                if verbose:
                    print(f"   Full Host ID: {host_id}")
                    print(f"   Authorized: {authorized_at}")
                    print(f"   Last Seen: {last_seen}")
                    print(f"   Authorized By: {host.get('authorized_by_host', 'Unknown')[:8]}...")
                else:
                    # Parse date for friendly display
                    try:
                        if authorized_at and authorized_at != 'Unknown':
                            auth_date = datetime.fromisoformat(authorized_at.replace('Z', '+00:00'))
                            days_ago = (datetime.now() - auth_date.replace(tzinfo=None)).days
                            if days_ago == 0:
                                friendly_date = "today"
                            elif days_ago == 1:
                                friendly_date = "yesterday"
                            else:
                                friendly_date = f"{days_ago} days ago"
                            print(f"   Authorized: {friendly_date}")
                    except:
                        print(f"   Authorized: {authorized_at}")
                
                print()
            
            return 0
            
        except Exception as e:
            print(f"Error: Failed to list hosts: {e}")
            return 1

    def revoke_host(self, host_id: str) -> int:
        """Revoke host access"""
        try:
            if not host_id:
                print("Error: Host ID is required")
                return 1
            
            # Allow partial host ID matching
            if len(host_id) < 8:
                print("Error: Host ID must be at least 8 characters")
                return 1
            
            # Find matching hosts
            trusted_hosts = self.host_trust.list_trusted_hosts()
            matches = []
            
            for host in trusted_hosts:
                full_id = host['host_id']
                if full_id.startswith(host_id) or full_id == host_id:
                    matches.append(host)
            
            if not matches:
                print(f"❌ No trusted host found matching '{host_id}'")
                return 1
            
            if len(matches) > 1:
                print(f"❌ Multiple hosts match '{host_id}':")
                for host in matches:
                    print(f"  {host['host_id'][:8]}...{host['host_id'][-4:]} - {host.get('description', 'No description')}")
                print("Please provide a more specific host ID")
                return 1
            
            # Single match found
            target_host = matches[0]
            full_host_id = target_host['host_id']
            description = target_host.get('description', 'No description')
            
            print(f"🚨 Revoking access for host: {full_host_id[:8]}...{full_host_id[-4:]}")
            print(f"   Description: {description}")
            print()
            
            # Confirm revocation
            try:
                confirm = input("Type 'yes' to confirm revocation: ").strip().lower()
                if confirm != 'yes':
                    print("Revocation cancelled")
                    return 0
            except KeyboardInterrupt:
                print("\nRevocation cancelled")
                return 0
            
            # Revoke access
            success = self.host_trust.revoke_host(full_host_id)
            
            if success:
                print(f"✅ Host {full_host_id[:8]}... access has been revoked")
                
                # Trigger key rotation to invalidate old keys
                print("🔄 Triggering key rotation to invalidate old keys...")
                key_rotation_success = self.key_manager.rotate_keys()
                if key_rotation_success:
                    print("✅ Key rotation completed")
                else:
                    print("⚠️  Key rotation failed - manual rotation may be needed")
                
                return 0
            else:
                print(f"❌ Failed to revoke host {full_host_id[:8]}...")
                return 1
                
        except Exception as e:
            print(f"Error: Revocation failed: {e}")
            return 1

    def security_status(self, verbose: bool = False) -> int:
        """Show security system status"""
        try:
            print("🔒 Claude-Sync Security Status")
            print("=" * 50)
            
            # Host identity info
            local_host_id = self.host_trust.get_host_identity()
            print(f"Local Host ID: {local_host_id}")
            
            # Trust statistics
            trust_stats = self.host_trust.get_trust_statistics()
            print(f"Trusted Hosts: {trust_stats['total_trusted_hosts']}")
            print(f"Recently Active: {trust_stats['recently_active_hosts']}")
            
            # Key management statistics  
            key_stats = self.key_manager.get_key_statistics()
            print(f"Current Key: {key_stats['current_key_id']}")
            print(f"Active Keys: {key_stats['active_keys_count']}")
            print(f"Key Retention: {key_stats['retention_days']} days")
            
            # Storage statistics
            storage_stats = self.secure_storage.get_storage_statistics()
            print(f"Learning Data: {storage_stats['total_files']} files ({storage_stats['total_size_bytes']} bytes)")
            print(f"Data Retention: {storage_stats['retention_days']} days")
            
            if verbose:
                print()
                print("Detailed Information:")
                print("-" * 30)
                
                # Hardware identity details
                identity_info = self.hardware_identity.get_identity_info()
                print("Hardware Sources:")
                for source, value in identity_info['hardware_sources'].items():
                    status = "✓" if value else "✗"
                    print(f"  {status} {source}: {value or 'Not available'}")
                
                print()
                print("Trust File Info:")
                print(f"  Created: {trust_stats['trust_file_created']}")
                print(f"  Modified: {trust_stats['last_modified']}")
                print(f"  Size: {trust_stats['trust_file_size_bytes']} bytes")
                
                print()
                print("Encryption Performance:")
                enc_stats = storage_stats.get('encryption_stats', {})
                total_enc = enc_stats.get('total_encryptions', 0)
                total_dec = enc_stats.get('total_decryptions', 0)
                avg_enc_time = enc_stats.get('avg_encryption_time_ms', 0)
                avg_dec_time = enc_stats.get('avg_decryption_time_ms', 0)
                
                print(f"  Total Encryptions: {total_enc}")
                print(f"  Total Decryptions: {total_dec}")
                print(f"  Avg Encryption Time: {avg_enc_time:.2f}ms")
                print(f"  Avg Decryption Time: {avg_dec_time:.2f}ms")
                
                # Performance warnings
                if avg_enc_time > 5:
                    print(f"  ⚠️  Encryption time above target (5ms)")
                if avg_dec_time > 5:
                    print(f"  ⚠️  Decryption time above target (5ms)")
            
            return 0
            
        except Exception as e:
            print(f"Error: Failed to get security status: {e}")
            return 1

    def rotate_keys(self) -> int:
        """Manually trigger key rotation"""
        try:
            print("🔄 Triggering manual key rotation...")
            
            success = self.key_manager.rotate_keys()
            
            if success:
                print("✅ Key rotation completed successfully")
                
                # Show new key info
                new_key_id = self.key_manager.get_current_key_id()
                print(f"New Key ID: {new_key_id}")
                
                # Show cleanup results
                removed_keys = self.key_manager.cleanup_old_keys()
                if removed_keys > 0:
                    print(f"🗑️  Removed {removed_keys} old keys")
                
                return 0
            else:
                print("❌ Key rotation failed")
                return 1
                
        except Exception as e:
            print(f"Error: Key rotation failed: {e}")
            return 1

    def audit_log(self, limit: int = 20) -> int:
        """Show security audit events"""
        try:
            events = self.host_trust.get_audit_events(limit)
            
            print("📋 Security Audit Log")
            print("=" * 50)
            
            if not events:
                print("No audit events found")
                return 0
            
            print(f"Showing last {len(events)} events:")
            print()
            
            for event in events:
                timestamp = event.get('timestamp', 'Unknown')
                event_type = event.get('event_type', 'unknown')
                details = event.get('details', {})
                
                # Parse timestamp for friendly display
                try:
                    if timestamp != 'Unknown':
                        event_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        friendly_time = event_time.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        friendly_time = timestamp
                except:
                    friendly_time = timestamp
                
                print(f"[{friendly_time}] {event_type.upper()}")
                
                # Show event details
                if details.get('event'):
                    print(f"  Event: {details['event']}")
                
                if details.get('host_id'):
                    host_id = details['host_id']
                    print(f"  Host: {host_id[:8]}...{host_id[-4:]}")
                
                if details.get('description'):
                    print(f"  Description: {details['description']}")
                
                if details.get('error'):
                    print(f"  Error: {details['error']}")
                
                if details.get('reason'):
                    print(f"  Reason: {details['reason']}")
                
                print()
            
            return 0
            
        except Exception as e:
            print(f"Error: Failed to show audit log: {e}")
            return 1

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Claude-Sync Security Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s request-access                    # Show this host's ID for authorization
  %(prog)s approve-host abc123def456        # Authorize a host
  %(prog)s list-hosts                       # List trusted hosts
  %(prog)s revoke-host abc123              # Revoke host access (partial ID ok)
  %(prog)s security-status                 # Show security system status
  %(prog)s rotate-keys                     # Manually rotate encryption keys
  %(prog)s audit-log                       # Show security audit events
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Security commands')
    
    # request-access command
    subparsers.add_parser('request-access', help='Show host ID for authorization')
    
    # approve-host command
    approve_parser = subparsers.add_parser('approve-host', help='Authorize a host')
    approve_parser.add_argument('host_id', help='Host ID to authorize')
    approve_parser.add_argument('--description', '-d', help='Description of the host')
    
    # list-hosts command
    list_parser = subparsers.add_parser('list-hosts', help='List trusted hosts')
    list_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    
    # revoke-host command
    revoke_parser = subparsers.add_parser('revoke-host', help='Revoke host access')
    revoke_parser.add_argument('host_id', help='Host ID to revoke (partial ID acceptable)')
    
    # security-status command
    status_parser = subparsers.add_parser('security-status', help='Show security system status')
    status_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    
    # rotate-keys command
    subparsers.add_parser('rotate-keys', help='Manually trigger key rotation')
    
    # audit-log command
    audit_parser = subparsers.add_parser('audit-log', help='Show security audit events')
    audit_parser.add_argument('--limit', '-l', type=int, default=20, help='Number of events to show')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = SecurityCLI()
    
    try:
        if args.command == 'request-access':
            return cli.request_access()
        elif args.command == 'approve-host':
            return cli.approve_host(args.host_id, args.description or "")
        elif args.command == 'list-hosts':
            return cli.list_hosts(args.verbose)
        elif args.command == 'revoke-host':
            return cli.revoke_host(args.host_id)
        elif args.command == 'security-status':
            return cli.security_status(args.verbose)
        elif args.command == 'rotate-keys':
            return cli.rotate_keys()
        elif args.command == 'audit-log':
            return cli.audit_log(args.limit)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())