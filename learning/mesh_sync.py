#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# ///
"""
Mesh Learning Sync - Secure P2P Learning Across Hosts
Handles mixed environments: Tailscale where available, SSH fallback everywhere
"""

import json
import sys
import time
import socket
import subprocess
import threading
from pathlib import Path
from collections import defaultdict
import hashlib
import base64

# Import our secure infrastructure
try:
    from encryption import get_secure_storage
    from abstraction import get_abstractor
    from mesh_discovery import get_mesh_discovery
except ImportError:
    print("Warning: Learning infrastructure not available", file=sys.stderr)
    sys.exit(0)

class MeshLearningSync:
    """Secure cross-host learning synchronization"""
    
    def __init__(self):
        self.storage = get_secure_storage()
        self.abstractor = get_abstractor()
        self.mesh = get_mesh_discovery()
        self.sync_enabled = True
        self.last_sync = 0
        self.sync_interval = 3600  # 1 hour
        self.max_sync_attempts = 3
        
    def should_sync(self):
        """Check if we should attempt sync now"""
        current_time = time.time()
        return (current_time - self.last_sync) > self.sync_interval
    
    def sync_learning_data(self, force=False):
        """Main sync orchestration"""
        if not force and not self.should_sync():
            return
        
        try:
            # Discover available peers
            peers = self.mesh.discover_peers()
            
            if not peers:
                # No peers available - this is fine for standalone hosts
                return
            
            # Get local learning data to share
            shareable_data = self._prepare_shareable_data()
            
            if not shareable_data:
                # Nothing to share yet
                return
            
            # Attempt sync with reachable peers
            successful_syncs = 0
            for peer_key, peer_info in peers.items():
                if self._sync_with_peer(peer_key, peer_info, shareable_data):
                    successful_syncs += 1
            
            self.last_sync = time.time()
            
            # Store sync statistics
            self._update_sync_stats(successful_syncs, len(peers))
            
        except Exception:
            # Sync failures should never break normal operation
            pass
    
    def _prepare_shareable_data(self):
        """Prepare anonymized learning data for sharing"""
        # Get local learning data
        bash_learning = self.storage.load_learning_data('bash_learning', {})
        ssh_topology = self.storage.load_learning_data('ssh_topology', {})
        performance_patterns = self.storage.load_learning_data('performance_patterns', {})
        
        if not any([bash_learning, ssh_topology, performance_patterns]):
            return None
        
        # Create shareable patterns (fully anonymized)
        shareable = {
            'version': 1,
            'timestamp': time.time(),
            'source_node': self._get_anonymous_node_id(),
            'patterns': {}
        }
        
        # Add bash optimization patterns
        if bash_learning and 'success_patterns' in bash_learning:
            shareable['patterns']['bash_optimizations'] = \
                self._anonymize_bash_patterns(bash_learning['success_patterns'])
        
        # Add performance patterns  
        if performance_patterns and 'command_performance' in performance_patterns:
            shareable['patterns']['performance_profiles'] = \
                self._anonymize_performance_patterns(performance_patterns['command_performance'])
        
        # Add network capability patterns (heavily abstracted)
        if ssh_topology and 'host_capabilities' in ssh_topology:
            shareable['patterns']['capability_profiles'] = \
                self._anonymize_capability_patterns(ssh_topology['host_capabilities'])
        
        return shareable
    
    def _anonymize_bash_patterns(self, success_patterns):
        """Anonymize bash patterns for sharing"""
        anonymized = {}
        
        for command, pattern_data in success_patterns.items():
            if pattern_data.get('success_rate', 0) > 0.8:  # Only share successful patterns
                # Create command category instead of actual command
                cmd_category = self._categorize_command(command)
                
                if cmd_category not in anonymized:
                    anonymized[cmd_category] = {
                        'successful_flags': defaultdict(int),
                        'file_type_patterns': defaultdict(int),
                        'success_rate': 0,
                        'sample_count': 0
                    }
                
                # Aggregate flags
                for flag, count in pattern_data.get('common_flags', {}).items():
                    anonymized[cmd_category]['successful_flags'][flag] += count
                
                # Aggregate file types
                for file_type, count in pattern_data.get('file_types', {}).items():
                    anonymized[cmd_category]['file_type_patterns'][file_type] += count
                
                # Update success rate
                anonymized[cmd_category]['sample_count'] += pattern_data.get('total_count', 0)
                anonymized[cmd_category]['success_rate'] = max(
                    anonymized[cmd_category]['success_rate'],
                    pattern_data.get('success_rate', 0)
                )
        
        return anonymized
    
    def _anonymize_performance_patterns(self, performance_data):
        """Anonymize performance patterns for sharing"""
        anonymized = {}
        
        for pattern_key, performances in performance_data.items():
            if len(performances) < 3:  # Need sufficient data
                continue
            
            # Extract command category
            cmd_category = pattern_key.split('_')[0] if '_' in pattern_key else pattern_key
            
            # Calculate aggregated metrics
            durations = [p.get('duration', 0) for p in performances[-10:]]  # Last 10 runs
            memory_deltas = [p.get('memory_delta', 0) for p in performances[-10:]]
            
            if durations:
                anonymized[cmd_category] = {
                    'avg_duration': sum(durations) / len(durations),
                    'max_duration': max(durations),
                    'avg_memory': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
                    'sample_count': len(durations),
                    'performance_tier': self._classify_performance(durations)
                }
        
        return anonymized
    
    def _anonymize_capability_patterns(self, capability_data):
        """Anonymize host capability patterns"""
        capabilities = defaultdict(int)
        
        # Aggregate capability types without exposing specific hosts
        for host_abstract, caps in capability_data.items():
            for capability, has_capability in caps.items():
                if has_capability:
                    capabilities[capability] += 1
        
        return dict(capabilities)
    
    def _categorize_command(self, command):
        """Categorize command for anonymous sharing"""
        # Map specific commands to categories
        categories = {
            'grep': 'text_search',
            'rg': 'text_search', 
            'find': 'file_search',
            'fd': 'file_search',
            'ls': 'file_list',
            'cat': 'file_view',
            'bat': 'file_view',
            'git': 'version_control',
            'R': 'r_computing',
            'Rscript': 'r_computing',
            'python': 'python_computing',
            'python3': 'python_computing',
            'sbatch': 'hpc_scheduling',
            'squeue': 'hpc_monitoring',
            'singularity': 'container_exec',
            'docker': 'container_exec'
        }
        
        return categories.get(command, 'general_command')
    
    def _classify_performance(self, durations):
        """Classify performance tier"""
        avg_duration = sum(durations) / len(durations)
        
        if avg_duration < 1:
            return 'fast'
        elif avg_duration < 60:
            return 'medium'
        elif avg_duration < 300:
            return 'slow'
        else:
            return 'very_slow'
    
    def _sync_with_peer(self, peer_key, peer_info, shareable_data):
        """Attempt to sync with a specific peer"""
        connection_method = peer_info.get('preferred_method')
        
        if connection_method == 'tailscale':
            return self._sync_via_tailscale(peer_info, shareable_data)
        elif connection_method == 'ssh':
            return self._sync_via_ssh(peer_info, shareable_data)
        
        return False
    
    def _sync_via_tailscale(self, peer_info, shareable_data):
        """Sync learning data via Tailscale"""
        try:
            # Create temporary sync file
            sync_data = self._encrypt_shareable_data(shareable_data)
            sync_file = f"/tmp/claude_sync_{int(time.time())}.enc"
            
            with open(sync_file, 'w') as f:
                json.dump(sync_data, f)
            
            # Send via Tailscale SSH
            tailscale_ip = peer_info.get('tailscale_ip')
            result = subprocess.run([
                'scp', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                sync_file, f'{tailscale_ip}:/tmp/claude_sync_incoming.enc'
            ], capture_output=True, timeout=30)
            
            # Cleanup
            Path(sync_file).unlink(missing_ok=True)
            
            if result.returncode == 0:
                # Signal peer to process sync data
                subprocess.run([
                    'ssh', '-o', 'ConnectTimeout=5', '-o', 'BatchMode=yes',
                    tailscale_ip, 'python3', '~/.claude/hooks/process_sync.py'
                ], capture_output=True, timeout=10)
                
                return True
                
        except Exception:
            pass
        
        return False
    
    def _sync_via_ssh(self, peer_info, shareable_data):
        """Sync learning data via SSH"""
        try:
            # Create temporary sync file
            sync_data = self._encrypt_shareable_data(shareable_data)
            sync_file = f"/tmp/claude_sync_{int(time.time())}.enc"
            
            with open(sync_file, 'w') as f:
                json.dump(sync_data, f)
            
            # Send via SSH
            hostname = peer_info.get('hostname')
            result = subprocess.run([
                'scp', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                sync_file, f'{hostname}:/tmp/claude_sync_incoming.enc'
            ], capture_output=True, timeout=30)
            
            # Cleanup
            Path(sync_file).unlink(missing_ok=True)
            
            if result.returncode == 0:
                # Signal peer to process sync data
                subprocess.run([
                    'ssh', '-o', 'ConnectTimeout=5', '-o', 'BatchMode=yes',
                    hostname, 'python3', '~/.claude/hooks/process_sync.py'
                ], capture_output=True, timeout=10)
                
                return True
                
        except Exception:
            pass
        
        return False
    
    def _encrypt_shareable_data(self, shareable_data):
        """Encrypt data for transmission"""
        # Use simplified encryption for P2P sharing
        data_json = json.dumps(shareable_data)
        
        # Simple XOR encryption with derived key
        key = self._derive_mesh_key()
        encrypted_bytes = []
        
        for i, byte in enumerate(data_json.encode()):
            encrypted_bytes.append(byte ^ key[i % len(key)])
        
        return {
            'encrypted_data': base64.b64encode(bytes(encrypted_bytes)).decode(),
            'timestamp': time.time(),
            'format_version': 1
        }
    
    def _derive_mesh_key(self):
        """Derive encryption key for mesh communication"""
        # Create a simple shared key based on known information
        # In production, this would use proper key exchange
        key_material = "claude-mesh-learning-2025"
        return hashlib.sha256(key_material.encode()).digest()[:32]
    
    def _get_anonymous_node_id(self):
        """Get anonymous identifier for this node"""
        hostname = socket.gethostname()
        # Create consistent but anonymous node ID
        node_data = f"{hostname}_{self.abstractor.session_id}"
        return hashlib.md5(node_data.encode()).hexdigest()[:8]
    
    def _update_sync_stats(self, successful_syncs, total_peers):
        """Update sync statistics"""
        stats = self.storage.load_learning_data('sync_stats', {
            'total_syncs': 0,
            'successful_syncs': 0,
            'last_sync_time': 0,
            'peer_count': 0
        })
        
        stats['total_syncs'] += 1
        stats['successful_syncs'] += successful_syncs
        stats['last_sync_time'] = time.time()
        stats['peer_count'] = total_peers
        
        self.storage.store_learning_data('sync_stats', stats)
    
    def process_incoming_sync(self, sync_file_path):
        """Process incoming sync data from peer"""
        try:
            if not Path(sync_file_path).exists():
                return
            
            with open(sync_file_path, 'r') as f:
                encrypted_data = json.load(f)
            
            # Decrypt data
            shareable_data = self._decrypt_shareable_data(encrypted_data)
            
            if shareable_data:
                # Merge learning patterns
                self._merge_peer_learning(shareable_data)
            
            # Cleanup
            Path(sync_file_path).unlink(missing_ok=True)
            
        except Exception:
            # Clean up on any error
            Path(sync_file_path).unlink(missing_ok=True)
    
    def _decrypt_shareable_data(self, encrypted_data):
        """Decrypt incoming data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data['encrypted_data'])
            key = self._derive_mesh_key()
            
            decrypted_bytes = []
            for i, byte in enumerate(encrypted_bytes):
                decrypted_bytes.append(byte ^ key[i % len(key)])
            
            decrypted_json = bytes(decrypted_bytes).decode()
            return json.loads(decrypted_json)
            
        except Exception:
            return None
    
    def _merge_peer_learning(self, peer_data):
        """Merge peer learning data with local data"""
        patterns = peer_data.get('patterns', {})
        
        # Merge bash optimizations
        if 'bash_optimizations' in patterns:
            self._merge_bash_patterns(patterns['bash_optimizations'])
        
        # Merge performance profiles
        if 'performance_profiles' in patterns:
            self._merge_performance_patterns(patterns['performance_profiles'])
        
        # Merge capability profiles
        if 'capability_profiles' in patterns:
            self._merge_capability_patterns(patterns['capability_profiles'])
    
    def _merge_bash_patterns(self, peer_patterns):
        """Merge peer bash patterns with local data"""
        local_data = self.storage.load_learning_data('mesh_bash_patterns', {})
        
        for cmd_category, pattern_data in peer_patterns.items():
            if cmd_category not in local_data:
                local_data[cmd_category] = {
                    'aggregated_flags': defaultdict(int),
                    'file_patterns': defaultdict(int),
                    'peer_count': 0,
                    'avg_success_rate': 0
                }
            
            # Merge flag patterns
            for flag, count in pattern_data.get('successful_flags', {}).items():
                local_data[cmd_category]['aggregated_flags'][flag] += count
            
            # Merge file patterns
            for file_type, count in pattern_data.get('file_type_patterns', {}).items():
                local_data[cmd_category]['file_patterns'][file_type] += count
            
            # Update peer statistics
            local_data[cmd_category]['peer_count'] += 1
            current_avg = local_data[cmd_category]['avg_success_rate']
            new_rate = pattern_data.get('success_rate', 0)
            peer_count = local_data[cmd_category]['peer_count']
            
            # Calculate running average
            local_data[cmd_category]['avg_success_rate'] = \
                ((current_avg * (peer_count - 1)) + new_rate) / peer_count
        
        self.storage.store_learning_data('mesh_bash_patterns', dict(local_data))
    
    def _merge_performance_patterns(self, peer_patterns):
        """Merge peer performance patterns"""
        local_data = self.storage.load_learning_data('mesh_performance_patterns', {})
        
        for cmd_category, perf_data in peer_patterns.items():
            if cmd_category not in local_data:
                local_data[cmd_category] = {
                    'duration_samples': [],
                    'memory_samples': [],
                    'performance_tiers': defaultdict(int)
                }
            
            # Add performance samples
            local_data[cmd_category]['duration_samples'].append(perf_data.get('avg_duration', 0))
            local_data[cmd_category]['memory_samples'].append(perf_data.get('avg_memory', 0))
            
            # Track performance tiers
            tier = perf_data.get('performance_tier', 'unknown')
            local_data[cmd_category]['performance_tiers'][tier] += 1
            
            # Keep only recent samples
            local_data[cmd_category]['duration_samples'] = \
                local_data[cmd_category]['duration_samples'][-50:]
            local_data[cmd_category]['memory_samples'] = \
                local_data[cmd_category]['memory_samples'][-50:]
        
        self.storage.store_learning_data('mesh_performance_patterns', dict(local_data))
    
    def _merge_capability_patterns(self, peer_patterns):
        """Merge peer capability patterns"""
        local_data = self.storage.load_learning_data('mesh_capability_patterns', {})
        
        for capability, count in peer_patterns.items():
            local_data[capability] = local_data.get(capability, 0) + count
        
        self.storage.store_learning_data('mesh_capability_patterns', local_data)

def get_mesh_sync():
    """Get a configured mesh sync instance"""
    return MeshLearningSync()

if __name__ == '__main__':
    # Test mesh sync
    mesh_sync = get_mesh_sync()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--process-incoming':
        # Process incoming sync data
        sync_file = '/tmp/claude_sync_incoming.enc'
        mesh_sync.process_incoming_sync(sync_file)
    else:
        # Perform sync
        mesh_sync.sync_learning_data(force=True)
        print("Mesh sync completed")