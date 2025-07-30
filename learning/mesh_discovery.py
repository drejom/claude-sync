#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# ///
"""
Mesh Discovery - Tailscale + SSH Topology Auto-Discovery
Self-learning peer discovery for encrypted cross-host intelligence sharing
"""

import json
import subprocess
import socket
import re
import time
from pathlib import Path
from collections import defaultdict
import threading
import hashlib

class MeshDiscovery:
    """Auto-discover and manage peer connections for learning sync"""
    
    def __init__(self):
        self.tailscale_peers = {}
        self.ssh_peers = {}
        self.reachable_peers = {}
        self.connection_cache = {}
        self.last_discovery = 0
        self.discovery_interval = 3600  # 1 hour
        
    def discover_peers(self, force_refresh=False):
        """Main discovery orchestration"""
        current_time = time.time()
        
        if not force_refresh and (current_time - self.last_discovery) < self.discovery_interval:
            return self.reachable_peers
        
        try:
            # Discover Tailscale peers
            self._discover_tailscale_peers()
            
            # Parse SSH configuration
            self._discover_ssh_peers()
            
            # Test connectivity
            self._test_peer_connectivity()
            
            self.last_discovery = current_time
            
        except Exception as e:
            # Fail gracefully - discovery is optional
            pass
        
        return self.reachable_peers
    
    def _discover_tailscale_peers(self):
        """Discover Tailscale peers on the tailnet"""
        try:
            # Run tailscale status to get peer list
            result = subprocess.run(
                ['tailscale', 'status', '--json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                status_data = json.loads(result.stdout)
                
                # Extract peer information
                for peer_id, peer_info in status_data.get('Peer', {}).items():
                    peer_data = {
                        'id': peer_id,
                        'hostname': peer_info.get('HostName', ''),
                        'tailscale_ip': peer_info.get('TailscaleIPs', [None])[0],
                        'dns_name': peer_info.get('DNSName', ''),
                        'online': peer_info.get('Online', False),
                        'connection_type': 'tailscale',
                        'last_seen': peer_info.get('LastSeen', ''),
                        'os': peer_info.get('OS', '')
                    }
                    
                    if peer_data['online'] and peer_data['tailscale_ip']:
                        # Create abstract peer identifier
                        peer_key = self._create_peer_key(peer_data['hostname'], 'tailscale')
                        self.tailscale_peers[peer_key] = peer_data
                        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
            # Tailscale not available or not configured
            pass
    
    def _discover_ssh_peers(self):
        """Parse SSH config for peer discovery"""
        ssh_config_path = Path.home() / '.ssh' / 'config'
        
        if not ssh_config_path.exists():
            return
        
        try:
            with open(ssh_config_path, 'r') as f:
                ssh_config = f.read()
            
            # Parse SSH hosts
            host_blocks = re.findall(r'Host\s+([^\s*?]+).*?(?=Host\s+|\Z)', ssh_config, re.DOTALL | re.IGNORECASE)
            
            for host_block in host_blocks:
                lines = host_block.split('\n')
                if not lines:
                    continue
                
                # Extract host name
                host_match = re.match(r'Host\s+([^\s*?]+)', lines[0])
                if not host_match:
                    continue
                
                hostname = host_match.group(1)
                
                # Parse host configuration
                host_config = {
                    'hostname': hostname,
                    'connection_type': 'ssh',
                    'config': {}
                }
                
                for line in lines[1:]:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if ' ' in line:
                        key, value = line.split(' ', 1)
                        host_config['config'][key.lower()] = value
                
                # Create abstract peer identifier
                peer_key = self._create_peer_key(hostname, 'ssh')
                self.ssh_peers[peer_key] = host_config
                
        except Exception:
            pass
    
    def _test_peer_connectivity(self):
        """Test connectivity to discovered peers"""
        # Test Tailscale peers
        for peer_key, peer_data in self.tailscale_peers.items():
            if self._test_tailscale_connectivity(peer_data):
                self.reachable_peers[peer_key] = {
                    **peer_data,
                    'reachable': True,
                    'preferred_method': 'tailscale',
                    'connection_latency': self._measure_latency(peer_data['tailscale_ip'])
                }
        
        # Test SSH peers (with rate limiting)
        tested_ssh = 0
        for peer_key, peer_data in self.ssh_peers.items():
            if tested_ssh >= 5:  # Limit SSH tests to avoid delays
                break
            
            if self._test_ssh_connectivity(peer_data):
                self.reachable_peers[peer_key] = {
                    **peer_data,
                    'reachable': True,
                    'preferred_method': 'ssh',
                    'connection_latency': self._measure_ssh_latency(peer_data['hostname'])
                }
                tested_ssh += 1
    
    def _test_tailscale_connectivity(self, peer_data):
        """Quick connectivity test for Tailscale peer"""
        try:
            # Simple ping test
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '2', peer_data['tailscale_ip']],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _test_ssh_connectivity(self, peer_data):
        """Quick connectivity test for SSH peer"""
        try:
            # Quick SSH connection test
            result = subprocess.run(
                ['ssh', '-o', 'ConnectTimeout=3', '-o', 'BatchMode=yes', 
                 peer_data['hostname'], 'echo', 'test'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _measure_latency(self, ip_address):
        """Measure network latency to peer"""
        try:
            result = subprocess.run(
                ['ping', '-c', '3', '-W', '2', ip_address],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Extract average latency
                latency_match = re.search(r'= [\d.]+/([\d.]+)/', result.stdout)
                if latency_match:
                    return float(latency_match.group(1))
            
        except:
            pass
        
        return 999.0  # High latency for failed tests
    
    def _measure_ssh_latency(self, hostname):
        """Measure SSH connection latency"""
        try:
            start_time = time.time()
            result = subprocess.run(
                ['ssh', '-o', 'ConnectTimeout=3', '-o', 'BatchMode=yes',
                 hostname, 'echo', 'latency_test'],
                capture_output=True,
                timeout=5
            )
            end_time = time.time()
            
            if result.returncode == 0:
                return (end_time - start_time) * 1000  # Convert to ms
                
        except:
            pass
        
        return 999.0  # High latency for failed tests
    
    def _create_peer_key(self, hostname, connection_type):
        """Create consistent peer identifier"""
        # Create abstract peer key for security
        peer_string = f"{hostname}_{connection_type}_{socket.gethostname()}"
        return hashlib.md5(peer_string.encode()).hexdigest()[:12]
    
    def get_optimal_peer_route(self, peer_key):
        """Get optimal connection route to peer"""
        if peer_key not in self.reachable_peers:
            return None
        
        peer = self.reachable_peers[peer_key]
        
        # Prefer Tailscale for lower latency and better security
        if peer.get('preferred_method') == 'tailscale':
            return {
                'method': 'tailscale',
                'address': peer.get('tailscale_ip'),
                'latency': peer.get('connection_latency', 999.0)
            }
        elif peer.get('preferred_method') == 'ssh':
            return {
                'method': 'ssh',
                'hostname': peer.get('hostname'),
                'latency': peer.get('connection_latency', 999.0)
            }
        
        return None
    
    def get_mesh_topology(self):
        """Get current mesh topology for learning sync"""
        topology = {
            'local_node': {
                'hostname': socket.gethostname(),
                'timestamp': time.time()
            },
            'reachable_peers': {},
            'connection_methods': {
                'tailscale': len([p for p in self.reachable_peers.values() 
                                if p.get('preferred_method') == 'tailscale']),
                'ssh': len([p for p in self.reachable_peers.values() 
                           if p.get('preferred_method') == 'ssh'])
            }
        }
        
        # Add reachable peers with abstract identifiers
        for peer_key, peer_data in self.reachable_peers.items():
            topology['reachable_peers'][peer_key] = {
                'connection_type': peer_data.get('preferred_method'),
                'latency': peer_data.get('connection_latency'),
                'last_seen': time.time(),
                'capabilities': self._detect_peer_capabilities(peer_data)
            }
        
        return topology
    
    def _detect_peer_capabilities(self, peer_data):
        """Detect peer capabilities for intelligent routing"""
        capabilities = []
        
        # Detect based on hostname patterns
        hostname = peer_data.get('hostname', '').lower()
        
        if any(pattern in hostname for pattern in ['gpu', 'cuda', 'tesla']):
            capabilities.append('gpu_computing')
        if any(pattern in hostname for pattern in ['data', 'storage']):
            capabilities.append('data_storage')
        if any(pattern in hostname for pattern in ['compute', 'hpc', 'cluster']):
            capabilities.append('hpc_computing')
        if any(pattern in hostname for pattern in ['login', 'gateway', 'head']):
            capabilities.append('gateway_node')
        
        # Detect based on OS
        os_info = peer_data.get('os', '').lower()
        if 'linux' in os_info:
            capabilities.append('linux_host')
        
        return capabilities

def get_mesh_discovery():
    """Get a configured mesh discovery instance"""
    return MeshDiscovery()

if __name__ == '__main__':
    # Test mesh discovery
    mesh = get_mesh_discovery()
    peers = mesh.discover_peers()
    
    print(f"Discovered {len(peers)} reachable peers:")
    for peer_key, peer_data in peers.items():
        method = peer_data.get('preferred_method')
        latency = peer_data.get('connection_latency', 0)
        print(f"  {peer_key}: {method} ({latency:.1f}ms)")
    
    print(f"\nMesh topology:")
    topology = mesh.get_mesh_topology()
    print(json.dumps(topology, indent=2))