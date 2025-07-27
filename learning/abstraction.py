#!/usr/bin/env python3
"""
Secure Data Abstraction - Convert sensitive data to safe learning patterns
Zero-knowledge design: useful to AI, useless to attackers
"""

import hashlib
import re
import socket
from pathlib import Path
from collections import defaultdict
import json

class SecureAbstractor:
    """Convert sensitive system data to safe semantic abstractions"""
    
    def __init__(self):
        self.session_id = self._generate_session_id()
        self.host_abstractions = {}
        self.path_abstractions = {}
        
    def _generate_session_id(self):
        """Generate unique session ID for consistent abstractions"""
        import time
        import os
        entropy = f"{time.time()}{os.getpid()}{socket.gethostname()}"
        return hashlib.md5(entropy.encode()).hexdigest()[:8]
    
    def abstract_hostname(self, hostname):
        """Convert hostname to semantic abstraction"""
        if not hostname or hostname in self.host_abstractions:
            return self.host_abstractions.get(hostname, 'unknown-host')
        
        # Detect host types from patterns (generic, not specific to your setup)
        host_type = 'compute-host'
        
        # Common patterns for different host types
        if any(pattern in hostname.lower() for pattern in ['login', 'head', 'gateway']):
            host_type = 'gateway-host'
        elif any(pattern in hostname.lower() for pattern in ['data', 'storage', 'nas']):
            host_type = 'storage-host'
        elif any(pattern in hostname.lower() for pattern in ['gpu', 'cuda', 'tesla']):
            host_type = 'gpu-host'
        elif any(pattern in hostname.lower() for pattern in ['compute', 'node', 'worker']):
            host_type = 'compute-host'
        elif any(pattern in hostname.lower() for pattern in ['web', 'www', 'http']):
            host_type = 'web-host'
        
        # Create semantic ID: type + session-specific hash
        host_hash = hashlib.md5(f"{hostname}{self.session_id}".encode()).hexdigest()[:4]
        abstraction = f"{host_type}-{host_hash}"
        
        self.host_abstractions[hostname] = abstraction
        return abstraction
    
    def abstract_path(self, path):
        """Convert filesystem path to semantic abstraction"""
        if not path:
            return 'unknown-path'
            
        path_str = str(path)
        
        # Already abstracted
        if path_str in self.path_abstractions:
            return self.path_abstractions[path_str]
        
        # Detect path types
        path_type = 'general-path'
        
        # Common path patterns (generic)
        if '/data/' in path_str or '/scratch/' in path_str:
            path_type = 'data-path'
        elif '/home/' in path_str or '/users/' in path_str:
            path_type = 'user-path'
        elif '/opt/' in path_str or '/usr/local/' in path_str:
            path_type = 'system-path'
        elif '/tmp/' in path_str or '/var/tmp/' in path_str:
            path_type = 'temp-path'
        elif any(pattern in path_str.lower() for pattern in ['/project/', '/work/', '/analysis/']):
            path_type = 'project-path'
        elif any(ext in path_str.lower() for ext in ['.fastq', '.bam', '.vcf', '.fasta']):
            path_type = 'genomics-data'
        elif path_str.endswith(('.R', '.py', '.sh', '.pl')):
            path_type = 'script-path'
        
        # Create semantic abstraction
        path_hash = hashlib.md5(f"{path_str}{self.session_id}".encode()).hexdigest()[:4]
        abstraction = f"{path_type}-{path_hash}"
        
        self.path_abstractions[path_str] = abstraction
        return abstraction
    
    def abstract_command(self, command):
        """Abstract command while preserving learning value"""
        if not command:
            return {}
        
        # Extract learning-valuable patterns without exposing sensitive data
        abstract_cmd = {
            'base_command': '',
            'flags': [],
            'file_types': [],
            'path_types': [],
            'complexity': 'simple'
        }
        
        # Get base command
        parts = command.split()
        if parts:
            abstract_cmd['base_command'] = parts[0]
        
        # Extract flags (safe to learn from)
        flags = re.findall(r'-[a-zA-Z]+', command)
        abstract_cmd['flags'] = list(set(flags))
        
        # Detect file types (learning valuable)
        file_extensions = re.findall(r'\.\w+', command)
        abstract_cmd['file_types'] = list(set(file_extensions))
        
        # Detect paths and abstract them
        paths = re.findall(r'/[^\s]+', command)
        abstract_cmd['path_types'] = [self.abstract_path(path) for path in paths]
        
        # Assess complexity
        if len(parts) > 10 or '|' in command or '&&' in command:
            abstract_cmd['complexity'] = 'complex'
        elif len(parts) > 5:
            abstract_cmd['complexity'] = 'medium'
        
        return abstract_cmd
    
    def abstract_environment(self):
        """Create abstract environment fingerprint"""
        try:
            import shutil
            import os
            
            env_profile = {
                'tools': {},
                'capabilities': [],
                'host_type': self.abstract_hostname(socket.gethostname()),
                'session_id': self.session_id
            }
            
            # Check for common tools (safe to expose)
            common_tools = [
                'bash', 'python3', 'R', 'git', 'docker', 'singularity',
                'sbatch', 'squeue', 'module', 'conda', 'pip',
                'wget', 'curl', 'rsync', 'ssh'
            ]
            
            for tool in common_tools:
                if shutil.which(tool):
                    env_profile['tools'][tool] = 'available'
            
            # Detect capabilities
            if 'sbatch' in env_profile['tools']:
                env_profile['capabilities'].append('hpc-scheduler')
            if 'singularity' in env_profile['tools'] or 'docker' in env_profile['tools']:
                env_profile['capabilities'].append('containers')
            if 'R' in env_profile['tools']:
                env_profile['capabilities'].append('r-computing')
            if 'module' in env_profile['tools']:
                env_profile['capabilities'].append('module-system')
            
            return env_profile
            
        except Exception:
            return {'host_type': 'unknown', 'tools': {}, 'capabilities': []}
    
    def abstract_ssh_connection(self, ssh_command):
        """Abstract SSH connection patterns"""
        # Extract learning patterns from SSH usage without exposing topology
        if not ssh_command.startswith('ssh '):
            return None
        
        parts = ssh_command.split()
        if len(parts) < 2:
            return None
        
        # Abstract the connection pattern
        connection = {
            'target_host_type': self.abstract_hostname(parts[1].split('@')[-1]),
            'has_user': '@' in parts[1],
            'has_command': len(parts) > 2,
            'connection_pattern': 'direct'
        }
        
        # Detect proxy/jump patterns
        if '-J' in ssh_command or 'ProxyJump' in ssh_command:
            connection['connection_pattern'] = 'proxied'
        
        # Abstract the remote command if present
        if len(parts) > 2:
            remote_cmd = ' '.join(parts[2:]).strip('\'"')
            connection['remote_command'] = self.abstract_command(remote_cmd)
        
        return connection

def get_abstractor():
    """Get a configured abstractor instance"""
    return SecureAbstractor()

if __name__ == '__main__':
    # Test abstraction
    abstractor = get_abstractor()
    
    # Test command abstraction
    test_cmd = "ssh user@host 'ls /data/project/*.fastq'"
    abstracted = abstractor.abstract_ssh_connection(test_cmd)
    print(f"SSH abstraction test: {json.dumps(abstracted, indent=2)}")
    
    # Test environment
    env = abstractor.abstract_environment()
    print(f"Environment profile: {json.dumps(env, indent=2)}")