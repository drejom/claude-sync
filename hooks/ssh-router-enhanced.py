#!/usr/bin/env python3
"""
Enhanced Self-Learning SSH Router Hook
Secure filesystem topology learning with host fingerprinting and AI intelligence.
"""

import json
import sys
import re
import os
from pathlib import Path
from collections import defaultdict, Counter
import pickle
import socket
import hashlib
import time

# Try to import secure learning infrastructure
LEARNING_DIR = Path.home() / '.claude' / 'learning'
HOOKS_DIR = Path(__file__).parent.parent / 'learning'

def import_learning_modules():
    """Import learning modules with fallback"""
    try:
        sys.path.insert(0, str(HOOKS_DIR))
        from encryption import get_secure_storage
        from abstraction import get_abstractor
        return get_secure_storage(), get_abstractor(), True
    except ImportError:
        return None, None, False

STORAGE, ABSTRACTOR, SECURE_MODE = import_learning_modules()

# Fallback storage
FALLBACK_DB = Path.home() / '.claude' / 'ssh_topology.pkl'

def main():
    # Handle command line arguments for standalone usage
    if len(sys.argv) > 1:
        if sys.argv[1] == '--stats':
            show_learning_stats()
            return
        elif sys.argv[1] == '--help':
            print("SSH Router Enhanced - Usage:")
            print("  --stats    Show learning statistics")
            print("  --help     Show this help")
            return
    
    # Read hook input (when called as a hook)
    try:
        hook_input = json.loads(sys.stdin.read())
    except:
        print("Error: Expected JSON input from Claude Code hook system")
        print("Use --stats or --help for standalone usage")
        sys.exit(1)
    
    if hook_input.get('tool_name') != 'Bash':
        sys.exit(0)
    
    command = hook_input.get('tool_input', {}).get('command', '')
    if not command:
        sys.exit(0)
    
    # Initialize host topology on first run
    initialize_host_topology()
    
    # Learn from SSH commands, suggest for local commands
    if command.strip().startswith('ssh '):
        learn_from_ssh_command(command)
    else:
        suggestion = suggest_routing(command)
        if suggestion:
            result = {
                'block': False,
                'message': suggestion
            }
            print(json.dumps(result))
    
    sys.exit(0)

def learn_from_ssh_command(command):
    """Learn secure filesystem topology and command patterns"""
    if SECURE_MODE and ABSTRACTOR:
        learn_secure_ssh_patterns(command)
    else:
        learn_basic_ssh_patterns(command)

def learn_secure_ssh_patterns(command):
    """Secure learning with abstraction and encryption"""
    # Abstract the SSH connection
    connection_data = ABSTRACTOR.abstract_ssh_connection(command)
    if not connection_data:
        return
    
    # Get current topology data
    topology = STORAGE.load_learning_data('ssh_topology', {
        'host_capabilities': defaultdict(dict),
        'connection_patterns': defaultdict(list),
        'command_patterns': defaultdict(Counter),
        'path_capabilities': defaultdict(set),
        'environment_profile': {}
    })
    
    target_host = connection_data['target_host_type']
    
    # Learn host capabilities
    if 'remote_command' in connection_data:
        remote_cmd = connection_data['remote_command']
        
        # Learn command capabilities
        if remote_cmd.get('base_command'):
            topology['command_patterns'][target_host][remote_cmd['base_command']] += 1
        
        # Learn path capabilities (abstracted)
        for path_type in remote_cmd.get('path_types', []):
            topology['path_capabilities'][target_host].add(path_type)
        
        # Learn file type capabilities
        for file_type in remote_cmd.get('file_types', []):
            if file_type in ['.fastq', '.bam', '.vcf', '.fasta', '.fa']:
                topology['host_capabilities'][target_host]['genomics'] = True
            elif file_type in ['.R', '.Rmd']:
                topology['host_capabilities'][target_host]['r_computing'] = True
        
        # Learn complexity patterns
        complexity = remote_cmd.get('complexity', 'simple')
        if complexity in ['medium', 'complex']:
            topology['host_capabilities'][target_host]['complex_computing'] = True
    
    # Learn connection patterns
    topology['connection_patterns'][target_host].append({
        'pattern': connection_data['connection_pattern'],
        'timestamp': time.time(),
        'has_user': connection_data['has_user']
    })
    
    # Update environment profile
    if not topology['environment_profile']:
        topology['environment_profile'] = ABSTRACTOR.abstract_environment()
    
    # Store securely
    STORAGE.store_learning_data('ssh_topology', dict(topology))

def learn_basic_ssh_patterns(command):
    """Fallback to basic learning without encryption"""
    # Extract host and remote command
    match = re.match(r'ssh\s+([^\s]+)\s+[\'"]?(.+?)[\'"]?$', command)
    if not match:
        return
    
    host, remote_cmd = match.groups()
    topology = load_basic_topology()
    
    # Extract paths from the remote command
    paths = extract_paths(remote_cmd)
    
    # Learn filesystem topology
    for path in paths:
        topology['host_paths'][host].add(path)
        topology['path_hosts'][path].add(host)
    
    # Learn command patterns
    cmd_base = remote_cmd.split()[0] if remote_cmd.split() else ''
    if cmd_base:
        topology['host_commands'][host][cmd_base] += 1
    
    # Learn path patterns (directory hierarchies)
    for path in paths:
        parts = Path(path).parts
        for i in range(len(parts)):
            prefix = str(Path(*parts[:i+1]))
            topology['host_path_patterns'][host].add(prefix)
    
    save_basic_topology(topology)

def suggest_routing(command):
    """Suggest remote execution based on learned patterns"""
    if SECURE_MODE and STORAGE:
        return suggest_secure_routing(command)
    else:
        return suggest_basic_routing(command)

def suggest_secure_routing(command):
    """Secure routing suggestions based on abstracted learning"""
    topology = STORAGE.load_learning_data('ssh_topology', {})
    if not topology:
        return None
    
    # Abstract the current command
    cmd_abstract = ABSTRACTOR.abstract_command(command)
    if not cmd_abstract:
        return None
    
    host_scores = defaultdict(int)
    suggestions = []
    
    # Score hosts based on capabilities
    for host, capabilities in topology.get('host_capabilities', {}).items():
        score = 0
        
        # Match genomics capabilities
        if any(ft in ['.fastq', '.bam', '.vcf'] for ft in cmd_abstract.get('file_types', [])):
            if capabilities.get('genomics'):
                score += 20
                suggestions.append(f"🧬 Genomics data detected → {host} has genomics capabilities")
        
        # Match R computing
        if cmd_abstract.get('base_command') == 'R' or '.R' in cmd_abstract.get('file_types', []):
            if capabilities.get('r_computing'):
                score += 15
                suggestions.append(f"📊 R computing → {host} has R environment")
        
        # Match complexity
        if cmd_abstract.get('complexity') in ['medium', 'complex']:
            if capabilities.get('complex_computing'):
                score += 10
                suggestions.append(f"⚡ Complex command → {host} handles complex tasks")
        
        # Match command patterns
        base_cmd = cmd_abstract.get('base_command', '')
        if base_cmd in topology.get('command_patterns', {}).get(host, {}):
            frequency = topology['command_patterns'][host][base_cmd]
            score += min(frequency, 15)  # Cap at 15 points
        
        if score > 0:
            host_scores[host] = score
    
    # Check path capabilities
    for path_type in cmd_abstract.get('path_types', []):
        for host, path_types in topology.get('path_capabilities', {}).items():
            if path_type in path_types:
                host_scores[host] += 5
                suggestions.append(f"📁 Path pattern {path_type} → known on {host}")
    
    if host_scores:
        return format_secure_suggestion(command, host_scores, suggestions)
    
    return None

def suggest_basic_routing(command):
    """Basic routing suggestions"""
    topology = load_basic_topology()
    
    if not topology['path_hosts']:
        return None
    
    paths = extract_paths(command)
    if not paths:
        return None
    
    host_scores = defaultdict(int)
    path_suggestions = []
    
    for path in paths:
        if path in topology['path_hosts']:
            for host in topology['path_hosts'][path]:
                host_scores[host] += 10
            path_suggestions.append(f"📁 `{path}` → known on: {list(topology['path_hosts'][path])}")
        elif not os.path.exists(path):
            similar_hosts = find_similar_paths(path, topology)
            if similar_hosts:
                for host, similarity in similar_hosts:
                    host_scores[host] += similarity
                path_suggestions.append(f"❓ `{path}` → might exist on: {[h for h, _ in similar_hosts]}")
    
    cmd_base = command.split()[0] if command.split() else ''
    if cmd_base:
        for host, commands in topology['host_commands'].items():
            if cmd_base in commands:
                host_scores[host] += commands[cmd_base]
    
    if host_scores:
        return format_basic_suggestion(command, host_scores, path_suggestions)
    
    return None

def format_secure_suggestion(command, host_scores, suggestions):
    """Format secure routing suggestions"""
    parts = ["🧠 **AI learned patterns suggest:**"]
    
    if suggestions:
        parts.extend(f"  {s}" for s in suggestions)
        parts.append("")
    
    top_hosts = sorted(host_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    if top_hosts:
        parts.append("**Recommended hosts:**")
        for host, score in top_hosts:
            parts.append(f"  • `ssh {host} '{command}'` (confidence: {score})")
    
    return "\n".join(parts)

def format_basic_suggestion(command, host_scores, path_suggestions):
    """Format basic routing suggestions"""
    parts = ["🗺️ **Filesystem topology suggests:**"]
    
    if path_suggestions:
        parts.extend(path_suggestions)
        parts.append("")
    
    top_hosts = sorted(host_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    if top_hosts:
        parts.append("**Recommended hosts:**")
        for host, score in top_hosts:
            parts.append(f"  • `ssh {host} '{command}'` (score: {score})")
    
    return "\n".join(parts)

def extract_paths(command):
    """Extract filesystem paths from a command"""
    paths = set()
    
    abs_paths = re.findall(r'/[a-zA-Z0-9_/.,-]+', command)
    paths.update(abs_paths)
    
    rel_patterns = [
        r'\./[a-zA-Z0-9_/.,-]+',
        r'[a-zA-Z0-9_]+/[a-zA-Z0-9_/.,-]+',
    ]
    
    for pattern in rel_patterns:
        rel_paths = re.findall(pattern, command)
        paths.update(rel_paths)
    
    cleaned_paths = set()
    for path in paths:
        cleaned = re.sub(r'[;,\'"]+$', '', path)
        if len(cleaned) > 2:
            cleaned_paths.add(cleaned)
    
    return cleaned_paths

def find_similar_paths(target_path, topology):
    """Find hosts with similar path patterns"""
    similar_hosts = []
    target_parts = Path(target_path).parts
    
    for host, path_patterns in topology['host_path_patterns'].items():
        max_similarity = 0
        
        for pattern in path_patterns:
            pattern_parts = Path(pattern).parts
            
            common_prefix = 0
            for i, (a, b) in enumerate(zip(target_parts, pattern_parts)):
                if a == b:
                    common_prefix += 1
                else:
                    break
            
            if common_prefix > 0:
                similarity = common_prefix / max(len(target_parts), len(pattern_parts))
                max_similarity = max(max_similarity, similarity)
        
        if max_similarity > 0.5:
            similar_hosts.append((host, int(max_similarity * 5)))
    
    return sorted(similar_hosts, key=lambda x: x[1], reverse=True)

def load_basic_topology():
    """Load basic filesystem topology"""
    default_topology = {
        'host_paths': defaultdict(set),
        'path_hosts': defaultdict(set),
        'host_path_patterns': defaultdict(set),
        'host_commands': defaultdict(Counter),
    }
    
    if not FALLBACK_DB.exists():
        return default_topology
    
    try:
        with open(FALLBACK_DB, 'rb') as f:
            topology = pickle.load(f)
        for key in default_topology:
            if key not in topology:
                topology[key] = default_topology[key]
        return topology
    except Exception:
        return default_topology

def save_basic_topology(topology):
    """Save basic filesystem topology"""
    try:
        FALLBACK_DB.parent.mkdir(exist_ok=True)
        with open(FALLBACK_DB, 'wb') as f:
            pickle.dump(topology, f)
    except Exception:
        pass

def initialize_host_topology():
    """Initialize host topology on first run"""
    if not SECURE_MODE or not STORAGE:
        # Initialize basic topology without secure storage
        initialize_basic_topology()
        return
    
    # Check if already initialized
    topology_data = STORAGE.load_learning_data('ssh_topology', {})
    if topology_data.get('initialized'):
        return  # Already done
    
    print("🔍 Initializing host topology discovery...", file=sys.stderr)
    
    # Learn about current host
    current_host = discover_current_host()
    if current_host:
        topology_data.setdefault('hosts', {})[current_host['hostname']] = current_host
    
    # Scan SSH config for known hosts
    ssh_hosts = discover_ssh_hosts()
    for host_info in ssh_hosts:
        topology_data.setdefault('hosts', {})[host_info['hostname']] = host_info
    
    # Mark as initialized
    topology_data['initialized'] = True
    topology_data['initialization_time'] = time.time()
    
    # Store the initialized topology
    STORAGE.store_learning_data('ssh_topology', topology_data)
    
    print(f"✅ Discovered {len(topology_data.get('hosts', {}))} hosts", file=sys.stderr)

def discover_current_host():
    """Discover information about the current host"""
    import socket
    import platform
    import subprocess
    
    try:
        hostname = socket.gethostname()
        fqdn = socket.getfqdn()
        
        # Get system info
        uname = platform.uname()
        
        # Try to get CPU info
        cpu_info = "unknown"
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_info = line.split(':')[1].strip()
                        break
        except:
            pass
        
        return {
            'hostname': hostname,
            'fqdn': fqdn,
            'system': uname.system,
            'machine': uname.machine,
            'processor': uname.processor,
            'cpu_model': cpu_info,
            'is_current_host': True,
            'discovered_at': time.time(),
            'connection_success_rate': 1.0,  # localhost always works
            'capabilities': ['local_execution', 'file_access']
        }
    except Exception:
        return None

def discover_ssh_hosts():
    """Discover hosts from SSH config"""
    import subprocess
    ssh_hosts = []
    
    try:
        # Read SSH config
        ssh_config_path = Path.home() / '.ssh' / 'config'
        if not ssh_config_path.exists():
            return ssh_hosts
        
        with open(ssh_config_path, 'r') as f:
            config_content = f.read()
        
        # Parse Host entries
        for line in config_content.split('\n'):
            line = line.strip()
            if line.startswith('Host ') and not line.startswith('Host *'):
                hostname = line.split()[1]
                if hostname and not hostname.startswith('*'):
                    ssh_hosts.append({
                        'hostname': hostname,
                        'source': 'ssh_config',
                        'discovered_at': time.time(),
                        'connection_success_rate': 0.5,  # Unknown, assume 50%
                        'capabilities': ['remote_execution', 'file_transfer']
                    })
    
    except Exception:
        pass
    
    return ssh_hosts

def initialize_basic_topology():
    """Initialize basic topology without secure storage"""
    # Check if already initialized
    if FALLBACK_DB.exists():
        try:
            with open(FALLBACK_DB, 'rb') as f:
                topology = pickle.load(f)
                if topology.get('initialized'):
                    return  # Already done
        except:
            pass
    
    print("🔍 Initializing basic host topology...", file=sys.stderr)
    
    # Learn about current host
    current_host = discover_current_host()
    ssh_hosts = discover_ssh_hosts()
    
    # Build basic topology
    topology = {
        'hosts': {},
        'initialized': True,
        'initialization_time': time.time()
    }
    
    if current_host:
        topology['hosts'][current_host['hostname']] = current_host
    
    for host_info in ssh_hosts:
        topology['hosts'][host_info['hostname']] = host_info
    
    # Store basic topology
    try:
        FALLBACK_DB.parent.mkdir(parents=True, exist_ok=True)
        with open(FALLBACK_DB, 'wb') as f:
            pickle.dump(topology, f)
    except Exception:
        pass
    
    print(f"✅ Discovered {len(topology.get('hosts', {}))} hosts", file=sys.stderr)

def show_learning_stats():
    """Show learning statistics for standalone usage"""
    print("SSH Router Enhanced - Learning Statistics")
    print("=" * 50)
    
    try:
        # Try to load learning data
        sys.path.insert(0, str(Path(__file__).parent.parent / 'learning'))
        from encryption import get_secure_storage
        storage = get_secure_storage()
        
        if storage:
            # Load SSH learning data
            ssh_data = storage.load_learning_data('ssh_topology', {})
            host_data = storage.load_learning_data('host_capabilities', {})
            
            print(f"Known hosts: {len(ssh_data.get('hosts', {}))}")
            print(f"SSH connections learned: {len(ssh_data.get('connections', []))}")
            print(f"Host capabilities: {len(host_data.get('capabilities', {}))}")
            
            if ssh_data.get('hosts'):
                print("\nHosts:")
                for host, info in ssh_data['hosts'].items():
                    print(f"  • {host}: {info.get('success_rate', 0):.1%} success rate")
        else:
            print("Learning system not available (missing encryption modules)")
            
    except ImportError:
        print("Learning system not available (missing dependencies)")
        
        # Fallback to simple file check
        if FALLBACK_DB.exists():
            try:
                with open(FALLBACK_DB, 'rb') as f:
                    topology = pickle.load(f)
                    hosts = topology.get('hosts', {})
                    print(f"Known hosts: {len(hosts)}")
                    print(f"Initialized: {topology.get('initialized', False)}")
                    
                    if hosts:
                        print("\nHosts:")
                        for hostname, info in hosts.items():
                            host_type = "current" if info.get('is_current_host') else "remote"
                            success_rate = info.get('connection_success_rate', 0)
                            capabilities = ', '.join(info.get('capabilities', []))
                            print(f"  • {hostname} ({host_type}): {success_rate:.1%} success, {capabilities}")
            except Exception as e:
                print(f"Error reading fallback data: {e}")
        else:
            print("No learning data found")
    
    except Exception as e:
        print(f"Error loading learning data: {e}")

if __name__ == '__main__':
    main()