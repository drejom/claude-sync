#!/usr/bin/env python3
"""
Self-Learning SSH Router Hook
Learns filesystem topology across hosts and command patterns for intelligent routing.
"""

import json
import sys
import re
import os
from pathlib import Path
from collections import defaultdict, Counter
import pickle

LEARNING_DB = Path.home() / '.claude' / 'ssh_topology.pkl'

def main():
    # Read hook input
    hook_input = json.loads(sys.stdin.read())
    
    if hook_input.get('tool_name') != 'Bash':
        sys.exit(0)
    
    command = hook_input.get('tool_input', {}).get('command', '')
    if not command:
        sys.exit(0)
    
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
    """Learn filesystem topology and command patterns from SSH usage"""
    # Extract host and remote command
    match = re.match(r'ssh\s+([^\s]+)\s+[\'"]?(.+?)[\'"]?$', command)
    if not match:
        return
    
    host, remote_cmd = match.groups()
    topology = load_topology()
    
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
    
    save_topology(topology)

def suggest_routing(command):
    """Suggest remote execution based on learned filesystem topology"""
    topology = load_topology()
    
    if not topology['path_hosts']:  # No learning data yet
        return None
    
    # Extract paths from current command
    paths = extract_paths(command)
    if not paths:
        return None
    
    # Find hosts that have these paths
    host_scores = defaultdict(int)
    path_suggestions = []
    
    for path in paths:
        # Check if we know this exact path
        if path in topology['path_hosts']:
            for host in topology['path_hosts'][path]:
                host_scores[host] += 10  # Exact match = high score
            path_suggestions.append(f"📁 `{path}` → known on: {list(topology['path_hosts'][path])}")
        
        # Check if local path doesn't exist but might exist remotely
        elif not os.path.exists(path):
            # Look for similar paths on hosts
            similar_hosts = find_similar_paths(path, topology)
            if similar_hosts:
                for host, similarity in similar_hosts:
                    host_scores[host] += similarity
                path_suggestions.append(f"❓ `{path}` → might exist on: {[h for h, _ in similar_hosts]}")
    
    # Add command pattern scoring
    cmd_base = command.split()[0] if command.split() else ''
    if cmd_base:
        for host, commands in topology['host_commands'].items():
            if cmd_base in commands:
                host_scores[host] += commands[cmd_base]  # Weight by frequency
    
    if host_scores:
        return format_routing_suggestion(command, host_scores, path_suggestions)
    
    return None

def extract_paths(command):
    """Extract filesystem paths from a command"""
    paths = set()
    
    # Find absolute paths
    abs_paths = re.findall(r'/[a-zA-Z0-9_/.,-]+', command)
    paths.update(abs_paths)
    
    # Find relative paths with common patterns
    rel_patterns = [
        r'\./[a-zA-Z0-9_/.,-]+',      # ./relative
        r'[a-zA-Z0-9_]+/[a-zA-Z0-9_/.,-]+',  # dir/file
    ]
    
    for pattern in rel_patterns:
        rel_paths = re.findall(pattern, command)
        paths.update(rel_paths)
    
    # Clean up paths (remove trailing punctuation)
    cleaned_paths = set()
    for path in paths:
        cleaned = re.sub(r'[;,\'"]+$', '', path)
        if len(cleaned) > 2:  # Ignore very short matches
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
            
            # Calculate path similarity
            common_prefix = 0
            for i, (a, b) in enumerate(zip(target_parts, pattern_parts)):
                if a == b:
                    common_prefix += 1
                else:
                    break
            
            if common_prefix > 0:
                similarity = common_prefix / max(len(target_parts), len(pattern_parts))
                max_similarity = max(max_similarity, similarity)
        
        if max_similarity > 0.5:  # Threshold for similarity
            similar_hosts.append((host, int(max_similarity * 5)))  # Scale for scoring
    
    return sorted(similar_hosts, key=lambda x: x[1], reverse=True)

def format_routing_suggestion(command, host_scores, path_suggestions):
    """Format the routing suggestion message"""
    parts = ["🗺️ **Filesystem topology suggests:**"]
    
    # Show path analysis
    if path_suggestions:
        parts.extend(path_suggestions)
        parts.append("")
    
    # Show top host suggestions
    top_hosts = sorted(host_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    if top_hosts:
        parts.append("**Recommended hosts:**")
        for host, score in top_hosts:
            parts.append(f"  • `ssh {host} '{command}'` (score: {score})")
    
    return "\n".join(parts)

def load_topology():
    """Load learned filesystem topology"""
    default_topology = {
        'host_paths': defaultdict(set),           # host -> set of known paths
        'path_hosts': defaultdict(set),           # path -> set of hosts that have it
        'host_path_patterns': defaultdict(set),   # host -> set of path prefixes
        'host_commands': defaultdict(Counter),    # host -> command frequency
    }
    
    if not LEARNING_DB.exists():
        return default_topology
    
    try:
        with open(LEARNING_DB, 'rb') as f:
            topology = pickle.load(f)
        # Ensure all expected keys exist
        for key in default_topology:
            if key not in topology:
                topology[key] = default_topology[key]
        return topology
    except Exception:
        return default_topology

def save_topology(topology):
    """Save learned filesystem topology"""
    try:
        LEARNING_DB.parent.mkdir(exist_ok=True)
        with open(LEARNING_DB, 'wb') as f:
            pickle.dump(topology, f)
    except Exception:
        pass  # Fail silently

if __name__ == '__main__':
    main()