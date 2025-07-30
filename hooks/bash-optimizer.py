#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# ///
"""
Universal Bash Command Optimizer Hook
Upgrades commands and adds safety features across all projects.
"""

import json
import sys
import re
import shutil
import os

def main():
    # Read hook input
    hook_input = json.loads(sys.stdin.read())
    
    if hook_input.get('tool_name') != 'Bash':
        sys.exit(0)
    
    command = hook_input.get('tool_input', {}).get('command', '')
    if not command:
        sys.exit(0)
    
    # Apply optimizations
    optimized_command = optimize_command(command)
    safety_warnings = check_safety(command)
    
    # If command was optimized or has warnings, provide feedback
    if optimized_command != command or safety_warnings:
        result = {
            'block': False,
            'message': format_feedback(command, optimized_command, safety_warnings)
        }
        print(json.dumps(result))
        sys.exit(0)
    
    # No changes needed
    sys.exit(0)

def optimize_command(command):
    """Apply smart command optimizations"""
    optimizations = [
        # Better search tools
        (r'\bgrep\b(?!\s+--)', 'rg'),
        (r'\bfind\b(?=\s+\S)', 'fd'),
        
        # Better file tools  
        (r'\bcat\b(?=\s+\S)', 'bat --plain'),
        (r'\bls\b(?!\s+-)', 'ls --color=auto'),
        
        # Add useful flags
        (r'\bdu\b(?!\s+-)', 'du -h'),
        (r'\bdf\b(?!\s+-)', 'df -h'),
        (r'\btree\b(?!\s+-)', 'tree -C'),
        
        # Git improvements
        (r'\bgit\s+log\b(?!\s+--)', 'git log --oneline --graph'),
        (r'\bgit\s+diff\b(?!\s+--)', 'git diff --color=always'),
        (r'\bgit\s+status\b(?!\s+--)', 'git status --short'),
    ]
    
    original = command
    for pattern, replacement in optimizations:
        # Only apply if the better tool exists
        tool = replacement.split()[0]
        if shutil.which(tool) or tool.startswith('git'):
            command = re.sub(pattern, replacement, command)
    
    return command

def check_safety(command):
    """Check for potentially dangerous operations"""
    warnings = []
    
    # Destructive operations
    if re.search(r'\brm\s+.*-rf?\s+/', command):
        warnings.append("⚠️  Recursive delete detected - be careful with paths")
    
    if re.search(r'\bmv\s+.*\s+/', command):
        warnings.append("⚠️  Moving to directory - verify target path")
    
    # Broad operations
    if re.search(r'(find|fd).*-exec.*rm', command):
        warnings.append("⚠️  Find + delete combo - test with -print first")
    
    if re.search(r'chmod.*777', command):
        warnings.append("⚠️  777 permissions - consider more restrictive permissions")
    
    # Remote operations without confirmation
    if re.search(r'ssh.*sudo', command):
        warnings.append("💡 Remote sudo detected - ensure you have proper access")
    
    return warnings

def format_feedback(original, optimized, warnings):
    """Format the feedback message"""
    parts = []
    
    if optimized != original:
        parts.append(f"🚀 **Optimized command:**\n```bash\n{optimized}\n```")
    
    if warnings:
        parts.append("**Safety notes:**\n" + "\n".join(f"  {w}" for w in warnings))
    
    return "\n\n".join(parts)

if __name__ == '__main__':
    main()