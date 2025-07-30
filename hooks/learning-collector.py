#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# ///
"""
Learning Data Collector - High-performance PostToolUse Hook
Target: <50ms execution time for silent learning
"""

import json
import sys
import os
import time

def main():
    """Ultra-fast learning data collection"""
    try:
        # Read input
        hook_input = json.loads(sys.stdin.read())
        
        # Extract basic execution data
        tool_input = hook_input.get('tool_input', {})
        tool_output = hook_input.get('tool_output', {})
        
        command = tool_input.get('command', '')
        if not command.strip():
            sys.exit(0)
        
        exit_code = tool_output.get('exit_code', 0)
        duration_ms = tool_output.get('duration_ms', 0)
        timestamp = time.time()
        
        # Fast learning record
        learning_record = {
            'command': command[:200],  # Truncate for performance
            'exit_code': exit_code,
            'duration_ms': duration_ms,
            'timestamp': timestamp,
            'success': exit_code == 0
        }
        
        # Ensure learning directory exists
        learning_dir = os.path.expanduser('~/.claude/learning')
        os.makedirs(learning_dir, exist_ok=True)
        
        # Append to daily file (fast I/O)
        date_str = time.strftime('%Y-%m-%d', time.localtime(timestamp))
        learning_file = os.path.join(learning_dir, f'commands_{date_str}.jsonl')
        
        with open(learning_file, 'a') as f:
            f.write(json.dumps(learning_record) + '\n')
        
        # Track failures for threshold triggering
        if exit_code != 0:
            failures_file = os.path.join(learning_dir, 'recent_failures.txt')
            with open(failures_file, 'a') as f:
                f.write(f"{timestamp},{command[:100]}\n")
        
        # PostToolUse hooks are silent - never output
        
    except Exception:
        # Silent failure - never break Claude Code
        pass
    
    sys.exit(0)

if __name__ == '__main__':
    main()