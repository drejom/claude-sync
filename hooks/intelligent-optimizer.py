#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0"
# ]
# ///
"""
Intelligent Command Optimizer - Ultra High-Performance PreToolUse Hook
Target: <10ms execution time for real-time responsiveness

Optimized version with minimal learning system overhead and fast fallbacks.
"""

import json
import sys
import time
from pathlib import Path

# Fast optimization patterns - precomputed for speed
FAST_OPTIMIZATIONS = {
    'grep ': ('rg ', 0.9, 'faster search with ripgrep'),
    'find ': ('fd ', 0.7, 'faster file finding with fd'),
    'cat ': ('bat ', 0.6, 'syntax highlighting with bat'),
}

# Critical safety patterns - checked first
SAFETY_PATTERNS = [
    ('rm -rf /', 'Attempting to delete entire filesystem'),
    ('dd if=', 'Direct disk writing - potential data loss'),
    ('> /dev/', 'Writing to device files'),
    ('mkfs', 'Filesystem creation - will destroy data'),
    ('chmod 777', 'Insecure permissions - security risk'),
]

# Context-specific quick tips (no learning required)
CONTEXT_TIPS = {
    'sbatch': ('🧬 **SLURM tip:** Consider adding `--mem=16G --time=4:00:00`', lambda cmd: '--mem=' not in cmd),
    'Rscript': ('📊 **R tip:** Consider `--vanilla` flag for reproducible execution', lambda cmd: '--vanilla' not in cmd),
    'singularity exec': ('📦 **Container tip:** Consider `--bind /data,/scratch`', lambda cmd: '--bind' not in cmd),
    'ssh ': ('🌐 **SSH tip:** Consider `-o ConnectTimeout=10` for faster failure detection', lambda cmd: 'ConnectTimeout' not in cmd),
}

# Performance limits
MAX_EXECUTION_TIME_MS = 10
MAX_SUGGESTIONS = 2

# Global learning cache (loaded once, reused)
_learning_cache = None
_cache_loaded = False

def get_cached_learning_data():
    """Get learning data with aggressive caching for performance"""
    global _learning_cache, _cache_loaded
    
    if _cache_loaded:
        return _learning_cache
    
    _cache_loaded = True
    
    try:
        # Ultra-fast check - if no recent learning data, skip
        learning_dir = Path.home() / '.claude' / 'learning'
        if not learning_dir.exists():
            return None
        
        # Look for today's command data only (fastest check)
        today_file = learning_dir / f'commands_{time.strftime("%Y-%m-%d")}.jsonl'
        if not today_file.exists() or today_file.stat().st_size == 0:
            return None
        
        # Load minimal recent patterns only (limit to 100 most recent)
        patterns = {}
        with today_file.open('r') as f:
            lines = f.readlines()
            for line in lines[-100:]:  # Only last 100 commands
                try:
                    data = json.loads(line.strip())
                    if data.get('success', False):  # Only successful commands
                        cmd = data.get('command', '')[:50]  # Truncate for speed
                        if cmd and not patterns.get(cmd):  # Avoid duplicates
                            patterns[cmd] = {
                                'count': patterns.get(cmd, {}).get('count', 0) + 1,
                                'avg_duration': data.get('duration_ms', 0)
                            }
                except:
                    continue
        
        _learning_cache = patterns if patterns else None
        return _learning_cache
        
    except Exception:
        # Never let learning system errors slow down the hook
        _learning_cache = None
        return None

def generate_fast_suggestions(command: str) -> list:
    """Generate suggestions with ultra-fast execution"""
    suggestions = []
    
    # Safety warnings (highest priority, fastest check)
    for pattern, warning in SAFETY_PATTERNS:
        if pattern in command:
            suggestions.append(f"⚠️ **Safety warning:** {warning}")
            return suggestions[:MAX_SUGGESTIONS]  # Exit immediately for safety
    
    # Fast optimizations (precomputed patterns)
    for old_cmd, (new_cmd, confidence, description) in FAST_OPTIMIZATIONS.items():
        if old_cmd in command and new_cmd.strip() not in command:
            optimized = command.replace(old_cmd, new_cmd)
            if confidence > 0.8:
                suggestions.append(f"🚀 **High-confidence optimization ({description}):**\n```bash\n{optimized}\n```")
            else:
                suggestions.append(f"💡 **Suggested optimization ({description}):**\n```bash\n{optimized}\n```")
            break  # Only first optimization for speed
    
    # Context-specific tips (no learning required)
    if len(suggestions) < MAX_SUGGESTIONS:
        for trigger, (tip, condition) in CONTEXT_TIPS.items():
            if trigger in command and condition(command):
                suggestions.append(tip)
                break
    
    return suggestions[:MAX_SUGGESTIONS]

def add_learning_insights(command: str, suggestions: list, max_time_remaining_ms: float) -> list:
    """Add learning insights if time permits"""
    if max_time_remaining_ms < 3:  # Need at least 3ms for learning lookup
        return suggestions
    
    if len(suggestions) >= MAX_SUGGESTIONS:
        return suggestions
    
    learning_data = get_cached_learning_data()
    if not learning_data:
        return suggestions
    
    try:
        # Quick pattern match
        cmd_key = command[:50]  # Same truncation as cache
        if cmd_key in learning_data:
            pattern = learning_data[cmd_key]
            if pattern['count'] > 2 and pattern['avg_duration'] > 5000:  # >5s avg
                suggestions.append(f"📊 **Learning insight:** This command typically takes {pattern['avg_duration']/1000:.1f}s")
        
    except Exception:
        # Never let learning errors slow down the hook
        pass
    
    return suggestions[:MAX_SUGGESTIONS]

def main():
    """Ultra-fast hook execution with learning integration"""
    start_time = time.perf_counter()
    
    try:
        # Read input
        hook_input = json.loads(sys.stdin.read())
        
        # Fast exit for non-Bash tools
        if hook_input.get('tool_name') != 'Bash':
            sys.exit(0)
        
        command = hook_input.get('tool_input', {}).get('command', '')
        if not command.strip():
            sys.exit(0)
        
        # Generate fast suggestions
        suggestions = generate_fast_suggestions(command)
        
        # Add learning insights if time permits
        time_elapsed_ms = (time.perf_counter() - start_time) * 1000
        time_remaining = MAX_EXECUTION_TIME_MS - time_elapsed_ms
        
        if time_remaining > 0:
            suggestions = add_learning_insights(command, suggestions, time_remaining)
        
        # Check final execution time
        final_time_ms = (time.perf_counter() - start_time) * 1000
        if final_time_ms > MAX_EXECUTION_TIME_MS:
            print(f"WARNING: intelligent-optimizer took {final_time_ms:.1f}ms (target: {MAX_EXECUTION_TIME_MS}ms)", file=sys.stderr)
        
        # Output result
        if suggestions:
            print(json.dumps({
                'block': False,
                'message': '\n\n'.join(suggestions)
            }))
        
    except Exception as e:
        # Log error but never break Claude Code
        print(f"intelligent-optimizer error: {e}", file=sys.stderr)
    
    sys.exit(0)

if __name__ == '__main__':
    main()