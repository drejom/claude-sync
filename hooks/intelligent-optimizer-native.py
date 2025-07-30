#!/usr/bin/env python3
"""
Native Python version for performance testing
"""

import json
import sys

# Fast optimization patterns - precomputed for speed
FAST_OPTIMIZATIONS = {
    'grep ': ('rg ', 0.9),
    'find ': ('fd ', 0.7), 
    'cat ': ('bat ', 0.6),
}

# Critical safety patterns - checked first
SAFETY_PATTERNS = [
    ('rm -rf /', 'Attempting to delete entire filesystem'),
    ('dd if=', 'Direct disk writing - potential data loss'),
    ('> /dev/', 'Writing to device files'),
    ('mkfs', 'Filesystem creation - will destroy data'),
]

def main():
    """Ultra-fast hook execution"""
    try:
        # Read input
        hook_input = json.loads(sys.stdin.read())
        
        # Fast exit for non-Bash tools
        if hook_input.get('tool_name') != 'Bash':
            sys.exit(0)
        
        command = hook_input.get('tool_input', {}).get('command', '')
        if not command.strip():
            sys.exit(0)
        
        suggestions = []
        
        # Safety check first (most critical)
        for pattern, warning in SAFETY_PATTERNS:
            if pattern in command:
                suggestions.append(f"⚠️ **Safety warning:** {warning}")
                break  # Only show first safety issue for speed
        
        # Fast optimizations
        for old_cmd, (new_cmd, confidence) in FAST_OPTIMIZATIONS.items():
            if old_cmd in command and new_cmd.strip() not in command:
                optimized = command.replace(old_cmd, new_cmd)
                if confidence > 0.8:
                    suggestions.append(f"🚀 **High-confidence optimization:**\n```bash\n{optimized}\n```")
                else:
                    suggestions.append(f"💡 **Suggested optimization (confidence: {confidence:.0%}):**\n```bash\n{optimized}\n```")
                break  # Only show first optimization for speed
        
        # SLURM quick suggestions
        if command.startswith('sbatch') and '--mem=' not in command:
            suggestions.append("🧬 **SLURM tip:** Consider adding `--mem=16G` for memory allocation")
        
        # R quick suggestions  
        if 'Rscript' in command and '--vanilla' not in command:
            suggestions.append("📊 **R tip:** Consider `--vanilla` flag for reproducible execution")
        
        # Container quick suggestions
        if 'singularity exec' in command and '--bind' not in command:
            suggestions.append("📦 **Container tip:** Consider `--bind /data,/scratch` for data access")
        
        # Output result
        if suggestions:
            print(json.dumps({
                'block': False,
                'message': '\n\n'.join(suggestions[:2])  # Limit to 2 suggestions for speed
            }))
        
    except Exception:
        # Silent failure - never break Claude Code
        pass
    
    sys.exit(0)

if __name__ == '__main__':
    main()