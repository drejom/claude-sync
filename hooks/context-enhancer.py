#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# ///
"""
Context Enhancer - High-performance UserPromptSubmit Hook
Target: <20ms execution time for responsive context injection
"""

import json
import sys
import os

# Fast context detection patterns - precomputed
CONTEXT_KEYWORDS = {
    'slurm': ['sbatch', 'slurm', 'queue', 'partition', 'hpc'],
    'container': ['singularity', 'container', 'docker', '.sif'],
    'r': ['rscript', 'r analysis', 'rstudio'],
    'network': ['ssh', 'remote', 'tailscale', 'connection'],
    'performance': ['slow', 'performance', 'optimize', 'faster'],
    'error': ['error', 'failed', 'debug', 'troubleshoot']
}

# Precomputed context responses for speed
CONTEXT_RESPONSES = {
    'slurm': """**SLURM optimization tips:**
- Add resource allocation: `--mem=16G --cpus-per-task=4`
- Set time limit: `--time=4:00:00`
- Use appropriate partition: `--partition=compute` or `--partition=gpu`""",
    
    'container': """**Container best practices:**
- Use bind mounts: `--bind /data,/scratch,/home`
- Store .sif files in persistent storage, not /tmp
- Check container permissions for data access""",
    
    'r': """**R workflow optimization:**
- Use `--vanilla` flag for reproducible execution
- Set memory limit: `--max-mem-size=8G`
- Common issues: memory exhaustion, missing packages""",
    
    'network': """**Network connection tips:**
- Tailscale provides better reliability than direct SSH
- Use connection timeouts: `-o ConnectTimeout=10`
- Consider rsync for large file transfers""",
    
    'performance': """**Performance optimization:**
- Use `rg` instead of `grep` for faster search
- Use `fd` instead of `find` for faster file search
- Consider SLURM resource allocation for compute tasks""",
    
    'error': """**Common troubleshooting:**
- Memory errors: Often from underestimated resource needs
- Permission errors: Check file ownership and bind mounts
- Network timeouts: Try Tailscale instead of direct SSH"""
}

def main():
    """Ultra-fast context enhancement"""
    try:
        # Read input
        hook_input = json.loads(sys.stdin.read())
        
        user_prompt = hook_input.get('user_prompt', '')
        if not user_prompt.strip():
            sys.exit(0)
        
        prompt_lower = user_prompt.lower()
        
        # Fast context detection
        detected_contexts = []
        for context_type, keywords in CONTEXT_KEYWORDS.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_contexts.append(context_type)
                if len(detected_contexts) >= 2:  # Limit for speed
                    break
        
        if not detected_contexts:
            sys.exit(0)
        
        # Generate context enhancement
        enhancements = []
        for context_type in detected_contexts:
            if context_type in CONTEXT_RESPONSES:
                enhancements.append(CONTEXT_RESPONSES[context_type])
        
        if enhancements:
            message = "🧠 **Added context from learning data:**\n\n" + "\n\n".join(enhancements)
            print(json.dumps({
                'block': False,
                'message': message
            }))
        
    except Exception:
        # Silent failure - never break Claude Code
        pass
    
    sys.exit(0)

if __name__ == '__main__':
    main()