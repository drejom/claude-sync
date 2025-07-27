#!/usr/bin/env python3
"""
Enhanced Universal Bash Command Optimizer Hook
Upgrades commands, adds safety features, and learns from success patterns.
"""

import json
import sys
import re
import shutil
import os
import time
from pathlib import Path
from collections import defaultdict, Counter

# Try to import secure learning infrastructure
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

def main():
    # Read hook input
    hook_input = json.loads(sys.stdin.read())
    
    if hook_input.get('tool_name') != 'Bash':
        sys.exit(0)
    
    command = hook_input.get('tool_input', {}).get('command', '')
    if not command:
        sys.exit(0)
    
    hook_event = hook_input.get('hook_event', 'PreToolUse')
    
    if hook_event == 'PreToolUse':
        handle_pre_tool_use(command, hook_input)
    elif hook_event == 'PostToolUse':
        handle_post_tool_use(command, hook_input)
    
    sys.exit(0)

def handle_pre_tool_use(command, hook_input):
    """Handle command optimization and safety checks"""
    # Apply optimizations
    optimized_command = optimize_command(command)
    safety_warnings = check_safety(command)
    learned_suggestions = get_learned_suggestions(command)
    
    # Compile feedback
    feedback_parts = []
    
    if optimized_command != command:
        feedback_parts.append(f"🚀 **Optimized command:**\n```bash\n{optimized_command}\n```")
    
    if safety_warnings:
        feedback_parts.append("**Safety notes:**\n" + "\n".join(f"  {w}" for w in safety_warnings))
    
    if learned_suggestions:
        feedback_parts.append(learned_suggestions)
    
    if feedback_parts:
        result = {
            'block': False,
            'message': "\n\n".join(feedback_parts)
        }
        print(json.dumps(result))

def handle_post_tool_use(command, hook_input):
    """Learn from successful command execution"""
    # Check if command was successful (simplified check)
    tool_output = hook_input.get('tool_output', {})
    exit_code = tool_output.get('exit_code', 0)
    
    if exit_code == 0:  # Success
        learn_successful_command(command)
    else:  # Failure
        learn_failed_command(command, exit_code)

def optimize_command(command):
    """Apply smart command optimizations with learned preferences"""
    # Get learned optimization preferences
    learned_optimizations = get_learned_optimizations()
    
    # Base optimizations
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
        
        # HPC/Scientific computing optimizations
        (r'\bsbatch\b(?!\s+--)', 'sbatch --parsable'),  # Get job ID for tracking
        (r'\bsqueue\b(?!\s+-)', 'squeue -u $USER'),     # Show only user's jobs
    ]
    
    # Add learned optimizations
    optimizations.extend(learned_optimizations)
    
    original = command
    for pattern, replacement in optimizations:
        # Only apply if the better tool exists
        tool = replacement.split()[0]
        if shutil.which(tool) or tool.startswith(('git', 'sbatch', 'squeue')):
            command = re.sub(pattern, replacement, command)
    
    # Domain-specific optimizations
    command = apply_domain_optimizations(command)
    
    return command

def apply_domain_optimizations(command):
    """Apply domain-specific optimizations for HPC/bioinformatics"""
    # R optimizations
    if command.startswith('R ') or 'Rscript' in command:
        # Suggest parallel processing hints
        if '--vanilla' not in command and len(command.split()) < 5:
            command = command.replace('R ', 'R --vanilla ')
    
    # Singularity optimizations
    if 'singularity exec' in command and '--bind' not in command:
        # Common bind mounts for HPC
        if '/data' in command or '/scratch' in command:
            command = command.replace('singularity exec', 'singularity exec --bind /data,/scratch')
    
    # FASTQ/genomics file handling
    if any(pattern in command for pattern in ['.fastq', '.fq', '.bam', '.vcf']):
        # Suggest compression awareness
        if 'zcat' not in command and '.gz' in command and command.startswith('cat'):
            command = command.replace('cat', 'zcat')
    
    return command

def check_safety(command):
    """Enhanced safety checks including HPC-specific warnings"""
    warnings = []
    
    # Basic destructive operations
    if re.search(r'\brm\s+.*-rf?\s+/', command):
        warnings.append("⚠️  Recursive delete detected - be careful with paths")
    
    if re.search(r'\bmv\s+.*\s+/', command):
        warnings.append("⚠️  Moving to directory - verify target path")
    
    # Broad operations
    if re.search(r'(find|fd).*-exec.*rm', command):
        warnings.append("⚠️  Find + delete combo - test with -print first")
    
    if re.search(r'chmod.*777', command):
        warnings.append("⚠️  777 permissions - consider more restrictive permissions")
    
    # Remote operations
    if re.search(r'ssh.*sudo', command):
        warnings.append("💡 Remote sudo detected - ensure you have proper access")
    
    # HPC-specific warnings
    if re.search(r'sbatch.*-N\s*[0-9]+', command):
        nodes = re.search(r'-N\s*([0-9]+)', command)
        if nodes and int(nodes.group(1)) > 10:
            warnings.append("🚀 Large node request detected - verify resource availability")
    
    if re.search(r'--mem.*[0-9]+[GT]', command):
        warnings.append("🧠 High memory request - ensure cluster has sufficient resources")
    
    # Data safety warnings
    if '/data/' in command and any(op in command for op in ['rm', 'mv', 'cp']):
        warnings.append("📁 Data directory operation - double-check paths")
    
    # Genomics workflow warnings
    if any(tool in command for tool in ['bcftools', 'samtools', 'bwa', 'gatk']):
        if not any(container in command for container in ['singularity', 'docker']):
            warnings.append("🧬 Genomics tool detected - consider using containers for reproducibility")
    
    return warnings

def get_learned_suggestions(command):
    """Get AI suggestions based on learned patterns"""
    if not SECURE_MODE or not STORAGE:
        return None
    
    suggestions = []
    
    # Get learning data
    learning_data = STORAGE.load_learning_data('bash_learning', {})
    if not learning_data:
        return None
    
    # Abstract current command
    if ABSTRACTOR:
        cmd_abstract = ABSTRACTOR.abstract_command(command)
        base_command = cmd_abstract.get('base_command', '')
        file_types = cmd_abstract.get('file_types', [])
        complexity = cmd_abstract.get('complexity', 'simple')
        
        # Get success patterns for this command type
        success_patterns = learning_data.get('success_patterns', {})
        if base_command in success_patterns:
            pattern_data = success_patterns[base_command]
            success_rate = pattern_data.get('success_rate', 0)
            common_flags = pattern_data.get('common_flags', [])
            
            if success_rate > 0.8 and common_flags:
                missing_flags = [flag for flag in common_flags if flag not in command]
                if missing_flags:
                    suggestions.append(f"💡 `{base_command}` often uses: {' '.join(missing_flags[:3])}")
        
        # File type specific suggestions
        genomics_files = [ft for ft in file_types if ft in ['.fastq', '.bam', '.vcf', '.fasta']]
        if genomics_files:
            suggestions.append("🧬 Genomics files detected - consider using parallel processing")
        
        # Complexity suggestions
        if complexity == 'complex':
            suggestions.append("⚡ Complex command - consider breaking into steps for debugging")
    
    if suggestions:
        return "🎯 **AI suggestions:**\n" + "\n".join(f"  {s}" for s in suggestions)
    
    return None

def learn_successful_command(command):
    """Learn from successful command execution"""
    if not SECURE_MODE or not STORAGE or not ABSTRACTOR:
        return
    
    # Abstract command for learning
    cmd_abstract = ABSTRACTOR.abstract_command(command)
    if not cmd_abstract:
        return
    
    # Get existing learning data
    learning_data = STORAGE.load_learning_data('bash_learning', {
        'success_patterns': defaultdict(dict),
        'failure_patterns': defaultdict(dict),
        'optimization_feedback': defaultdict(list),
        'environment_patterns': {}
    })
    
    base_command = cmd_abstract.get('base_command', '')
    if not base_command:
        return
    
    # Update success patterns
    pattern_key = base_command
    if pattern_key not in learning_data['success_patterns']:
        learning_data['success_patterns'][pattern_key] = {
            'success_count': 0,
            'total_count': 0,
            'common_flags': Counter(),
            'file_types': Counter(),
            'complexity_patterns': Counter()
        }
    
    pattern_data = learning_data['success_patterns'][pattern_key]
    pattern_data['success_count'] += 1
    pattern_data['total_count'] += 1
    
    # Learn flag patterns
    for flag in cmd_abstract.get('flags', []):
        pattern_data['common_flags'][flag] += 1
    
    # Learn file type patterns
    for file_type in cmd_abstract.get('file_types', []):
        pattern_data['file_types'][file_type] += 1
    
    # Learn complexity patterns
    complexity = cmd_abstract.get('complexity', 'simple')
    pattern_data['complexity_patterns'][complexity] += 1
    
    # Calculate success rate
    pattern_data['success_rate'] = pattern_data['success_count'] / pattern_data['total_count']
    
    # Store updated learning data
    STORAGE.store_learning_data('bash_learning', dict(learning_data))

def learn_failed_command(command, exit_code):
    """Learn from failed command execution"""
    if not SECURE_MODE or not STORAGE or not ABSTRACTOR:
        return
    
    # Abstract command for learning
    cmd_abstract = ABSTRACTOR.abstract_command(command)
    if not cmd_abstract:
        return
    
    # Get existing learning data
    learning_data = STORAGE.load_learning_data('bash_learning', {
        'success_patterns': defaultdict(dict),
        'failure_patterns': defaultdict(dict),
        'optimization_feedback': defaultdict(list),
        'environment_patterns': {}
    })
    
    base_command = cmd_abstract.get('base_command', '')
    if not base_command:
        return
    
    # Update failure patterns
    pattern_key = base_command
    if pattern_key not in learning_data['failure_patterns']:
        learning_data['failure_patterns'][pattern_key] = {
            'failure_count': 0,
            'exit_codes': Counter(),
            'problematic_flags': Counter()
        }
    
    failure_data = learning_data['failure_patterns'][pattern_key]
    failure_data['failure_count'] += 1
    failure_data['exit_codes'][exit_code] += 1
    
    # Learn problematic flag combinations
    for flag in cmd_abstract.get('flags', []):
        failure_data['problematic_flags'][flag] += 1
    
    # Update success pattern total count if exists
    if pattern_key in learning_data['success_patterns']:
        learning_data['success_patterns'][pattern_key]['total_count'] += 1
        # Recalculate success rate
        success_data = learning_data['success_patterns'][pattern_key]
        success_data['success_rate'] = success_data['success_count'] / success_data['total_count']
    
    # Store updated learning data
    STORAGE.store_learning_data('bash_learning', dict(learning_data))

def get_learned_optimizations():
    """Get learned optimization patterns"""
    if not SECURE_MODE or not STORAGE:
        return []
    
    learning_data = STORAGE.load_learning_data('bash_learning', {})
    optimizations = []
    
    # Extract learned optimizations from success patterns
    for command, pattern_data in learning_data.get('success_patterns', {}).items():
        if pattern_data.get('success_rate', 0) > 0.9:  # High success rate
            common_flags = pattern_data.get('common_flags', {})
            if common_flags:
                # Get most common flag
                most_common_flag = common_flags.most_common(1)[0][0]
                if most_common_flag and common_flags[most_common_flag] > 5:  # Used frequently
                    pattern = rf'\b{re.escape(command)}\b(?!\s+.*{re.escape(most_common_flag)})'
                    replacement = f"{command} {most_common_flag}"
                    optimizations.append((pattern, replacement))
    
    return optimizations

if __name__ == '__main__':
    main()