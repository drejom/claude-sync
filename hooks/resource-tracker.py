#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "psutil>=5.9.0"
# ]
# requires-python = ">=3.10"
# ///
"""
Resource Tracker Hook
Monitors command performance and learns optimal resource patterns for AI suggestions.
"""

import json
import sys
import time
import psutil
import os
import re
from pathlib import Path
from collections import defaultdict

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

# Performance tracking state
PROCESS_TRACKING = {}

def main():
    hook_input = json.loads(sys.stdin.read())
    
    if hook_input.get('tool_name') != 'Bash':
        sys.exit(0)
    
    hook_event = hook_input.get('hook_event', '')
    command = hook_input.get('tool_input', {}).get('command', '')
    
    if not command:
        sys.exit(0)
    
    if hook_event == 'PreToolUse':
        start_tracking(command, hook_input)
    elif hook_event == 'PostToolUse':
        end_tracking(command, hook_input)
    
    sys.exit(0)

def start_tracking(command, hook_input):
    """Start performance tracking for a command"""
    try:
        # Generate tracking ID
        tracking_id = f"{os.getpid()}_{int(time.time() * 1000)}"
        
        # Get initial system state
        initial_state = {
            'start_time': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_used': psutil.virtual_memory().used,
            'disk_io': get_disk_io(),
            'command': command,
            'tracking_id': tracking_id
        }
        
        # Abstract command for learning
        if SECURE_MODE and ABSTRACTOR:
            initial_state['command_abstract'] = ABSTRACTOR.abstract_command(command)
        
        PROCESS_TRACKING[tracking_id] = initial_state
        
        # Store tracking ID for PostToolUse
        print(json.dumps({
            'block': False,
            'tracking_id': tracking_id
        }))
        
    except Exception:
        pass  # Fail silently

def end_tracking(command, hook_input):
    """End performance tracking and learn patterns"""
    try:
        # Find matching tracking entry
        tracking_id = find_tracking_id(command)
        if not tracking_id or tracking_id not in PROCESS_TRACKING:
            return
        
        initial_state = PROCESS_TRACKING.pop(tracking_id)
        end_time = time.time()
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(initial_state, end_time)
        
        # Learn patterns
        if SECURE_MODE and STORAGE:
            learn_secure_performance_patterns(metrics)
        else:
            learn_basic_performance_patterns(metrics)
        
        # Provide feedback if significant resource usage
        feedback = generate_performance_feedback(metrics)
        if feedback:
            print(json.dumps({
                'block': False,
                'message': feedback
            }))
        
    except Exception:
        pass  # Fail silently

def find_tracking_id(command):
    """Find tracking ID for command (simple heuristic)"""
    # In real implementation, would use better tracking
    # For now, return the most recent tracking ID
    if PROCESS_TRACKING:
        return max(PROCESS_TRACKING.keys())
    return None

def calculate_performance_metrics(initial_state, end_time):
    """Calculate performance metrics"""
    duration = end_time - initial_state['start_time']
    
    try:
        final_cpu = psutil.cpu_percent()
        final_memory = psutil.virtual_memory().used
        final_disk_io = get_disk_io()
        
        metrics = {
            'duration': duration,
            'cpu_usage': final_cpu - initial_state['cpu_percent'],
            'memory_delta': final_memory - initial_state['memory_used'],
            'disk_io_delta': final_disk_io - initial_state['disk_io'],
            'command': initial_state['command'],
            'timestamp': time.time()
        }
        
        if 'command_abstract' in initial_state:
            metrics['command_abstract'] = initial_state['command_abstract']
        
        return metrics
        
    except Exception:
        return {
            'duration': duration,
            'command': initial_state['command'],
            'timestamp': time.time()
        }

def get_disk_io():
    """Get current disk I/O stats"""
    try:
        disk_io = psutil.disk_io_counters()
        return disk_io.read_bytes + disk_io.write_bytes if disk_io else 0
    except:
        return 0

def learn_secure_performance_patterns(metrics):
    """Learn performance patterns securely"""
    # Get existing performance data
    perf_data = STORAGE.load_learning_data('performance_patterns', {
        'command_performance': defaultdict(list),
        'resource_patterns': defaultdict(dict),
        'optimization_suggestions': defaultdict(list)
    })
    
    # Abstract command for secure storage
    cmd_abstract = metrics.get('command_abstract', {})
    if not cmd_abstract:
        return
    
    base_command = cmd_abstract.get('base_command', 'unknown')
    complexity = cmd_abstract.get('complexity', 'simple')
    file_types = cmd_abstract.get('file_types', [])
    
    # Store performance pattern
    pattern_key = f"{base_command}_{complexity}"
    perf_data['command_performance'][pattern_key].append({
        'duration': metrics['duration'],
        'memory_delta': metrics.get('memory_delta', 0),
        'cpu_usage': metrics.get('cpu_usage', 0),
        'timestamp': metrics['timestamp']
    })
    
    # Learn file type patterns
    for file_type in file_types:
        if file_type in ['.fastq', '.bam', '.vcf', '.fasta']:
            # Genomics file - typically memory/IO intensive
            perf_data['resource_patterns'][f"genomics{file_type}"] = {
                'expected_memory': 'high',
                'expected_duration': 'long',
                'optimization': 'consider_hpc'
            }
        elif file_type in ['.R', '.Rmd']:
            # R files - can be memory intensive
            perf_data['resource_patterns'][f"r_computing{file_type}"] = {
                'expected_memory': 'medium',
                'expected_duration': 'medium',
                'optimization': 'consider_parallel'
            }
    
    # Generate optimization suggestions based on patterns
    if metrics['duration'] > 60:  # Long running command
        suggestion_key = f"{base_command}_optimization"
        if metrics.get('memory_delta', 0) > 1024 * 1024 * 1024:  # >1GB memory
            perf_data['optimization_suggestions'][suggestion_key].append({
                'type': 'memory_intensive',
                'suggestion': 'Consider using HPC cluster for memory-intensive tasks',
                'timestamp': metrics['timestamp']
            })
        
        if any(ft in ['.fastq', '.bam'] for ft in file_types):
            perf_data['optimization_suggestions'][suggestion_key].append({
                'type': 'genomics_workflow',
                'suggestion': 'Consider using Singularity containers for genomics workflows',
                'timestamp': metrics['timestamp']
            })
    
    # Keep only recent data (last 100 entries per pattern)
    for pattern in perf_data['command_performance']:
        perf_data['command_performance'][pattern] = \
            perf_data['command_performance'][pattern][-100:]
    
    # Store securely
    STORAGE.store_learning_data('performance_patterns', dict(perf_data))

def learn_basic_performance_patterns(metrics):
    """Basic performance pattern learning (fallback)"""
    # Simple file-based learning for basic mode
    perf_file = Path.home() / '.claude' / 'performance_patterns.json'
    
    try:
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                perf_data = json.load(f)
        else:
            perf_data = {'patterns': []}
        
        # Add new pattern
        perf_data['patterns'].append({
            'command': metrics['command'][:50],  # Truncate for privacy
            'duration': metrics['duration'],
            'timestamp': metrics['timestamp']
        })
        
        # Keep only last 50 patterns
        perf_data['patterns'] = perf_data['patterns'][-50:]
        
        with open(perf_file, 'w') as f:
            json.dump(perf_data, f)
            
    except Exception:
        pass

def generate_performance_feedback(metrics):
    """Generate performance feedback for Claude"""
    feedback_parts = []
    
    # Duration feedback
    if metrics['duration'] > 300:  # 5 minutes
        feedback_parts.append("⏱️ **Long-running command detected** (>5min)")
        
        # Check if it's a genomics workflow
        if any(pattern in metrics['command'].lower() 
               for pattern in ['fastq', 'bam', 'vcf', 'fasta', 'genomics']):
            feedback_parts.append("🧬 Genomics workflow → Consider using HPC cluster")
        
        # Check if it's R computing
        if 'R ' in metrics['command'] or '.R' in metrics['command']:
            feedback_parts.append("📊 R computing → Consider parallel processing or more memory")
    
    # Memory feedback
    memory_delta = metrics.get('memory_delta', 0)
    if memory_delta > 2 * 1024 * 1024 * 1024:  # >2GB
        feedback_parts.append("🧠 **High memory usage** (>2GB)")
        feedback_parts.append("💡 Consider using a high-memory cluster node")
    
    # HPC suggestions
    if metrics['duration'] > 60 and any(term in metrics['command'].lower() 
                                       for term in ['sbatch', 'squeue', 'slurm']):
        feedback_parts.append("🚀 **HPC job detected** → Monitor with `squeue` for progress")
    
    if feedback_parts:
        return "\n".join(feedback_parts)
    
    return None

def get_performance_suggestions(command):
    """Get performance suggestions based on learned patterns"""
    if not SECURE_MODE or not STORAGE:
        return None
    
    perf_data = STORAGE.load_learning_data('performance_patterns', {})
    if not perf_data:
        return None
    
    # Abstract current command
    cmd_abstract = ABSTRACTOR.abstract_command(command)
    base_command = cmd_abstract.get('base_command', '')
    file_types = cmd_abstract.get('file_types', [])
    
    suggestions = []
    
    # Check for known patterns
    for pattern_key, performances in perf_data.get('command_performance', {}).items():
        if base_command in pattern_key and performances:
            avg_duration = sum(p['duration'] for p in performances[-10:]) / len(performances[-10:])
            if avg_duration > 60:
                suggestions.append(f"⏱️ `{base_command}` typically takes {avg_duration:.1f}s")
    
    # Check optimization suggestions
    for opt_key, opts in perf_data.get('optimization_suggestions', {}).items():
        if base_command in opt_key and opts:
            latest_opt = opts[-1]
            suggestions.append(f"💡 {latest_opt['suggestion']}")
    
    if suggestions:
        return "🎯 **Performance insights:**\n" + "\n".join(f"  {s}" for s in suggestions)
    
    return None

if __name__ == '__main__':
    main()