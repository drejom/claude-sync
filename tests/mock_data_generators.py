#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "typing-extensions>=4.0.0"
# ]
# ///
"""
Mock Data Generators for Claude-Sync Testing

Comprehensive mock data generators that create realistic Claude Code hook inputs
for testing all components of the claude-sync system.

This module provides:
- Realistic hook input data for PreToolUse, PostToolUse, and UserPromptSubmit
- Common HPC/bioinformatics command patterns
- Various execution contexts and environments
- Performance and error scenarios
"""

import json
import time
import random
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from interfaces import CommandExecutionData, InformationTypes, AgentNames

# ============================================================================
# Mock Data Configuration
# ============================================================================

# Common HPC and bioinformatics commands
HPC_COMMANDS = [
    "sbatch --partition=compute --time=04:00:00 --mem=16GB analysis.sh",
    "squeue -u $USER",
    "scancel 12345",
    "sbatch --partition=gpu --gres=gpu:1 --mem=32GB deep_learning.py",
    "sinfo -p compute",
    "sacct -j 12345 --format=JobID,JobName,State,ExitCode,MaxRSS",
    "sbatch --array=1-100 --partition=compute array_job.sh",
    "srun --partition=interactive --time=1:00:00 --pty bash"
]

R_COMMANDS = [
    "Rscript analysis.R",
    "Rscript --vanilla --max-mem-size=8G genomics_analysis.R",
    "R CMD BATCH analysis.R",
    "Rscript -e 'install.packages(\"ggplot2\")'",
    "Rscript differential_expression.R --input data.csv --output results/",
    "R --slave --vanilla < pipeline.R > pipeline.log 2>&1"
]

CONTAINER_COMMANDS = [
    "singularity exec biotools.sif python analysis.py",
    "singularity exec --bind /data:/data,/scratch:/scratch r-analysis.sif Rscript pipeline.R",
    "singularity run --nv gpu-pytorch.sif python train_model.py",
    "singularity shell biotools.sif",
    "singularity exec --containall clean-env.sif ./process_data.sh"
]

SSH_COMMANDS = [
    "ssh hpc-login01 'sbatch job.sh'",
    "ssh -J jumphost compute-node-01 'ps aux | grep python'",
    "ssh gpu-cluster 'nvidia-smi'",
    "scp large_dataset.tar.gz hpc:/scratch/",
    "rsync -avz --progress data/ hpc:/data/project/",
    "ssh -L 8888:localhost:8888 hpc-login 'jupyter notebook --no-browser'"
]

DATA_PROCESSING_COMMANDS = [
    "grep -r 'AGATCTCGGAA' /data/fastq/",
    "find /data -name '*.bam' -exec samtools index {} \\;",
    "awk '{print $1,$3}' results.txt | sort | uniq -c",
    "sed 's/old_pattern/new_pattern/g' file.txt > file_corrected.txt",
    "cut -f1,3,5 data.tsv | head -1000",
    "sort -k2,2nr results.txt | head -20"
]

NETWORK_COMMANDS = [
    "tailscale status",
    "tailscale ping hpc-cluster",
    "curl -s https://api.example.com/data | jq '.'",
    "wget --continue https://data.repo.com/large_file.tar.gz",
    "ping -c 4 8.8.8.8",
    "netstat -tulpn | grep :22"
]

# Host types and their characteristics
HOST_TYPES = {
    'hpc-login': {
        'type': 'login_node',
        'capabilities': ['slurm_client', 'ssh_gateway', 'file_storage'],
        'performance_tier': 'medium',
        'resource_limits': {'cpu_cores': 8, 'memory_gb': 32, 'storage_gb': 1000}
    },
    'hpc-compute': {
        'type': 'compute_node',
        'capabilities': ['high_cpu', 'slurm_worker', 'singularity'],
        'performance_tier': 'high',
        'resource_limits': {'cpu_cores': 48, 'memory_gb': 256, 'storage_gb': 500}
    },
    'hpc-gpu': {
        'type': 'gpu_node',
        'capabilities': ['gpu_compute', 'cuda', 'tensorflow', 'pytorch'],
        'performance_tier': 'high',
        'resource_limits': {'cpu_cores': 24, 'memory_gb': 128, 'gpu_count': 4, 'storage_gb': 1000}
    },
    'workstation': {
        'type': 'local_workstation',
        'capabilities': ['development', 'data_analysis', 'visualization'],
        'performance_tier': 'medium',
        'resource_limits': {'cpu_cores': 16, 'memory_gb': 64, 'storage_gb': 2000}
    },
    'laptop': {
        'type': 'mobile_device',
        'capabilities': ['development', 'ssh_client', 'light_analysis'],
        'performance_tier': 'low',
        'resource_limits': {'cpu_cores': 8, 'memory_gb': 16, 'storage_gb': 500}
    }
}

# File system patterns
FILESYSTEM_PATTERNS = {
    'genomics_data': ['/data/genomics/', '/scratch/seq_data/', '/project/genomes/'],
    'analysis_results': ['/results/', '/output/', '/analysis_output/'],
    'scratch_space': ['/scratch/', '/tmp/', '/local_scratch/'],
    'home_directory': ['/home/user/', '/users/researcher/', '~/'],
    'project_data': ['/project/', '/shared/', '/group_data/']
}

# ============================================================================
# Mock Data Generators
# ============================================================================

class MockDataGenerator:
    """Base class for generating mock testing data"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.session_id = str(uuid.uuid4())
    
    def random_choice_weighted(self, choices: List[Tuple[Any, float]]) -> Any:
        """Choose randomly with weights"""
        total_weight = sum(weight for _, weight in choices)
        rand_val = random.uniform(0, total_weight)
        
        current_weight = 0
        for choice, weight in choices:
            current_weight += weight
            if rand_val <= current_weight:
                return choice
        
        return choices[0][0]  # Fallback

class HookInputGenerator(MockDataGenerator):
    """Generate realistic Claude Code hook inputs"""
    
    def generate_pretooluse_input(self, 
                                 command_type: str = "random",
                                 tool_name: str = "Bash") -> Dict[str, Any]:
        """Generate PreToolUse hook input"""
        
        # Select command based on type
        if command_type == "random":
            command_type = random.choice(['hpc', 'r_analysis', 'container', 'ssh', 'data_processing', 'network'])
        
        command_map = {
            'hpc': HPC_COMMANDS,
            'r_analysis': R_COMMANDS,
            'container': CONTAINER_COMMANDS,
            'ssh': SSH_COMMANDS,
            'data_processing': DATA_PROCESSING_COMMANDS,
            'network': NETWORK_COMMANDS
        }
        
        command = random.choice(command_map.get(command_type, HPC_COMMANDS))
        
        # Generate context
        host_info = random.choice(list(HOST_TYPES.items()))
        host_name, host_specs = host_info
        
        working_dir = random.choice([
            '/home/user/projects/genomics',
            '/scratch/analysis_run_001',
            '/data/project_alpha',
            '/users/researcher/current_work',
            '/project/shared/pipeline'
        ])
        
        return {
            'hook_type': 'PreToolUse',
            'tool_name': tool_name,
            'tool_input': {
                'command': command,
                'description': f'Execute {command_type} command',
                'timeout': random.choice([60000, 120000, 300000, 600000])  # ms
            },
            'context': {
                'session_id': self.session_id,
                'timestamp': time.time(),
                'working_directory': working_dir,
                'user_id': 'test_user',
                'host_info': {
                    'abstract_host_id': f"{host_specs['type']}-{random.randint(1000, 9999)}",
                    'host_type': host_specs['type'],
                    'capabilities': host_specs['capabilities'],
                    'performance_tier': host_specs['performance_tier']
                },
                'environment': {
                    'SLURM_JOB_ID': str(random.randint(100000, 999999)),
                    'CUDA_VISIBLE_DEVICES': '0,1,2,3' if 'gpu' in host_name else None,
                    'PATH': '/usr/local/bin:/usr/bin:/bin'
                }
            },
            'user_prompt': f"Please run this {command_type} command for me"
        }
    
    def generate_posttooluse_input(self, 
                                  command_type: str = "random",
                                  success: bool = True,
                                  execution_time_ms: Optional[int] = None) -> Dict[str, Any]:
        """Generate PostToolUse hook input"""
        
        # Start with PreToolUse input structure
        base_input = self.generate_pretooluse_input(command_type)
        
        # Add execution results
        if execution_time_ms is None:
            execution_time_ms = self.random_choice_weighted([
                (random.randint(10, 100), 0.4),      # Fast commands
                (random.randint(100, 1000), 0.3),    # Medium commands
                (random.randint(1000, 10000), 0.2),  # Slow commands
                (random.randint(10000, 60000), 0.1)  # Very slow commands
            ])
        
        exit_code = 0 if success else random.choice([1, 2, 127, 130, 255])
        
        # Generate realistic output based on command type
        output_text = self._generate_command_output(base_input['tool_input']['command'], success)
        
        base_input.update({
            'hook_type': 'PostToolUse',
            'tool_output': {
                'exit_code': exit_code,
                'stdout': output_text,
                'stderr': '' if success else f"Error executing command: exit code {exit_code}",
                'duration_ms': execution_time_ms,
                'start_time': time.time() - (execution_time_ms / 1000),
                'end_time': time.time()
            },
            'execution_metrics': {
                'cpu_usage_percent': random.uniform(10, 95),
                'memory_usage_mb': random.randint(100, 2048),
                'disk_io_mb': random.randint(0, 500),
                'network_io_kb': random.randint(0, 1024)
            }
        })
        
        return base_input
    
    def generate_userpromptsubmit_input(self, prompt_type: str = "random") -> Dict[str, Any]:
        """Generate UserPromptSubmit hook input"""
        
        prompt_templates = {
            'hpc_help': [
                "How do I optimize my SLURM jobs for better queue times?",
                "What's the best partition to use for genomics analysis?",
                "My sbatch job keeps failing with memory errors, what should I do?",
                "How can I check the status of my running jobs?"
            ],
            'r_analysis': [
                "Help me optimize this R script for large datasets",
                "My R analysis is running out of memory, what can I do?",
                "What's the best way to parallelize R computations on this cluster?",
                "How do I install R packages on the HPC system?"
            ],
            'container_workflow': [
                "How do I run this Docker container with Singularity?",
                "What bind mounts should I use for my container workflow?",
                "My containerized pipeline is slow, how can I optimize it?",
                "How do I troubleshoot container permissions issues?"
            ],
            'ssh_networking': [
                "I can't connect to the HPC cluster, help me debug",
                "How do I set up SSH tunneling for Jupyter notebooks?",
                "What's the best way to transfer large files between hosts?",
                "My SSH connection keeps dropping, what could be wrong?"
            ],
            'performance_optimization': [
                "This command is running much slower than expected",
                "How can I make my data processing pipeline faster?",
                "What tools should I use instead of grep for large files?",
                "My analysis is taking forever, any suggestions for optimization?"
            ],
            'troubleshooting': [
                "I'm getting strange errors in my pipeline, can you help debug?",
                "This worked yesterday but is failing today, what changed?",
                "My job was killed by the scheduler, what does that mean?",
                "The results look wrong, how do I troubleshoot my analysis?"
            ]
        }
        
        if prompt_type == "random":
            prompt_type = random.choice(list(prompt_templates.keys()))
        
        user_prompt = random.choice(prompt_templates.get(prompt_type, prompt_templates['hpc_help']))
        
        # Add context that would influence what learning data is relevant
        context_hints = []
        if 'slurm' in user_prompt.lower() or 'job' in user_prompt.lower():
            context_hints.append('slurm_patterns')
        if 'r ' in user_prompt.lower() or 'rscript' in user_prompt.lower():
            context_hints.append('r_patterns')
        if 'container' in user_prompt.lower() or 'singularity' in user_prompt.lower():
            context_hints.append('container_patterns')
        if 'ssh' in user_prompt.lower() or 'connect' in user_prompt.lower():
            context_hints.append('network_patterns')
        if 'slow' in user_prompt.lower() or 'optimize' in user_prompt.lower():
            context_hints.append('performance_insights')
        if 'error' in user_prompt.lower() or 'debug' in user_prompt.lower():
            context_hints.append('error_patterns')
        
        return {
            'hook_type': 'UserPromptSubmit',
            'user_prompt': user_prompt,
            'context': {
                'session_id': self.session_id,
                'timestamp': time.time(),
                'working_directory': random.choice([
                    '/home/user/analysis',
                    '/scratch/current_project', 
                    '/data/genomics_study'
                ]),
                'recent_commands': self._generate_recent_commands_context(),
                'context_hints': context_hints,
                'conversation_history': f"User has been working on {prompt_type.replace('_', ' ')} for the past session"
            }
        }
    
    def _generate_command_output(self, command: str, success: bool) -> str:
        """Generate realistic command output"""
        if not success:
            return ""
        
        if command.startswith('sbatch'):
            return f"Submitted batch job {random.randint(100000, 999999)}"
        elif command.startswith('squeue'):
            return """JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
 123456   compute analysis     user  R       1:23      1 compute-001
 123457       gpu ml_train     user PD       0:00      1 (Resources)
 123458   compute  process     user  R      15:42      2 compute-[002-003]"""
        elif command.startswith('scancel'):
            return f"Job {random.randint(100000, 999999)} cancelled"
        elif command.startswith('Rscript'):
            return """Loading required package: ggplot2
Reading data file...
Processing 15,432 rows...
Analysis complete. Results saved to output/"""
        elif 'singularity exec' in command:
            return """Container execution started
Loading dataset...
Running analysis pipeline...
Results written to /data/output/"""
        elif command.startswith('ssh'):
            return "Connection established successfully"
        elif 'grep' in command:
            return f"Found {random.randint(1, 100)} matches in {random.randint(1, 20)} files"
        elif 'find' in command:
            files = [f"/data/file{i}.bam" for i in range(1, random.randint(5, 25))]
            return "\n".join(files)
        else:
            return "Command executed successfully"
    
    def _generate_recent_commands_context(self) -> List[str]:
        """Generate context of recent commands for prompt enhancement"""
        recent_commands = []
        for _ in range(random.randint(3, 8)):
            command_type = random.choice(['hpc', 'r_analysis', 'data_processing'])
            command_map = {
                'hpc': HPC_COMMANDS,
                'r_analysis': R_COMMANDS,
                'data_processing': DATA_PROCESSING_COMMANDS
            }
            recent_commands.append(random.choice(command_map[command_type]))
        return recent_commands

class LearningDataGenerator(MockDataGenerator):
    """Generate mock learning data for testing"""
    
    def generate_command_execution_data(self, count: int = 10) -> List[CommandExecutionData]:
        """Generate multiple command execution data points"""
        hook_generator = HookInputGenerator(seed=42)  # Consistent seed for testing
        data_points = []
        
        for _ in range(count):
            # Generate varied execution scenarios
            success_rate = 0.85  # 85% success rate
            success = random.random() < success_rate
            
            hook_input = hook_generator.generate_posttooluse_input(success=success)
            execution_data = CommandExecutionData.from_hook_input(hook_input)
            data_points.append(execution_data)
        
        return data_points
    
    def generate_optimization_patterns(self, count: int = 5) -> List[Dict[str, Any]]:
        """Generate mock optimization patterns"""
        patterns = []
        
        optimization_examples = [
            {
                'original': 'grep -r "pattern" /data/',
                'optimized': 'rg "pattern" /data/',
                'confidence': 0.95,
                'success_rate': 0.98,
                'category': 'tool_upgrade'
            },
            {
                'original': 'find /data -name "*.bam"',
                'optimized': 'fd "*.bam" /data',
                'confidence': 0.87,
                'success_rate': 0.92,
                'category': 'tool_upgrade'
            },
            {
                'original': 'sbatch --mem=8GB job.sh',
                'optimized': 'sbatch --mem=16GB job.sh',
                'confidence': 0.73,
                'success_rate': 0.89,
                'category': 'resource_optimization'
            }
        ]
        
        for i in range(count):
            base_pattern = optimization_examples[i % len(optimization_examples)]
            patterns.append({
                'pattern_id': f"opt_pattern_{i:03d}",
                'original_pattern': base_pattern['original'],
                'optimized_pattern': base_pattern['optimized'],
                'confidence': base_pattern['confidence'] + random.uniform(-0.1, 0.1),
                'success_rate': base_pattern['success_rate'] + random.uniform(-0.05, 0.05),
                'application_count': random.randint(1, 50),
                'created_at': time.time() - random.randint(0, 86400 * 30),  # Last 30 days
                'last_used': time.time() - random.randint(0, 86400 * 7),   # Last 7 days
                'categories': [base_pattern['category']]
            })
        
        return patterns
    
    def generate_performance_metrics(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate mock performance metrics"""
        metrics = []
        
        for i in range(count):
            metrics.append({
                'timestamp': time.time() - random.randint(0, 86400),  # Last 24 hours
                'operation_type': random.choice(['hook_execution', 'learning_operation', 'security_operation']),
                'duration_ms': random.uniform(1, 100),
                'memory_mb': random.uniform(5, 50),
                'success': random.random() < 0.95,
                'host_type': random.choice(['hpc-login', 'hpc-compute', 'workstation']),
                'context': {
                    'command_complexity': random.randint(1, 10),
                    'data_size_mb': random.randint(1, 1000),
                    'concurrent_operations': random.randint(1, 5)
                }
            })
        
        return metrics

class ErrorScenarioGenerator(MockDataGenerator):
    """Generate error scenarios for comprehensive testing"""
    
    def generate_hook_error_scenarios(self) -> List[Dict[str, Any]]:
        """Generate various hook execution error scenarios"""
        scenarios = []
        
        # Hook execution timeout
        scenarios.append({
            'scenario_name': 'hook_timeout',
            'description': 'Hook execution exceeds time limit',
            'hook_input': HookInputGenerator().generate_pretooluse_input(),
            'expected_behavior': 'graceful_timeout',
            'timeout_ms': 5000,
            'expected_result': {'block': False, 'message': None}
        })
        
        # Invalid hook input
        scenarios.append({
            'scenario_name': 'invalid_hook_input',
            'description': 'Malformed or missing hook input data',
            'hook_input': {'invalid': 'data'},
            'expected_behavior': 'handle_gracefully',
            'expected_result': {'block': False, 'message': None}
        })
        
        # Learning system unavailable
        scenarios.append({
            'scenario_name': 'learning_system_unavailable', 
            'description': 'Learning storage/components not available',
            'hook_input': HookInputGenerator().generate_posttooluse_input(),
            'expected_behavior': 'fallback_gracefully',
            'mock_failures': ['learning_storage', 'threshold_manager'],
            'expected_result': {'block': False, 'message': None}
        })
        
        # Security system failure
        scenarios.append({
            'scenario_name': 'security_system_failure',
            'description': 'Encryption/decryption operations fail',
            'hook_input': HookInputGenerator().generate_posttooluse_input(),
            'expected_behavior': 'fail_safe',
            'mock_failures': ['encryption', 'host_identity'],
            'expected_result': {'block': False, 'message': None}
        })
        
        # Memory pressure
        scenarios.append({
            'scenario_name': 'memory_pressure',
            'description': 'System under memory pressure during hook execution',
            'hook_input': HookInputGenerator().generate_pretooluse_input(),
            'expected_behavior': 'graceful_degradation',
            'memory_limit_mb': 10,
            'expected_result': {'block': False}
        })
        
        return scenarios
    
    def generate_integration_error_scenarios(self) -> List[Dict[str, Any]]:
        """Generate integration error scenarios"""
        scenarios = []
        
        # Database corruption
        scenarios.append({
            'scenario_name': 'learning_data_corruption',
            'description': 'Learning data files are corrupted',
            'setup': 'corrupt_learning_files',
            'expected_behavior': 'rebuild_from_scratch',
            'recovery_strategy': 'automatic'
        })
        
        # Key rotation failure
        scenarios.append({
            'scenario_name': 'key_rotation_failure',
            'description': 'Daily key rotation process fails',
            'setup': 'simulate_key_rotation_failure',
            'expected_behavior': 'continue_with_old_keys',
            'recovery_strategy': 'manual_intervention'
        })
        
        # Network partition
        scenarios.append({
            'scenario_name': 'network_partition',
            'description': 'Network connectivity lost during mesh sync',
            'setup': 'simulate_network_failure',
            'expected_behavior': 'work_offline',
            'recovery_strategy': 'automatic_retry'
        })
        
        return scenarios

# ============================================================================
# Realistic Test Data Sets
# ============================================================================

class RealisticDataSets:
    """Pre-generated realistic test data sets for consistent testing"""
    
    @staticmethod
    def bioinformatics_workflow() -> List[Dict[str, Any]]:
        """Generate a realistic bioinformatics workflow sequence"""
        generator = HookInputGenerator(seed=12345)
        
        workflow_steps = [
            generator.generate_pretooluse_input("ssh"),  # Connect to cluster
            generator.generate_pretooluse_input("hpc"),  # Submit preprocessing job
            generator.generate_posttooluse_input("hpc", success=True, execution_time_ms=5000),
            generator.generate_pretooluse_input("container"),  # Run analysis in container
            generator.generate_posttooluse_input("container", success=True, execution_time_ms=15000),
            generator.generate_pretooluse_input("r_analysis"),  # Statistical analysis
            generator.generate_posttooluse_input("r_analysis", success=True, execution_time_ms=30000),
            generator.generate_userpromptsubmit_input("performance_optimization")  # User asks for help
        ]
        
        return workflow_steps
    
    @staticmethod
    def hpc_troubleshooting_session() -> List[Dict[str, Any]]:
        """Generate an HPC troubleshooting session with failures"""
        generator = HookInputGenerator(seed=54321)
        
        troubleshooting_steps = [
            generator.generate_pretooluse_input("hpc"),  # Submit job
            generator.generate_posttooluse_input("hpc", success=False, execution_time_ms=100),  # Job fails
            generator.generate_userpromptsubmit_input("troubleshooting"),  # User asks for help
            generator.generate_pretooluse_input("hpc"),  # Try with more memory
            generator.generate_posttooluse_input("hpc", success=True, execution_time_ms=8000),  # Success
        ]
        
        return troubleshooting_steps
    
    @staticmethod
    def performance_degradation_scenario() -> List[Dict[str, Any]]:
        """Generate scenario showing performance degradation over time"""
        generator = HookInputGenerator(seed=98765)
        
        # Same command with increasing execution times
        base_command = "grep -r 'AGATCTCGGAA' /data/fastq/"
        execution_times = [1500, 2000, 3500, 5000, 8000]  # Gradually slower
        
        scenario = []
        for i, exec_time in enumerate(execution_times):
            scenario.append(generator.generate_posttooluse_input(
                "data_processing", 
                success=True, 
                execution_time_ms=exec_time
            ))
        
        # User notices and asks for help
        scenario.append(generator.generate_userpromptsubmit_input("performance_optimization"))
        
        return scenario

# ============================================================================
# Test Data Validation
# ============================================================================

def validate_hook_input(hook_input: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate that generated hook input follows expected structure"""
    errors = []
    
    # Check required fields
    if 'hook_type' not in hook_input:
        errors.append("Missing 'hook_type' field")
    
    if 'context' not in hook_input:
        errors.append("Missing 'context' field")
    
    # Validate based on hook type
    hook_type = hook_input.get('hook_type')
    
    if hook_type == 'PreToolUse':
        if 'tool_name' not in hook_input:
            errors.append("PreToolUse missing 'tool_name'")
        if 'tool_input' not in hook_input:
            errors.append("PreToolUse missing 'tool_input'")
    
    elif hook_type == 'PostToolUse':
        if 'tool_output' not in hook_input:
            errors.append("PostToolUse missing 'tool_output'")
    
    elif hook_type == 'UserPromptSubmit':
        if 'user_prompt' not in hook_input:
            errors.append("UserPromptSubmit missing 'user_prompt'")
    
    # Validate context structure
    context = hook_input.get('context', {})
    required_context_fields = ['session_id', 'timestamp', 'working_directory']
    
    for field in required_context_fields:
        if field not in context:
            errors.append(f"Context missing '{field}'")
    
    return len(errors) == 0, errors

# ============================================================================
# Main Testing Function
# ============================================================================

def main():
    """Test the mock data generators"""
    print("🧪 Testing Mock Data Generators")
    print("=" * 50)
    
    # Test hook input generation
    generator = HookInputGenerator(seed=42)
    
    print("📝 Testing PreToolUse input generation...")
    pretool_input = generator.generate_pretooluse_input("hpc")
    valid, errors = validate_hook_input(pretool_input)
    print(f"✅ Valid: {valid} | Errors: {len(errors)}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    print("📝 Testing PostToolUse input generation...")
    posttool_input = generator.generate_posttooluse_input("r_analysis")
    valid, errors = validate_hook_input(posttool_input)
    print(f"✅ Valid: {valid} | Errors: {len(errors)}")
    
    print("📝 Testing UserPromptSubmit input generation...")
    prompt_input = generator.generate_userpromptsubmit_input("hpc_help")
    valid, errors = validate_hook_input(prompt_input)
    print(f"✅ Valid: {valid} | Errors: {len(errors)}")
    
    # Test realistic data sets
    print("\n📊 Testing realistic data sets...")
    bio_workflow = RealisticDataSets.bioinformatics_workflow()
    print(f"✅ Bioinformatics workflow: {len(bio_workflow)} steps")
    
    hpc_troubleshooting = RealisticDataSets.hpc_troubleshooting_session()
    print(f"✅ HPC troubleshooting: {len(hpc_troubleshooting)} steps")
    
    perf_degradation = RealisticDataSets.performance_degradation_scenario()
    print(f"✅ Performance degradation: {len(perf_degradation)} steps")
    
    # Test learning data generation
    print("\n🧠 Testing learning data generation...")
    learning_gen = LearningDataGenerator(seed=42)
    execution_data = learning_gen.generate_command_execution_data(5)
    print(f"✅ Command execution data: {len(execution_data)} points")
    
    optimization_patterns = learning_gen.generate_optimization_patterns(3)
    print(f"✅ Optimization patterns: {len(optimization_patterns)} patterns")
    
    # Test error scenarios
    print("\n❌ Testing error scenario generation...")
    error_gen = ErrorScenarioGenerator(seed=42)
    hook_errors = error_gen.generate_hook_error_scenarios()
    print(f"✅ Hook error scenarios: {len(hook_errors)} scenarios")
    
    integration_errors = error_gen.generate_integration_error_scenarios()
    print(f"✅ Integration error scenarios: {len(integration_errors)} scenarios")
    
    print("\n🎯 Mock data generation testing complete!")

if __name__ == "__main__":
    main()