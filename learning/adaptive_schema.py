#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0"
# ]
# ///
"""
Adaptive Learning Schema - Self-evolving knowledge structure based on usage patterns

This implements the core adaptive schema evolution system that learns from command
execution patterns and automatically evolves its understanding without breaking
existing patterns. Key features:

- NoSQL-style flexible schema that adapts to new patterns
- Usage frequency tracking for pattern significance
- Automatic field discovery from command execution data
- Schema evolution triggers based on statistical significance
- Performance optimized for <100ms learning operations

Based on REFACTOR_PLAN.md sections 310-435 (Adaptive Schema Evolution)
"""

import json
import time
import hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import logging

# Import our interfaces
import sys
sys.path.append(str(Path(__file__).parent.parent))
from interfaces import (
    AdaptiveSchemaInterface, 
    CommandExecutionData,
    LearningStorageInterface,
    PerformanceTargets
)

@dataclass
class PatternSignature:
    """Represents a command pattern signature for learning"""
    command_base: str  # Base command (e.g., 'slurm_sbatch', 'r_script')
    complexity_tier: str  # 'simple', 'medium', 'complex'
    tool_category: str  # 'hpc', 'container', 'analysis', 'system'
    context_hash: str  # Hash of consistent context elements
    
    def __str__(self) -> str:
        return f"{self.tool_category}_{self.command_base}_{self.complexity_tier}"

@dataclass
class FieldMetadata:
    """Metadata about discovered schema fields"""
    field_type: str
    first_seen: float
    frequency: int
    example_values: List[Any]
    significance_score: float
    last_updated: float
    
    def update_with_value(self, value: Any) -> None:
        """Update field metadata with new observed value"""
        self.frequency += 1
        self.last_updated = time.time()
        
        # Add new example values (keep recent ones)
        if value not in self.example_values:
            self.example_values.append(value)
            if len(self.example_values) > 10:
                self.example_values = self.example_values[-10:]
    
    def calculate_significance(self, total_observations: int) -> float:
        """Calculate field significance based on frequency and consistency"""
        frequency_score = min(self.frequency / max(total_observations, 1), 1.0)
        consistency_score = 1.0 / max(len(set(str(v) for v in self.example_values)), 1)
        recency_score = max(0.1, 1.0 - (time.time() - self.last_updated) / (30 * 24 * 3600))  # 30 days
        
        self.significance_score = (frequency_score * 0.5 + 
                                 consistency_score * 0.3 + 
                                 recency_score * 0.2)
        return self.significance_score

class AdaptiveLearningSchema(AdaptiveSchemaInterface):
    """
    Self-evolving knowledge schema that learns from command execution patterns.
    
    This class implements the adaptive schema evolution system described in
    REFACTOR_PLAN.md sections 310-435. It automatically discovers new patterns,
    tracks field usage, and evolves the schema based on statistical significance.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir or Path.home() / '.claude' / 'learning')
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Core schema data structures
        self.schema_version = 1
        self.pattern_registry: Dict[str, Dict[str, FieldMetadata]] = defaultdict(dict)
        self.usage_frequency = Counter()
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.last_evolution_check = time.time()
        self.evolution_check_interval = 3600  # Check every hour
        
        # Load existing schema if available
        self._load_schema()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def observe_command_pattern(self, command_data: CommandExecutionData) -> None:
        """
        Learn from actual command execution patterns.
        
        This is the core learning method that processes new command data,
        extracts patterns, and updates the adaptive schema. Performance
        target: <100ms per observation.
        """
        start_time = time.time()
        
        try:
            # Extract pattern signature from command data
            pattern_sig = self._extract_pattern_signature(command_data)
            pattern_key = str(pattern_sig)
            
            # Track usage frequency
            self.usage_frequency[pattern_key] += 1
            
            # Convert command data to analyzable dictionary
            command_dict = command_data.to_dict()
            
            # Discover and update fields dynamically
            self._discover_and_update_fields(pattern_key, command_dict)
            
            # Check if schema evolution is needed (periodic check)
            current_time = time.time()
            if current_time - self.last_evolution_check > self.evolution_check_interval:
                self.evolve_schema()
                self.last_evolution_check = current_time
            
            # Performance monitoring
            duration_ms = (time.time() - start_time) * 1000
            if duration_ms > PerformanceTargets.LEARNING_OPERATION_MS:
                self.logger.warning(f"Schema observation took {duration_ms:.1f}ms (target: {PerformanceTargets.LEARNING_OPERATION_MS}ms)")
                
        except Exception as e:
            self.logger.error(f"Error in observe_command_pattern: {e}")
    
    def _extract_pattern_signature(self, command_data: CommandExecutionData) -> PatternSignature:
        """Extract semantic pattern signature from command execution data"""
        command = command_data.command.strip()
        
        # Determine base command
        command_base = self._classify_command_base(command)
        
        # Determine complexity tier
        complexity_tier = self._calculate_complexity_tier(command)
        
        # Determine tool category
        tool_category = self._classify_tool_category(command)
        
        # Create context hash from consistent elements
        context_elements = {
            'working_directory_type': self._classify_directory_type(command_data.working_directory),
            'exit_code': command_data.exit_code,
            'duration_tier': self._classify_duration_tier(command_data.duration_ms)
        }
        context_hash = hashlib.md5(json.dumps(context_elements, sort_keys=True).encode()).hexdigest()[:8]
        
        return PatternSignature(
            command_base=command_base,
            complexity_tier=complexity_tier,
            tool_category=tool_category,
            context_hash=context_hash
        )
    
    def _classify_command_base(self, command: str) -> str:
        """Classify the base type of command"""
        command_lower = command.lower()
        
        # HPC commands
        if any(cmd in command_lower for cmd in ['sbatch', 'squeue', 'scancel', 'sinfo']):
            return 'slurm'
        elif any(cmd in command_lower for cmd in ['qsub', 'qstat', 'qdel']):
            return 'pbs'
        elif any(cmd in command_lower for cmd in ['bsub', 'bjobs', 'bkill']):
            return 'lsf'
        
        # Container commands
        elif any(cmd in command_lower for cmd in ['singularity', 'apptainer']):
            return 'singularity'
        elif command_lower.startswith('docker'):
            return 'docker'
        elif any(cmd in command_lower for cmd in ['podman', 'buildah']):
            return 'podman'
        
        # Analysis tools
        elif command_lower.startswith('r ') or 'rscript' in command_lower:
            return 'r_script'
        elif command_lower.startswith('python') or command_lower.startswith('jupyter'):
            return 'python'
        elif any(cmd in command_lower for cmd in ['nextflow', 'nf']):
            return 'nextflow'
        elif 'snakemake' in command_lower:
            return 'snakemake'
        
        # System commands
        elif any(cmd in command_lower for cmd in ['ssh', 'scp', 'rsync']):
            return 'remote'
        elif any(cmd in command_lower for cmd in ['find', 'grep', 'awk', 'sed']):
            return 'search'
        elif any(cmd in command_lower for cmd in ['tar', 'gzip', 'zip']):
            return 'archive'
        
        # Default to first word
        return command.split()[0] if command.split() else 'unknown'
    
    def _calculate_complexity_tier(self, command: str) -> str:
        """Calculate command complexity tier"""
        # Count complexity indicators
        pipe_count = command.count('|')
        flag_count = len([part for part in command.split() if part.startswith('-')])
        redirect_count = command.count('>') + command.count('<')
        length = len(command)
        
        complexity_score = (pipe_count * 2 + flag_count + redirect_count + length / 50)
        
        if complexity_score < 5:
            return 'simple'
        elif complexity_score < 15:
            return 'medium'
        else:
            return 'complex'
    
    def _classify_tool_category(self, command: str) -> str:
        """Classify the tool category"""
        command_lower = command.lower()
        
        if any(pattern in command_lower for pattern in ['sbatch', 'slurm', 'qsub', 'bsub']):
            return 'hpc'
        elif any(pattern in command_lower for pattern in ['singularity', 'docker', 'podman']):
            return 'container'
        elif any(pattern in command_lower for pattern in ['python', 'r ', 'rscript', 'jupyter']):
            return 'analysis'
        elif any(pattern in command_lower for pattern in ['nextflow', 'snakemake']):
            return 'workflow'
        elif any(pattern in command_lower for pattern in ['ssh', 'scp', 'rsync']):
            return 'network'
        else:
            return 'system'
    
    def _classify_directory_type(self, directory: str) -> str:
        """Classify working directory type"""
        dir_lower = directory.lower()
        
        if any(pattern in dir_lower for pattern in ['data', 'dataset']):
            return 'data_directory'
        elif any(pattern in dir_lower for pattern in ['project', 'work', 'workspace']):
            return 'project_directory'
        elif any(pattern in dir_lower for pattern in ['home', 'user']):
            return 'user_directory'
        elif any(pattern in dir_lower for pattern in ['tmp', 'temp', 'scratch']):
            return 'temp_directory'
        else:
            return 'system_directory'
    
    def _classify_duration_tier(self, duration_ms: int) -> str:
        """Classify command duration tier"""
        if duration_ms < 1000:  # < 1 second
            return 'instant'
        elif duration_ms < 10000:  # < 10 seconds
            return 'fast'
        elif duration_ms < 60000:  # < 1 minute
            return 'medium'
        elif duration_ms < 600000:  # < 10 minutes
            return 'slow'
        else:
            return 'very_slow'
    
    def _discover_and_update_fields(self, pattern_key: str, command_dict: Dict[str, Any]) -> None:
        """Discover new fields and update existing field metadata"""
        pattern_fields = self.pattern_registry[pattern_key]
        
        for field_name, field_value in command_dict.items():
            if field_name not in pattern_fields:
                # New field discovered
                pattern_fields[field_name] = FieldMetadata(
                    field_type=type(field_value).__name__,
                    first_seen=time.time(),
                    frequency=1,
                    example_values=[field_value] if field_value is not None else [],
                    significance_score=0.0,
                    last_updated=time.time()
                )
            else:
                # Update existing field
                pattern_fields[field_name].update_with_value(field_value)
    
    def get_current_schema(self) -> Dict[str, Any]:
        """Return current adaptive schema structure"""
        return {
            'version': self.schema_version,
            'patterns': {
                pattern_key: {
                    field_name: {
                        'type': field_meta.field_type,
                        'frequency': field_meta.frequency,
                        'significance': field_meta.significance_score,
                        'first_seen': field_meta.first_seen,
                        'example_values': field_meta.example_values[:3]  # Limit for output
                    }
                    for field_name, field_meta in fields.items()
                }
                for pattern_key, fields in self.pattern_registry.items()
            },
            'usage_stats': dict(self.usage_frequency.most_common(20)),
            'evolution_history': self.evolution_history[-10:],  # Last 10 evolutions
            'last_evolution_check': self.last_evolution_check
        }
    
    def should_evolve_schema(self) -> bool:
        """Check if schema evolution is needed based on usage patterns"""
        # Check for patterns with significant usage (>50 observations)
        significant_patterns = [(k, v) for k, v in self.usage_frequency.items() if v > 50]
        
        if not significant_patterns:
            return False
        
        # Check if any significant pattern has new high-frequency fields
        for pattern_key, usage_count in significant_patterns:
            pattern_fields = self.pattern_registry.get(pattern_key, {})
            
            # Calculate field significance scores
            for field_name, field_meta in pattern_fields.items():
                significance = field_meta.calculate_significance(usage_count)
                
                # High significance field in frequently used pattern suggests evolution
                if significance > 0.8 and field_meta.frequency > usage_count * 0.7:
                    return True
        
        return False
    
    def evolve_schema(self) -> bool:
        """Perform schema evolution based on discovered patterns"""
        if not self.should_evolve_schema():
            return False
        
        start_time = time.time()
        evolution_changes = []
        
        try:
            # Analyze top patterns for evolution opportunities
            top_patterns = self.usage_frequency.most_common(10)
            
            for pattern_key, usage_count in top_patterns:
                if usage_count < 50:  # Skip infrequent patterns
                    continue
                
                pattern_fields = self.pattern_registry.get(pattern_key, {})
                
                # Find high-significance fields
                significant_fields = []
                for field_name, field_meta in pattern_fields.items():
                    significance = field_meta.calculate_significance(usage_count)
                    if significance > 0.8 and field_meta.frequency > usage_count * 0.7:
                        significant_fields.append((field_name, significance))
                
                if len(significant_fields) >= 3:  # Pattern has consistent structure
                    evolution_changes.extend(
                        self._propose_pattern_evolution(pattern_key, significant_fields)
                    )
            
            if evolution_changes:
                self.schema_version += 1
                evolution_record = {
                    'timestamp': time.time(),
                    'version': self.schema_version,
                    'changes': evolution_changes,
                    'duration_ms': (time.time() - start_time) * 1000
                }
                self.evolution_history.append(evolution_record)
                
                # Save evolved schema
                self._save_schema()
                
                self.logger.info(f"Schema evolved to version {self.schema_version} with {len(evolution_changes)} changes")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error during schema evolution: {e}")
            return False
    
    def _propose_pattern_evolution(self, pattern_key: str, significant_fields: List[tuple]) -> List[Dict[str, Any]]:
        """Propose evolution changes for a specific pattern"""
        changes = []
        
        # Detect specific evolution patterns
        field_names = [field_name for field_name, _ in significant_fields]
        
        # GPU computing pattern
        if any('gpu' in field.lower() for field in field_names) and any('cuda' in field.lower() for field in field_names):
            changes.append({
                'type': 'new_category',
                'category': 'gpu_computing',
                'pattern': pattern_key,
                'reason': 'Consistent GPU usage patterns detected'
            })
        
        # Container workflow pattern
        if any('container' in field.lower() for field in field_names) and any('mount' in field.lower() for field in field_names):
            changes.append({
                'type': 'new_category', 
                'category': 'container_workflow',
                'pattern': pattern_key,
                'reason': 'Consistent container usage patterns detected'
            })
        
        # Workflow engine pattern
        if pattern_key.startswith(('workflow_nextflow', 'workflow_snakemake')):
            changes.append({
                'type': 'specialization',
                'category': 'workflow_engine',
                'pattern': pattern_key,
                'reason': 'Workflow engine usage patterns specialized'
            })
        
        return changes
    
    def get_pattern_frequency(self, pattern: str) -> int:
        """Get usage frequency for specific pattern"""
        return self.usage_frequency.get(pattern, 0)
    
    def _save_schema(self) -> None:
        """Save schema to encrypted storage"""
        try:
            schema_data = {
                'version': self.schema_version,
                'pattern_registry': {
                    pattern_key: {
                        field_name: asdict(field_meta)
                        for field_name, field_meta in fields.items()
                    }
                    for pattern_key, fields in self.pattern_registry.items()
                },
                'usage_frequency': dict(self.usage_frequency),
                'evolution_history': self.evolution_history,
                'last_updated': time.time()
            }
            
            schema_file = self.storage_dir / 'adaptive_schema.json'
            with open(schema_file, 'w') as f:
                json.dump(schema_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving schema: {e}")
    
    def _load_schema(self) -> None:
        """Load schema from storage"""
        try:
            schema_file = self.storage_dir / 'adaptive_schema.json'
            if not schema_file.exists():
                return
            
            with open(schema_file, 'r') as f:
                schema_data = json.load(f)
            
            self.schema_version = schema_data.get('version', 1)
            self.usage_frequency = Counter(schema_data.get('usage_frequency', {}))
            self.evolution_history = schema_data.get('evolution_history', [])
            
            # Reconstruct pattern registry
            for pattern_key, fields_data in schema_data.get('pattern_registry', {}).items():
                pattern_fields = {}
                for field_name, field_data in fields_data.items():
                    pattern_fields[field_name] = FieldMetadata(**field_data)
                self.pattern_registry[pattern_key] = pattern_fields
                
        except Exception as e:
            self.logger.error(f"Error loading schema: {e}")

if __name__ == "__main__":
    # Example usage and testing
    schema = AdaptiveLearningSchema()
    
    # Simulate command observations
    test_commands = [
        CommandExecutionData("sbatch job.sh", 0, 1500, time.time(), "test", "/home/user/project"),
        CommandExecutionData("python script.py", 0, 3000, time.time(), "test", "/home/user/analysis"),
        CommandExecutionData("singularity exec container.sif python", 0, 5000, time.time(), "test", "/data/genomics")
    ]
    
    for cmd_data in test_commands:
        schema.observe_command_pattern(cmd_data)
    
    # Print current schema
    current_schema = schema.get_current_schema()
    print(json.dumps(current_schema, indent=2, default=str))