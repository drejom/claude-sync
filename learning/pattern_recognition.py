#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0"
# ]
# ///
"""
Pattern Recognition - Workflow sequence analysis and optimization patterns

This implements advanced workflow pattern recognition that goes beyond individual
command analysis to identify multi-step workflow sequences, recurring patterns,
and optimization opportunities across command sequences. Key features:

- Workflow sequence detection and classification
- Multi-command pattern analysis with temporal relationships
- Success correlation analysis across workflow steps
- Optimization recommendations for workflow sequences
- Integration with adaptive schema for pattern evolution

Based on REFACTOR_PLAN.md workflow pattern analysis requirements
"""

import json
import time
import hashlib
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import logging

# Import our interfaces and components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from interfaces import (
    CommandExecutionData,
    OptimizationPattern,
    LearningPattern
)

@dataclass
class WorkflowStep:
    """Individual step in a workflow sequence"""
    step_index: int
    command_category: str
    subcategory: str
    command_signature: str
    duration_ms: int
    success: bool
    timestamp: float
    file_types_involved: List[str]
    resource_usage: Dict[str, Any]

@dataclass
class WorkflowSequence:
    """Detected workflow sequence pattern"""
    sequence_id: str
    workflow_type: str  # 'bioinformatics', 'data_analysis', 'ml_pipeline', etc.
    steps: List[WorkflowStep]
    total_duration_ms: int
    success_rate: float
    frequency: int
    first_seen: float
    last_seen: float
    optimization_opportunities: List[str]
    typical_resource_requirements: Dict[str, Any]

@dataclass
class PatternCorrelation:
    """Correlation between workflow patterns"""
    pattern1_id: str
    pattern2_id: str
    correlation_strength: float
    co_occurrence_frequency: int
    typical_sequence_order: str  # 'pattern1_then_pattern2', 'simultaneous', 'either_order'
    success_correlation: float

class WorkflowPatternRecognizer:
    """
    Advanced workflow pattern recognition system.
    
    This class analyzes sequences of commands to identify recurring workflow
    patterns, success correlations, and optimization opportunities across
    multi-step processes.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir or Path.home() / '.claude' / 'learning')
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Workflow pattern storage
        self.workflow_sequences: Dict[str, WorkflowSequence] = {}
        self.pattern_correlations: Dict[Tuple[str, str], PatternCorrelation] = {}
        
        # Command sequence tracking
        self.active_sessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))  # Last 20 commands per session
        self.session_timeouts: Dict[str, float] = {}
        
        # Pattern detection parameters
        self.sequence_timeout_seconds = 3600  # 1 hour session timeout
        self.min_sequence_length = 3
        self.min_pattern_frequency = 5
        
        # Workflow type classifiers
        self.workflow_classifiers = self._build_workflow_classifiers()
        
        # Load existing patterns
        self._load_patterns()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_command_sequence(self, command_data: CommandExecutionData, command_analysis: Any) -> Dict[str, Any]:
        """
        Analyze command in context of workflow sequences.
        
        This method adds the command to active session tracking and analyzes
        for workflow patterns, returning insights about detected sequences.
        """
        start_time = time.time()
        
        try:
            session_id = command_data.session_id
            current_time = time.time()
            
            # Clean up expired sessions
            self._cleanup_expired_sessions(current_time)
            
            # Create workflow step from command data
            workflow_step = self._create_workflow_step(command_data, command_analysis)
            
            # Add to active session
            self.active_sessions[session_id].append(workflow_step)
            self.session_timeouts[session_id] = current_time + self.sequence_timeout_seconds
            
            # Analyze current sequence for patterns
            sequence_insights = self._analyze_current_sequence(session_id)
            
            # Check for completed workflow sequences
            completed_sequences = self._detect_completed_sequences(session_id)
            
            # Update pattern database
            for sequence in completed_sequences:
                self._register_workflow_sequence(sequence)
            
            # Generate recommendations
            recommendations = self._generate_sequence_recommendations(session_id, workflow_step)
            
            return {
                'workflow_insights': sequence_insights,
                'completed_sequences': [seq.sequence_id for seq in completed_sequences],
                'recommendations': recommendations,
                'active_sequence_length': len(self.active_sessions[session_id]),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing command sequence: {e}")
            return {'error': str(e)}
    
    def get_workflow_optimization_patterns(self, workflow_type: str = None) -> List[OptimizationPattern]:
        """Get optimization patterns for specific workflow types"""
        try:
            patterns = []
            
            for sequence_id, sequence in self.workflow_sequences.items():
                if workflow_type and sequence.workflow_type != workflow_type:
                    continue
                
                if sequence.frequency >= self.min_pattern_frequency:
                    # Convert workflow sequence to optimization patterns
                    for opportunity in sequence.optimization_opportunities:
                        pattern = OptimizationPattern(
                            original_pattern=f"workflow:{sequence.workflow_type}",
                            optimized_pattern=opportunity,
                            confidence=min(sequence.success_rate, 1.0),
                            success_rate=sequence.success_rate,
                            application_count=sequence.frequency,
                            created_at=sequence.first_seen,
                            last_used=sequence.last_seen,
                            categories=[sequence.workflow_type]
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting workflow optimization patterns: {e}")
            return []
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get comprehensive workflow pattern statistics"""
        try:
            stats = {
                'total_sequences': len(self.workflow_sequences),
                'workflow_types': Counter(),
                'average_sequence_length': 0,
                'success_rates_by_type': {},
                'most_common_patterns': [],
                'optimization_opportunities': Counter(),
                'active_sessions': len(self.active_sessions)
            }
            
            if not self.workflow_sequences:
                return stats
            
            # Analyze workflow sequences
            total_steps = 0
            for sequence in self.workflow_sequences.values():
                stats['workflow_types'][sequence.workflow_type] += sequence.frequency
                total_steps += len(sequence.steps) * sequence.frequency
                
                # Track optimization opportunities
                for opportunity in sequence.optimization_opportunities:
                    stats['optimization_opportunities'][opportunity] += sequence.frequency
            
            # Calculate averages
            total_sequences = sum(seq.frequency for seq in self.workflow_sequences.values())
            if total_sequences > 0:
                stats['average_sequence_length'] = total_steps / total_sequences
            
            # Success rates by workflow type
            type_stats = defaultdict(lambda: {'total_freq': 0, 'weighted_success': 0})
            for sequence in self.workflow_sequences.values():
                type_stats[sequence.workflow_type]['total_freq'] += sequence.frequency
                type_stats[sequence.workflow_type]['weighted_success'] += sequence.success_rate * sequence.frequency
            
            for workflow_type, data in type_stats.items():
                if data['total_freq'] > 0:
                    stats['success_rates_by_type'][workflow_type] = data['weighted_success'] / data['total_freq']
            
            # Most common patterns
            pattern_frequencies = [(seq.sequence_id, seq.frequency, seq.workflow_type) 
                                 for seq in self.workflow_sequences.values()]
            stats['most_common_patterns'] = sorted(pattern_frequencies, 
                                                 key=lambda x: x[1], reverse=True)[:10]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting workflow statistics: {e}")
            return {'error': str(e)}
    
    def _create_workflow_step(self, command_data: CommandExecutionData, command_analysis: Any) -> WorkflowStep:
        """Create workflow step from command execution data"""
        # Extract file types from command
        file_types = self._extract_file_types(command_data.command)
        
        # Extract resource usage hints
        resource_usage = self._extract_resource_hints(command_data.command)
        
        return WorkflowStep(
            step_index=0,  # Will be set when added to sequence
            command_category=getattr(command_analysis, 'command_category', 'unknown'),
            subcategory=getattr(command_analysis, 'subcategory', 'unknown'),
            command_signature=self._create_command_signature(command_data.command),
            duration_ms=command_data.duration_ms,
            success=command_data.exit_code == 0,
            timestamp=command_data.timestamp,
            file_types_involved=file_types,
            resource_usage=resource_usage
        )
    
    def _create_command_signature(self, command: str) -> str:
        """Create abstract signature for command pattern matching"""
        # Extract key elements for pattern matching
        parts = command.split()
        if not parts:
            return 'empty_command'
        
        base_command = parts[0]
        
        # Extract significant flags
        significant_flags = []
        for part in parts[1:]:
            if part.startswith('-') and len(part) <= 10:  # Reasonable flag length
                significant_flags.append(part)
        
        # Create signature
        signature_parts = [base_command]
        if significant_flags:
            signature_parts.extend(sorted(significant_flags[:5]))  # Top 5 flags
        
        return '_'.join(signature_parts)
    
    def _extract_file_types(self, command: str) -> List[str]:
        """Extract file types involved in command"""
        import re
        
        # Common bioinformatics and data file extensions
        file_extensions = re.findall(r'\.\w+', command)
        
        # Filter to relevant extensions
        relevant_extensions = []
        important_extensions = {
            '.fastq', '.fq', '.bam', '.sam', '.vcf', '.bed', '.gff', '.gtf',
            '.fa', '.fasta', '.fna', '.csv', '.tsv', '.txt', '.json', '.xml',
            '.R', '.py', '.sh', '.pl', '.sql', '.h5', '.hdf5', '.parquet'
        }
        
        for ext in file_extensions:
            if ext.lower() in important_extensions:
                relevant_extensions.append(ext.lower())
        
        return list(set(relevant_extensions))
    
    def _extract_resource_hints(self, command: str) -> Dict[str, Any]:
        """Extract resource usage hints from command"""
        import re
        
        resource_hints = {}
        
        # Memory specifications
        memory_match = re.search(r'--mem[=\s]+(\d+[GM]?)', command, re.IGNORECASE)
        if memory_match:
            resource_hints['memory_requested'] = memory_match.group(1)
        
        # CPU/thread specifications
        cpu_patterns = [
            r'--cpus-per-task[=\s]+(\d+)',
            r'--threads[=\s]+(\d+)',
            r'-j\s*(\d+)',
            r'--cores[=\s]+(\d+)'
        ]
        
        for pattern in cpu_patterns:
            cpu_match = re.search(pattern, command, re.IGNORECASE)
            if cpu_match:
                resource_hints['cpu_requested'] = int(cpu_match.group(1))
                break
        
        # Time specifications
        time_match = re.search(r'--time[=\s]+(\d+:[\d:]+)', command, re.IGNORECASE)
        if time_match:
            resource_hints['time_requested'] = time_match.group(1)
        
        # GPU requirements
        if any(gpu_hint in command.lower() for gpu_hint in ['--gres=gpu', 'cuda', 'nvidia']):
            resource_hints['gpu_required'] = True
        
        return resource_hints
    
    def _analyze_current_sequence(self, session_id: str) -> Dict[str, Any]:
        """Analyze current command sequence for patterns"""
        sequence = list(self.active_sessions[session_id])
        
        if len(sequence) < 2:
            return {'pattern_detected': False}
        
        # Set step indices
        for i, step in enumerate(sequence):
            step.step_index = i
        
        # Classify workflow type
        workflow_type = self._classify_workflow_type(sequence)
        
        # Check for known patterns
        similar_sequences = self._find_similar_sequences(sequence)
        
        # Analyze success trajectory
        success_trajectory = [step.success for step in sequence]
        current_success_rate = sum(success_trajectory) / len(success_trajectory)
        
        return {
            'pattern_detected': True,
            'workflow_type': workflow_type,
            'sequence_length': len(sequence),
            'similar_patterns': len(similar_sequences),
            'current_success_rate': current_success_rate,
            'predicted_next_steps': self._predict_next_steps(sequence, workflow_type)
        }
    
    def _classify_workflow_type(self, sequence: List[WorkflowStep]) -> str:
        """Classify the type of workflow based on command sequence"""
        # Analyze command categories and subcategories
        categories = [step.command_category for step in sequence]
        subcategories = [step.subcategory for step in sequence]
        file_types = []
        for step in sequence:
            file_types.extend(step.file_types_involved)
        
        # Bioinformatics workflow detection
        if any(cat in ['bioinformatics'] for cat in categories):
            bio_subcats = [sub for sub in subcategories if 'bio' in sub or sub in [
                'quality_control', 'read_mapping', 'variant_calling', 'sequence_processing'
            ]]
            if len(bio_subcats) >= 2 or any(ft in file_types for ft in ['.fastq', '.bam', '.vcf']):
                return 'bioinformatics_pipeline'
        
        # R data analysis workflow
        if any('r_' in sub for sub in subcategories):
            return 'r_data_analysis'
        
        # Python data science workflow
        if any(cat == 'data_analysis' and 'python' in sub for cat, sub in zip(categories, subcategories)):
            return 'python_data_science'
        
        # Machine learning workflow
        if any('machine_learning' in sub or 'ml' in sub for sub in subcategories):
            return 'ml_pipeline'
        
        # HPC workflow
        if any(cat == 'hpc_job_submission' for cat in categories):
            return 'hpc_workflow'
        
        # Container workflow
        if any(cat == 'container_execution' for cat in categories):
            return 'container_workflow'
        
        # Workflow engine
        if any(cat == 'workflow_execution' for cat in categories):
            return 'workflow_engine_pipeline'
        
        # System administration
        if any(cat == 'system_admin' for cat in categories):
            return 'system_administration'
        
        return 'general_workflow'
    
    def _find_similar_sequences(self, current_sequence: List[WorkflowStep]) -> List[WorkflowSequence]:
        """Find similar sequences in the pattern database"""
        similar = []
        
        current_signatures = [step.command_signature for step in current_sequence]
        
        for sequence in self.workflow_sequences.values():
            stored_signatures = [step.command_signature for step in sequence.steps]
            
            # Calculate sequence similarity
            similarity = self._calculate_sequence_similarity(current_signatures, stored_signatures)
            
            if similarity > 0.6:  # 60% similarity threshold
                similar.append(sequence)
        
        return similar
    
    def _calculate_sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two command signature sequences"""
        if not seq1 or not seq2:
            return 0.0
        
        # Use longest common subsequence approach
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        return (2 * lcs_length) / (m + n)
    
    def _predict_next_steps(self, sequence: List[WorkflowStep], workflow_type: str) -> List[str]:
        """Predict likely next steps based on workflow patterns"""
        predictions = []
        
        # Look for similar sequences that are longer
        current_signatures = [step.command_signature for step in sequence]
        
        for stored_sequence in self.workflow_sequences.values():
            if stored_sequence.workflow_type != workflow_type:
                continue
            
            stored_signatures = [step.command_signature for step in stored_sequence.steps]
            
            # Check if current sequence is a prefix of stored sequence
            if len(stored_signatures) > len(current_signatures):
                # Check if stored sequence starts with current sequence
                if stored_signatures[:len(current_signatures)] == current_signatures:
                    next_signature = stored_signatures[len(current_signatures)]
                    if next_signature not in predictions:
                        predictions.append(next_signature)
        
        return predictions[:5]  # Top 5 predictions
    
    def _detect_completed_sequences(self, session_id: str) -> List[WorkflowSequence]:
        """Detect completed workflow sequences"""
        sequence = list(self.active_sessions[session_id])
        
        if len(sequence) < self.min_sequence_length:
            return []
        
        completed_sequences = []
        
        # Look for natural breakpoints in the sequence
        breakpoints = self._find_sequence_breakpoints(sequence)
        
        for start, end in breakpoints:
            subsequence = sequence[start:end]
            if len(subsequence) >= self.min_sequence_length:
                workflow_seq = self._create_workflow_sequence(subsequence)
                completed_sequences.append(workflow_seq)
        
        return completed_sequences
    
    def _find_sequence_breakpoints(self, sequence: List[WorkflowStep]) -> List[Tuple[int, int]]:
        """Find natural breakpoints in command sequences"""
        breakpoints = []
        
        if len(sequence) < self.min_sequence_length:
            return breakpoints
        
        # Look for time gaps (more than 10 minutes between commands)
        time_gaps = []
        for i in range(1, len(sequence)):
            time_diff = sequence[i].timestamp - sequence[i-1].timestamp
            if time_diff > 600:  # 10 minutes
                time_gaps.append(i)
        
        # Look for workflow type changes
        workflow_changes = []
        current_types = []
        for i, step in enumerate(sequence):
            step_type = f"{step.command_category}_{step.subcategory}"
            if current_types and step_type not in current_types[-3:]:  # Different from recent types
                workflow_changes.append(i)
            current_types.append(step_type)
        
        # Combine breakpoints
        all_breakpoints = sorted(time_gaps + workflow_changes)
        
        # Create sequence ranges
        start = 0
        for breakpoint in all_breakpoints:
            if breakpoint - start >= self.min_sequence_length:
                breakpoints.append((start, breakpoint))
            start = breakpoint
        
        # Add final sequence
        if len(sequence) - start >= self.min_sequence_length:
            breakpoints.append((start, len(sequence)))
        
        return breakpoints
    
    def _create_workflow_sequence(self, steps: List[WorkflowStep]) -> WorkflowSequence:
        """Create workflow sequence from steps"""
        workflow_type = self._classify_workflow_type(steps)
        
        # Calculate metrics
        total_duration = sum(step.duration_ms for step in steps)
        success_count = sum(1 for step in steps if step.success)
        success_rate = success_count / len(steps) if steps else 0.0
        
        # Create sequence ID
        signatures = [step.command_signature for step in steps]
        sequence_signature = '_'.join(signatures[:5])  # First 5 commands
        sequence_id = hashlib.md5(f"{workflow_type}_{sequence_signature}".encode()).hexdigest()[:12]
        
        # Generate optimization opportunities
        optimizations = self._identify_sequence_optimizations(steps, workflow_type)
        
        # Calculate typical resource requirements
        resource_reqs = self._calculate_typical_resources(steps)
        
        return WorkflowSequence(
            sequence_id=sequence_id,
            workflow_type=workflow_type,
            steps=steps,
            total_duration_ms=total_duration,
            success_rate=success_rate,
            frequency=1,  # Will be updated if pattern repeats
            first_seen=steps[0].timestamp,
            last_seen=steps[-1].timestamp,
            optimization_opportunities=optimizations,
            typical_resource_requirements=resource_reqs
        )
    
    def _identify_sequence_optimizations(self, steps: List[WorkflowStep], workflow_type: str) -> List[str]:
        """Identify optimization opportunities for workflow sequence"""
        optimizations = []
        
        # Parallel execution opportunities
        parallel_candidates = []
        for i, step in enumerate(steps[:-1]):
            next_step = steps[i + 1]
            # Check if steps could be parallelized (different file types, independent operations)
            if (not any(ft in step.file_types_involved for ft in next_step.file_types_involved) and
                step.command_category != next_step.command_category):
                parallel_candidates.append((i, i + 1))
        
        if parallel_candidates:
            optimizations.append(f"Consider parallelizing steps: {parallel_candidates[:3]}")
        
        # Resource optimization
        high_memory_steps = [i for i, step in enumerate(steps) 
                           if step.duration_ms > 60000 and 'memory_requested' not in step.resource_usage]
        if high_memory_steps:
            optimizations.append(f"Consider explicit memory allocation for long-running steps: {high_memory_steps}")
        
        # Tool upgrade opportunities
        for step in steps:
            if 'grep' in step.command_signature and 'rg' not in step.command_signature:
                optimizations.append("Consider upgrading grep to ripgrep (rg) for better performance")
            if 'find' in step.command_signature and 'fd' not in step.command_signature:
                optimizations.append("Consider upgrading find to fd for better performance")
        
        # Workflow-specific optimizations
        if workflow_type == 'bioinformatics_pipeline':
            if any('quality_control' in step.subcategory for step in steps):
                if not any('parallel' in step.command_signature for step in steps):
                    optimizations.append("Consider using parallel versions of QC tools")
        
        return optimizations
    
    def _calculate_typical_resources(self, steps: List[WorkflowStep]) -> Dict[str, Any]:
        """Calculate typical resource requirements for sequence"""
        resources = {
            'peak_memory_gb': 0,
            'total_cpu_hours': 0,
            'typical_duration_minutes': sum(step.duration_ms for step in steps) / (1000 * 60),
            'gpu_required': False
        }
        
        # Analyze resource hints from steps
        memory_requests = []
        cpu_requests = []
        
        for step in steps:
            if 'memory_requested' in step.resource_usage:
                mem_str = step.resource_usage['memory_requested']
                if mem_str.endswith('G'):
                    memory_requests.append(int(mem_str[:-1]))
                elif mem_str.endswith('M'):
                    memory_requests.append(int(mem_str[:-1]) / 1024)
            
            if 'cpu_requested' in step.resource_usage:
                cpu_requests.append(step.resource_usage['cpu_requested'])
            
            if step.resource_usage.get('gpu_required'):
                resources['gpu_required'] = True
        
        if memory_requests:
            resources['peak_memory_gb'] = max(memory_requests)
        
        if cpu_requests:
            resources['total_cpu_hours'] = sum(cpu_requests) * resources['typical_duration_minutes'] / 60
        
        return resources
    
    def _register_workflow_sequence(self, sequence: WorkflowSequence) -> None:
        """Register or update workflow sequence in pattern database"""
        if sequence.sequence_id in self.workflow_sequences:
            # Update existing pattern
            existing = self.workflow_sequences[sequence.sequence_id]
            existing.frequency += 1
            existing.last_seen = sequence.last_seen
            
            # Update success rate (weighted average)
            total_observations = existing.frequency
            weight = 1.0 / total_observations
            existing.success_rate = (existing.success_rate * (1 - weight)) + (sequence.success_rate * weight)
            
            # Update duration estimate
            existing.total_duration_ms = int(
                (existing.total_duration_ms * (1 - weight)) + (sequence.total_duration_ms * weight)
            )
        else:
            # Register new pattern
            self.workflow_sequences[sequence.sequence_id] = sequence
    
    def _generate_sequence_recommendations(self, session_id: str, current_step: WorkflowStep) -> List[str]:
        """Generate recommendations based on current sequence context"""
        recommendations = []
        
        sequence = list(self.active_sessions[session_id])
        if len(sequence) < 2:
            return recommendations
        
        # Find similar successful sequences
        workflow_type = self._classify_workflow_type(sequence)
        similar_sequences = [seq for seq in self.workflow_sequences.values()
                           if seq.workflow_type == workflow_type and seq.success_rate > 0.8]
        
        if similar_sequences:
            # Recommend next steps from successful patterns
            next_steps = self._predict_next_steps(sequence, workflow_type)
            if next_steps:
                recommendations.append(f"Based on successful {workflow_type} patterns, consider: {next_steps[0]}")
        
        # Resource recommendations
        if current_step.duration_ms > 30000 and not current_step.resource_usage:
            recommendations.append("Consider specifying resource requirements for long-running commands")
        
        return recommendations
    
    def _cleanup_expired_sessions(self, current_time: float) -> None:
        """Clean up expired command sequences"""
        expired_sessions = []
        for session_id, timeout in self.session_timeouts.items():
            if current_time > timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            del self.session_timeouts[session_id]
    
    def _build_workflow_classifiers(self) -> Dict[str, Dict[str, Any]]:
        """Build workflow classification patterns"""
        return {
            'bioinformatics_pipeline': {
                'required_categories': ['bioinformatics'],
                'common_file_types': ['.fastq', '.bam', '.vcf', '.bed'],
                'typical_steps': ['quality_control', 'read_mapping', 'variant_calling'],
                'min_sequence_length': 3
            },
            'r_data_analysis': {
                'required_categories': ['data_analysis'],
                'required_subcategories': ['r_'],
                'common_file_types': ['.csv', '.tsv', '.R'],
                'min_sequence_length': 2
            },
            'ml_pipeline': {
                'required_categories': ['machine_learning'],
                'common_file_types': ['.h5', '.parquet', '.csv'],
                'typical_steps': ['preprocessing', 'training', 'evaluation'],
                'min_sequence_length': 3
            }
        }
    
    def _save_patterns(self) -> None:
        """Save workflow patterns to storage"""
        try:
            patterns_data = {
                'workflow_sequences': {
                    seq_id: asdict(seq) for seq_id, seq in self.workflow_sequences.items()
                },
                'pattern_correlations': {
                    f"{corr[0]}_{corr[1]}": asdict(pattern) 
                    for corr, pattern in self.pattern_correlations.items()
                },
                'last_updated': time.time()
            }
            
            patterns_file = self.storage_dir / 'workflow_patterns.json'
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
        
        except Exception as e:
            self.logger.error(f"Error saving workflow patterns: {e}")
    
    def _load_patterns(self) -> None:
        """Load workflow patterns from storage"""
        try:
            patterns_file = self.storage_dir / 'workflow_patterns.json'
            if not patterns_file.exists():
                return
            
            with open(patterns_file, 'r') as f:
                patterns_data = json.load(f)
            
            # Restore workflow sequences
            for seq_id, seq_data in patterns_data.get('workflow_sequences', {}).items():
                try:
                    # Reconstruct WorkflowStep objects
                    steps = []
                    for step_data in seq_data['steps']:
                        step = WorkflowStep(**step_data)
                        steps.append(step)
                    
                    seq_data['steps'] = steps
                    sequence = WorkflowSequence(**seq_data)
                    self.workflow_sequences[seq_id] = sequence
                    
                except Exception as e:
                    self.logger.warning(f"Error loading sequence {seq_id}: {e}")
            
            self.logger.info(f"Loaded {len(self.workflow_sequences)} workflow patterns")
            
        except Exception as e:
            self.logger.error(f"Error loading workflow patterns: {e}")
    
    def shutdown(self) -> None:
        """Save patterns and cleanup"""
        self._save_patterns()

if __name__ == "__main__":
    # Example usage and testing
    recognizer = WorkflowPatternRecognizer()
    
    # Simulate a bioinformatics workflow
    from interfaces import CommandExecutionData
    from learning.command_abstractor import AdvancedCommandAbstractor
    
    abstractor = AdvancedCommandAbstractor()
    
    # Test workflow sequence
    test_commands = [
        CommandExecutionData("fastqc sample.fastq", 0, 5000, time.time(), "bio_session", "/data/seq"),
        CommandExecutionData("trimmomatic PE sample.fastq trimmed.fastq", 0, 15000, time.time() + 60, "bio_session", "/data/seq"),
        CommandExecutionData("bwa mem reference.fa trimmed.fastq | samtools sort -o aligned.bam", 0, 120000, time.time() + 120, "bio_session", "/data/seq"),
        CommandExecutionData("gatk HaplotypeCaller -I aligned.bam -O variants.vcf", 0, 300000, time.time() + 240, "bio_session", "/data/seq")
    ]
    
    # Process workflow sequence
    for cmd_data in test_commands:
        analysis = abstractor.analyze_command_comprehensive(cmd_data.command)
        result = recognizer.analyze_command_sequence(cmd_data, analysis)
        print(f"Command: {cmd_data.command[:50]}...")
        print(f"Workflow type: {result.get('workflow_insights', {}).get('workflow_type', 'unknown')}")
        print(f"Sequence length: {result.get('workflow_insights', {}).get('sequence_length', 0)}")
        print()
    
    # Get workflow statistics
    stats = recognizer.get_workflow_statistics()
    print("Workflow Statistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    # Cleanup
    recognizer.shutdown()