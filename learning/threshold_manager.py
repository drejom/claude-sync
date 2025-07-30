#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0"
# ]
# ///
"""
Information Threshold Manager - Adaptive agent triggering based on information density

This implements the intelligent threshold system that accumulates information from
command executions and triggers specialized agents when information density reaches
significance thresholds. Key features:

- Weighted information accumulation by significance type
- Adaptive threshold adjustment based on analysis effectiveness  
- Multi-agent coordination (learning-analyst, hpc-advisor, troubleshooting-detective)
- Information significance calculation based on command complexity and context
- Performance optimized for <1ms threshold checks

Based on REFACTOR_PLAN.md sections 1360-1655 (Information Threshold Management)
"""

import json
import time
import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import logging

# Import our interfaces
import sys
sys.path.append(str(Path(__file__).parent.parent))
from interfaces import (
    InformationThresholdInterface,
    InformationTypes,
    AgentNames,
    PerformanceTargets,
    CommandExecutionData
)

@dataclass
class InformationAccumulator:
    """Tracks accumulated information for threshold calculations"""
    total_value: float = 0.0
    recent_additions: deque = None
    last_reset: float = 0.0
    peak_value: float = 0.0
    
    def __post_init__(self):
        if self.recent_additions is None:
            self.recent_additions = deque(maxlen=100)  # Keep last 100 additions
        if self.last_reset == 0.0:
            self.last_reset = time.time()

@dataclass
class AgentThresholdConfig:
    """Configuration for agent-specific information thresholds"""
    base_threshold: float
    weight_factors: Dict[str, float]
    effectiveness_history: deque = None
    last_triggered: float = 0.0
    trigger_count: int = 0
    
    def __post_init__(self):
        if self.effectiveness_history is None:
            self.effectiveness_history = deque(maxlen=10)  # Keep last 10 effectiveness scores

@dataclass
class ThresholdEvent:
    """Records a threshold trigger event for analysis"""
    timestamp: float
    agent_name: str
    trigger_score: float
    information_breakdown: Dict[str, float]
    context: Optional[Dict[str, Any]] = None

class InformationThresholdManager(InformationThresholdInterface):
    """
    Manages adaptive information thresholds for intelligent agent triggering.
    
    This class implements the information density-based agent triggering system
    described in REFACTOR_PLAN.md sections 1360-1655. It accumulates weighted
    information from various sources and triggers specialized agents when
    significance thresholds are reached.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir or Path.home() / '.claude' / 'learning')
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Information accumulators by type
        self.info_accumulators: Dict[str, InformationAccumulator] = {
            info_type: InformationAccumulator()
            for info_type in [
                InformationTypes.NEW_COMMANDS,
                InformationTypes.FAILURES,
                InformationTypes.OPTIMIZATIONS,
                InformationTypes.HOST_CHANGES,
                InformationTypes.PERFORMANCE_SHIFTS
            ]
        }
        
        # Agent-specific threshold configurations
        self.agent_configs: Dict[str, AgentThresholdConfig] = {
            AgentNames.LEARNING_ANALYST: AgentThresholdConfig(
                base_threshold=50.0,
                weight_factors={
                    InformationTypes.NEW_COMMANDS: 2.0,
                    InformationTypes.FAILURES: 3.0,
                    InformationTypes.OPTIMIZATIONS: 1.5,
                    InformationTypes.PERFORMANCE_SHIFTS: 4.0,
                    InformationTypes.HOST_CHANGES: 1.5
                }
            ),
            AgentNames.HPC_ADVISOR: AgentThresholdConfig(
                base_threshold=30.0,
                weight_factors={
                    InformationTypes.NEW_COMMANDS: 3.0,  # HPC commands are complex
                    InformationTypes.FAILURES: 4.0,     # SLURM failures are expensive
                    InformationTypes.HOST_CHANGES: 2.0, # New hosts need optimization
                    InformationTypes.PERFORMANCE_SHIFTS: 3.0,
                    InformationTypes.OPTIMIZATIONS: 1.0
                }
            ),
            AgentNames.TROUBLESHOOTING_DETECTIVE: AgentThresholdConfig(
                base_threshold=15.0,  # Quick response to failures
                weight_factors={
                    InformationTypes.FAILURES: 5.0,         # Primary trigger
                    InformationTypes.NEW_COMMANDS: 1.0,     # Less relevant
                    InformationTypes.PERFORMANCE_SHIFTS: 3.0,
                    InformationTypes.OPTIMIZATIONS: 0.5,
                    InformationTypes.HOST_CHANGES: 1.0
                }
            )
        }
        
        # Event tracking
        self.threshold_events: List[ThresholdEvent] = []
        self.agent_callbacks: Dict[str, Callable] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing state
        self._load_state()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def accumulate_information(self, info_type: str, significance: float = 1.0, context: Optional[Dict] = None) -> None:
        """
        Accumulate information and check if any agent thresholds are reached.
        
        This is the core method that processes new information and determines
        if specialized agents should be triggered. Performance target: <1ms.
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Validate info_type
                if info_type not in self.info_accumulators:
                    self.logger.warning(f"Unknown information type: {info_type}")
                    return
                
                # Accumulate information
                accumulator = self.info_accumulators[info_type]
                accumulator.total_value += significance
                accumulator.recent_additions.append((time.time(), significance))
                accumulator.peak_value = max(accumulator.peak_value, accumulator.total_value)
                
                # Check each agent's threshold
                for agent_name in self.agent_configs:
                    if self.should_trigger_agent(agent_name):
                        self._trigger_agent(agent_name, context)
                
                # Performance monitoring
                duration_ms = (time.time() - start_time) * 1000
                if duration_ms > PerformanceTargets.PATTERN_LOOKUP_MS:
                    self.logger.warning(f"Information accumulation took {duration_ms:.1f}ms (target: {PerformanceTargets.PATTERN_LOOKUP_MS}ms)")
                    
        except Exception as e:
            self.logger.error(f"Error in accumulate_information: {e}")
    
    def calculate_weighted_score(self, agent_name: str) -> float:
        """Calculate current weighted information score for specific agent"""
        if agent_name not in self.agent_configs:
            return 0.0
        
        config = self.agent_configs[agent_name]
        score = 0.0
        
        # Calculate weighted score across all information types
        for info_type, accumulator in self.info_accumulators.items():
            weight = config.weight_factors.get(info_type, 1.0)
            score += accumulator.total_value * weight
        
        return score
    
    def should_trigger_agent(self, agent_name: str) -> bool:
        """Check if agent analysis should be triggered"""
        if agent_name not in self.agent_configs:
            return False
        
        config = self.agent_configs[agent_name]
        current_score = self.calculate_weighted_score(agent_name)
        
        # Check cooldown period (don't trigger same agent too frequently)
        time_since_last = time.time() - config.last_triggered
        min_cooldown = 300  # 5 minutes minimum between triggers
        
        if time_since_last < min_cooldown:
            return False
        
        return current_score >= config.base_threshold
    
    def _trigger_agent(self, agent_name: str, context: Optional[Dict] = None) -> None:
        """Trigger agent analysis and reset relevant counters"""
        try:
            config = self.agent_configs[agent_name]
            trigger_score = self.calculate_weighted_score(agent_name)
            
            # Create information breakdown
            info_breakdown = {}
            for info_type, accumulator in self.info_accumulators.items():
                weight = config.weight_factors.get(info_type, 1.0)
                weighted_value = accumulator.total_value * weight
                if weighted_value > 0:
                    info_breakdown[info_type] = weighted_value
            
            # Record threshold event
            event = ThresholdEvent(
                timestamp=time.time(),
                agent_name=agent_name,
                trigger_score=trigger_score,
                information_breakdown=info_breakdown,
                context=context
            )
            self.threshold_events.append(event)
            
            # Update agent config
            config.last_triggered = time.time()
            config.trigger_count += 1
            
            # Reset counters for this agent
            self.reset_counters_for_agent(agent_name)
            
            # Call agent callback if registered
            if agent_name in self.agent_callbacks:
                try:
                    self.agent_callbacks[agent_name](event)
                except Exception as e:
                    self.logger.error(f"Error calling agent callback for {agent_name}: {e}")
            
            self.logger.info(f"Triggered {agent_name} analysis (score: {trigger_score:.1f}, threshold: {config.base_threshold:.1f})")
            
        except Exception as e:
            self.logger.error(f"Error triggering agent {agent_name}: {e}")
    
    def reset_counters_for_agent(self, agent_name: str) -> None:
        """Reset information counters after agent analysis"""
        with self._lock:
            # Partial reset based on agent's focus areas
            if agent_name not in self.agent_configs:
                return
            
            config = self.agent_configs[agent_name]
            
            # Reset counters with higher weights more aggressively
            for info_type, accumulator in self.info_accumulators.items():
                weight = config.weight_factors.get(info_type, 1.0)
                
                # Higher weight = more aggressive reset
                reset_factor = min(0.9, weight / 5.0)  # Reset 0-90% based on weight
                accumulator.total_value *= (1.0 - reset_factor)
                accumulator.last_reset = time.time()
    
    def adapt_threshold(self, agent_name: str, effectiveness_score: float) -> None:
        """Adapt threshold based on analysis effectiveness"""
        if agent_name not in self.agent_configs:
            return
        
        with self._lock:
            config = self.agent_configs[agent_name]
            config.effectiveness_history.append(effectiveness_score)
            
            # Calculate adaptation based on recent effectiveness
            if len(config.effectiveness_history) >= 3:
                recent_avg = sum(list(config.effectiveness_history)[-3:]) / 3
                
                if recent_avg > 0.8:
                    # Analysis very useful - lower threshold (more sensitive)
                    adaptation_factor = 0.95
                elif recent_avg < 0.3:
                    # Analysis not useful - raise threshold (less sensitive)
                    adaptation_factor = 1.1
                else:
                    # Moderate effectiveness - small adjustment toward optimal
                    target_effectiveness = 0.6
                    adaptation_factor = 1.0 + (target_effectiveness - recent_avg) * 0.05
                
                config.base_threshold *= adaptation_factor
                
                # Keep thresholds within reasonable bounds
                config.base_threshold = max(5.0, min(200.0, config.base_threshold))
                
                self.logger.info(f"Adapted {agent_name} threshold to {config.base_threshold:.1f} (effectiveness: {recent_avg:.2f})")
    
    def register_agent_callback(self, agent_name: str, callback: Callable) -> None:
        """Register callback function for agent trigger events"""
        self.agent_callbacks[agent_name] = callback
    
    def get_status(self) -> Dict[str, Any]:
        """Get current threshold manager status"""
        with self._lock:
            return {
                'information_accumulators': {
                    info_type: {
                        'total_value': acc.total_value,
                        'peak_value': acc.peak_value,
                        'recent_activity': len(acc.recent_additions),
                        'last_reset': acc.last_reset
                    }
                    for info_type, acc in self.info_accumulators.items()
                },
                'agent_configs': {
                    agent_name: {
                        'base_threshold': config.base_threshold,
                        'current_score': self.calculate_weighted_score(agent_name),
                        'trigger_ready': self.should_trigger_agent(agent_name),
                        'last_triggered': config.last_triggered,
                        'trigger_count': config.trigger_count,
                        'avg_effectiveness': sum(config.effectiveness_history) / max(len(config.effectiveness_history), 1)
                    }
                    for agent_name, config in self.agent_configs.items()
                },
                'recent_events': [
                    {
                        'timestamp': event.timestamp,
                        'agent_name': event.agent_name,
                        'trigger_score': event.trigger_score,
                        'info_breakdown': event.information_breakdown
                    }
                    for event in self.threshold_events[-10:]  # Last 10 events
                ]
            }
    
    def calculate_information_significance(self, command_data: CommandExecutionData) -> Dict[str, float]:
        """
        Calculate information significance for different types based on command data.
        
        Enhanced algorithm with sophisticated weighting based on:
        - Temporal patterns and recency
        - Command rarity and novelty scoring
        - Context-aware failure analysis
        - Multi-factor performance deviation detection
        - Host capability correlation analysis
        """
        significance_scores = {}
        
        # Enhanced base significance with temporal weighting
        base_significance = self._calculate_temporal_significance()
        
        # Multi-factor complexity analysis
        complexity_factor = self._calculate_enhanced_command_complexity(command_data.command)
        
        # Context-aware significance calculation
        context_multiplier = self._calculate_context_significance(command_data)
        
        # NEW_COMMANDS significance with rarity scoring
        novelty_score = self._calculate_command_novelty(command_data.command)
        if novelty_score > 0.3:  # Threshold for novel commands
            # Sophisticated novelty weighting
            tool_sophistication = self._assess_tool_sophistication(command_data.command)
            workflow_impact = self._assess_workflow_impact(command_data)
            
            new_command_significance = (
                base_significance * 
                complexity_factor * 
                novelty_score * 
                (1 + tool_sophistication * 0.5) *
                (1 + workflow_impact * 0.3) *
                context_multiplier
            )
            significance_scores[InformationTypes.NEW_COMMANDS] = new_command_significance
        
        # FAILURES significance with context analysis
        if command_data.exit_code != 0:
            failure_base = base_significance * 2.0
            
            # Enhanced failure analysis
            historical_success_rate = self._get_historical_success_rate(command_data.command)
            failure_rarity = max(0.1, historical_success_rate)  # Rare failures are more significant
            
            # Command criticality assessment
            criticality = self._assess_command_criticality(command_data.command)
            
            # Time-of-day and workflow stage impact
            temporal_impact = self._calculate_failure_temporal_impact(command_data.timestamp)
            
            failure_significance = (
                failure_base *
                failure_rarity *
                (1 + criticality * 0.8) *
                (1 + temporal_impact * 0.4) *
                complexity_factor *
                context_multiplier
            )
            
            significance_scores[InformationTypes.FAILURES] = failure_significance
        
        # Enhanced PERFORMANCE_SHIFTS with multi-factor analysis
        performance_significance = self._calculate_performance_shift_significance(command_data)
        if performance_significance > 0:
            significance_scores[InformationTypes.PERFORMANCE_SHIFTS] = performance_significance
        
        # Advanced HOST_CHANGES with capability correlation
        host_change_significance = self._calculate_host_change_significance(command_data)
        if host_change_significance > 0:
            significance_scores[InformationTypes.HOST_CHANGES] = host_change_significance
        
        # OPTIMIZATIONS significance (new category)
        optimization_significance = self._calculate_optimization_significance(command_data)
        if optimization_significance > 0:
            significance_scores[InformationTypes.OPTIMIZATIONS] = optimization_significance
        
        return significance_scores
    
    def _calculate_command_complexity(self, command: str) -> float:
        """Calculate command complexity factor"""
        base_complexity = 1.0
        
        # Pipe complexity
        pipe_count = command.count('|')
        base_complexity += pipe_count * 0.3
        
        # Flag complexity
        flag_count = len([part for part in command.split() if part.startswith('-')])
        base_complexity += flag_count * 0.1
        
        # Length complexity
        length_factor = min(len(command) / 100, 2.0)
        base_complexity += length_factor * 0.2
        
        return min(base_complexity, 5.0)  # Cap at 5x complexity
    
    def _is_novel_command(self, command: str) -> bool:
        """Check if command represents a novel pattern"""
        # This would integrate with the adaptive schema to check novelty
        # For now, use simple heuristics
        
        # Check for uncommon command patterns
        uncommon_commands = ['nextflow', 'snakemake', 'singularity', 'apptainer']
        return any(cmd in command.lower() for cmd in uncommon_commands)
    
    def _is_usually_successful_command(self, command: str) -> bool:
        """Check if this command usually succeeds (failure is anomalous)"""
        # This would check historical success rates
        # For now, assume basic commands usually succeed
        basic_commands = ['ls', 'cp', 'mv', 'cat', 'echo']
        return any(command.strip().startswith(cmd) for cmd in basic_commands)
    
    def _predict_command_duration(self, command: str) -> int:
        """Predict expected duration for command (in ms)"""
        # Simple duration predictions based on command type
        if any(cmd in command.lower() for cmd in ['ls', 'echo', 'pwd']):
            return 100  # Very fast commands
        elif any(cmd in command.lower() for cmd in ['find', 'grep']):
            return 2000  # Search commands
        elif any(cmd in command.lower() for cmd in ['sbatch', 'python', 'r ']):
            return 10000  # Analysis commands
        else:
            return 1000  # Default
    
    def _detect_new_host_context(self, command_data: CommandExecutionData) -> bool:
        """Detect if command was executed in a new host context"""
        # Check for SSH commands or new working directories
        command = command_data.command
        return 'ssh' in command.lower() or command_data.working_directory.startswith('/tmp')
    
    def _calculate_temporal_significance(self) -> float:
        """Calculate temporal weighting based on time of day and recent activity"""
        current_hour = time.localtime().tm_hour
        
        # Working hours (9-17) have higher significance
        if 9 <= current_hour <= 17:
            time_factor = 1.2
        elif 6 <= current_hour <= 21:  # Extended work hours
            time_factor = 1.0
        else:  # Night hours - potentially critical operations
            time_factor = 1.5
        
        # Recent activity boost (more active periods = higher significance)
        recent_operations = sum(1 for acc in self.info_accumulators.values() 
                               for addition in acc.recent_additions 
                               if time.time() - addition[0] < 3600)  # Last hour
        activity_factor = min(1.5, 1.0 + (recent_operations / 100))
        
        return time_factor * activity_factor
    
    def _calculate_enhanced_command_complexity(self, command: str) -> float:
        """Enhanced command complexity calculation with domain-specific weights"""
        base_complexity = self._calculate_command_complexity(command)
        
        # Domain-specific complexity multipliers
        domain_multipliers = {
            'bioinformatics': 1.8,  # Bio tools are inherently complex
            'hpc': 1.6,            # HPC commands are resource-critical
            'container': 1.4,       # Container operations are environment-critical
            'workflow': 2.0,        # Workflow engines are highly complex
            'ml': 1.7              # ML operations are compute-intensive
        }
        
        # Detect domain
        command_lower = command.lower()
        domain_factor = 1.0
        
        for domain, multiplier in domain_multipliers.items():
            domain_patterns = {
                'bioinformatics': ['blast', 'bwa', 'samtools', 'gatk', 'fastqc'],
                'hpc': ['sbatch', 'squeue', 'scancel', 'slurm'],
                'container': ['singularity', 'docker', 'podman'],
                'workflow': ['nextflow', 'snakemake', 'cwl'],
                'ml': ['tensorflow', 'pytorch', 'sklearn', 'keras']
            }
            
            if any(pattern in command_lower for pattern in domain_patterns.get(domain, [])):
                domain_factor = max(domain_factor, multiplier)
                break
        
        return base_complexity * domain_factor
    
    def _calculate_context_significance(self, command_data: CommandExecutionData) -> float:
        """Calculate context-based significance multiplier"""
        multiplier = 1.0
        
        # Working directory significance
        work_dir = command_data.working_directory.lower()
        if any(pattern in work_dir for pattern in ['data', 'project', 'analysis']):
            multiplier *= 1.2
        elif any(pattern in work_dir for pattern in ['tmp', 'temp', 'scratch']):
            multiplier *= 0.8
        
        # Host context significance
        if command_data.host_context:
            hostname = command_data.host_context.get('hostname', '').lower()
            if any(pattern in hostname for pattern in ['gpu', 'hpc', 'cluster']):
                multiplier *= 1.3
        
        return multiplier
    
    def _calculate_command_novelty(self, command: str) -> float:
        """Calculate command novelty score based on historical frequency"""
        command_lower = command.lower()
        
        # Common commands have low novelty
        common_commands = ['ls', 'cp', 'mv', 'cd', 'pwd', 'cat', 'echo', 'grep', 'find']
        if any(command_lower.startswith(cmd) for cmd in common_commands):
            return 0.1
        
        # Uncommon but known commands
        uncommon_commands = ['sbatch', 'singularity', 'docker', 'python', 'R']
        if any(cmd in command_lower for cmd in uncommon_commands):
            return 0.6
        
        # Rare/specialized commands
        specialized_commands = ['nextflow', 'snakemake', 'gatk', 'blast', 'bwa']
        if any(cmd in command_lower for cmd in specialized_commands):
            return 0.9
        
        # Unknown commands are highly novel
        return 0.8
    
    def _assess_tool_sophistication(self, command: str) -> float:
        """Assess sophistication level of tools used"""
        command_lower = command.lower()
        
        sophistication_scores = {
            # High sophistication tools
            'nextflow': 1.0, 'snakemake': 1.0, 'gatk': 0.9, 'blast': 0.8,
            'singularity': 0.8, 'docker': 0.7,
            
            # Medium sophistication
            'python': 0.6, 'R': 0.6, 'sbatch': 0.7, 'slurm': 0.7,
            
            # Basic tools
            'grep': 0.3, 'find': 0.3, 'ls': 0.1, 'cp': 0.1
        }
        
        max_sophistication = 0.0
        for tool, score in sophistication_scores.items():
            if tool in command_lower:
                max_sophistication = max(max_sophistication, score)
        
        return max_sophistication
    
    def _assess_workflow_impact(self, command_data: CommandExecutionData) -> float:
        """Assess potential impact on workflow"""
        impact = 0.0
        
        # Long-running commands have higher impact
        if command_data.duration_ms > 60000:  # > 1 minute
            impact += 0.5
        
        # Commands in project directories have higher impact
        if any(pattern in command_data.working_directory.lower() 
               for pattern in ['project', 'analysis', 'data']):
            impact += 0.3
        
        # Commands with resource specifications have higher impact
        if any(resource in command_data.command.lower() 
               for resource in ['--mem', '--time', '--cpus']):
            impact += 0.4
        
        return min(impact, 1.0)
    
    def _get_historical_success_rate(self, command: str) -> float:
        """Get historical success rate for command pattern"""
        command_lower = command.lower()
        
        # Assign estimated success rates based on command type
        if any(basic in command_lower for basic in ['ls', 'cp', 'mv', 'cat']):
            return 0.95  # Basic commands usually succeed
        elif any(complex_cmd in command_lower for complex_cmd in ['sbatch', 'singularity']):
            return 0.7   # Complex commands fail more often
        elif any(analysis in command_lower for analysis in ['gatk', 'blast', 'bwa']):
            return 0.6   # Analysis tools are error-prone
        
        return 0.8  # Default success rate
    
    def _assess_command_criticality(self, command: str) -> float:
        """Assess how critical a command failure would be"""
        command_lower = command.lower()
        
        # High criticality operations
        if any(critical in command_lower for critical in ['rm -rf', 'dd if=', 'format']):
            return 1.0
        
        # Medium criticality
        elif any(medium in command_lower for medium in ['sbatch', 'docker run', 'singularity']):
            return 0.7
        
        # Analysis operations
        elif any(analysis in command_lower for analysis in ['gatk', 'blast', 'nextflow']):
            return 0.6
        
        # Low criticality
        return 0.3
    
    def _calculate_failure_temporal_impact(self, timestamp: float) -> float:
        """Calculate temporal impact of failure"""
        failure_hour = time.localtime(timestamp).tm_hour
        
        # Failures during working hours are more impactful
        if 9 <= failure_hour <= 17:
            return 0.8
        elif 18 <= failure_hour <= 21:  # Evening work
            return 0.6
        else:  # Night/early morning
            return 0.3
    
    def _calculate_performance_shift_significance(self, command_data: CommandExecutionData) -> float:
        """Calculate performance shift significance with multi-factor analysis"""
        expected_duration = self._predict_command_duration(command_data.command)
        if expected_duration <= 0:
            return 0.0
        
        actual_duration = command_data.duration_ms
        performance_ratio = abs(actual_duration - expected_duration) / expected_duration
        
        if performance_ratio < 0.2:  # Less than 20% deviation
            return 0.0
        
        # Base significance from performance deviation
        base_significance = performance_ratio * 2.0
        
        # Command importance multiplier
        importance = self._assess_command_criticality(command_data.command)
        
        # Resource context multiplier
        resource_context = 1.0
        if 'mem' in command_data.command.lower() or 'cpu' in command_data.command.lower():
            resource_context = 1.3
        
        # Temporal pattern (degradation vs improvement)
        temporal_factor = 1.0
        if actual_duration > expected_duration:  # Performance degradation
            temporal_factor = 1.5
        else:  # Performance improvement
            temporal_factor = 0.8
        
        return base_significance * (1 + importance * 0.5) * resource_context * temporal_factor
    
    def _calculate_host_change_significance(self, command_data: CommandExecutionData) -> float:
        """Calculate host change significance with capability correlation"""
        if not self._detect_new_host_context(command_data):
            return 0.0
        
        base_significance = 1.0
        
        # Command-host capability matching
        command_lower = command_data.command.lower()
        host_context = command_data.host_context or {}
        hostname = host_context.get('hostname', '').lower()
        
        # GPU commands on non-GPU hosts
        if any(gpu in command_lower for gpu in ['cuda', 'nvidia', '--gres=gpu']):
            if not any(gpu_pattern in hostname for gpu_pattern in ['gpu', 'cuda']):
                base_significance *= 2.0  # Significant mismatch
        
        # HPC commands on non-HPC hosts
        elif any(hpc in command_lower for hpc in ['sbatch', 'slurm']):
            if not any(hpc_pattern in hostname for hpc_pattern in ['hpc', 'cluster', 'node']):
                base_significance *= 1.5
        
        # Container commands with different environments
        elif any(container in command_lower for container in ['singularity', 'docker']):
            base_significance *= 1.2  # Container portability reduces significance
        
        return base_significance
    
    def _calculate_optimization_significance(self, command_data: CommandExecutionData) -> float:
        """Calculate significance of potential optimizations"""
        optimization_significance = 0.0
        command_lower = command_data.command.lower()
        
        # Suboptimal tool usage
        if 'grep' in command_lower and 'rg' not in command_lower:
            optimization_significance += 0.8
        
        if 'find' in command_lower and 'fd' not in command_lower:
            optimization_significance += 0.7
        
        # Missing resource specifications for long commands
        if command_data.duration_ms > 30000:  # > 30 seconds
            if not any(resource in command_lower for resource in ['--mem', '--time', '--cpus']):
                optimization_significance += 1.0
        
        # Inefficient patterns
        if 'cat' in command_lower and '|' in command_lower:
            optimization_significance += 0.5
        
        # Serial processing opportunities
        if command_data.duration_ms > 60000:  # > 1 minute
            if not any(parallel in command_lower for parallel in ['-j', '--parallel', '--threads']):
                optimization_significance += 0.6
        
        return min(optimization_significance, 2.0)
    
    def _save_state(self) -> None:
        """Save threshold manager state to storage"""
        try:
            state_data = {
                'info_accumulators': {
                    info_type: {
                        'total_value': acc.total_value,
                        'peak_value': acc.peak_value,
                        'last_reset': acc.last_reset,
                        'recent_additions': list(acc.recent_additions)
                    }
                    for info_type, acc in self.info_accumulators.items()
                },
                'agent_configs': {
                    agent_name: {
                        'base_threshold': config.base_threshold,
                        'weight_factors': config.weight_factors,
                        'last_triggered': config.last_triggered,
                        'trigger_count': config.trigger_count,
                        'effectiveness_history': list(config.effectiveness_history)
                    }
                    for agent_name, config in self.agent_configs.items()
                },
                'threshold_events': [asdict(event) for event in self.threshold_events[-50:]],  # Keep last 50
                'last_updated': time.time()
            }
            
            state_file = self.storage_dir / 'threshold_manager_state.json'
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving threshold manager state: {e}")
    
    def _load_state(self) -> None:
        """Load threshold manager state from storage"""
        try:
            state_file = self.storage_dir / 'threshold_manager_state.json'
            if not state_file.exists():
                return
            
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore information accumulators
            for info_type, acc_data in state_data.get('info_accumulators', {}).items():
                if info_type in self.info_accumulators:
                    acc = self.info_accumulators[info_type]
                    acc.total_value = acc_data.get('total_value', 0.0)
                    acc.peak_value = acc_data.get('peak_value', 0.0)
                    acc.last_reset = acc_data.get('last_reset', time.time())
                    acc.recent_additions = deque(acc_data.get('recent_additions', []), maxlen=100)
            
            # Restore agent configs
            for agent_name, config_data in state_data.get('agent_configs', {}).items():
                if agent_name in self.agent_configs:
                    config = self.agent_configs[agent_name]
                    config.base_threshold = config_data.get('base_threshold', config.base_threshold)
                    config.last_triggered = config_data.get('last_triggered', 0.0)
                    config.trigger_count = config_data.get('trigger_count', 0)
                    config.effectiveness_history = deque(config_data.get('effectiveness_history', []), maxlen=10)
            
            # Restore events
            for event_data in state_data.get('threshold_events', []):
                try:
                    event = ThresholdEvent(**event_data)
                    self.threshold_events.append(event)
                except:
                    pass  # Skip invalid events
                    
        except Exception as e:
            self.logger.error(f"Error loading threshold manager state: {e}")
    
    def __del__(self):
        """Save state on cleanup"""
        try:
            self._save_state()
        except:
            pass

if __name__ == "__main__":
    # Example usage and testing
    threshold_manager = InformationThresholdManager()
    
    # Register simple callback for testing
    def agent_callback(event: ThresholdEvent):
        print(f"Agent {event.agent_name} triggered with score {event.trigger_score:.1f}")
    
    for agent_name in [AgentNames.LEARNING_ANALYST, AgentNames.HPC_ADVISOR, AgentNames.TROUBLESHOOTING_DETECTIVE]:
        threshold_manager.register_agent_callback(agent_name, agent_callback)
    
    # Simulate information accumulation
    threshold_manager.accumulate_information(InformationTypes.NEW_COMMANDS, 5.0)
    threshold_manager.accumulate_information(InformationTypes.FAILURES, 8.0)
    threshold_manager.accumulate_information(InformationTypes.PERFORMANCE_SHIFTS, 6.0)
    
    # Print status
    status = threshold_manager.get_status()
    print(json.dumps(status, indent=2, default=str))