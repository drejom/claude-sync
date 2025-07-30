#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0",
#   "psutil>=5.9.0"
# ]
# ///
"""
Learning Engine - Main integration layer for the claude-sync learning system

This is the primary interface for the adaptive learning system that integrates
all learning components into a cohesive intelligence engine. Key features:

- Unified interface for all learning operations
- Automatic coordination between adaptive schema, threshold management, and storage
- Real-time performance monitoring and optimization
- Agent trigger coordination and feedback loops
- Hook integration points for PreToolUse, PostToolUse, and UserPromptSubmit
- Cross-host learning preparation and mesh sync coordination

Based on REFACTOR_PLAN.md complete learning system architecture
"""

import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
import logging

# Import all learning components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from interfaces import (
    CommandExecutionData,
    OptimizationPattern,
    LearningPattern,
    HookResult,
    PerformanceTargets,
    InformationTypes,
    AgentNames
)

from learning.adaptive_schema import AdaptiveLearningSchema
from learning.threshold_manager import InformationThresholdManager, ThresholdEvent
from learning.learning_storage import LearningStorage
from learning.command_abstractor import AdvancedCommandAbstractor
from learning.performance_monitor import PerformanceMonitor, PerformanceAlert

class LearningEngine:
    """
    Main learning engine that coordinates all learning system components.
    
    This class provides the primary interface for the claude-sync learning
    system, integrating adaptive schema evolution, information threshold
    management, secure storage, and performance monitoring into a unified
    intelligence engine.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        self.storage_dir = Path(storage_dir or Path.home() / '.claude' / 'learning')
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # Initialize core components
        self.schema = AdaptiveLearningSchema(self.storage_dir)
        self.threshold_manager = InformationThresholdManager(self.storage_dir)
        self.storage = LearningStorage(self.storage_dir)
        self.abstractor = AdvancedCommandAbstractor()
        self.performance_monitor = PerformanceMonitor(self.storage_dir)
        
        # Agent coordination
        self.agent_callbacks: Dict[str, Callable] = {}
        self.agent_feedback_handlers: Dict[str, Callable] = {}
        
        # Performance tracking
        self.operation_stats = {
            'commands_processed': 0,
            'patterns_learned': 0,
            'optimizations_suggested': 0,
            'agent_triggers': 0,
            'schema_evolutions': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Setup component integration
        self._setup_component_integration()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Learning engine initialized")
    
    def process_command_execution(self, command_data: CommandExecutionData) -> Dict[str, Any]:
        """
        Main entry point for processing command execution data.
        
        This method coordinates all learning operations when a command is executed,
        including pattern recognition, threshold management, and agent triggering.
        Performance target: <100ms total processing time.
        """
        if not self.enabled:
            return {'processed': False, 'reason': 'learning_disabled'}
        
        start_time = time.time()
        
        try:
            with self._lock:
                # Start performance monitoring
                operation_id = f"cmd_process_{int(time.time())}_{hash(command_data.command) % 1000}"
                self.performance_monitor.start_operation_monitoring(
                    operation_id, 'command_processing', 
                    {'command': command_data.command[:50]}  # First 50 chars for context
                )
                
                # 1. Comprehensive command analysis
                analysis_start = time.time()
                command_analysis = self.abstractor.analyze_command_comprehensive(
                    command_data.command, 
                    command_data.host_context
                )
                analysis_duration = (time.time() - analysis_start) * 1000
                self.performance_monitor.record_learning_operation('command_analysis', int(analysis_duration))
                
                # 2. Store execution data with adaptive schema learning
                storage_start = time.time()
                storage_success = self.storage.store_command_execution(command_data)
                storage_duration = (time.time() - storage_start) * 1000
                self.performance_monitor.record_learning_operation('data_storage', int(storage_duration))
                
                # 3. Update information thresholds
                threshold_start = time.time()
                self._accumulate_command_information(command_data, command_analysis)
                threshold_duration = (time.time() - threshold_start) * 1000
                self.performance_monitor.record_learning_operation('threshold_update', int(threshold_duration))
                
                # 4. Check for optimization opportunities
                optimization_start = time.time()
                optimizations = self._identify_optimizations(command_data, command_analysis)
                optimization_duration = (time.time() - optimization_start) * 1000
                self.performance_monitor.record_learning_operation('optimization_analysis', int(optimization_duration))
                
                # 5. Update statistics
                self.operation_stats['commands_processed'] += 1
                if optimizations:
                    self.operation_stats['optimizations_suggested'] += len(optimizations)
                
                # End performance monitoring
                total_duration = self.performance_monitor.end_operation_monitoring(operation_id, storage_success)
                
                # Return processing results
                return {
                    'processed': True,
                    'storage_success': storage_success,
                    'analysis': {
                        'category': command_analysis.command_category,
                        'complexity': command_analysis.complexity_score,
                        'learning_significance': command_analysis.learning_significance,
                        'safety_concerns': command_analysis.safety_concerns
                    },
                    'optimizations': optimizations,
                    'performance': {
                        'total_duration_ms': total_duration,
                        'target_met': total_duration <= PerformanceTargets.LEARNING_OPERATION_MS
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error processing command execution: {e}")
            return {'processed': False, 'error': str(e)}
    
    def get_command_suggestions(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get optimization suggestions for a command before execution.
        
        This method provides real-time suggestions for PreToolUse hooks.
        Performance target: <10ms for PreToolUse compatibility.
        """
        start_time = time.time()
        
        try:
            # Quick analysis for suggestions
            analysis = self.abstractor.analyze_command_comprehensive(command, context)
            
            # Get stored optimization patterns
            optimization_patterns = self.storage.get_optimization_patterns(command)
            
            # Get command statistics
            command_stats = self.storage.get_command_statistics(command)
            
            # Combine suggestions
            suggestions = {
                'optimization_opportunities': analysis.optimization_opportunities,
                'safety_warnings': analysis.safety_concerns,
                'stored_patterns': [
                    {
                        'optimized_command': pattern.optimized_pattern,
                        'confidence': pattern.confidence,
                        'success_rate': pattern.success_rate
                    }
                    for pattern in optimization_patterns
                ],
                'command_stats': command_stats,
                'complexity_analysis': {
                    'score': analysis.complexity_score,
                    'category': analysis.command_category,
                    'resource_indicators': analysis.resource_indicators
                }
            }
            
            # Performance tracking
            duration_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_learning_operation('command_suggestions', int(duration_ms))
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting command suggestions: {e}")
            return {'error': str(e)}
    
    def get_learning_context_for_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get relevant learning context to enhance user prompts.
        
        This method provides context enhancement for UserPromptSubmit hooks.
        Performance target: <20ms for UserPromptSubmit compatibility.
        """
        start_time = time.time()
        
        try:
            # Analyze prompt for learning relevance
            if not self._is_learning_relevant_prompt(prompt):
                return None
            
            # Extract relevant learning insights
            learning_context = []
            
            # Recent command patterns
            recent_patterns = self._get_recent_command_patterns()
            if recent_patterns:
                learning_context.append(f"Recent command patterns: {', '.join(recent_patterns[:3])}")
            
            # Performance insights
            perf_stats = self.performance_monitor.get_performance_stats(1)  # Last hour
            if perf_stats.get('total_operations', 0) > 0:
                violations = perf_stats.get('target_violations', 0)
                if violations > 0:
                    learning_context.append(f"Note: {violations} recent performance issues detected")
            
            # Schema evolution insights
            schema_info = self.schema.get_current_schema()
            if schema_info.get('evolution_history'):
                recent_evolution = schema_info['evolution_history'][-1]
                learning_context.append(f"Recent learning evolution: {recent_evolution.get('changes', [])[0].get('reason', 'pattern improvement')}")
            
            # Combine context
            if learning_context:
                context_text = "\n[Learning Context]\n" + "\n".join(f"- {item}" for item in learning_context)
                
                # Performance tracking
                duration_ms = (time.time() - start_time) * 1000
                self.performance_monitor.record_learning_operation('prompt_enhancement', int(duration_ms))
                
                return context_text
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting learning context for prompt: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive learning system status"""
        try:
            return {
                'enabled': self.enabled,
                'operation_stats': self.operation_stats,
                'schema_status': {
                    'version': self.schema.schema_version,
                    'pattern_count': len(self.schema.pattern_registry),
                    'top_patterns': list(self.schema.usage_frequency.most_common(5))
                },
                'threshold_status': self.threshold_manager.get_status(),
                'storage_status': {
                    'total_patterns': len(self.storage.command_patterns),
                    'cache_stats': {
                        'optimization_cache_size': len(self.storage.optimization_cache),
                        'statistics_cache_size': len(self.storage.statistics_cache)
                    }
                },
                'performance_status': self.performance_monitor.get_real_time_status(),
                'health_score': self._calculate_overall_health()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def trigger_schema_evolution(self) -> bool:
        """Manually trigger schema evolution"""
        try:
            if self.schema.should_evolve_schema():
                success = self.schema.evolve_schema()
                if success:
                    self.operation_stats['schema_evolutions'] += 1
                return success
            return False
            
        except Exception as e:
            self.logger.error(f"Error triggering schema evolution: {e}")
            return False
    
    def register_agent_callback(self, agent_name: str, callback: Callable) -> None:
        """Register callback for agent trigger events"""
        self.agent_callbacks[agent_name] = callback
        
        # Also register with threshold manager
        self.threshold_manager.register_agent_callback(agent_name, callback)
    
    def provide_agent_feedback(self, agent_name: str, analysis_id: str, effectiveness_score: float) -> None:
        """Provide feedback on agent analysis effectiveness"""
        try:
            # Update threshold manager with effectiveness
            self.threshold_manager.adapt_threshold(agent_name, effectiveness_score)
            
            # Update operation stats
            if effectiveness_score > 0.7:  # Good effectiveness
                self.operation_stats['agent_triggers'] += 1
            
        except Exception as e:
            self.logger.error(f"Error providing agent feedback: {e}")
    
    def cleanup_expired_data(self, retention_days: int = 30) -> Dict[str, int]:
        """Clean up expired learning data across all components"""
        try:
            results = {}
            
            # Clean up storage
            results['storage_patterns'] = self.storage.cleanup_expired_data(retention_days)
            
            # Clean up performance data
            # (This would be implemented based on specific needs)
            results['performance_metrics'] = 0  # Placeholder
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired data: {e}")
            return {'error': str(e)}
    
    def enable_learning(self) -> None:
        """Enable learning system"""
        self.enabled = True
        self.logger.info("Learning system enabled")
    
    def disable_learning(self) -> None:
        """Disable learning system"""
        self.enabled = False
        self.logger.info("Learning system disabled")
    
    def _setup_component_integration(self) -> None:
        """Setup integration between components"""
        # Register performance alert handler
        def performance_alert_handler(alert: PerformanceAlert):
            # Convert performance issues to threshold information
            if alert.severity == 'critical':
                self.threshold_manager.accumulate_information(
                    InformationTypes.PERFORMANCE_SHIFTS, 
                    3.0,  # High significance for critical alerts
                    {'alert_type': alert.alert_type, 'operation': alert.operation_type}
                )
        
        self.performance_monitor.register_alert_callback(performance_alert_handler)
        
        # Register threshold event handler
        def threshold_event_handler(event: ThresholdEvent):
            self.logger.info(f"Agent {event.agent_name} triggered by threshold system")
            if event.agent_name in self.agent_callbacks:
                try:
                    self.agent_callbacks[event.agent_name](event)
                except Exception as e:
                    self.logger.error(f"Error in agent callback: {e}")
        
        # Register the handler with threshold manager
        for agent_name in [AgentNames.LEARNING_ANALYST, AgentNames.HPC_ADVISOR, AgentNames.TROUBLESHOOTING_DETECTIVE]:
            self.threshold_manager.register_agent_callback(agent_name, threshold_event_handler)
    
    def _accumulate_command_information(self, command_data: CommandExecutionData, analysis) -> None:
        """Accumulate information from command execution for threshold system"""
        try:
            # Calculate significance scores based on analysis
            significance_scores = self.threshold_manager.calculate_information_significance(command_data)
            
            # Add analysis-based significance
            if analysis.learning_significance > 2.0:
                significance_scores[InformationTypes.NEW_COMMANDS] = analysis.learning_significance
            
            if analysis.safety_concerns:
                significance_scores[InformationTypes.FAILURES] = len(analysis.safety_concerns) * 1.5
            
            if analysis.optimization_opportunities:
                significance_scores[InformationTypes.OPTIMIZATIONS] = len(analysis.optimization_opportunities) * 1.2
            
            # Accumulate information
            for info_type, significance in significance_scores.items():
                self.threshold_manager.accumulate_information(info_type, significance)
                
        except Exception as e:
            self.logger.error(f"Error accumulating command information: {e}")
    
    def _identify_optimizations(self, command_data: CommandExecutionData, analysis) -> List[Dict[str, Any]]:
        """Identify optimization opportunities from analysis"""
        optimizations = []
        
        # Add analysis-based optimizations
        for opportunity in analysis.optimization_opportunities:
            optimizations.append({
                'type': 'analysis_suggestion',
                'description': opportunity,
                'confidence': 0.8,
                'source': 'command_analysis'
            })
        
        # Add stored pattern optimizations
        stored_patterns = self.storage.get_optimization_patterns(command_data.command)
        for pattern in stored_patterns:
            optimizations.append({
                'type': 'learned_pattern',
                'description': f"Consider: {pattern.optimized_pattern}",
                'confidence': pattern.confidence,
                'source': 'stored_patterns'
            })
        
        return optimizations
    
    def _is_learning_relevant_prompt(self, prompt: str) -> bool:
        """Check if prompt is relevant for learning context enhancement"""
        learning_keywords = [
            'command', 'script', 'run', 'execute', 'optimize', 'performance',
            'slurm', 'sbatch', 'container', 'singularity', 'docker',
            'analysis', 'workflow', 'pipeline', 'bioinformatics'
        ]
        
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in learning_keywords)
    
    def _get_recent_command_patterns(self) -> List[str]:
        """Get recent command patterns for context"""
        try:
            # Get top patterns from schema
            top_patterns = self.schema.usage_frequency.most_common(10)
            return [pattern for pattern, count in top_patterns if count > 5]
        except:
            return []
    
    def _calculate_overall_health(self) -> float:
        """Calculate overall learning system health score"""
        try:
            health_components = []
            
            # Performance health
            perf_health = self.performance_monitor._calculate_system_health()
            health_components.append(perf_health * 0.4)  # 40% weight
            
            # Schema health (based on evolution activity)
            schema_health = 100.0
            if len(self.schema.evolution_history) > 0:
                recent_evolution = self.schema.evolution_history[-1]
                time_since_evolution = time.time() - recent_evolution.get('timestamp', 0)
                if time_since_evolution < 86400:  # Less than 24 hours
                    schema_health = 100.0
                elif time_since_evolution < 604800:  # Less than week
                    schema_health = 80.0
                else:
                    schema_health = 60.0
            health_components.append(schema_health * 0.3)  # 30% weight
            
            # Storage health (based on successful operations)
            storage_perf = self.storage.get_performance_statistics()
            storage_health = 100.0
            if 'avg_store_time_ms' in storage_perf:
                if storage_perf['avg_store_time_ms'] > PerformanceTargets.LEARNING_OPERATION_MS:
                    storage_health = 70.0
            health_components.append(storage_health * 0.3)  # 30% weight
            
            return sum(health_components)
            
        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return 50.0
    
    def shutdown(self) -> None:
        """Shutdown learning engine and all components"""
        try:
            self.logger.info("Shutting down learning engine...")
            
            # Shutdown performance monitor
            self.performance_monitor.shutdown()
            
            # Save final state for all components
            # (Components have their own shutdown/save mechanisms)
            
            self.logger.info("Learning engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except:
            pass

# Factory function for creating learning engine instances
def create_learning_engine(storage_dir: Optional[Path] = None, config: Optional[Dict[str, Any]] = None) -> LearningEngine:
    """Factory function to create learning engine instance"""
    return LearningEngine(storage_dir, config)

if __name__ == "__main__":
    # Example usage and testing
    engine = create_learning_engine()
    
    # Register a simple agent callback
    def test_agent_callback(event):
        print(f"Agent triggered: {event.agent_name} with score {event.trigger_score:.1f}")
    
    engine.register_agent_callback(AgentNames.LEARNING_ANALYST, test_agent_callback)
    
    # Test command processing
    test_commands = [
        CommandExecutionData("sbatch --mem=32G job.sh", 0, 1500, time.time(), "test", "/home/user/project"),
        CommandExecutionData("grep -r pattern *.txt", 0, 800, time.time(), "test", "/home/user/data"),
        CommandExecutionData("python analysis.py", 1, 5000, time.time(), "test", "/home/user/analysis")
    ]
    
    for cmd_data in test_commands:
        result = engine.process_command_execution(cmd_data)
        print(f"Processed command: {result['processed']}")
        if result.get('optimizations'):
            print(f"  Optimizations: {len(result['optimizations'])}")
    
    # Test command suggestions
    suggestions = engine.get_command_suggestions("grep pattern file.txt")
    print(f"Suggestions for grep command: {len(suggestions.get('optimization_opportunities', []))}")
    
    # Test prompt enhancement
    context = engine.get_learning_context_for_prompt("Help me optimize my SLURM jobs")
    if context:
        print(f"Learning context provided: {context[:100]}...")
    
    # Get system status
    status = engine.get_system_status()
    print(f"System health: {status.get('health_score', 0):.1f}%")
    
    # Cleanup
    engine.shutdown()