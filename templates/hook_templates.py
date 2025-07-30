#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "typing-extensions>=4.0.0"
# ]
# ///
"""
Claude-Sync Hook Templates

Templates showing exact implementation patterns for the three main hooks:
- intelligent-optimizer.py (PreToolUse)
- learning-collector.py (PostToolUse)  
- context-enhancer.py (UserPromptSubmit)

These templates demonstrate the interfaces, performance patterns, and 
error handling required for production hooks.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from interfaces import (
    HookResult, CommandExecutionData, OptimizationPattern,
    PreToolUseHookInterface, PostToolUseHookInterface, 
    UserPromptSubmitHookInterface, PerformanceTargets
)

# ============================================================================
# Performance Monitoring Decorator
# ============================================================================

def performance_monitored(max_duration_ms: int):
    """Decorator to monitor hook performance and enforce limits"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Log performance if over target
                if duration_ms > max_duration_ms:
                    print(f"WARNING: {func.__name__} took {duration_ms:.1f}ms (target: {max_duration_ms}ms)", 
                          file=sys.stderr)
                
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                print(f"ERROR: {func.__name__} failed after {duration_ms:.1f}ms: {e}", 
                      file=sys.stderr)
                # Never let hook failures break Claude Code
                return HookResult(block=False, message=None)
        return wrapper
    return decorator

# ============================================================================
# PreToolUse Hook Template - intelligent-optimizer.py
# ============================================================================

class IntelligentOptimizerHook(PreToolUseHookInterface):
    """
    PreToolUse hook template for command optimization
    
    Performance Target: <10ms execution
    Responsibilities:
    - Analyze commands for optimization opportunities
    - Provide safety warnings for dangerous operations
    - Suggest alternative commands based on learned patterns
    """
    
    def __init__(self):
        self.learning_storage = self._get_learning_storage()
        self.abstractor = self._get_abstractor()
        self.performance_monitor = self._get_performance_monitor()
    
    def _get_learning_storage(self):
        """Get learning storage with fallback"""
        try:
            # Import learning modules
            return None  # TODO: Implement
        except ImportError:
            return None
    
    def _get_abstractor(self):
        """Get abstraction system with fallback"""
        try:
            return None  # TODO: Implement
        except ImportError:
            return None
    
    def _get_performance_monitor(self):
        """Get performance monitor with fallback"""
        try:
            return None  # TODO: Implement
        except ImportError:
            return None
    
    @performance_monitored(PerformanceTargets.PRE_TOOL_USE_HOOK_MS)
    def execute(self, hook_input: Dict[str, Any]) -> HookResult:
        """Main hook execution entry point"""
        try:
            # Extract command information
            tool_name = hook_input.get('tool_name')
            if tool_name != 'Bash':
                return HookResult(block=False)
            
            command = hook_input.get('tool_input', {}).get('command', '')
            if not command.strip():
                return HookResult(block=False)
            
            # Analyze command
            analysis = self.analyze_command(command, hook_input.get('context', {}))
            
            # Generate response
            suggestions = []
            
            # Add optimization suggestions
            optimizations = self.suggest_optimizations(command)
            if optimizations:
                best_optimization = max(optimizations, key=lambda x: x.confidence)
                if best_optimization.confidence > 0.8:
                    suggestions.append(
                        f"🚀 **High-confidence optimization:**\n```bash\n{best_optimization.optimized_pattern}\n```"
                    )
                elif best_optimization.confidence > 0.6:
                    suggestions.append(
                        f"💡 **Suggested optimization (confidence: {best_optimization.confidence:.0%}):**\n```bash\n{best_optimization.optimized_pattern}\n```"
                    )
            
            # Add safety warnings
            safety_issues = self.check_safety(command)
            if safety_issues:
                warnings = "⚠️ **Safety analysis:**\n" + "\n".join(f"  {issue}" for issue in safety_issues)
                suggestions.append(warnings)
            
            # Add performance insights
            if analysis.get('performance_insights'):
                suggestions.append(f"📊 **Performance insight:** {analysis['performance_insights']}")
            
            if suggestions:
                message = "\n\n".join(suggestions)
                return HookResult(block=False, message=message)
            
            return HookResult(block=False)
            
        except Exception as e:
            # Never break Claude Code execution
            print(f"IntelligentOptimizer error: {e}", file=sys.stderr)
            return HookResult(block=False)
    
    def analyze_command(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze command for patterns and characteristics"""
        analysis = {
            'command_type': self._classify_command(command),
            'complexity': self._calculate_complexity(command),
            'risk_level': self._assess_risk(command),
            'optimization_potential': True
        }
        
        # Add context-specific analysis
        if 'working_directory' in context:
            analysis['context_aware'] = True
        
        # Add performance insights if available
        if self.learning_storage:
            historical_data = self.learning_storage.get_command_statistics(command)
            if historical_data.get('avg_duration_ms', 0) > 5000:
                analysis['performance_insights'] = f"This command typically takes {historical_data['avg_duration_ms']/1000:.1f}s"
        
        return analysis
    
    def suggest_optimizations(self, command: str) -> List[OptimizationPattern]:
        """Suggest command optimizations based on learned patterns"""
        optimizations = []
        
        # Basic optimizations (always available)
        basic_optimizations = self._get_basic_optimizations(command)
        optimizations.extend(basic_optimizations)
        
        # Learned optimizations (if learning system available)
        if self.learning_storage:
            learned_patterns = self.learning_storage.get_optimization_patterns(command)
            optimizations.extend(learned_patterns)
        
        return optimizations
    
    def check_safety(self, command: str) -> List[str]:
        """Check for potential safety issues"""
        warnings = []
        
        # Destructive operations
        if any(dangerous in command for dangerous in ['rm -rf', 'dd if=', '> /dev/', 'mkfs']):
            warnings.append("Potentially destructive operation detected")
        
        # Privilege escalation
        if any(priv in command for priv in ['sudo', 'su -', 'chmod 777']):
            warnings.append("Privilege escalation detected")
        
        # Network operations with security implications
        if any(net in command for net in ['curl -k', 'wget --no-check-certificate']):
            warnings.append("Insecure network operation detected")
        
        return warnings
    
    def _classify_command(self, command: str) -> str:
        """Classify command type"""
        if command.startswith('sbatch'):
            return 'slurm_submission'
        elif command.startswith('ssh'):
            return 'remote_connection'
        elif 'singularity' in command:
            return 'container_execution'
        elif command.startswith('Rscript'):
            return 'r_execution'
        elif any(cmd in command for cmd in ['grep', 'find', 'awk', 'sed']):
            return 'text_processing'
        else:
            return 'general_bash'
    
    def _calculate_complexity(self, command: str) -> int:
        """Calculate command complexity (1-10 scale)"""
        complexity = 1
        complexity += command.count('|')  # Pipes
        complexity += command.count('&&') + command.count('||')  # Logical operators
        complexity += len([part for part in command.split() if part.startswith('-')])  # Flags
        return min(complexity, 10)
    
    def _assess_risk(self, command: str) -> str:
        """Assess command risk level"""
        if any(danger in command for danger in ['rm -rf', 'dd', 'mkfs', '> /dev/']):
            return 'high'
        elif any(caution in command for caution in ['sudo', 'chmod', 'chown']):
            return 'medium'
        else:
            return 'low'
    
    def _get_basic_optimizations(self, command: str) -> List[OptimizationPattern]:
        """Get basic optimizations that don't require learning"""
        optimizations = []
        
        # grep -> rg optimization
        if 'grep' in command and 'rg' not in command:
            optimized = command.replace('grep', 'rg')
            optimizations.append(OptimizationPattern(
                original_pattern=command,
                optimized_pattern=optimized,
                confidence=0.9,
                success_rate=0.95,
                application_count=0,
                created_at=time.time(),
                last_used=time.time(),
                categories=['performance', 'tool_upgrade']
            ))
        
        # find -> fd optimization
        if command.startswith('find') and 'fd' not in command:
            # Simple find -> fd conversion
            if '-name' in command:
                optimized = command.replace('find', 'fd').replace('-name', '')
                optimizations.append(OptimizationPattern(
                    original_pattern=command,
                    optimized_pattern=optimized,
                    confidence=0.7,
                    success_rate=0.9,
                    application_count=0,
                    created_at=time.time(),
                    last_used=time.time(),
                    categories=['performance', 'tool_upgrade']
                ))
        
        return optimizations
    
    def get_execution_time_limit_ms(self) -> int:
        """Return execution time limit"""
        return PerformanceTargets.PRE_TOOL_USE_HOOK_MS


# ============================================================================
# PostToolUse Hook Template - learning-collector.py
# ============================================================================

class LearningCollectorHook(PostToolUseHookInterface):
    """
    PostToolUse hook template for learning data collection
    
    Performance Target: <50ms execution
    Responsibilities:
    - Extract learning data from command execution
    - Update success/failure patterns
    - Trigger adaptive schema evolution
    - Background learning operations
    """
    
    def __init__(self):
        self.learning_storage = self._get_learning_storage()
        self.abstractor = self._get_abstractor()
        self.threshold_manager = self._get_threshold_manager()
    
    def _get_learning_storage(self):
        """Get learning storage with fallback"""
        return None  # TODO: Implement
    
    def _get_abstractor(self):
        """Get abstraction system with fallback"""
        return None  # TODO: Implement
    
    def _get_threshold_manager(self):
        """Get threshold manager with fallback"""
        return None  # TODO: Implement
    
    @performance_monitored(PerformanceTargets.POST_TOOL_USE_HOOK_MS)
    def execute(self, hook_input: Dict[str, Any]) -> HookResult:
        """Main hook execution entry point"""
        try:
            # Extract learning data
            execution_data = self.extract_learning_data(hook_input)
            
            # Store learning data
            if self.learning_storage:
                self.learning_storage.store_command_execution(execution_data)
            
            # Analyze performance
            performance_analysis = self.analyze_performance(execution_data)
            
            # Update patterns
            self.update_patterns(execution_data)
            
            # Accumulate information for threshold system
            if self.threshold_manager:
                self._accumulate_threshold_information(execution_data)
            
            # PostToolUse hooks should not show messages to user
            # All learning happens silently in background
            return HookResult(block=False)
            
        except Exception as e:
            print(f"LearningCollector error: {e}", file=sys.stderr)
            return HookResult(block=False)
    
    def extract_learning_data(self, hook_input: Dict[str, Any]) -> CommandExecutionData:
        """Extract learning data from hook input"""
        return CommandExecutionData.from_hook_input(hook_input)
    
    def analyze_performance(self, execution_data: CommandExecutionData) -> Dict[str, Any]:
        """Analyze command performance"""
        analysis = {
            'duration_category': self._categorize_duration(execution_data.duration_ms),
            'success': execution_data.exit_code == 0,
            'performance_tier': self._assess_performance_tier(execution_data)
        }
        
        # Compare with historical performance if available
        if self.learning_storage:
            historical = self.learning_storage.get_command_statistics(execution_data.command)
            if historical.get('avg_duration_ms'):
                ratio = execution_data.duration_ms / historical['avg_duration_ms']
                if ratio > 1.5:
                    analysis['performance_anomaly'] = 'slower_than_expected'
                elif ratio < 0.5:
                    analysis['performance_anomaly'] = 'faster_than_expected'
        
        return analysis
    
    def update_patterns(self, execution_data: CommandExecutionData) -> bool:
        """Update learning patterns based on execution"""
        if not self.learning_storage:
            return False
        
        try:
            # Update success/failure patterns
            success = execution_data.exit_code == 0
            
            # Abstract the command for pattern learning
            if self.abstractor:
                abstract_command = self.abstractor.abstract_command(execution_data.command)
                
                # Store abstracted pattern (safe for learning)
                pattern_data = {
                    'abstract_command': abstract_command,
                    'success': success,
                    'duration_ms': execution_data.duration_ms,
                    'timestamp': execution_data.timestamp,
                    'context': self.abstractor.abstract_execution_context(
                        execution_data.host_context or {}
                    )
                }
                
                # This would update the learning patterns
                # Implementation depends on learning storage design
            
            return True
            
        except Exception as e:
            print(f"Pattern update error: {e}", file=sys.stderr)
            return False
    
    def _accumulate_threshold_information(self, execution_data: CommandExecutionData) -> None:
        """Accumulate information for threshold-based agent triggering"""
        if not self.threshold_manager:
            return
        
        # Calculate significance of this execution
        significance = 1.0
        
        # Failures are more significant
        if execution_data.exit_code != 0:
            significance = 3.0
            self.threshold_manager.accumulate_information('failures', significance)
            
        # New command patterns are significant
        command_novelty = self._assess_command_novelty(execution_data.command)
        if command_novelty > 0.5:
            self.threshold_manager.accumulate_information('new_commands', command_novelty * 2.0)
        
        # Performance anomalies are significant
        performance_anomaly = self._detect_performance_anomaly(execution_data)
        if performance_anomaly > 0.3:
            self.threshold_manager.accumulate_information('performance_shifts', performance_anomaly * 4.0)
    
    def _categorize_duration(self, duration_ms: int) -> str:
        """Categorize command duration"""
        if duration_ms < 100:
            return 'instant'
        elif duration_ms < 1000:
            return 'fast'
        elif duration_ms < 10000:
            return 'medium'
        elif duration_ms < 60000:
            return 'slow'
        else:
            return 'very_slow'
    
    def _assess_performance_tier(self, execution_data: CommandExecutionData) -> str:
        """Assess overall performance tier"""
        if execution_data.exit_code != 0:
            return 'failed'
        elif execution_data.duration_ms < 1000:
            return 'high'
        elif execution_data.duration_ms < 10000:
            return 'medium'
        else:
            return 'low'
    
    def _assess_command_novelty(self, command: str) -> float:
        """Assess how novel this command is (0.0 = common, 1.0 = completely new)"""
        # Simplified novelty assessment
        # Real implementation would check against learning data
        if not self.learning_storage:
            return 0.0
        
        # Check if we've seen this command pattern before
        patterns = self.learning_storage.get_optimization_patterns(command)
        if not patterns:
            return 1.0  # Completely new
        
        # Check frequency
        total_executions = sum(p.application_count for p in patterns)
        if total_executions < 5:
            return 0.7  # Relatively new
        elif total_executions < 20:
            return 0.3  # Somewhat familiar
        else:
            return 0.1  # Very familiar
    
    def _detect_performance_anomaly(self, execution_data: CommandExecutionData) -> float:
        """Detect performance anomaly (0.0 = normal, 1.0 = major anomaly)"""
        if not self.learning_storage:
            return 0.0
        
        stats = self.learning_storage.get_command_statistics(execution_data.command)
        if not stats.get('avg_duration_ms'):
            return 0.0
        
        expected = stats['avg_duration_ms']
        actual = execution_data.duration_ms
        
        if expected == 0:
            return 0.0
        
        ratio = abs(actual - expected) / expected
        return min(ratio, 1.0)
    
    def get_execution_time_limit_ms(self) -> int:
        """Return execution time limit"""
        return PerformanceTargets.POST_TOOL_USE_HOOK_MS


# ============================================================================
# UserPromptSubmit Hook Template - context-enhancer.py
# ============================================================================

class ContextEnhancerHook(UserPromptSubmitHookInterface):
    """
    UserPromptSubmit hook template for context enhancement
    
    Performance Target: <20ms execution
    Responsibilities:
    - Detect when user needs context from learning data
    - Inject relevant historical patterns
    - Enhance prompts with learned insights
    """
    
    def __init__(self):
        self.learning_storage = self._get_learning_storage()
        self.agent_knowledge = self._get_agent_knowledge()
    
    def _get_learning_storage(self):
        """Get learning storage with fallback"""
        return None  # TODO: Implement
    
    def _get_agent_knowledge(self):
        """Get agent knowledge interface with fallback"""
        return None  # TODO: Implement
    
    @performance_monitored(PerformanceTargets.USER_PROMPT_SUBMIT_HOOK_MS)
    def execute(self, hook_input: Dict[str, Any]) -> HookResult:
        """Main hook execution entry point"""
        try:
            user_prompt = hook_input.get('user_prompt', '')
            if not user_prompt.strip():
                return HookResult(block=False)
            
            # Detect what context might be helpful
            context_needs = self.detect_context_needs(user_prompt)
            
            if not context_needs:
                return HookResult(block=False)
            
            # Enhance with relevant context
            context_enhancement = self.enhance_with_context(user_prompt)
            
            if context_enhancement:
                message = f"🧠 **Added context from learning data:**\n{context_enhancement}"
                return HookResult(block=False, message=message)
            
            return HookResult(block=False)
            
        except Exception as e:
            print(f"ContextEnhancer error: {e}", file=sys.stderr)
            return HookResult(block=False)
    
    def detect_context_needs(self, prompt: str) -> List[str]:
        """Detect what context might be helpful"""
        context_needs = []
        prompt_lower = prompt.lower()
        
        # SLURM/HPC context
        if any(keyword in prompt_lower for keyword in ['sbatch', 'slurm', 'queue', 'partition', 'hpc']):
            context_needs.append('slurm_patterns')
        
        # Container context
        if any(keyword in prompt_lower for keyword in ['singularity', 'container', 'docker']):
            context_needs.append('container_patterns')
        
        # R/data analysis context
        if any(keyword in prompt_lower for keyword in ['rscript', 'r analysis', 'data analysis']):
            context_needs.append('r_patterns')
        
        # SSH/networking context
        if any(keyword in prompt_lower for keyword in ['ssh', 'remote', 'tailscale', 'connection']):
            context_needs.append('network_patterns')
        
        # Performance troubleshooting
        if any(keyword in prompt_lower for keyword in ['slow', 'performance', 'optimize', 'faster']):
            context_needs.append('performance_insights')
        
        # Error troubleshooting
        if any(keyword in prompt_lower for keyword in ['error', 'failed', 'debug', 'troubleshoot']):
            context_needs.append('error_patterns')
        
        return context_needs
    
    def enhance_with_context(self, prompt: str) -> Optional[str]:
        """Add relevant learning context to prompt"""
        if not self.learning_storage:
            return None
        
        context_needs = self.detect_context_needs(prompt)
        enhancements = []
        
        for need in context_needs:
            enhancement = self._get_context_for_need(need)
            if enhancement:
                enhancements.append(enhancement)
        
        if enhancements:
            return "\n\n".join(enhancements)
        
        return None
    
    def _get_context_for_need(self, need: str) -> Optional[str]:
        """Get specific context based on detected need"""
        if need == 'slurm_patterns':
            return self._get_slurm_context()
        elif need == 'container_patterns':
            return self._get_container_context()
        elif need == 'r_patterns':
            return self._get_r_context()
        elif need == 'network_patterns':
            return self._get_network_context()
        elif need == 'performance_insights':
            return self._get_performance_context()
        elif need == 'error_patterns':
            return self._get_error_context()
        
        return None
    
    def _get_slurm_context(self) -> Optional[str]:
        """Get SLURM-related context from learning data"""
        # This would query learning data for SLURM patterns
        # For template, return example structure
        return """**Learned SLURM patterns on this cluster:**
- Most successful partition: `compute` (95% success rate)
- Typical memory allocation: 16-32GB for genomics workflows
- Average queue time: 5 minutes on compute, 15 minutes on gpu"""
    
    def _get_container_context(self) -> Optional[str]:
        """Get container-related context"""
        return """**Common container patterns:**
- Preferred bind mounts: `/data`, `/scratch`, `/home`
- Most used containers: `r-analysis.sif`, `biotools.sif`
- Success rate: 88% with proper bind mounts"""
    
    def _get_r_context(self) -> Optional[str]:
        """Get R-related context"""
        return """**R workflow patterns:**
- Memory usage: Average 4GB, peak 8GB for typical analyses
- Common flags: `--vanilla`, `--max-mem-size=8G`
- Failure modes: memory exhaustion (45%), missing packages (30%)"""
    
    def _get_network_context(self) -> Optional[str]:
        """Get network-related context"""
        return """**Network connection patterns:**
- Tailscale: 96% reliability, average 50Mbps
- SSH fallback: 95% reliability, average 15Mbps
- Best for large transfers: rsync via Tailscale"""
    
    def _get_performance_context(self) -> Optional[str]:
        """Get performance-related context"""
        return """**Performance insights:**
- Commands >1GB data: Use compute nodes, not login nodes
- R analyses: Memory usage scales ~2x with input size
- Container overhead: ~10% performance penalty"""
    
    def _get_error_context(self) -> Optional[str]:
        """Get error pattern context"""
        return """**Common error patterns:**
- Memory errors: 60% from underestimating R memory needs
- Connection failures: 25% during peak hours (2-4pm)
- Container issues: Usually bind mount problems"""
    
    def get_execution_time_limit_ms(self) -> int:
        """Return execution time limit"""
        return PerformanceTargets.USER_PROMPT_SUBMIT_HOOK_MS


# ============================================================================
# Hook Factory Functions
# ============================================================================

def create_intelligent_optimizer_hook() -> IntelligentOptimizerHook:
    """Factory function for creating intelligent optimizer hook"""
    return IntelligentOptimizerHook()

def create_learning_collector_hook() -> LearningCollectorHook:  
    """Factory function for creating learning collector hook"""
    return LearningCollectorHook()

def create_context_enhancer_hook() -> ContextEnhancerHook:
    """Factory function for creating context enhancer hook"""
    return ContextEnhancerHook()

# ============================================================================
# Main Function for Testing
# ============================================================================

def main():
    """Test function for hook templates"""
    print("Claude-Sync Hook Templates")
    print("=" * 50)
    
    # Test hook creation
    optimizer = create_intelligent_optimizer_hook()
    collector = create_learning_collector_hook()
    enhancer = create_context_enhancer_hook()
    
    print(f"✅ IntelligentOptimizerHook - Target: {optimizer.get_execution_time_limit_ms()}ms")
    print(f"✅ LearningCollectorHook - Target: {collector.get_execution_time_limit_ms()}ms")
    print(f"✅ ContextEnhancerHook - Target: {enhancer.get_execution_time_limit_ms()}ms")
    
    # Test with sample input
    sample_hook_input = {
        'tool_name': 'Bash',
        'tool_input': {'command': 'grep -r "pattern" /data/'},
        'tool_output': {'exit_code': 0, 'duration_ms': 1500},
        'context': {'working_directory': '/home/user', 'timestamp': time.time()}
    }
    
    print("\n📊 Testing hooks with sample input...")
    
    # Test optimizer
    result = optimizer.execute(sample_hook_input)
    print(f"Optimizer result: block={result.block}, has_message={result.message is not None}")
    
    # Test collector
    result = collector.execute(sample_hook_input)
    print(f"Collector result: block={result.block}, has_message={result.message is not None}")
    
    # Test enhancer
    prompt_input = {'user_prompt': 'How do I optimize my SLURM jobs?'}
    result = enhancer.execute(prompt_input)
    print(f"Enhancer result: block={result.block}, has_message={result.message is not None}")

if __name__ == "__main__":
    main()