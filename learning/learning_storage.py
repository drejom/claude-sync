#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0"
# ]
# ///
"""
Learning Storage - Encrypted persistence and fast pattern lookups for AI learning data

This implements the main learning storage system that integrates adaptive schema
evolution, information threshold management, and encrypted persistence. Key features:

- Encrypted learning data storage with automatic key rotation
- Fast pattern lookups (<1ms target for common queries)
- Integration with AdaptiveLearningSchema for schema evolution
- Command execution data abstraction and storage
- Cross-host learning data preparation
- Performance monitoring and optimization

Based on REFACTOR_PLAN.md sections 310-508 (Knowledge Storage Architecture)
"""

import json
import time
import hashlib
import pickle
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import threading

# Import our interfaces and components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from interfaces import (
    LearningStorageInterface,
    CommandExecutionData,
    OptimizationPattern,
    LearningPattern,
    PerformanceTargets
)

# Import our learning components
from learning.adaptive_schema import AdaptiveLearningSchema
from learning.threshold_manager import InformationThresholdManager
from learning.abstraction import SecureAbstractor
from learning.encryption import SecureLearningStorage as SecureStorage

@dataclass
class CommandPattern:
    """Abstracted command pattern for learning"""
    pattern_id: str
    abstract_command: str
    command_category: str
    complexity_score: float
    success_rate: float
    average_duration_ms: int
    execution_count: int
    first_seen: float
    last_seen: float
    optimization_suggestions: List[str]
    
    def update_with_execution(self, execution_data: CommandExecutionData) -> None:
        """Update pattern with new execution data"""
        self.execution_count += 1
        self.last_seen = time.time()
        
        # Update success rate
        if execution_data.exit_code == 0:
            # Weighted average for success rate
            weight = 1.0 / self.execution_count
            self.success_rate = (self.success_rate * (1 - weight)) + weight
        else:
            weight = 1.0 / self.execution_count  
            self.success_rate = self.success_rate * (1 - weight)
        
        # Update average duration
        weight = min(0.1, 1.0 / self.execution_count)  # Limit weight for stability
        self.average_duration_ms = int(
            self.average_duration_ms * (1 - weight) + 
            execution_data.duration_ms * weight
        )

class LearningStorage(LearningStorageInterface):
    """
    Main learning storage system with encrypted persistence and fast lookups.
    
    This class provides the primary interface for storing and retrieving learning
    data, integrating adaptive schema evolution, information threshold management,
    and secure storage systems.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir or Path.home() / '.claude' / 'learning')
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component systems
        self.adaptive_schema = AdaptiveLearningSchema(self.storage_dir)
        self.threshold_manager = InformationThresholdManager(self.storage_dir)
        self.abstractor = SecureAbstractor()
        self.secure_storage = SecureStorage(self.storage_dir)
        
        # In-memory caches for fast lookups
        self.command_patterns: Dict[str, CommandPattern] = {}
        self.optimization_cache: Dict[str, List[OptimizationPattern]] = {}
        self.statistics_cache: Dict[str, Dict[str, Any]] = {}
        
        # Cache management
        self.cache_last_updated = time.time()
        self.cache_update_interval = 300  # Update cache every 5 minutes
        
        # Performance tracking
        self.performance_stats = {
            'store_operations': [],
            'lookup_operations': [],
            'schema_evolutions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing data
        self._load_cached_data()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def store_command_execution(self, data: CommandExecutionData) -> bool:
        """
        Store command execution data with automatic abstraction and learning.
        
        This method processes new command execution data, abstracts sensitive
        information, updates learning patterns, and triggers schema evolution
        if needed. Performance target: <100ms.
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Abstract sensitive data
                abstracted_data = self._abstract_command_data(data)
                
                # Update adaptive schema (this may trigger evolution)
                self.adaptive_schema.observe_command_pattern(data)
                
                # Update or create command pattern
                pattern_id = self._get_pattern_id(abstracted_data)
                self._update_command_pattern(pattern_id, data, abstracted_data)
                
                # Accumulate information for threshold system
                self._accumulate_threshold_information(data)
                
                # Store encrypted data
                success = self._store_encrypted_execution_data(abstracted_data)
                
                # Update caches if needed
                self._update_caches_if_needed()
                
                # Performance tracking
                duration_ms = (time.time() - start_time) * 1000
                self.performance_stats['store_operations'].append(duration_ms)
                
                if duration_ms > PerformanceTargets.LEARNING_OPERATION_MS:
                    self.logger.warning(f"Store operation took {duration_ms:.1f}ms (target: {PerformanceTargets.LEARNING_OPERATION_MS}ms)")
                
                return success
                
        except Exception as e:
            self.logger.error(f"Error storing command execution: {e}")
            return False
    
    def get_optimization_patterns(self, command: str) -> List[OptimizationPattern]:
        """
        Retrieve optimization patterns for command with fast lookup.
        
        Performance target: <1ms for cached lookups.
        """
        start_time = time.time()
        
        try:
            # Create cache key
            cache_key = self._create_command_cache_key(command)
            
            # Check cache first
            if cache_key in self.optimization_cache:
                self.performance_stats['cache_hits'] += 1
                return self.optimization_cache[cache_key]
            
            self.performance_stats['cache_misses'] += 1
            
            # Find matching patterns
            patterns = self._find_optimization_patterns(command)
            
            # Cache the result
            self.optimization_cache[cache_key] = patterns
            
            # Performance tracking
            duration_ms = (time.time() - start_time) * 1000
            self.performance_stats['lookup_operations'].append(duration_ms)
            
            if duration_ms > PerformanceTargets.PATTERN_LOOKUP_MS:
                self.logger.warning(f"Pattern lookup took {duration_ms:.1f}ms (target: {PerformanceTargets.PATTERN_LOOKUP_MS}ms)")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error retrieving optimization patterns: {e}")
            return []
    
    def get_success_rate(self, command_pattern: str) -> float:
        """Get historical success rate for command pattern"""
        try:
            cache_key = self._create_command_cache_key(command_pattern)
            
            # Find matching command patterns
            matching_patterns = []
            for pattern_id, pattern in self.command_patterns.items():
                if self._patterns_match(pattern.abstract_command, command_pattern):
                    matching_patterns.append(pattern)
            
            if not matching_patterns:
                return 0.5  # Default neutral success rate
            
            # Calculate weighted success rate
            total_executions = sum(p.execution_count for p in matching_patterns)
            if total_executions == 0:
                return 0.5
            
            weighted_success_rate = sum(
                p.success_rate * (p.execution_count / total_executions)
                for p in matching_patterns
            )
            
            return weighted_success_rate
            
        except Exception as e:
            self.logger.error(f"Error calculating success rate: {e}")
            return 0.5
    
    def get_command_statistics(self, command_pattern: str) -> Dict[str, Any]:
        """Get comprehensive statistics for command pattern"""
        try:
            cache_key = self._create_command_cache_key(command_pattern)
            
            if cache_key in self.statistics_cache:
                return self.statistics_cache[cache_key]
            
            # Find matching patterns
            matching_patterns = []
            for pattern_id, pattern in self.command_patterns.items():
                if self._patterns_match(pattern.abstract_command, command_pattern):
                    matching_patterns.append(pattern)
            
            if not matching_patterns:
                return {'error': 'No matching patterns found'}
            
            # Calculate comprehensive statistics
            total_executions = sum(p.execution_count for p in matching_patterns)
            avg_success_rate = sum(p.success_rate * p.execution_count for p in matching_patterns) / total_executions
            avg_duration = sum(p.average_duration_ms * p.execution_count for p in matching_patterns) / total_executions
            
            # Get complexity distribution
            complexity_scores = [p.complexity_score for p in matching_patterns]
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            
            # Get category distribution
            categories = Counter(p.command_category for p in matching_patterns)
            
            # Get optimization suggestions
            all_suggestions = []
            for pattern in matching_patterns:
                all_suggestions.extend(pattern.optimization_suggestions)
            common_suggestions = Counter(all_suggestions).most_common(5)
            
            stats = {
                'total_executions': total_executions,
                'unique_patterns': len(matching_patterns),
                'success_rate': avg_success_rate,
                'average_duration_ms': int(avg_duration),
                'complexity_score': avg_complexity,
                'category_distribution': dict(categories),
                'common_optimizations': [suggestion for suggestion, count in common_suggestions],
                'first_seen': min(p.first_seen for p in matching_patterns),
                'last_seen': max(p.last_seen for p in matching_patterns)
            }
            
            # Cache the result
            self.statistics_cache[cache_key] = stats
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating command statistics: {e}")
            return {'error': str(e)}
    
    def evolve_schema_if_needed(self) -> bool:
        """Trigger adaptive schema evolution based on usage patterns"""
        try:
            if self.adaptive_schema.should_evolve_schema():
                evolution_result = self.adaptive_schema.evolve_schema()
                if evolution_result:
                    self.performance_stats['schema_evolutions'] += 1
                    self._invalidate_caches()  # Schema changes invalidate caches
                    self.logger.info("Schema evolution completed")
                return evolution_result
            return False
            
        except Exception as e:
            self.logger.error(f"Error during schema evolution: {e}")
            return False
    
    def cleanup_expired_data(self, retention_days: int = 30) -> int:
        """Remove expired learning data, return count removed"""
        try:
            current_time = time.time()
            retention_seconds = retention_days * 24 * 3600
            cutoff_time = current_time - retention_seconds
            
            removed_count = 0
            
            # Clean up command patterns
            expired_patterns = []
            for pattern_id, pattern in self.command_patterns.items():
                if pattern.last_seen < cutoff_time:
                    expired_patterns.append(pattern_id)
            
            for pattern_id in expired_patterns:
                del self.command_patterns[pattern_id]
                removed_count += 1
            
            # Clean up caches
            self._invalidate_caches()
            
            # Clean up encrypted storage
            # (This would be implemented by the SecureStorage class)
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} expired learning patterns")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired data: {e}")
            return 0
    
    def _abstract_command_data(self, data: CommandExecutionData) -> Dict[str, Any]:
        """Abstract sensitive information from command execution data"""
        try:
            # Use the abstractor to safely convert sensitive data
            abstracted = {
                'abstract_command': self.abstractor.abstract_command(data.command),
                'abstract_hostname': self.abstractor.abstract_hostname(data.host_context.get('hostname', '') if data.host_context else ''),
                'abstract_working_directory': self.abstractor.abstract_path(data.working_directory),
                'exit_code': data.exit_code,
                'duration_ms': data.duration_ms,
                'timestamp': data.timestamp,
                'session_id': data.session_id,
                'abstract_context': self.abstractor.abstract_execution_context(data.host_context or {})
            }
            
            return abstracted
            
        except Exception as e:
            self.logger.error(f"Error abstracting command data: {e}")
            return {}
    
    def _get_pattern_id(self, abstracted_data: Dict[str, Any]) -> str:
        """Generate unique pattern ID from abstracted data"""
        try:
            # Create pattern signature for consistent ID generation
            signature_data = {
                'command': abstracted_data.get('abstract_command', {}),
                'category': abstracted_data.get('abstract_command', {}).get('category', 'unknown'),
                'complexity': abstracted_data.get('abstract_command', {}).get('complexity', 'unknown')
            }
            
            signature_str = json.dumps(signature_data, sort_keys=True)
            pattern_id = hashlib.md5(signature_str.encode()).hexdigest()
            
            return pattern_id
            
        except Exception as e:
            self.logger.error(f"Error generating pattern ID: {e}")
            return 'unknown_pattern'
    
    def _update_command_pattern(self, pattern_id: str, execution_data: CommandExecutionData, abstracted_data: Dict[str, Any]) -> None:
        """Update or create command pattern with new execution data"""
        try:
            if pattern_id in self.command_patterns:
                # Update existing pattern
                pattern = self.command_patterns[pattern_id]
                pattern.update_with_execution(execution_data)
            else:
                # Create new pattern
                abstract_command_info = abstracted_data.get('abstract_command', {})
                
                pattern = CommandPattern(
                    pattern_id=pattern_id,
                    abstract_command=json.dumps(abstract_command_info),
                    command_category=abstract_command_info.get('category', 'unknown'),
                    complexity_score=abstract_command_info.get('complexity', 1.0),
                    success_rate=1.0 if execution_data.exit_code == 0 else 0.0,
                    average_duration_ms=execution_data.duration_ms,
                    execution_count=1,
                    first_seen=time.time(),
                    last_seen=time.time(),
                    optimization_suggestions=[]
                )
                
                self.command_patterns[pattern_id] = pattern
            
            # Generate optimization suggestions based on pattern
            self._update_optimization_suggestions(pattern, execution_data)
            
        except Exception as e:
            self.logger.error(f"Error updating command pattern: {e}")
    
    def _accumulate_threshold_information(self, data: CommandExecutionData) -> None:
        """Accumulate information for the threshold management system"""
        try:
            # Calculate information significance scores
            significance_scores = self.threshold_manager.calculate_information_significance(data)
            
            # Accumulate information by type
            for info_type, significance in significance_scores.items():
                self.threshold_manager.accumulate_information(info_type, significance)
                
        except Exception as e:
            self.logger.error(f"Error accumulating threshold information: {e}")
    
    def _store_encrypted_execution_data(self, abstracted_data: Dict[str, Any]) -> bool:
        """Store abstracted execution data using encrypted storage"""
        try:
            # Use the secure storage system to encrypt and store data
            storage_key = f"execution_{int(time.time())}_{hash(json.dumps(abstracted_data, sort_keys=True)) % 10000}"
            
            encrypted_data = self.secure_storage.encrypt_data(abstracted_data)
            
            # Store with timestamp-based file naming for automatic cleanup
            storage_file = self.storage_dir / f"execution_data_{int(time.time())}.enc"
            
            with open(storage_file, 'wb') as f:
                f.write(encrypted_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing encrypted execution data: {e}")
            return False
    
    def _find_optimization_patterns(self, command: str) -> List[OptimizationPattern]:
        """Find optimization patterns matching the given command"""
        try:
            patterns = []
            abstract_command = self.abstractor.abstract_command(command)
            
            # Look for patterns in stored command patterns
            for pattern_id, pattern in self.command_patterns.items():
                if self._patterns_match(pattern.abstract_command, json.dumps(abstract_command)):
                    # Convert pattern to optimization patterns
                    for suggestion in pattern.optimization_suggestions:
                        opt_pattern = OptimizationPattern(
                            original_pattern=command,
                            optimized_pattern=suggestion,
                            confidence=pattern.success_rate,
                            success_rate=pattern.success_rate,
                            application_count=pattern.execution_count,
                            created_at=pattern.first_seen,
                            last_used=pattern.last_seen,
                            categories=[pattern.command_category]
                        )
                        patterns.append(opt_pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error finding optimization patterns: {e}")
            return []
    
    def _patterns_match(self, pattern1: str, pattern2: str) -> bool:
        """Check if two abstracted patterns match"""
        try:
            # Simple string similarity for now
            # In a full implementation, this would use semantic similarity
            
            if pattern1 == pattern2:
                return True
            
            # Parse JSON patterns for structural comparison
            try:
                p1_data = json.loads(pattern1) if isinstance(pattern1, str) and pattern1.startswith('{') else {'raw': pattern1}
                p2_data = json.loads(pattern2) if isinstance(pattern2, str) and pattern2.startswith('{') else {'raw': pattern2}
                
                # Compare key characteristics
                if p1_data.get('category') == p2_data.get('category'):
                    return True
                
            except json.JSONDecodeError:
                pass
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error matching patterns: {e}")
            return False
    
    def _update_optimization_suggestions(self, pattern: CommandPattern, execution_data: CommandExecutionData) -> None:
        """Update optimization suggestions for a command pattern"""
        try:
            # Generate suggestions based on command analysis
            command = execution_data.command
            suggestions = []
            
            # Common optimization patterns
            if 'grep' in command.lower() and 'rg' not in command.lower():
                suggestions.append(command.replace('grep', 'rg'))
            
            if 'find' in command.lower() and 'fd' not in command.lower():
                suggestions.append(command.replace('find', 'fd'))
            
            # SLURM optimizations
            if 'sbatch' in command.lower():
                if '--mem=' not in command.lower():
                    suggestions.append(f"{command} --mem=4G")
                if '--time=' not in command.lower():
                    suggestions.append(f"{command} --time=1:00:00")
            
            # Add new unique suggestions
            for suggestion in suggestions:
                if suggestion not in pattern.optimization_suggestions:
                    pattern.optimization_suggestions.append(suggestion)
            
            # Keep only the most recent suggestions (limit to 5)
            if len(pattern.optimization_suggestions) > 5:
                pattern.optimization_suggestions = pattern.optimization_suggestions[-5:]
                
        except Exception as e:
            self.logger.error(f"Error updating optimization suggestions: {e}")
    
    def _create_command_cache_key(self, command: str) -> str:
        """Create cache key for command"""
        return hashlib.md5(command.encode()).hexdigest()
    
    def _update_caches_if_needed(self) -> None:
        """Update caches if enough time has passed"""
        current_time = time.time()
        if current_time - self.cache_last_updated > self.cache_update_interval:
            self._save_cached_data()
            self.cache_last_updated = current_time
    
    def _invalidate_caches(self) -> None:
        """Invalidate all caches"""
        self.optimization_cache.clear()
        self.statistics_cache.clear()
    
    def _save_cached_data(self) -> None:
        """Save cached data to storage"""
        try:
            cache_data = {
                'command_patterns': {
                    pattern_id: asdict(pattern)
                    for pattern_id, pattern in self.command_patterns.items()
                },
                'performance_stats': self.performance_stats,
                'last_updated': time.time()
            }
            
            cache_file = self.storage_dir / 'learning_cache.json'
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving cached data: {e}")
    
    def _load_cached_data(self) -> None:
        """Load cached data from storage"""
        try:
            cache_file = self.storage_dir / 'learning_cache.json'
            if not cache_file.exists():
                return
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Restore command patterns
            for pattern_id, pattern_data in cache_data.get('command_patterns', {}).items():
                try:
                    pattern = CommandPattern(**pattern_data)
                    self.command_patterns[pattern_id] = pattern
                except Exception as e:
                    self.logger.warning(f"Error loading pattern {pattern_id}: {e}")
            
            # Restore performance stats
            self.performance_stats.update(cache_data.get('performance_stats', {}))
            
            self.logger.info(f"Loaded {len(self.command_patterns)} command patterns from cache")
            
        except Exception as e:
            self.logger.error(f"Error loading cached data: {e}")
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the learning system"""
        stats = self.performance_stats.copy()
        
        # Calculate averages
        if stats['store_operations']:
            stats['avg_store_time_ms'] = sum(stats['store_operations']) / len(stats['store_operations'])
            stats['max_store_time_ms'] = max(stats['store_operations'])
        
        if stats['lookup_operations']:
            stats['avg_lookup_time_ms'] = sum(stats['lookup_operations']) / len(stats['lookup_operations'])
            stats['max_lookup_time_ms'] = max(stats['lookup_operations'])
        
        # Cache statistics
        total_cache_operations = stats['cache_hits'] + stats['cache_misses']
        if total_cache_operations > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_operations
        
        # System statistics
        stats['total_patterns'] = len(self.command_patterns)
        stats['cache_sizes'] = {
            'optimization_cache': len(self.optimization_cache),
            'statistics_cache': len(self.statistics_cache)
        }
        
        return stats

if __name__ == "__main__":
    # Example usage and testing
    storage = LearningStorage()
    
    # Test storing command execution data
    test_commands = [
        CommandExecutionData("sbatch job.sh", 0, 1500, time.time(), "test", "/home/user/project"),
        CommandExecutionData("grep pattern file.txt", 0, 200, time.time(), "test", "/home/user/data"),
        CommandExecutionData("python script.py", 1, 5000, time.time(), "test", "/home/user/analysis")
    ]
    
    for cmd_data in test_commands:
        success = storage.store_command_execution(cmd_data)
        print(f"Stored command execution: {success}")
    
    # Test retrieving optimization patterns
    patterns = storage.get_optimization_patterns("grep test data.txt")
    print(f"Found {len(patterns)} optimization patterns")
    
    # Test getting statistics
    stats = storage.get_command_statistics("python")
    print(f"Command statistics: {json.dumps(stats, indent=2, default=str)}")
    
    # Test performance statistics
    perf_stats = storage.get_performance_statistics()
    print(f"Performance statistics: {json.dumps(perf_stats, indent=2, default=str)}")