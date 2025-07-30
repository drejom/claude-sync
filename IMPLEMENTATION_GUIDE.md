# Claude-Sync Implementation Guide
*System Architect Implementation Patterns*
*Version: 1.0*
*Date: 2025-07-30*

## 🎯 **Implementation Strategy Overview**

This guide provides concrete implementation patterns and examples for each specialist team to ensure consistent, high-performance, and secure components that integrate seamlessly.

---

## 🔧 **1. UV Script Architecture Patterns**

### **Standard UV Script Template**

```python
#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0",
#   "typing-extensions>=4.0.0"
# ]
# ///
"""
Claude-Sync Hook Template
High-performance, self-contained implementation
"""

import json
import sys
import time
from typing import Dict, Any, Optional
from pathlib import Path

# Global lazy-loaded caches for performance
_pattern_cache = None
_learning_client = None
_performance_monitor = None

def get_pattern_cache():
    """Lazy-load pattern cache to minimize startup time"""
    global _pattern_cache
    if _pattern_cache is None:
        _pattern_cache = initialize_pattern_cache()
    return _pattern_cache

def get_learning_client():
    """Lazy-load learning client only when needed"""
    global _learning_client
    if _learning_client is None:
        _learning_client = initialize_learning_client()
    return _learning_client

def get_performance_monitor():
    """Lazy-load performance monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = initialize_performance_monitor()
    return _performance_monitor

def main():
    """Main execution with performance monitoring and error handling"""
    start_time = time.perf_counter()
    
    try:
        # Read hook input
        hook_input = json.loads(sys.stdin.read())
        
        # Execute hook logic
        result = execute_hook_logic(hook_input)
        
        # Performance validation
        duration_ms = (time.perf_counter() - start_time) * 1000
        if result:
            result['_performance_ms'] = duration_ms
        
        # Performance warning for optimization
        if duration_ms > get_performance_target():
            log_performance_warning(duration_ms)
        
        # Output result
        if result:
            print(json.dumps(result))
        
    except Exception as e:
        # Graceful degradation - never break Claude Code
        error_result = create_safe_error_result()
        print(json.dumps(error_result))
        log_error(e)
    
    sys.exit(0)

def create_safe_error_result() -> Dict[str, Any]:
    """Create safe fallback result that won't break Claude Code"""
    return {'block': False, 'message': None}

def get_performance_target() -> int:
    """Return performance target in milliseconds"""
    # Override in specific hooks
    return 50  # Default target

def execute_hook_logic(hook_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Hook-specific logic - implement in each hook"""
    raise NotImplementedError("Implement in specific hook")

if __name__ == '__main__':
    main()
```

### **Performance Optimization Patterns**

```python
# Pattern 1: Fast-path execution for common cases
def execute_hook_logic(hook_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Optimized execution with fast path"""
    command = hook_input.get('tool_input', {}).get('command', '')
    
    # Fast path for common patterns (no learning lookup needed)
    if is_trivial_command(command):
        return None  # No suggestions needed
    
    if is_cached_pattern(command):
        return get_cached_result(command)
    
    # Full analysis path
    return full_analysis(command, hook_input)

def is_trivial_command(command: str) -> bool:
    """Quick check for commands that don't need optimization"""
    trivial_patterns = ['ls', 'pwd', 'cd ', 'echo ', 'cat ']
    return any(command.strip().startswith(pattern) for pattern in trivial_patterns)

def is_cached_pattern(command: str) -> bool:
    """Check if pattern is in hot cache"""
    cache = get_pattern_cache()
    pattern_key = extract_pattern_key(command)
    return pattern_key in cache

# Pattern 2: Lazy loading with shared state
class SharedLearningClient:
    """Shared learning client to avoid repeated initialization"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.learning_storage = None
            self.pattern_cache = {}
            self._initialized = True
    
    def get_learning_storage(self):
        """Lazy initialize learning storage"""
        if self.learning_storage is None:
            self.learning_storage = create_learning_storage()
        return self.learning_storage

# Pattern 3: Circuit breaker for reliability
class CircuitBreaker:
    """Protect against cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                return self._fallback_result()
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
            
            return self._fallback_result()
    
    def _fallback_result(self):
        """Safe fallback when circuit is open"""
        return {'block': False, 'message': None}
```

---

## 🧠 **2. Learning System Implementation Patterns**

### **Secure Learning Storage Pattern**

```python
#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0"
# ]
# ///
"""
Secure Learning Storage Implementation
Zero-knowledge learning with encryption
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
import sqlite3

@dataclass
class AbstractedCommandData:
    """Safe abstraction of command execution data"""
    command_category: str
    success: bool
    duration_tier: str  # 'fast', 'medium', 'slow'
    complexity_score: float
    timestamp: float
    environment_type: str
    performance_characteristics: Dict[str, Any]
    
    def to_storage_dict(self) -> Dict[str, Any]:
        """Convert to storage format"""
        return asdict(self)

class SecureLearningStorage:
    """Learning storage with encryption and abstraction"""
    
    def __init__(self, encryption_key: bytes, storage_dir: Path):
        self.fernet = Fernet(encryption_key)
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite for structured queries
        self.db_path = storage_dir / "patterns.db"
        self._init_database()
        
        # In-memory cache for performance
        self.pattern_cache = {}
        self.cache_size_limit = 1000
    
    def _init_database(self):
        """Initialize encrypted pattern database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS command_patterns (
                    id INTEGER PRIMARY KEY,
                    pattern_hash TEXT UNIQUE,
                    encrypted_data BLOB,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pattern_hash 
                ON command_patterns(pattern_hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON command_patterns(last_accessed)
            """)
    
    def store_abstracted_data(self, data: AbstractedCommandData) -> bool:
        """Store abstracted command data securely"""
        try:
            # Create pattern hash for deduplication
            pattern_key = self._create_pattern_key(data)
            pattern_hash = hashlib.sha256(pattern_key.encode()).hexdigest()
            
            # Encrypt data
            json_data = json.dumps(data.to_storage_dict())
            encrypted_data = self.fernet.encrypt(json_data.encode())
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO command_patterns 
                    (pattern_hash, encrypted_data, created_at, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, 
                           COALESCE((SELECT access_count + 1 FROM command_patterns WHERE pattern_hash = ?), 1))
                """, (pattern_hash, encrypted_data, time.time(), time.time(), pattern_hash))
            
            # Update cache
            self._update_cache(pattern_hash, data)
            
            return True
            
        except Exception as e:
            # Log error but don't fail the learning system
            self._log_storage_error(e)
            return False
    
    def get_patterns_for_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve patterns for command category"""
        try:
            # Check cache first
            cached_patterns = self._get_cached_patterns(category, limit)
            if cached_patterns:
                return cached_patterns
            
            # Query database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT encrypted_data, last_accessed, access_count
                    FROM command_patterns
                    ORDER BY access_count DESC, last_accessed DESC
                    LIMIT ?
                """, (limit * 2,))  # Get extra for filtering
                
                patterns = []
                for row in cursor:
                    try:
                        # Decrypt and parse
                        decrypted_data = self.fernet.decrypt(row[0])
                        pattern_data = json.loads(decrypted_data.decode())
                        
                        # Filter by category
                        if pattern_data.get('command_category') == category:
                            patterns.append(pattern_data)
                            
                        if len(patterns) >= limit:
                            break
                            
                    except Exception:
                        # Skip corrupted entries
                        continue
                
                # Update cache
                self._cache_patterns(category, patterns)
                return patterns
                
        except Exception as e:
            self._log_retrieval_error(e)
            return []
    
    def _create_pattern_key(self, data: AbstractedCommandData) -> str:
        """Create unique pattern key for deduplication"""
        key_components = [
            data.command_category,
            data.environment_type,
            str(int(data.complexity_score))  # Rounded for grouping
        ]
        return ':'.join(key_components)
    
    def _update_cache(self, pattern_hash: str, data: AbstractedCommandData):
        """Update in-memory cache with LRU eviction"""
        if len(self.pattern_cache) >= self.cache_size_limit:
            # Remove oldest entry
            oldest_key = min(self.pattern_cache.keys(), 
                           key=lambda k: self.pattern_cache[k]['last_accessed'])
            del self.pattern_cache[oldest_key]
        
        self.pattern_cache[pattern_hash] = {
            'data': data,
            'last_accessed': time.time()
        }

# Abstraction implementation
class CommandAbstractor:
    """Convert sensitive commands to safe abstractions"""
    
    def __init__(self):
        self.command_categories = {
            'slurm_submission': ['sbatch', 'srun', 'salloc'],
            'container_execution': ['singularity', 'docker'],
            'remote_execution': ['ssh'],
            'r_analysis': ['Rscript', 'R CMD'],
            'file_operations': ['cp', 'mv', 'rsync', 'scp'],
            'text_processing': ['grep', 'sed', 'awk', 'find']
        }
        
        self.data_patterns = {
            'genomics_data': ['.fastq', '.fq', '.bam', '.sam', '.vcf'],
            'analysis_data': ['.csv', '.tsv', '.rds', '.RData'],
            'archive_data': ['.tar', '.gz', '.zip'],
            'image_data': ['.png', '.jpg', '.pdf', '.svg']
        }
    
    def abstract_command(self, command: str, context: Dict[str, Any]) -> AbstractedCommandData:
        """Create safe abstraction of command execution"""
        
        # Categorize command
        command_category = self._categorize_command(command)
        
        # Abstract performance characteristics
        duration_ms = context.get('duration_ms', 0)
        duration_tier = self._categorize_duration(duration_ms)
        
        # Calculate complexity without revealing specifics
        complexity_score = self._calculate_abstract_complexity(command)
        
        # Abstract environment
        environment_type = self._abstract_environment(context)
        
        return AbstractedCommandData(
            command_category=command_category,
            success=context.get('exit_code', 0) == 0,
            duration_tier=duration_tier,
            complexity_score=complexity_score,
            timestamp=context.get('timestamp', time.time()),
            environment_type=environment_type,
            performance_characteristics=self._abstract_performance(context)
        )
    
    def _categorize_command(self, command: str) -> str:
        """Categorize command without revealing specifics"""
        command_lower = command.lower()
        
        for category, keywords in self.command_categories.items():
            if any(keyword in command_lower for keyword in keywords):
                return category
        
        return 'general_command'
    
    def _categorize_duration(self, duration_ms: int) -> str:
        """Categorize execution duration"""
        if duration_ms < 1000:  # < 1 second
            return 'fast'
        elif duration_ms < 30000:  # < 30 seconds
            return 'medium'
        else:
            return 'slow'
    
    def _abstract_environment(self, context: Dict[str, Any]) -> str:
        """Abstract execution environment"""
        # Check for indicators without revealing specifics
        if 'SLURM_JOB_ID' in context.get('environment', {}):
            return 'hpc_compute'
        elif 'SSH_CONNECTION' in context.get('environment', {}):
            return 'remote_host'
        else:
            return 'local_host'
```

### **Adaptive Schema Evolution Pattern**

```python
#!/usr/bin/env -S uv run
# /// script  
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Adaptive Schema Evolution Implementation
NoSQL-style schema that evolves with usage patterns
"""

import json
import time
from collections import defaultdict, Counter
from typing import Dict, Any, List, Set
from pathlib import Path

class AdaptiveSchemaManager:
    """NoSQL-style schema that adapts to usage patterns"""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Schema tracking
        self.pattern_registry = defaultdict(dict)  # pattern_sig -> field_stats
        self.usage_frequency = Counter()           # pattern_sig -> count
        self.schema_version = "1.0"
        self.last_evolution = 0
        
        # Evolution configuration
        self.evolution_config = {
            'min_pattern_frequency': 50,        # Pattern must appear 50+ times
            'min_field_consistency': 0.8,      # Field must appear in 80%+ of patterns
            'evolution_cooldown_hours': 24,    # Wait 24 hours between evolutions
            'max_schema_versions': 10          # Keep 10 schema versions
        }
        
        # Load existing schema
        self._load_schema_state()
    
    def observe_command_pattern(self, command_data: Dict[str, Any]) -> None:
        """Learn from command execution pattern"""
        
        # Extract pattern signature
        pattern_sig = self._extract_pattern_signature(command_data)
        
        # Update usage frequency
        self.usage_frequency[pattern_sig] += 1
        
        # Track field presence and types
        self._track_field_statistics(pattern_sig, command_data)
        
        # Check for evolution trigger
        if self._should_trigger_evolution():
            evolution_result = self.evolve_schema()
            if evolution_result['evolved']:
                self._save_schema_state()
    
    def _extract_pattern_signature(self, command_data: Dict[str, Any]) -> str:
        """Create signature for pattern recognition"""
        key_fields = [
            command_data.get('command_category', 'unknown'),
            command_data.get('environment_type', 'unknown'),
            str(command_data.get('success', False))
        ]
        return ':'.join(key_fields)
    
    def _track_field_statistics(self, pattern_sig: str, data: Dict[str, Any]) -> None:
        """Track field presence and consistency"""
        
        if pattern_sig not in self.pattern_registry:
            self.pattern_registry[pattern_sig] = {}
        
        pattern_stats = self.pattern_registry[pattern_sig]
        
        # Track each field in the data
        for field_name, field_value in data.items():
            if field_name not in pattern_stats:
                pattern_stats[field_name] = {
                    'type': type(field_value).__name__,
                    'first_seen': time.time(),
                    'frequency': 0,
                    'sample_values': []
                }
            
            field_stats = pattern_stats[field_name]
            field_stats['frequency'] += 1
            
            # Keep sample values for analysis
            if len(field_stats['sample_values']) < 10:
                if field_value not in field_stats['sample_values']:
                    field_stats['sample_values'].append(field_value)
    
    def _should_trigger_evolution(self) -> bool:
        """Determine if schema evolution should be triggered"""
        
        # Cooldown check
        if time.time() - self.last_evolution < self.evolution_config['evolution_cooldown_hours'] * 3600:
            return False
        
        # Check for patterns with high consistency
        for pattern_sig, frequency in self.usage_frequency.items():
            if frequency >= self.evolution_config['min_pattern_frequency']:
                consistency_score = self._calculate_pattern_consistency(pattern_sig)
                if consistency_score >= self.evolution_config['min_field_consistency']:
                    return True
        
        return False
    
    def _calculate_pattern_consistency(self, pattern_sig: str) -> float:
        """Calculate field consistency for a pattern"""
        if pattern_sig not in self.pattern_registry:
            return 0.0
        
        pattern_stats = self.pattern_registry[pattern_sig]
        total_occurrences = self.usage_frequency[pattern_sig]
        
        if total_occurrences == 0:
            return 0.0
        
        # Calculate average field consistency
        field_consistencies = []
        for field_name, field_stats in pattern_stats.items():
            field_consistency = field_stats['frequency'] / total_occurrences
            field_consistencies.append(field_consistency)
        
        return sum(field_consistencies) / len(field_consistencies) if field_consistencies else 0.0
    
    def evolve_schema(self) -> Dict[str, Any]:
        """Perform schema evolution based on observed patterns"""
        
        evolution_summary = {
            'evolved': False,
            'new_categories': [],
            'enhanced_patterns': [],
            'deprecated_fields': [],
            'schema_version': self.schema_version
        }
        
        try:
            # Find patterns ready for evolution
            evolution_candidates = self._find_evolution_candidates()
            
            if not evolution_candidates:
                return evolution_summary
            
            # Process each candidate
            for pattern_sig in evolution_candidates:
                evolution_result = self._evolve_pattern(pattern_sig)
                
                if evolution_result['category_created']:
                    evolution_summary['new_categories'].append(evolution_result['category_name'])
                
                if evolution_result['pattern_enhanced']:
                    evolution_summary['enhanced_patterns'].append(pattern_sig)
            
            # Update schema version
            if evolution_summary['new_categories'] or evolution_summary['enhanced_patterns']:
                self.schema_version = self._increment_schema_version()
                evolution_summary['schema_version'] = self.schema_version
                evolution_summary['evolved'] = True
                self.last_evolution = time.time()
            
            return evolution_summary
            
        except Exception as e:
            # Log evolution error but don't break system
            self._log_evolution_error(e)
            return evolution_summary
    
    def _find_evolution_candidates(self) -> List[str]:
        """Find patterns ready for schema evolution"""
        candidates = []
        
        for pattern_sig, frequency in self.usage_frequency.items():
            if frequency >= self.evolution_config['min_pattern_frequency']:
                consistency = self._calculate_pattern_consistency(pattern_sig)
                if consistency >= self.evolution_config['min_field_consistency']:
                    candidates.append(pattern_sig)
        
        return candidates
    
    def _evolve_pattern(self, pattern_sig: str) -> Dict[str, Any]:
        """Evolve a specific pattern based on its statistics"""
        
        result = {
            'category_created': False,
            'category_name': '',
            'pattern_enhanced': False
        }
        
        pattern_stats = self.pattern_registry[pattern_sig]
        
        # Check for specialization opportunities
        specialized_fields = self._find_specialized_fields(pattern_stats)
        
        if len(specialized_fields) >= 3:  # Threshold for new category
            # Create specialized category
            category_name = self._generate_category_name(pattern_sig, specialized_fields)
            
            if self._create_specialized_category(category_name, specialized_fields):
                result['category_created'] = True
                result['category_name'] = category_name
        
        # Enhance existing pattern
        if self._enhance_pattern_structure(pattern_sig, pattern_stats):
            result['pattern_enhanced'] = True
        
        return result
    
    def _find_specialized_fields(self, pattern_stats: Dict[str, Any]) -> List[str]:
        """Find fields that suggest pattern specialization"""
        specialized_fields = []
        
        for field_name, field_stats in pattern_stats.items():
            # High frequency fields with diverse values suggest specialization
            if (field_stats['frequency'] >= 10 and  # Appears frequently
                len(field_stats['sample_values']) >= 3):  # Has variety
                specialized_fields.append(field_name)
        
        return specialized_fields
    
    def get_current_schema(self) -> Dict[str, Any]:
        """Get current schema structure"""
        return {
            'version': self.schema_version,
            'patterns': dict(self.pattern_registry),
            'usage_stats': dict(self.usage_frequency),
            'last_evolution': self.last_evolution,
            'evolution_config': self.evolution_config
        }
```

### **Information Threshold System Pattern**

```python
#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Information Threshold Management
Adaptive agent triggering based on information density
"""

import time
import json
from collections import defaultdict
from typing import Dict, Any, Optional, List
from pathlib import Path

class InformationThresholdManager:
    """Manages adaptive thresholds for agent triggering"""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Information accumulation
        self.accumulated_info = defaultdict(float)
        
        # Information weights (importance multipliers)
        self.info_weights = {
            'new_commands': 2.0,        # Novel patterns valuable
            'failures': 4.0,            # Failures need attention
            'optimizations': 1.5,       # Success patterns matter
            'host_changes': 2.5,        # Environment changes important
            'performance_shifts': 5.0   # Performance issues critical
        }
        
        # Agent-specific thresholds (adaptive)
        self.agent_thresholds = {
            'learning-analyst': 50.0,        # Comprehensive analysis
            'hpc-advisor': 30.0,             # HPC-specific insights  
            'troubleshooting-detective': 15.0 # Rapid failure response
        }
        
        # Agent-specific weight factors
        self.agent_weights = {
            'learning-analyst': {
                'new_commands': 1.0,
                'optimizations': 1.2,
                'performance_shifts': 0.8,
                'failures': 0.6,
                'host_changes': 0.7
            },
            'hpc-advisor': {
                'new_commands': 1.5,
                'failures': 1.0,
                'host_changes': 1.3,
                'performance_shifts': 1.1,
                'optimizations': 0.8
            },
            'troubleshooting-detective': {
                'failures': 2.0,
                'performance_shifts': 1.5,
                'new_commands': 0.5,
                'optimizations': 0.3,
                'host_changes': 0.8
            }
        }
        
        # Effectiveness tracking for adaptive thresholds
        self.effectiveness_history = defaultdict(list)
        self.analysis_queue = []
        
        # Load state
        self._load_threshold_state()
    
    def accumulate_information(self, info_type: str, significance: float = 1.0, 
                             context: Optional[Dict] = None) -> None:
        """Accumulate weighted information and check thresholds"""
        
        # Apply base weight
        base_weight = self.info_weights.get(info_type, 1.0)
        weighted_significance = significance * base_weight
        
        # Context-based adjustment
        if context:
            context_multiplier = self._calculate_context_multiplier(context, info_type)
            weighted_significance *= context_multiplier
        
        # Accumulate information
        self.accumulated_info[info_type] += weighted_significance
        
        # Check all agent thresholds
        for agent_name in self.agent_thresholds:
            if self.should_trigger_agent(agent_name):
                self._trigger_agent_analysis(agent_name)
                self.reset_counters_for_agent(agent_name)
    
    def should_trigger_agent(self, agent_name: str) -> bool:
        """Check if agent analysis should be triggered"""
        current_score = self.calculate_agent_score(agent_name)
        threshold = self.agent_thresholds[agent_name]
        return current_score >= threshold
    
    def calculate_agent_score(self, agent_name: str) -> float:
        """Calculate information score specific to agent"""
        
        if agent_name not in self.agent_weights:
            # Fallback to simple sum
            return sum(self.accumulated_info.values())
        
        # Calculate weighted score for specific agent
        score = 0.0
        agent_weight_config = self.agent_weights[agent_name]
        
        for info_type, accumulated_amount in self.accumulated_info.items():
            agent_weight = agent_weight_config.get(info_type, 1.0)
            score += accumulated_amount * agent_weight
        
        return score
    
    def _calculate_context_multiplier(self, context: Dict, info_type: str) -> float:
        """Calculate context-based significance multiplier"""
        multiplier = 1.0
        
        # Command complexity increases significance
        if 'complexity_score' in context:
            complexity = context['complexity_score']
            multiplier *= (1.0 + complexity * 0.2)  # Up to 2x for very complex
        
        # Frequency affects significance
        if 'command_frequency' in context:
            frequency = context['command_frequency']
            if frequency < 5:  # Rare commands more significant
                multiplier *= 1.5
            elif frequency > 100:  # Very common commands less significant
                multiplier *= 0.8
        
        # Performance anomalies are highly significant
        if info_type == 'performance_shifts' and 'performance_change' in context:
            perf_change = abs(context['performance_change'])
            multiplier *= (1.0 + perf_change)  # Linear with performance change
        
        # Failure context
        if info_type == 'failures' and 'failure_context' in context:
            failure_context = context['failure_context']
            
            # Previously successful patterns failing is very significant
            if failure_context.get('historical_success_rate', 0) > 0.8:
                multiplier *= 2.0
            
            # Critical system failures are extremely significant
            if failure_context.get('system_impact') == 'critical':
                multiplier *= 3.0
        
        return min(multiplier, 5.0)  # Cap at 5x
    
    def _trigger_agent_analysis(self, agent_name: str) -> None:
        """Queue agent analysis with context"""
        
        analysis_context = {
            'agent_name': agent_name,
            'trigger_reason': f"Information threshold reached: {self.calculate_agent_score(agent_name):.1f}",
            'accumulated_info': dict(self.accumulated_info),
            'priority': self._calculate_analysis_priority(agent_name),
            'timestamp': time.time()
        }
        
        # Add to analysis queue
        self.analysis_queue.append(analysis_context)
        
        # Persist analysis request
        self._persist_analysis_request(analysis_context)
    
    def _calculate_analysis_priority(self, agent_name: str) -> str:
        """Calculate priority level for analysis"""
        
        current_score = self.calculate_agent_score(agent_name)
        threshold = self.agent_thresholds[agent_name]
        
        ratio = current_score / threshold
        
        if ratio >= 3.0:
            return 'URGENT'
        elif ratio >= 2.0:
            return 'HIGH'
        elif ratio >= 1.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def reset_counters_for_agent(self, agent_name: str) -> None:
        """Reset information counters after agent analysis"""
        
        # Partial reset based on agent's focus areas
        if agent_name == 'troubleshooting-detective':
            # Reset failure-related counters
            self.accumulated_info['failures'] = 0
            self.accumulated_info['performance_shifts'] *= 0.5  # Partial reset
        
        elif agent_name == 'learning-analyst':
            # Reset learning-related counters
            self.accumulated_info['new_commands'] = 0
            self.accumulated_info['optimizations'] = 0
        
        elif agent_name == 'hpc-advisor':
            # Reset HPC-related counters
            self.accumulated_info['host_changes'] = 0
            self.accumulated_info['new_commands'] *= 0.7  # Partial reset
        
        # Always save state after reset
        self._save_threshold_state()
    
    def adapt_threshold(self, agent_name: str, effectiveness_score: float) -> None:
        """Adapt threshold based on analysis effectiveness"""
        
        if agent_name not in self.agent_thresholds:
            return
        
        # Track effectiveness history
        self.effectiveness_history[agent_name].append({
            'score': effectiveness_score,
            'timestamp': time.time()
        })
        
        # Keep only recent history (last 10 analyses)
        if len(self.effectiveness_history[agent_name]) > 10:
            self.effectiveness_history[agent_name] = self.effectiveness_history[agent_name][-10:]
        
        # Calculate recent average effectiveness
        recent_scores = [entry['score'] for entry in self.effectiveness_history[agent_name][-5:]]
        avg_effectiveness = sum(recent_scores) / len(recent_scores)
        
        # Adjust threshold based on effectiveness
        current_threshold = self.agent_thresholds[agent_name]
        
        if avg_effectiveness > 0.8:
            # High effectiveness - lower threshold (more sensitive)
            new_threshold = current_threshold * 0.9
        elif avg_effectiveness < 0.3:
            # Low effectiveness - raise threshold (less sensitive)
            new_threshold = current_threshold * 1.2
        else:
            # Moderate effectiveness - small adjustment
            adjustment = (avg_effectiveness - 0.5) * 0.2
            new_threshold = current_threshold * (1.0 - adjustment)
        
        # Keep thresholds within reasonable bounds
        min_threshold = 5.0   # Minimum sensitivity
        max_threshold = 200.0 # Maximum threshold
        
        self.agent_thresholds[agent_name] = max(min_threshold, min(max_threshold, new_threshold))
        
        # Persist updated thresholds
        self._save_threshold_state()
    
    def get_threshold_status(self) -> Dict[str, Any]:
        """Get current threshold status for all agents"""
        
        status = {}
        
        for agent_name, threshold in self.agent_thresholds.items():
            current_score = self.calculate_agent_score(agent_name)
            
            status[agent_name] = {
                'threshold': threshold,
                'current_score': current_score,
                'progress_ratio': current_score / threshold,
                'ready_for_analysis': current_score >= threshold,
                'accumulated_info': dict(self.accumulated_info)
            }
        
        return status
```

This implementation guide provides concrete patterns and examples that specialist teams can follow to create high-performance, secure, and well-integrated components. Each pattern includes error handling, performance optimization, and integration points that align with the overall system architecture.