# Claude-Sync Core System Architecture
*Designed by: System Architect*
*Version: 1.0*
*Date: 2025-07-30*

## 🏗️ **Executive Summary**

Claude-sync implements a **performance-first, security-by-design** architecture that integrates seamlessly with Claude Code through hooks and agents. The system achieves sub-10ms hook execution while providing sophisticated learning capabilities through UV script self-contained modules.

**Key Architectural Principles:**
- **Performance**: Sub-10ms hook execution, 1ms pattern lookups
- **Security**: Hardware-based identity, daily key rotation, zero-knowledge abstractions  
- **Integration**: Symlink-based activation, atomic operations, settings preservation
- **Scalability**: Adaptive schemas, information-threshold triggers, P2P mesh learning

---

## 🔧 **Component Architecture**

### **1. Hook System Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claude Code Hook Lifecycle                  │
├─────────────────────────────────────────────────────────────────┤
│ UserPromptSubmit → PreToolUse → Tool Execution → PostToolUse   │
│       ↓               ↓              ↓              ↓          │
│   Context         Optimization    Command         Learning     │
│   Enhancement     & Safety        Execution       Collection   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Claude-Sync Hook Implementation                   │
├─────────────────────────────────────────────────────────────────┤
│ ~/.claude/hooks/                                                │
│ ├── context-enhancer.py        (UserPromptSubmit)             │
│ ├── intelligent-optimizer.py   (PreToolUse)                   │
│ └── learning-collector.py      (PostToolUse)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Performance Requirements                    │
├─────────────────────────────────────────────────────────────────┤
│ UserPromptSubmit:  < 20ms (context enhancement)               │
│ PreToolUse:        < 10ms (optimization suggestions)          │
│ PostToolUse:       < 50ms (learning data collection)          │
│ Memory per hook:   < 10MB                                      │
└─────────────────────────────────────────────────────────────────┘
```

#### **Hook Performance Architecture**

```python
# High-performance hook execution pattern
class PerformanceOptimizedHook:
    def __init__(self):
        # Pre-load frequently accessed data
        self.pattern_cache = LRUCache(maxsize=1000)
        self.learning_client = LazyLoadedLearningClient()
        self.performance_monitor = PerformanceMonitor()
    
    def execute(self, hook_input: Dict[str, Any]) -> HookResult:
        """Execute with performance tracking and circuit breaker"""
        start_time = time.perf_counter()
        
        try:
            # Fast path for common patterns
            result = self._fast_path_execution(hook_input)
            if result:
                return result
            
            # Full execution with timeout protection
            with timeout(self.get_execution_time_limit_ms() / 1000):
                result = self._full_execution(hook_input)
            
        except TimeoutError:
            # Circuit breaker - return minimal response
            result = HookResult(block=False, message=None)
        except Exception:
            # Graceful degradation - never break Claude Code
            result = HookResult(block=False, message=None)
        finally:
            # Performance tracking
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_hook_execution(
                self.__class__.__name__, duration_ms
            )
        
        return result
```

### **2. Learning System Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                   Adaptive Learning Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Command Execution → Abstraction → Encrypted Storage → Analysis │
│        ↓                ↓              ↓               ↓       │
│   Raw Hook Data    Safe Patterns   ~/.claude/learning/*.enc   │
│                                                                 │
│ Pattern Recognition ← Schema Evolution ← Threshold Analysis     │
│        ↓                    ↓                 ↓                │
│ Optimization Rules     New Categories    Agent Triggers        │
└─────────────────────────────────────────────────────────────────┘
```

#### **Data Flow Architecture**

```python
# Learning data flow with security-first design
class SecureLearningPipeline:
    """Zero-knowledge learning pipeline"""
    
    def process_command_execution(self, raw_data: CommandExecutionData):
        """Transform raw data through security layers"""
        
        # 1. Immediate abstraction (no sensitive data stored)
        abstracted_data = self.abstraction_layer.abstract_execution_context(
            raw_data.to_dict()
        )
        
        # 2. Pattern extraction with privacy preservation
        patterns = self.pattern_extractor.extract_safe_patterns(abstracted_data)
        
        # 3. Encrypted storage with automatic rotation
        encrypted_patterns = self.security_layer.encrypt_data(
            patterns, context="learning_storage"
        )
        
        # 4. Schema-aware storage with evolution tracking
        self.adaptive_storage.store_with_schema_evolution(encrypted_patterns)
        
        # 5. Information threshold tracking for agent triggers
        self.threshold_manager.accumulate_information(
            info_type=self._classify_information_type(patterns),
            significance=self._calculate_significance(patterns)
        )
```

#### **Adaptive Schema Evolution**

```
┌─────────────────────────────────────────────────────────────────┐
│                  NoSQL-Style Schema Evolution                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Week 1: Basic Patterns                                          │
│ {'slurm_job': ['command', 'memory', 'time']}                   │
│                        ↓                                        │
│ Week 4: Usage-Driven Evolution                                  │
│ {'slurm_job': ['command', 'memory', 'time', 'gpu_type'],       │
│  'gpu_slurm_job': ['cuda_version', 'memory_per_gpu']}          │
│                        ↓                                        │
│ Week 8: Domain-Specific Specialization                         │
│ {'bioinformatics_gpu_job': ['blast_db_size', 'sequence_type']} │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **3. Security Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                  Military-Grade Security Stack                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Hardware Identity Layer                                         │
│ ├── CPU Serial + Motherboard UUID → Stable Host ID            │
│ └── Survives OS reinstalls, uniquely identifies hardware       │
│                        ↓                                        │
│ Key Management Layer                                            │
│ ├── Daily automatic key rotation (PBKDF2 + HKDF)              │
│ ├── Host-specific entropy generation                           │
│ └── 7-day key retention with automatic cleanup                 │
│                        ↓                                        │
│ Encryption Layer                                                │
│ ├── Fernet symmetric encryption (AES-128)                     │
│ ├── Context-specific keys for different data types             │
│ └── Automatic data expiration (30-day default)                 │
│                        ↓                                        │
│ Trust Network Layer                                             │
│ ├── Simple host authorization (binary trust)                   │
│ ├── P2P mesh discovery (Tailscale + SSH)                      │
│ └── Abstracted pattern sharing only                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### **Hardware-Based Identity System**

```python
class HardwareIdentitySystem:
    """Stable host identity that survives OS reinstalls"""
    
    def generate_stable_host_id(self) -> str:
        """Create deterministic ID from hardware characteristics"""
        
        # Collect stable hardware identifiers
        hardware_sources = [
            self._get_cpu_identifier(),      # CPU serial/model
            self._get_motherboard_uuid(),    # Motherboard UUID  
            self._get_bios_uuid(),          # BIOS UUID
            self._get_machine_id()          # systemd machine-id (Linux)
        ]
        
        # Combine available sources (graceful degradation)
        available_sources = [src for src in hardware_sources if src]
        if not available_sources:
            raise SecurityError("No stable hardware identifiers available")
        
        # Create deterministic hash
        combined = ':'.join(sorted(available_sources))
        host_id = hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        return f"cs-{host_id}"  # claude-sync prefix
```

#### **Automatic Key Rotation System**

```python
class AutomaticKeyRotation:
    """Daily key rotation with secure derivation"""
    
    def get_current_encryption_key(self) -> bytes:
        """Get today's key - auto-generates if needed"""
        
        today = datetime.now().strftime('%Y-%m-%d')
        key_cache_path = self.key_cache_dir / f"key_{today}.cache"
        
        if key_cache_path.exists():
            return self._load_cached_key(key_cache_path)
        
        # Generate deterministic key for today
        key = self._generate_daily_key(today)
        self._cache_key(key_cache_path, key)
        self._cleanup_old_keys()
        
        return key
    
    def _generate_daily_key(self, date_str: str) -> bytes:
        """Deterministic key generation from host ID + date"""
        
        # Seed material: hardware identity + date + salt
        seed_material = f"{self.host_identity}:{date_str}:claude-sync-v1"
        
        # PBKDF2 key derivation (100,000 iterations)
        derived_key = hashlib.pbkdf2_hmac(
            'sha256',
            seed_material.encode(),
            b'claude-sync-learning-encryption',
            100000,  # iterations
            32       # key length
        )
        
        return base64.urlsafe_b64encode(derived_key)
```

### **4. Integration Architecture with Claude Code**

```
┌─────────────────────────────────────────────────────────────────┐
│                 Claude Code Integration Pattern                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ User's Existing Setup                                           │
│ ~/.claude/settings.json  ← NEVER directly modified             │
│ ~/.claude/hooks/         ← Target for symlinks                 │
│                                                                 │
│                        ↓                                        │
│ Claude-Sync Integration                                         │
│ ~/.claude/claude-sync/                                          │
│ ├── hooks/*.py          ← Source files                         │
│ ├── templates/settings.global.json  ← Settings template        │
│ └── backups/settings_backup_*.json  ← Safety backups           │
│                                                                 │
│                        ↓                                        │
│ Activation Process (Atomic)                                     │
│ 1. Backup user's settings                                       │
│ 2. Create hook symlinks                                         │
│ 3. Merge settings (JSON deep merge)                            │
│ 4. Verify activation integrity                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### **Symlink-Based Activation**

```python
class SymlinkActivationManager:
    """Atomic activation/deactivation with full rollback"""
    
    def activate_global(self) -> ActivationResult:
        """Atomically activate claude-sync globally"""
        
        backup_paths = []
        created_symlinks = []
        actions = []
        
        try:
            # 1. Create safety backup
            backup_path = self._backup_user_settings()
            backup_paths.append(backup_path)
            actions.append(f"Created backup: {backup_path}")
            
            # 2. Create hook symlinks
            symlinks = self._create_hook_symlinks()
            created_symlinks.extend(symlinks)
            actions.append(f"Created {len(symlinks)} hook symlinks")
            
            # 3. Merge settings (preserving user config)
            merged_settings = self._merge_settings_safely()
            self._write_merged_settings(merged_settings)
            actions.append("Merged hook settings with user config")
            
            # 4. Verify activation
            verification = self._verify_activation()
            if not verification['success']:
                raise ActivationError(f"Verification failed: {verification['errors']}")
            
            actions.append("Verified activation integrity")
            
            return ActivationResult(
                success=True,
                message="Claude-sync activated successfully",
                actions_performed=actions,
                backups_created=backup_paths,
                errors=[]
            )
            
        except Exception as e:
            # Atomic rollback on any failure
            self._rollback_activation(created_symlinks, backup_paths)
            
            return ActivationResult(
                success=False,
                message=f"Activation failed: {str(e)}",
                actions_performed=actions,
                backups_created=backup_paths,
                errors=[str(e)],
                rollback_required=True
            )
```

#### **Settings Preservation Strategy**

```python
class SafeSettingsMerger:
    """JSON settings merger that preserves user configuration"""
    
    def merge_hook_settings(self, user_settings: Dict, claude_sync_settings: Dict) -> Dict:
        """Deep merge preserving user's existing hooks and settings"""
        
        merged = copy.deepcopy(user_settings)
        
        # Handle hooks section carefully
        if 'hooks' not in merged:
            merged['hooks'] = {}
        
        for hook_type, hook_configs in claude_sync_settings.get('hooks', {}).items():
            if hook_type not in merged['hooks']:
                # No existing hooks of this type - direct assignment
                merged['hooks'][hook_type] = hook_configs
            else:
                # Merge with existing hooks
                merged['hooks'][hook_type] = self._merge_hook_type_configs(
                    merged['hooks'][hook_type], hook_configs
                )
        
        return merged
    
    def _merge_hook_type_configs(self, existing: List, claude_sync: List) -> List:
        """Merge hook configurations avoiding duplicates"""
        
        # Build set of existing hook commands for deduplication
        existing_commands = set()
        for hook_config in existing:
            for hook in hook_config.get('hooks', []):
                if hook.get('type') == 'command':
                    existing_commands.add(hook.get('command'))
        
        # Add claude-sync hooks that don't conflict
        merged = existing.copy()
        for claude_sync_config in claude_sync:
            should_add = True
            for hook in claude_sync_config.get('hooks', []):
                if hook.get('type') == 'command':
                    if hook.get('command') in existing_commands:
                        should_add = False
                        break
            
            if should_add:
                merged.append(claude_sync_config)
        
        return merged
```

### **5. UV Script Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Self-Contained UV Scripts                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Standard Script Header:                                         │
│ #!/usr/bin/env -S uv run                                        │
│ # /// script                                                    │
│ # requires-python = ">=3.10"                                    │
│ # dependencies = [                                              │
│ #   "cryptography>=41.0.0",                                     │
│ #   "psutil>=5.9.0"                                             │
│ # ]                                                             │
│ # ///                                                           │
│                                                                 │
│ Benefits:                                                       │
│ ├── No global environment pollution                             │
│ ├── Reproducible dependency resolution                          │
│ ├── Fast startup with uv caching                               │
│ ├── Version isolation per script                               │
│ └── Simplified deployment                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### **Performance-Optimized UV Script Pattern**

```python
#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0"
# ]
# ///
"""
High-performance hook with lazy loading and caching
"""

import json
import sys
import time
from pathlib import Path

# Lazy-loaded global caches (initialized on first use)
_pattern_cache = None
_learning_client = None

def get_pattern_cache():
    """Lazy-load pattern cache to minimize startup time"""
    global _pattern_cache
    if _pattern_cache is None:
        _pattern_cache = initialize_pattern_cache()
    return _pattern_cache

def get_learning_client():
    """Lazy-load learning client to minimize memory usage"""
    global _learning_client
    if _learning_client is None:
        _learning_client = initialize_learning_client()
    return _learning_client

def main():
    """Main hook execution with performance monitoring"""
    start_time = time.perf_counter()
    
    try:
        hook_input = json.loads(sys.stdin.read())
        result = execute_hook_logic(hook_input)
        
        # Performance validation
        duration_ms = (time.perf_counter() - start_time) * 1000
        if duration_ms > 10:  # Performance target
            # Log performance warning for optimization
            log_performance_warning(duration_ms)
        
        if result:
            print(json.dumps(result.to_json()))
        
    except Exception as e:
        # Graceful degradation - never break Claude Code
        error_result = {'block': False, 'message': None}
        print(json.dumps(error_result))
    
    sys.exit(0)

if __name__ == '__main__':
    main()
```

---

## 🔄 **Data Flow Patterns**

### **Real-Time Hook Data Flow**

```
User Command → Claude Code → PreToolUse Hook
                                ↓
           Pattern Cache ← Learning Storage → Optimization Engine
                 ↓              ↓                    ↓
         Fast Suggestions   Background          Safety Checks
                            Learning
                                ↓
           Command Execution (Modified/Original)
                                ↓
           PostToolUse Hook → Learning Collection
                                ↓
           Encrypted Storage → Adaptive Schema → Threshold Monitor
                                                       ↓
                                              Agent Trigger (if needed)
```

### **Information Threshold Architecture**

```python
class InformationThresholdSystem:
    """Adaptive agent triggering based on information density"""
    
    def __init__(self):
        self.info_weights = {
            'new_commands': 2.0,      # Novel patterns are valuable
            'failures': 4.0,          # Failures need attention
            'optimizations': 1.5,     # Success patterns matter
            'performance_shifts': 5.0, # Performance issues critical
            'host_changes': 2.5       # Environment changes important
        }
        
        self.agent_thresholds = {
            'learning-analyst': 50.0,        # Comprehensive analysis
            'hpc-advisor': 30.0,             # HPC-specific insights
            'troubleshooting-detective': 15.0 # Rapid failure response
        }
        
        self.accumulated_info = defaultdict(float)
    
    def accumulate_information(self, info_type: str, significance: float = 1.0):
        """Accumulate weighted information and check triggers"""
        
        weighted_significance = significance * self.info_weights.get(info_type, 1.0)
        self.accumulated_info[info_type] += weighted_significance
        
        # Check if any agents should be triggered
        for agent_name, threshold in self.agent_thresholds.items():
            current_score = self._calculate_agent_score(agent_name)
            if current_score >= threshold:
                self._trigger_agent_analysis(agent_name, current_score)
                self._reset_counters_for_agent(agent_name)
    
    def _calculate_agent_score(self, agent_name: str) -> float:
        """Calculate information score specific to agent type"""
        
        # Different agents weight information types differently
        if agent_name == 'learning-analyst':
            return (
                self.accumulated_info['new_commands'] * 1.0 +
                self.accumulated_info['optimizations'] * 1.2 +
                self.accumulated_info['performance_shifts'] * 0.8
            )
        elif agent_name == 'hpc-advisor':
            return (
                self.accumulated_info['new_commands'] * 1.5 +
                self.accumulated_info['failures'] * 1.0 +
                self.accumulated_info['host_changes'] * 1.3
            )
        elif agent_name == 'troubleshooting-detective':
            return (
                self.accumulated_info['failures'] * 2.0 +
                self.accumulated_info['performance_shifts'] * 1.5
            )
        
        return sum(self.accumulated_info.values())
```

---

## 🚀 **Performance Architecture**

### **Hook Performance Optimization**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sub-10ms Hook Execution                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Startup Optimization                                            │
│ ├── Lazy loading of heavy dependencies                         │
│ ├── Pre-compiled pattern caches                                │
│ ├── UV caching for rapid script startup                        │
│ └── Minimal import statements                                   │
│                                                                 │
│ Execution Optimization                                          │
│ ├── LRU caches for frequent pattern lookups                    │
│ ├── Circuit breakers for timeout protection                    │
│ ├── Fast-path execution for common patterns                    │
│ └── Background processing for heavy operations                 │
│                                                                 │
│ Memory Optimization                                             │
│ ├── Shared caches across hook instances                        │
│ ├── Automatic cache eviction policies                          │  
│ ├── Lazy deserialization of learning data                      │
│ └── Memory-mapped file access for large datasets               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### **Performance Monitoring Integration**

```python
class PerformanceMonitor:
    """Real-time performance tracking and optimization"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.performance_targets = {
            'PreToolUse': 10,    # milliseconds
            'PostToolUse': 50,   # milliseconds  
            'UserPromptSubmit': 20  # milliseconds
        }
    
    def record_hook_execution(self, hook_name: str, duration_ms: float):
        """Record execution time and trigger optimization if needed"""
        
        self.metrics[hook_name].append({
            'duration_ms': duration_ms,
            'timestamp': time.time()
        })
        
        # Keep only recent metrics (last 1000 executions)
        if len(self.metrics[hook_name]) > 1000:
            self.metrics[hook_name] = self.metrics[hook_name][-1000:]
        
        # Check if performance is degrading
        if self._is_performance_degrading(hook_name, duration_ms):
            self._trigger_performance_optimization(hook_name)
    
    def _is_performance_degrading(self, hook_name: str, current_duration: float) -> bool:
        """Detect performance degradation patterns"""
        
        target = self.performance_targets.get(hook_name, 100)
        
        # Check if current execution exceeds target
        if current_duration > target:
            return True
        
        # Check if recent average is trending upward
        recent_metrics = self.metrics[hook_name][-10:]  # Last 10 executions
        if len(recent_metrics) >= 5:
            recent_avg = sum(m['duration_ms'] for m in recent_metrics) / len(recent_metrics)
            if recent_avg > target * 0.8:  # 80% of target
                return True
        
        return False
```

---

## 🔐 **Security Integration Patterns**

### **Zero-Knowledge Learning Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│              Zero-Knowledge Learning Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Raw Command Data (NEVER STORED)                                │
│ "ssh user@hpc.university.edu '/data/genomics/analysis.sh'"     │
│                        ↓                                        │
│ Immediate Abstraction                                           │
│ {'command_type': 'remote_execution',                           │
│  'host_type': 'compute-cluster',                               │
│  'data_pattern': 'genomics-analysis'}                          │
│                        ↓                                        │
│ Pattern Extraction                                              │
│ {'success_pattern': 'ssh_genomics_compute',                    │
│  'performance_tier': 'medium',                                 │
│  'confidence': 0.87}                                           │
│                        ↓                                        │
│ Encrypted Storage                                               │
│ ~/.claude/learning/patterns_2025-07-30.enc                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### **Abstraction Implementation**

```python
class CommandAbstractor:
    """Convert sensitive commands to safe learning patterns"""
    
    def abstract_command(self, command: str) -> Dict[str, Any]:
        """Create safe abstraction of command for learning"""
        
        abstraction = {
            'command_category': self._categorize_command(command),
            'complexity_score': self._calculate_complexity(command),
            'tool_pattern': self._extract_tool_pattern(command),
            'flag_patterns': self._abstract_flags(command),
            'data_pattern': self._abstract_data_references(command)
        }
        
        return abstraction
    
    def _categorize_command(self, command: str) -> str:
        """Categorize command type without revealing specifics"""
        
        if 'sbatch' in command:
            return 'slurm_submission'
        elif 'singularity' in command:
            return 'container_execution'
        elif command.startswith('ssh'):
            return 'remote_execution'
        elif 'Rscript' in command:
            return 'r_analysis'
        else:
            return 'general_command'
    
    def _abstract_data_references(self, command: str) -> str:
        """Abstract file paths and data references"""
        
        # Pattern-based abstraction without storing real paths
        if '/data/' in command:
            return 'shared_data_access'
        elif '/scratch/' in command:
            return 'temporary_storage'
        elif '.fastq' in command or '.bam' in command:
            return 'genomics_data'
        elif '.csv' in command or '.rds' in command:
            return 'analysis_data'
        else:
            return 'general_files'
```

---

## 📋 **Implementation Interface Contracts**

### **Component Dependency Graph**

```
┌─────────────────────────────────────────────────────────────────┐
│                Component Dependencies                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ SecurityInterface ←─┐                                          │
│                     │                                          │
│ AbstractionInterface ←── LearningStorageInterface              │
│                     │            ↑                            │
│ PerformanceMonitor ──┘            │                            │
│                                   │                            │
│ HookInterface ←─── AdaptiveSchemaInterface                     │
│      ↑                            │                            │
│      │                            │                            │
│ InformationThresholdInterface ←───┘                            │
│      ↑                                                         │
│      │                                                         │
│ AgentKnowledgeInterface                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **Critical Implementation Contracts**

#### **1. Hook Interface Contract**
```python
# MANDATORY: All hooks must implement this pattern
class ClaudeSyncHook:
    def execute(self, hook_input: Dict[str, Any]) -> HookResult:
        """MUST complete in <10ms for PreToolUse, <50ms for PostToolUse"""
        pass
    
    def get_execution_time_limit_ms(self) -> int:
        """MUST return realistic timeout for circuit breaker"""
        pass
```

#### **2. Security Interface Contract**
```python  
# MANDATORY: All learning data must be encrypted
class SecurityRequirements:
    def encrypt_data(self, data: Dict[str, Any]) -> bytes:
        """MUST use Fernet encryption with daily-rotated keys"""
        pass
    
    def generate_host_identity(self) -> str:
        """MUST create stable ID from hardware characteristics"""
        pass
```

#### **3. Performance Interface Contract**
```python
# MANDATORY: All components must track performance
class PerformanceRequirements:
    def record_execution_time(self, operation: str, duration_ms: float):
        """MUST record all operations >1ms for optimization"""
        pass
    
    def check_performance_targets(self) -> bool:
        """MUST validate against PerformanceTargets constants"""
        pass
```

---

## 🎯 **Integration Points & Handoff Protocols**

### **Cross-Component Communication**

```python
# Standard event bus for component coordination
class ComponentEventBus:
    """Decoupled communication between system components"""
    
    def __init__(self):
        self.event_handlers = defaultdict(list)
    
    def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish event to all registered handlers"""
        for handler in self.event_handlers[event_type]:
            try:
                handler(data)
            except Exception as e:
                # Log but don't break other handlers
                logger.warning(f"Event handler failed: {e}")
    
    def subscribe(self, event_type: str, handler: Callable):
        """Register handler for event type"""
        self.event_handlers[event_type].append(handler)

# Key event types for coordination
EVENT_TYPES = {
    'command_executed': 'learning_data_available',
    'pattern_learned': 'optimization_available', 
    'performance_issue': 'optimization_needed',
    'security_event': 'audit_required',
    'threshold_reached': 'agent_analysis_needed'
}
```

### **Handoff Protocol for Implementation Specialists**

```python
# Standard handoff interface for parallel development
class ComponentHandoff:
    """Standardized handoff between specialists"""
    
    @dataclass
    class HandoffPackage:
        component_name: str
        interfaces_implemented: List[str]
        test_coverage: float
        performance_validated: bool
        security_audited: bool
        documentation_complete: bool
        integration_points: List[str]
        
    def validate_handoff(self, package: HandoffPackage) -> bool:
        """Validate component ready for integration"""
        checks = [
            package.test_coverage >= 0.9,      # 90% test coverage
            package.performance_validated,      # Performance targets met
            package.security_audited,           # Security review complete
            package.documentation_complete,     # Docs written
            len(package.integration_points) > 0 # Integration defined
        ]
        return all(checks)
```

---

## 🔧 **Implementation Priorities & Critical Path**

### **Phase 1: Core Infrastructure (Week 1)**
1. **SecurityInterface** implementation (hardware identity, key rotation)
2. **PerformanceMonitor** with hook timing
3. **Basic LearningStorage** with encryption
4. **ActivationManager** with symlink management
5. **ComponentFactory** for dependency injection

### **Phase 2: Hook System (Week 2)**  
1. **PreToolUse Hook** - intelligent-optimizer.py
2. **PostToolUse Hook** - learning-collector.py
3. **UserPromptSubmit Hook** - context-enhancer.py
4. **Hook testing framework** with mock data
5. **Performance validation** and optimization

### **Phase 3: Learning System (Week 3)**
1. **AdaptiveSchema** with evolution logic
2. **AbstractionInterface** for zero-knowledge patterns
3. **InformationThreshold** system with agent triggers
4. **Learning pattern storage** with schema versioning
5. **Cross-host mesh synchronization**

### **Phase 4: Integration & Testing (Week 4)**
1. **Component integration** testing
2. **Performance optimization** and validation  
3. **Security audit** and validation
4. **Documentation** completion
5. **User acceptance testing**

---

## 🎯 **Success Metrics & Validation**

### **Performance Targets**
- **Hook Execution**: <10ms (95th percentile)
- **Pattern Lookup**: <1ms average
- **Memory Usage**: <50MB total system
- **Learning Operations**: <100ms background
- **Installation Time**: <30 seconds full activation

### **Security Validation**
- **Zero sensitive data** in repository or logs
- **Hardware-based identity** survives OS reinstall
- **Daily key rotation** automatic and reliable  
- **Encrypted storage** for all learning data
- **P2P mesh** only shares abstracted patterns

### **Integration Success**
- **Zero impact** on existing Claude Code setup
- **Atomic operations** for activate/deactivate
- **Settings preservation** through JSON merge
- **Symlink management** with proper cleanup
- **Full rollback** capability on any failure

---

This architecture provides the foundation for a high-performance, secure, and seamlessly integrated learning system for Claude Code. Each component has clear interfaces, performance targets, and security requirements that enable parallel development by specialist teams.

**Next Step**: Project Orchestrator should use this architecture to create detailed implementation tasks for each specialist team.