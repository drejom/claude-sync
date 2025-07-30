# Claude-Sync System Architecture

## Overview

Claude-sync is a learning-enabled command optimization system that integrates with Claude Code through hooks and agents. The system learns from command execution patterns to provide intelligent optimization suggestions, safety warnings, and workflow improvements.

## Core Component Interfaces

### 1. Hook System Interface

All hooks follow the Claude Code JSON protocol with strict performance requirements:

#### Hook Input Schema
```json
{
  "tool_name": "Bash",
  "tool_input": {
    "command": "sbatch --mem=32G run_analysis.sh",
    "description": "Submit SLURM job"
  },
  "tool_output": {
    "exit_code": 0,
    "stdout": "...",
    "stderr": "...",
    "duration_ms": 3500
  },
  "context": {
    "working_directory": "/home/user/project",
    "timestamp": 1704067200,
    "session_id": "abc123"
  }
}
```

#### Hook Output Schema
```json
{
  "block": false,
  "message": "🚀 **Optimization suggestion:**\n```bash\noptimized_command\n```"
}
```

#### Performance Contract
- **PreToolUse hooks**: <10ms execution (95th percentile)
- **PostToolUse hooks**: <50ms execution (background learning)
- **UserPromptSubmit hooks**: <20ms execution
- **Memory usage**: <10MB per hook execution
- **Error handling**: Graceful degradation, never break Claude Code

### 2. Learning System Interface

#### Learning Data Storage Contract
```python
class LearningDataInterface:
    """Interface for learning data storage and retrieval"""
    
    def store_command_execution(self, data: CommandExecutionData) -> bool:
        """Store command execution data with automatic abstraction"""
        pass
    
    def get_optimization_patterns(self, command: str) -> List[OptimizationPattern]:
        """Retrieve optimization patterns for command"""
        pass
    
    def get_success_rate(self, command_pattern: str) -> float:
        """Get historical success rate for command pattern"""
        pass
    
    def evolve_schema_if_needed(self) -> bool:
        """Trigger adaptive schema evolution"""
        pass
```

#### Data Types
```python
@dataclass
class CommandExecutionData:
    command: str
    exit_code: int
    duration_ms: int
    host_context: HostContext
    timestamp: float
    abstracted_patterns: Dict[str, Any]

@dataclass
class OptimizationPattern:
    original_pattern: str
    optimized_pattern: str
    confidence: float
    success_rate: float
    application_count: int

@dataclass
class HostContext:
    abstract_host_id: str
    capabilities: List[str]
    resource_availability: Dict[str, Any]
    network_context: NetworkContext
```

### 3. Security System Interface

#### Encryption Interface
```python
class SecureStorageInterface:
    """Interface for encrypted learning data storage"""
    
    def encrypt_and_store(self, data: Dict, key_id: str) -> bool:
        """Encrypt data with host-specific key"""
        pass
    
    def decrypt_and_load(self, key_id: str) -> Optional[Dict]:
        """Decrypt and load data"""
        pass
    
    def rotate_keys(self) -> bool:
        """Perform daily key rotation"""
        pass
    
    def get_current_key_id(self) -> str:
        """Get current encryption key identifier"""
        pass
```

#### Host Authorization Interface
```python
class HostAuthorizationInterface:
    """Interface for cross-host trust management"""
    
    def get_host_identity(self) -> str:
        """Get hardware-based host identity"""
        pass
    
    def is_trusted_host(self, host_id: str) -> bool:
        """Check if host is authorized"""
        pass
    
    def authorize_host(self, host_id: str) -> bool:
        """Add host to trust list"""
        pass
    
    def revoke_host(self, host_id: str) -> bool:
        """Remove host from trust list"""
        pass
```

### 4. Agent System Interface

#### Agent Knowledge Interface
```python
class AgentKnowledgeInterface:
    """Interface for providing learning data to agents"""
    
    def get_learning_summary(self, agent_name: str, context: Optional[Dict] = None) -> Dict:
        """Provide relevant learning data to specific agent"""
        pass
    
    def trigger_agent_analysis(self, agent_name: str, priority: str, context: Dict) -> str:
        """Queue background agent analysis"""
        pass
    
    def update_from_agent_insights(self, agent_name: str, insights: Dict) -> bool:
        """Update learning patterns from agent analysis"""
        pass
```

#### Information Threshold Interface
```python
class InformationThresholdInterface:
    """Interface for adaptive agent triggering"""
    
    def accumulate_information(self, info_type: str, significance: float = 1.0) -> None:
        """Accumulate information and check thresholds"""
        pass
    
    def calculate_weighted_score(self, agent_name: str) -> float:
        """Calculate current information score for agent"""
        pass
    
    def adapt_threshold(self, agent_name: str, effectiveness: float) -> None:
        """Adapt threshold based on analysis effectiveness"""
        pass
```

## Data Flow Architecture

### Command Execution Flow
```
User Command Input
        ↓
UserPromptSubmit Hook → Context Enhancement → Inject Learning Context
        ↓
Claude Code Planning
        ↓
PreToolUse Hooks → [Command Optimization] → [Safety Checks] → [Resource Planning]
        ↓
Tool Execution (Bash, SSH, etc.)
        ↓
PostToolUse Hooks → [Learning Collection] → [Performance Analysis] → [Background Sync]
        ↓
Learning Data Storage → Encrypted Local Storage → Abstraction Layer
        ↓
Information Threshold System → Weighted Significance → Agent Triggering
        ↓
Background Agent Analysis → Pattern Updates → Hook Optimization Updates
```

### Learning Data Pipeline
```
Raw Command Data
        ↓
Security Abstraction → Remove Sensitive Information → Generate Semantic Patterns
        ↓
Local Encryption → Host-Specific Keys → Daily Rotation
        ↓
Adaptive Schema → NoSQL Evolution → Pattern Recognition
        ↓
Cross-Host Sync → P2P Mesh → Encrypted Pattern Exchange
        ↓
Agent Analysis → Deep Insights → Hook Pattern Updates
```

### Information Threshold Flow
```
Command Execution Events
        ↓
Significance Calculation → Based on Novelty, Failure, Performance
        ↓
Weighted Accumulation → Agent-Specific Weights → Threshold Monitoring
        ↓
Threshold Trigger → Background Agent Analysis → Context-Rich Investigation
        ↓
Analysis Results → Hook Pattern Updates → Threshold Adaptation
```

## Integration Patterns

### Claude Code Integration

#### Directory Structure
```
~/.claude/
├── settings.json          # User's main settings (preserved)
├── claude-sync/           # Our installation
│   ├── hooks/            # Hook implementations
│   ├── learning/         # Learning modules
│   ├── templates/        # Settings templates
│   └── backups/          # Settings backups
├── hooks/                # Active hooks (symlinked)
│   ├── intelligent-optimizer.py -> ../claude-sync/hooks/intelligent-optimizer.py
│   ├── learning-collector.py -> ../claude-sync/hooks/learning-collector.py
│   └── context-enhancer.py -> ../claude-sync/hooks/context-enhancer.py
└── learning/             # Encrypted learning data (gitignored)
    ├── learning_YYYY-MM-DD.enc
    ├── keys/
    └── trusted_hosts
```

#### Settings Merging Strategy
```python
class SettingsMerger:
    """Safely merge claude-sync settings without overwriting user config"""
    
    def merge_settings(self, user_settings: Dict, sync_settings: Dict) -> Dict:
        """Deep merge preserving user preferences"""
        # Strategy: Append our hooks to existing hook arrays
        # Never overwrite user's hooks, permissions, or other settings
        pass
    
    def backup_user_settings(self) -> Path:
        """Create timestamped backup of user settings"""
        pass
    
    def restore_user_settings(self, backup_path: Path) -> bool:
        """Restore user settings from backup"""
        pass
```

### Activation/Deactivation Patterns

#### Atomic Operations
```python
class ActivationManager:
    """Manage clean activation/deactivation"""
    
    def activate_global(self) -> ActivationResult:
        """Atomically activate global hooks"""
        # 1. Backup user settings
        # 2. Create hook symlinks
        # 3. Merge settings
        # 4. Verify activation
        # 5. Rollback on any failure
        pass
    
    def deactivate(self, purge_data: bool = False) -> DeactivationResult:
        """Atomically deactivate and optionally purge data"""
        # 1. Remove hook symlinks
        # 2. Restore original settings
        # 3. Optionally purge learning data
        # 4. Verify clean state
        pass
    
    def verify_activation(self) -> ActivationStatus:
        """Verify system is properly activated"""
        pass
```

## Performance Requirements

### Hook Performance Targets
- **PreToolUse**: <10ms (critical path)
- **PostToolUse**: <50ms (background acceptable)
- **UserPromptSubmit**: <20ms (user experience)
- **Memory per hook**: <10MB
- **Storage overhead**: <1MB per day
- **Network sync**: <1KB/hour background

### Learning System Performance
- **Pattern matching**: <1ms lookup
- **Data encryption**: <5ms per operation
- **Schema evolution**: <100ms (background)
- **Cross-host sync**: <10KB/sync, max hourly
- **Agent triggering**: <1ms threshold calculation

### Resource Budgets
- **Total memory footprint**: <50MB
- **Disk usage**: ~1-5MB per host for learning data
- **CPU overhead**: <1% during normal operation
- **Network usage**: <1MB/day for mesh sync

## Error Handling & Reliability

### Graceful Degradation
- **No learning data**: System works with basic optimizations
- **Encryption failure**: Fall back to memory-only patterns
- **Network unavailable**: Local-only learning continues
- **Agent failure**: Hooks continue working independently
- **Hook failure**: Never block Claude Code execution

### Error Recovery
- **Corrupt learning data**: Auto-rebuild from recent patterns
- **Failed key rotation**: Continue with previous keys
- **Partial activation**: Complete rollback to clean state
- **Settings corruption**: Restore from automatic backups

## Security Architecture

### Defense in Depth
1. **Data Abstraction**: No sensitive data in learning patterns
2. **Local Encryption**: All learning data encrypted at rest
3. **Hardware Identity**: Host authentication via hardware fingerprint
4. **Daily Key Rotation**: Automatic cryptographic key refresh
5. **P2P Only**: No external services or API calls
6. **Audit Logging**: All security operations logged

### Threat Model
- **Protect against**: Learning data exposure, unauthorized access, network interception
- **Assume compromised**: Individual hosts may be compromised
- **Guarantee**: Repository never contains sensitive data
- **Maintain**: Zero-knowledge learning architecture

## Development & Testing Standards

### Code Standards
- **UV Scripts**: All Python scripts self-contained with inline dependencies
- **Type Hints**: Full type annotation required
- **Error Handling**: Comprehensive exception handling
- **Documentation**: Docstrings for all public interfaces
- **Performance**: Benchmark all critical paths

### Testing Requirements
- **Unit Tests**: >90% coverage for core components
- **Integration Tests**: Full activation/deactivation cycles
- **Performance Tests**: Benchmark against targets
- **Security Tests**: Encryption, key rotation, authorization
- **Compatibility Tests**: Multiple Claude Code configurations

### Quality Gates
- **Hook performance**: Must meet <10ms target
- **Security audit**: All encryption code reviewed
- **Integration test**: Clean install/uninstall
- **Cross-platform**: macOS, Linux, WSL support
- **Documentation**: Actionable user guides

This architecture provides the foundation for parallel development while ensuring all components integrate seamlessly into a cohesive, high-performance system.