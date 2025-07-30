#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "typing-extensions>=4.0.0"
# ]
# ///
"""
Claude-Sync Component Interfaces

These interfaces define the contracts between system components, enabling
parallel development and clean integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Protocol
from pathlib import Path
import time

# ============================================================================
# Data Types & Structures
# ============================================================================

@dataclass
class CommandExecutionData:
    """Raw command execution data from hooks"""
    command: str
    exit_code: int
    duration_ms: int
    timestamp: float
    session_id: str
    working_directory: str
    host_context: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'command': self.command,
            'exit_code': self.exit_code,
            'duration_ms': self.duration_ms,
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'working_directory': self.working_directory,
            'host_context': self.host_context or {},
            'tool_output': self.tool_output or {}
        }
    
    @classmethod
    def from_hook_input(cls, hook_input: Dict[str, Any]) -> 'CommandExecutionData':
        """Create from Claude Code hook input"""
        tool_input = hook_input.get('tool_input', {})
        tool_output = hook_input.get('tool_output', {})
        context = hook_input.get('context', {})
        
        return cls(
            command=tool_input.get('command', ''),
            exit_code=tool_output.get('exit_code', 0),
            duration_ms=tool_output.get('duration_ms', 0),
            timestamp=context.get('timestamp', time.time()),
            session_id=context.get('session_id', 'unknown'),
            working_directory=context.get('working_directory', ''),
            host_context=context,
            tool_output=tool_output
        )

@dataclass
class OptimizationPattern:
    """Command optimization pattern with confidence metrics"""
    original_pattern: str
    optimized_pattern: str
    confidence: float
    success_rate: float
    application_count: int
    created_at: float
    last_used: float
    categories: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []

@dataclass
class HostCapabilities:
    """Abstract host capability information"""
    abstract_host_id: str
    host_type: str  # 'gpu-host', 'compute-host', 'storage-host'
    capabilities: List[str]
    performance_tier: str  # 'high', 'medium', 'low'
    available_tools: List[str]
    network_context: Dict[str, Any]
    resource_limits: Dict[str, Any]

@dataclass
class LearningPattern:
    """Abstracted learning pattern for cross-host sharing"""
    pattern_id: str
    command_category: str
    success_rate: float
    performance_characteristics: Dict[str, Any]
    optimization_suggestions: List[str]
    confidence: float
    usage_frequency: int

@dataclass
class HookResult:
    """Standard hook result structure"""
    block: bool = False
    message: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON for Claude Code"""
        result = {'block': self.block}
        if self.message:
            result['message'] = self.message
        if self.modifications:
            result['modifications'] = self.modifications
        return result
    
    def meets_performance_targets(self, hook_type: str) -> bool:
        """Check if execution meets performance targets"""
        if hook_type == 'PreToolUse':
            return self.execution_time_ms <= PerformanceTargets.PRE_TOOL_USE_HOOK_MS
        elif hook_type == 'PostToolUse':
            return self.execution_time_ms <= PerformanceTargets.POST_TOOL_USE_HOOK_MS
        elif hook_type == 'UserPromptSubmit':
            return self.execution_time_ms <= PerformanceTargets.USER_PROMPT_SUBMIT_HOOK_MS
        return True

@dataclass
class SystemState:
    """Represents current system state for diagnostics"""
    is_activated: bool
    hooks_installed: List[str]
    learning_data_size_mb: float
    last_key_rotation: float
    trusted_hosts_count: int
    performance_metrics: Dict[str, float]
    errors: List[str]
    warnings: List[str]

@dataclass
class ActivationResult:
    """Result of activation/deactivation operation"""
    success: bool
    message: str
    actions_performed: List[str]
    backups_created: List[Path]
    errors: List[str]
    rollback_required: bool = False

@dataclass
class NetworkTopology:
    """Abstract network topology information"""
    abstract_host_id: str
    peer_hosts: List[str]
    connection_quality: Dict[str, float]  # host_id -> quality score
    preferred_routes: Dict[str, str]  # host_id -> route_type
    last_updated: float

# ============================================================================
# Core System Interfaces
# ============================================================================

class HookInterface(Protocol):
    """Interface for all Claude Code hooks"""
    
    def execute(self, hook_input: Dict[str, Any]) -> HookResult:
        """Execute hook logic and return result"""
        ...
    
    def get_execution_time_limit_ms(self) -> int:
        """Return maximum allowed execution time in milliseconds"""
        ...

class LearningStorageInterface(ABC):
    """Interface for learning data storage and retrieval"""
    
    @abstractmethod
    def store_command_execution(self, data: CommandExecutionData) -> bool:
        """Store command execution data with automatic abstraction"""
        pass
    
    @abstractmethod
    def get_optimization_patterns(self, command: str) -> List[OptimizationPattern]:
        """Retrieve optimization patterns for command"""
        pass
    
    @abstractmethod
    def get_success_rate(self, command_pattern: str) -> float:
        """Get historical success rate for command pattern"""
        pass
    
    @abstractmethod
    def get_command_statistics(self, command_pattern: str) -> Dict[str, Any]:
        """Get comprehensive statistics for command pattern"""
        pass
    
    @abstractmethod
    def evolve_schema_if_needed(self) -> bool:
        """Trigger adaptive schema evolution based on usage patterns"""
        pass
    
    @abstractmethod
    def cleanup_expired_data(self, retention_days: int = 30) -> int:
        """Remove expired learning data, return count removed"""
        pass

class SecurityInterface(ABC):
    """Interface for encryption and security operations"""
    
    @abstractmethod
    def encrypt_data(self, data: Dict[str, Any], context: str = "default") -> bytes:
        """Encrypt data with current key"""
        pass
    
    @abstractmethod
    def decrypt_data(self, encrypted_data: bytes, context: str = "default") -> Optional[Dict[str, Any]]:
        """Decrypt data, return None if failed"""
        pass
    
    @abstractmethod
    def rotate_keys(self) -> bool:
        """Perform daily key rotation"""
        pass
    
    @abstractmethod
    def get_current_key_id(self) -> str:
        """Get current encryption key identifier"""
        pass
    
    @abstractmethod
    def cleanup_old_keys(self, retention_days: int = 7) -> int:
        """Remove old encryption keys, return count removed"""
        pass

class HostAuthorizationInterface(ABC):
    """Interface for cross-host trust management"""
    
    @abstractmethod
    def get_host_identity(self) -> str:
        """Get stable hardware-based host identity"""
        pass
    
    @abstractmethod
    def is_trusted_host(self, host_id: str) -> bool:
        """Check if host is in trust list"""
        pass
    
    @abstractmethod
    def authorize_host(self, host_id: str, host_description: str = "") -> bool:
        """Add host to trust list"""
        pass
    
    @abstractmethod
    def revoke_host(self, host_id: str) -> bool:
        """Remove host from trust list"""
        pass
    
    @abstractmethod
    def list_trusted_hosts(self) -> List[Dict[str, str]]:
        """Get list of trusted hosts with metadata"""
        pass

class AbstractionInterface(ABC):
    """Interface for converting sensitive data to safe abstractions"""
    
    @abstractmethod
    def abstract_command(self, command: str) -> Dict[str, Any]:
        """Convert command to safe abstraction"""
        pass
    
    @abstractmethod
    def abstract_hostname(self, hostname: str) -> str:
        """Convert hostname to semantic type"""
        pass
    
    @abstractmethod
    def abstract_path(self, path: str) -> str:
        """Convert file path to category"""
        pass
    
    @abstractmethod
    def abstract_execution_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Abstract execution environment data"""
        pass

class AdaptiveSchemaInterface(ABC):
    """Interface for adaptive learning schema evolution"""
    
    @abstractmethod
    def observe_command_pattern(self, command_data: CommandExecutionData) -> None:
        """Learn from command execution pattern"""
        pass
    
    @abstractmethod
    def get_current_schema(self) -> Dict[str, Any]:
        """Get current schema structure"""
        pass
    
    @abstractmethod
    def should_evolve_schema(self) -> bool:
        """Check if schema evolution is needed"""
        pass
    
    @abstractmethod
    def evolve_schema(self) -> bool:
        """Perform schema evolution"""
        pass
    
    @abstractmethod
    def get_pattern_frequency(self, pattern: str) -> int:
        """Get usage frequency for pattern"""
        pass

class InformationThresholdInterface(ABC):
    """Interface for adaptive agent triggering based on information density"""
    
    @abstractmethod
    def accumulate_information(self, info_type: str, significance: float = 1.0, context: Optional[Dict] = None) -> None:
        """Accumulate information and check thresholds"""
        pass
    
    @abstractmethod
    def calculate_weighted_score(self, agent_name: str) -> float:
        """Calculate current information score for specific agent"""
        pass
    
    @abstractmethod
    def should_trigger_agent(self, agent_name: str) -> bool:
        """Check if agent analysis should be triggered"""
        pass
    
    @abstractmethod
    def adapt_threshold(self, agent_name: str, effectiveness_score: float) -> None:
        """Adapt threshold based on analysis effectiveness"""
        pass
    
    @abstractmethod
    def reset_counters_for_agent(self, agent_name: str) -> None:
        """Reset information counters after agent analysis"""
        pass

class AgentKnowledgeInterface(ABC):
    """Interface for providing learning data to Claude Code agents"""
    
    @abstractmethod
    def get_learning_summary(self, agent_name: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Provide relevant learning data to specific agent"""
        pass
    
    @abstractmethod
    def trigger_agent_analysis(self, agent_name: str, priority: str, context: Dict) -> str:
        """Queue background agent analysis"""
        pass
    
    @abstractmethod
    def update_from_agent_insights(self, agent_name: str, insights: Dict) -> bool:
        """Update learning patterns from agent analysis"""
        pass
    
    @abstractmethod
    def measure_agent_effectiveness(self, agent_name: str, analysis_id: str) -> float:
        """Measure how effective an agent analysis was"""
        pass

class MeshSyncInterface(ABC):
    """Interface for cross-host learning data synchronization"""
    
    @abstractmethod
    def discover_peers(self) -> List[str]:
        """Discover other claude-sync hosts in mesh"""
        pass
    
    @abstractmethod
    def sync_with_peer(self, peer_host_id: str) -> bool:
        """Synchronize learning patterns with peer"""
        pass
    
    @abstractmethod
    def share_pattern(self, pattern: LearningPattern, target_hosts: Optional[List[str]] = None) -> int:
        """Share learning pattern with mesh, return count of successful shares"""
        pass
    
    @abstractmethod
    def receive_pattern(self, pattern: LearningPattern, source_host: str) -> bool:
        """Receive and integrate pattern from peer"""
        pass

class ActivationManagerInterface(ABC):
    """Interface for system activation and deactivation"""
    
    @abstractmethod
    def activate_global(self) -> Dict[str, Any]:
        """Activate claude-sync globally"""
        pass
    
    @abstractmethod
    def activate_project(self, project_path: Path) -> Dict[str, Any]:
        """Activate claude-sync for specific project"""
        pass
    
    @abstractmethod
    def deactivate(self, purge_data: bool = False) -> Dict[str, Any]:
        """Deactivate claude-sync and optionally purge data"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current activation status"""
        pass
    
    @abstractmethod
    def verify_installation(self) -> Dict[str, Any]:
        """Verify system is properly installed and configured"""
        pass

# ============================================================================
# Specific Hook Interfaces
# ============================================================================

class PreToolUseHookInterface(HookInterface):
    """Specialized interface for PreToolUse hooks"""
    
    def analyze_command(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze command before execution"""
        ...
    
    def suggest_optimizations(self, command: str) -> List[OptimizationPattern]:
        """Suggest command optimizations"""
        ...
    
    def check_safety(self, command: str) -> List[str]:
        """Check for potential safety issues"""
        ...

class PostToolUseHookInterface(HookInterface):
    """Specialized interface for PostToolUse hooks"""
    
    def extract_learning_data(self, hook_input: Dict[str, Any]) -> CommandExecutionData:
        """Extract learning data from execution"""
        ...
    
    def analyze_performance(self, execution_data: CommandExecutionData) -> Dict[str, Any]:
        """Analyze command performance"""
        ...
    
    def update_patterns(self, execution_data: CommandExecutionData) -> bool:
        """Update learning patterns based on execution"""
        ...

class UserPromptSubmitHookInterface(HookInterface):
    """Specialized interface for UserPromptSubmit hooks"""
    
    def detect_context_needs(self, prompt: str) -> List[str]:
        """Detect what context might be helpful"""
        ...
    
    def enhance_with_context(self, prompt: str) -> Optional[str]:
        """Add relevant learning context to prompt"""
        ...

# ============================================================================
# Performance & Monitoring Interfaces
# ============================================================================

class PerformanceMonitorInterface(ABC):
    """Interface for system performance monitoring"""
    
    @abstractmethod
    def record_hook_execution(self, hook_name: str, duration_ms: int) -> None:
        """Record hook execution time"""
        pass
    
    @abstractmethod
    def record_learning_operation(self, operation: str, duration_ms: int) -> None:
        """Record learning system operation time"""
        pass
    
    @abstractmethod
    def get_performance_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics for recent period"""
        pass
    
    @abstractmethod
    def check_performance_targets(self) -> Dict[str, bool]:
        """Check if system meets performance targets"""
        pass

class DiagnosticsInterface(ABC):
    """Interface for system health diagnostics"""
    
    @abstractmethod
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive system health check"""
        pass
    
    @abstractmethod
    def check_hook_status(self) -> Dict[str, Any]:
        """Check status of all hooks"""
        pass
    
    @abstractmethod
    def check_learning_system(self) -> Dict[str, Any]:
        """Check learning system health"""
        pass
    
    @abstractmethod
    def check_security_system(self) -> Dict[str, Any]:
        """Check security system status"""
        pass

# ============================================================================
# Factory Interface for Dependency Injection
# ============================================================================

class ComponentFactoryInterface(ABC):
    """Factory interface for creating system components"""
    
    @abstractmethod
    def create_learning_storage(self) -> LearningStorageInterface:
        """Create learning storage implementation"""
        pass
    
    @abstractmethod
    def create_security_system(self) -> SecurityInterface:
        """Create security system implementation"""
        pass
    
    @abstractmethod
    def create_abstraction_system(self) -> AbstractionInterface:
        """Create data abstraction implementation"""
        pass
    
    @abstractmethod
    def create_threshold_manager(self) -> InformationThresholdInterface:
        """Create information threshold manager"""
        pass
    
    @abstractmethod
    def create_performance_monitor(self) -> PerformanceMonitorInterface:
        """Create performance monitoring system"""
        pass
    
    @abstractmethod
    def create_adaptive_schema(self) -> AdaptiveSchemaInterface:
        """Create adaptive schema manager"""
        pass
    
    @abstractmethod
    def create_mesh_sync(self) -> MeshSyncInterface:
        """Create mesh synchronization system"""
        pass
    
    @abstractmethod
    def create_activation_manager(self) -> ActivationManagerInterface:
        """Create activation management system"""
        pass

class HardwareIdentityInterface(ABC):
    """Interface for hardware-based host identification"""
    
    @abstractmethod
    def get_cpu_serial(self) -> Optional[str]:
        """Get CPU serial number or identifier"""
        pass
    
    @abstractmethod
    def get_motherboard_uuid(self) -> Optional[str]:
        """Get motherboard UUID"""
        pass
    
    @abstractmethod
    def get_network_mac_primary(self) -> Optional[str]:
        """Get primary network interface MAC address"""
        pass
    
    @abstractmethod
    def generate_stable_host_id(self) -> str:
        """Generate stable host ID from hardware characteristics"""
        pass
    
    @abstractmethod
    def validate_host_identity(self, claimed_id: str) -> bool:
        """Validate claimed host identity against hardware"""
        pass

class SettingsMergerInterface(ABC):
    """Interface for safely merging Claude Code settings"""
    
    @abstractmethod
    def backup_user_settings(self) -> Path:
        """Create timestamped backup of user settings"""
        pass
    
    @abstractmethod
    def merge_hook_settings(self, claude_sync_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Merge claude-sync hooks into existing user settings"""
        pass
    
    @abstractmethod
    def restore_from_backup(self, backup_path: Path) -> bool:
        """Restore user settings from backup"""
        pass
    
    @abstractmethod
    def validate_merged_settings(self, settings: Dict[str, Any]) -> bool:
        """Validate merged settings are syntactically correct"""
        pass
    
    @abstractmethod
    def extract_claude_sync_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only claude-sync related settings"""
        pass

class SymlinkManagerInterface(ABC):
    """Interface for managing hook symlinks"""
    
    @abstractmethod
    def create_hook_symlinks(self, source_dir: Path, target_dir: Path) -> List[Path]:
        """Create symlinks for all hooks, return created symlinks"""
        pass
    
    @abstractmethod
    def remove_hook_symlinks(self, target_dir: Path) -> List[Path]:
        """Remove claude-sync symlinks, return removed paths"""
        pass
    
    @abstractmethod
    def verify_symlinks(self, target_dir: Path) -> Dict[str, bool]:
        """Verify all expected symlinks exist and are valid"""
        pass
    
    @abstractmethod
    def list_claude_sync_symlinks(self, target_dir: Path) -> List[Path]:
        """List all claude-sync related symlinks"""
        pass

# ============================================================================
# Constants & Configuration
# ============================================================================

class PerformanceTargets:
    """System performance targets"""
    # Hook execution time limits (95th percentile)
    PRE_TOOL_USE_HOOK_MS = 10
    POST_TOOL_USE_HOOK_MS = 50
    USER_PROMPT_SUBMIT_HOOK_MS = 20
    
    # Learning system operation limits
    LEARNING_OPERATION_MS = 100
    PATTERN_LOOKUP_MS = 1
    SCHEMA_EVOLUTION_MS = 200
    
    # Security operation limits
    ENCRYPTION_OPERATION_MS = 5
    KEY_ROTATION_MS = 1000
    HOST_IDENTITY_GENERATION_MS = 100
    
    # Memory usage limits
    HOOK_MEMORY_MB = 10
    LEARNING_CACHE_MB = 50
    TOTAL_SYSTEM_MEMORY_MB = 100
    
    # Storage limits
    DAILY_LEARNING_DATA_MB = 1
    MAX_LEARNING_DATA_MB = 30
    
    # Network limits
    DAILY_MESH_SYNC_KB = 1000
    PEER_DISCOVERY_TIMEOUT_MS = 5000

class InformationTypes:
    """Types of information for threshold system"""
    NEW_COMMANDS = "new_commands"
    FAILURES = "failures"
    OPTIMIZATIONS = "optimizations"
    HOST_CHANGES = "host_changes"
    PERFORMANCE_SHIFTS = "performance_shifts"

class AgentNames:
    """Standard agent names"""
    LEARNING_ANALYST = "learning-analyst"
    HPC_ADVISOR = "hpc-advisor"
    TROUBLESHOOTING_DETECTIVE = "troubleshooting-detective"

# ============================================================================
# Utility Functions
# ============================================================================

def validate_hook_result(result: HookResult) -> bool:
    """Validate hook result structure"""
    if not isinstance(result.block, bool):
        return False
    if result.message is not None and not isinstance(result.message, str):
        return False
    return True

def create_hook_result(block: bool = False, message: str = None) -> HookResult:
    """Create standardized hook result"""
    return HookResult(block=block, message=message)

def calculate_command_complexity(command: str) -> float:
    """Calculate relative complexity of command for significance weighting"""
    # Basic complexity heuristics
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

if __name__ == "__main__":
    print("Claude-Sync Component Interfaces")
    print("These interfaces define the contracts for system components.")
    print("Use these for parallel development and testing.")