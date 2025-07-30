# Claude-Sync Data Flow Architecture

This document specifies the detailed data flow patterns for the claude-sync system, defining how data moves through hooks, learning storage, agent systems, and cross-host synchronization.

## 1. Hook Execution Lifecycle

### 1.1 UserPromptSubmit Hook Flow

```mermaid
graph TD
    A[User Prompt Input] --> B[UserPromptSubmit Hook]
    B --> C{Context Detection}
    C -->|SLURM Context| D1[Load SLURM Patterns]
    C -->|Container Context| D2[Load Container Patterns]
    C -->|R/Analysis Context| D3[Load R Workflow Patterns]
    C -->|No Context| D4[Skip Enhancement]
    D1 --> E[Inject Context]
    D2 --> E
    D3 --> E
    D4 --> F[Return to Claude]
    E --> F
    F --> G[Claude Code Planning]
```

**Data Structures:**
```python
# Input to UserPromptSubmit hook
{
    "user_prompt": "Help me optimize this SLURM job",
    "context": {
        "timestamp": 1704067200,
        "session_id": "abc123",
        "working_directory": "/home/user/project"
    }
}

# Output from UserPromptSubmit hook
{
    "block": false,
    "message": "🧠 **Added SLURM context:**\nOptimal partition: compute (avg queue time: 5min)\nMemory efficiency: 80% utilization typical\nRecommended flags: --mem=32G --cpus-per-task=8"
}
```

### 1.2 PreToolUse Hook Flow

```mermaid
graph TD
    A[Claude Code Tool Execution] --> B[PreToolUse Hook]
    B --> C[Extract Command]
    C --> D[Load Optimization Patterns]
    D --> E[Calculate Confidence]
    E --> F{Confidence > 0.8?}
    F -->|Yes| G[High-Confidence Suggestion]
    F -->|No| H{Confidence > 0.5?}
    H -->|Yes| I[Medium-Confidence Suggestion]
    H -->|No| J[Skip Suggestion]
    G --> K[Safety Check]
    I --> K
    J --> L[Return to Claude]
    K --> M{Safety Issues?}
    M -->|Yes| N[Add Warning]
    M -->|No| L
    N --> L
```

**Performance Contract:**
- Execution time: <10ms (95th percentile)
- Memory usage: <10MB
- Pattern lookup: <1ms
- Safety check: <5ms

**Data Flow:**
```python
# PreToolUse input
{
    "tool_name": "Bash",
    "tool_input": {
        "command": "sbatch --mem=64G run_analysis.sh",
        "description": "Submit SLURM job"
    },
    "context": {...}
}

# PreToolUse processing steps
1. Extract command pattern: "sbatch --mem=* *.sh"
2. Lookup optimization patterns: get_patterns("slurm_job")
3. Calculate confidence: 0.92 (high confidence)
4. Generate suggestion: "--mem=32G based on historical usage"
5. Safety check: No dangerous flags detected
6. Format response with confidence level

# PreToolUse output
{
    "block": false,
    "message": "🚀 **High-confidence optimization:**\n```bash\nsbatch --mem=32G --cpus-per-task=8 run_analysis.sh\n```\n📊 Success rate: 92% | Avg savings: 15min queue time"
}
```

### 1.3 PostToolUse Hook Flow

```mermaid
graph TD
    A[Tool Execution Complete] --> B[PostToolUse Hook]
    B --> C[Extract Execution Data]
    C --> D[Abstract Sensitive Data]
    D --> E[Store Learning Data]
    E --> F[Update Success Patterns]
    F --> G[Calculate Information Significance]
    G --> H[Update Threshold Counters]
    H --> I{Threshold Reached?}
    I -->|Yes| J[Queue Agent Analysis]
    I -->|No| K[Background Schema Evolution]
    J --> K
    K --> L[Update Performance Metrics]
    L --> M[Return Success]
```

**Data Processing Pipeline:**
```python
# Step 1: Extract execution data
execution_data = CommandExecutionData(
    command="sbatch --mem=32G run_analysis.sh",
    exit_code=0,
    duration_ms=3500,
    timestamp=1704067200,
    session_id="abc123",
    working_directory="/home/user/project",
    host_context={
        "slurm_partition": "compute",
        "available_memory": "512GB",
        "queue_time": "2min"
    }
)

# Step 2: Abstract sensitive data
abstracted_data = {
    "command_pattern": "slurm_job_submission",
    "resource_pattern": "medium_memory_job",
    "success_indicators": ["job_accepted", "normal_completion"],
    "performance_tier": "efficient",
    "host_type": "compute_cluster"
}

# Step 3: Store with encryption
encrypted_storage.store_pattern(abstracted_data, context="slurm_learning")

# Step 4: Update information thresholds
threshold_manager.accumulate_information(
    info_type="optimizations",
    significance=2.0,  # Successful optimization applied
    context={"command_type": "slurm", "optimization_accepted": True}
)
```

## 2. Learning Data Pipeline

### 2.1 Data Abstraction Flow

```mermaid
graph TD
    A[Raw Command Data] --> B[Sensitivity Scanner]
    B --> C{Contains Sensitive Data?}
    C -->|Yes| D[Apply Abstraction Rules]
    C -->|No| E[Direct Pattern Generation]
    D --> F[Generate Semantic Patterns]
    E --> F
    F --> G[Validate Abstraction]
    G --> H{Validation Passed?}
    H -->|Yes| I[Store Abstracted Pattern]
    H -->|No| J[Reject Data]
    I --> K[Update Pattern Registry]
    J --> L[Log Rejection Reason]
```

**Abstraction Rules:**
```python
class AbstractionRules:
    """Rules for converting sensitive data to safe patterns"""
    
    HOSTNAME_PATTERNS = {
        r'.*gpu.*': 'gpu-host',
        r'.*compute.*': 'compute-host',
        r'.*storage.*': 'storage-host',
        r'.*login.*': 'login-host'
    }
    
    PATH_PATTERNS = {
        r'/data/.*genomics.*': 'genomics-data',
        r'/scratch/.*': 'scratch-space',
        r'/home/.*': 'user-home',
        r'.*\.fastq.*': 'sequence-data'
    }
    
    COMMAND_PATTERNS = {
        r'sbatch.*--mem=\d+G.*': 'slurm_memory_job',
        r'singularity exec.*\.sif.*': 'container_execution',
        r'Rscript.*\.R.*': 'r_script_execution'
    }
```

### 2.2 Encrypted Storage Pipeline

```mermaid
graph TD
    A[Abstracted Learning Data] --> B[Get Current Encryption Key]
    B --> C[Serialize Data]
    C --> D[Encrypt with Fernet]
    D --> E[Add Metadata Header]
    E --> F[Store to Daily File]
    F --> G[Update Index]
    G --> H{Daily Rotation?}
    H -->|Yes| I[Rotate Keys]
    H -->|No| J[Complete]
    I --> K[Cleanup Old Keys]
    K --> J
```

**Storage Format:**
```python
# Encrypted file structure
{
    "header": {
        "version": "1.0",
        "key_id": "2024-01-01",
        "host_id_hash": "abc123...",
        "created_at": 1704067200,
        "data_type": "learning_patterns"
    },
    "encrypted_payload": b"gAAAAABh..."  # Fernet encrypted data
}

# Decrypted payload structure
{
    "patterns": [
        {
            "pattern_id": "slurm_001",
            "command_category": "slurm_job_submission",
            "success_rate": 0.92,
            "optimization_suggestions": ["reduce_memory", "use_compute_partition"],
            "performance_characteristics": {
                "avg_duration": 3600,
                "memory_efficiency": 0.8,
                "queue_time": 300
            }
        }
    ],
    "schema_version": "1.2",
    "last_updated": 1704067200
}
```

### 2.3 Adaptive Schema Evolution Flow

```mermaid
graph TD
    A[New Command Pattern] --> B[Schema Observer]
    B --> C[Extract Pattern Signature]
    C --> D[Check Existing Schema]
    D --> E{Pattern Exists?}
    E -->|Yes| F[Update Frequency Counter]
    E -->|No| G[Add New Pattern Fields]
    F --> H[Check Evolution Triggers]
    G --> H
    H --> I{Evolution Needed?}
    I -->|Yes| J[Propose Schema Changes]
    I -->|No| K[Continue Learning]
    J --> L[Validate Changes]
    L --> M[Update Schema Version]
    M --> K
```

**Schema Evolution Example:**
```python
# Week 1: Basic SLURM pattern
initial_schema = {
    "slurm_job": {
        "required_fields": ["command", "memory", "time", "partition"],
        "optional_fields": [],
        "frequency": 25
    }
}

# Week 4: GPU usage detected in 80% of jobs
evolved_schema = {
    "slurm_job": {
        "required_fields": ["command", "memory", "time", "partition"],
        "optional_fields": ["gpu_type", "gpu_count"],  # NEW
        "frequency": 156
    },
    "gpu_slurm_job": {  # NEW specialized pattern
        "required_fields": ["command", "memory", "time", "partition", "gpu_type"],
        "optional_fields": ["cuda_version", "gpu_memory"],
        "frequency": 78
    }
}
```

## 3. Information Threshold System

### 3.1 Information Accumulation Flow

```mermaid
graph TD
    A[Command Execution Event] --> B[Calculate Significance]
    B --> C[Apply Weighted Scoring]
    C --> D[Update Agent Counters]
    D --> E{Check Each Agent}
    E --> F[Learning Analyst]
    E --> G[HPC Advisor]
    E --> H[Troubleshooting Detective]
    F --> I{Threshold Reached?}
    G --> J{Threshold Reached?}
    H --> K{Threshold Reached?}
    I -->|Yes| L[Queue Analysis]
    J -->|Yes| M[Queue Analysis]
    K -->|Yes| N[Queue Analysis]
    I -->|No| O[Continue Accumulation]
    J -->|No| O
    K -->|No| O
    L --> P[Reset Counters]
    M --> P
    N --> P
```

**Significance Calculation:**
```python
def calculate_significance(execution_data: CommandExecutionData) -> float:
    """Calculate information significance with contextual weighting"""
    base_significance = 1.0
    
    # Command novelty (higher = more significant)
    command_frequency = get_command_frequency(execution_data.command)
    if command_frequency < 5:
        base_significance *= 2.0  # Rare command
    
    # Failure significance
    if execution_data.exit_code != 0:
        historical_success = get_success_rate(execution_data.command)
        if historical_success > 0.8:
            base_significance *= 3.0  # Usually successful command failed
        else:
            base_significance *= 1.5  # Expected failure
    
    # Performance anomaly
    expected_duration = predict_duration(execution_data.command)
    actual_duration = execution_data.duration_ms
    performance_ratio = abs(actual_duration - expected_duration) / expected_duration
    if performance_ratio > 0.5:
        base_significance *= (1 + performance_ratio)
    
    # Optimization impact
    if was_optimization_applied(execution_data):
        base_significance *= 1.8
    
    return min(base_significance, 10.0)  # Cap at 10x significance
```

### 3.2 Agent Triggering Flow

```mermaid
graph TD
    A[Threshold Reached] --> B[Create Analysis Context]
    B --> C[Determine Priority]
    C --> D[Queue Background Task]
    D --> E[Agent Analysis Begins]
    E --> F[Load Relevant Learning Data]
    F --> G[Perform Deep Analysis]
    G --> H[Generate Insights]
    H --> I[Update Hook Patterns]
    I --> J[Measure Effectiveness]
    J --> K[Adapt Thresholds]
    K --> L[Complete Analysis]
```

**Agent Context Creation:**
```python
def create_agent_context(agent_name: str, trigger_info: Dict) -> Dict[str, Any]:
    """Create rich context for agent analysis"""
    return {
        "agent_name": agent_name,
        "trigger_reason": f"Information threshold: {trigger_info['score']:.1f}",
        "accumulated_info": trigger_info["counters"],
        "priority": calculate_priority(trigger_info),
        "relevant_patterns": get_relevant_patterns(agent_name),
        "recent_failures": get_recent_failures() if agent_name == "troubleshooting-detective" else [],
        "performance_trends": get_performance_trends() if agent_name == "learning-analyst" else {},
        "analysis_scope": {
            "time_window_hours": 168,  # 1 week
            "max_patterns": 1000,
            "focus_areas": get_agent_focus_areas(agent_name)
        }
    }
```

## 4. Cross-Host Mesh Synchronization

### 4.1 Peer Discovery Flow

```mermaid
graph TD
    A[Mesh Sync Trigger] --> B[Discover Tailscale Peers]
    B --> C[Filter for Claude-Sync Hosts]
    C --> D[Check Trust Status]
    D --> E{Trusted Peers Found?}
    E -->|Yes| F[Test Connectivity]
    E -->|No| G[SSH Fallback Discovery]
    F --> H[Rank by Quality]
    G --> I[Check SSH Config]
    I --> J[Test SSH Connectivity]
    J --> H
    H --> K[Select Sync Targets]
    K --> L[Begin Pattern Exchange]
```

### 4.2 Pattern Synchronization Flow

```mermaid
graph TD
    A[Pattern Sync Request] --> B[Abstract Local Patterns]
    B --> C[Create Sync Payload]
    C --> D[Encrypt for Transit]
    D --> E[Send to Peer]
    E --> F[Peer Validates Patterns]
    F --> G{Validation Passed?}
    G -->|Yes| H[Merge with Local Patterns]
    G -->|No| I[Reject Patterns]
    H --> J[Send Acknowledgment]
    I --> K[Send Rejection Reason]
    J --> L[Update Sync Metadata]
    K --> L
```

**Sync Pattern Structure:**
```python
# Patterns safe for cross-host sharing
sync_pattern = {
    "pattern_id": "cmd_optimization_001",
    "command_category": "text_search",  # NOT real command
    "success_rate": 0.92,
    "performance_tier": "fast",
    "optimization_type": "tool_upgrade", 
    "confidence": 0.85,
    "usage_frequency": 47,
    "effectiveness_metrics": {
        "speed_improvement": 3.2,
        "error_reduction": 0.15
    },
    "created_by_host_type": "development_machine",
    "applicable_host_types": ["development_machine", "compute_cluster"]
}
```

## 5. Performance Monitoring Flow

### 5.1 Real-time Performance Tracking

```mermaid
graph TD
    A[Hook Execution Start] --> B[Record Start Time]
    B --> C[Monitor Memory Usage]
    C --> D[Execute Hook Logic]
    D --> E[Record End Time]
    E --> F[Calculate Metrics]
    F --> G{Performance Target Met?}
    G -->|Yes| H[Log Success]
    G -->|No| I[Log Performance Issue]
    H --> J[Update Performance Stats]
    I --> K[Trigger Performance Alert]
    K --> J
    J --> L[Return to Caller]
```

**Performance Metrics Collection:**
```python
@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    hook_name: str
    execution_time_ms: float
    memory_peak_mb: float
    cpu_time_ms: float
    pattern_lookups: int
    encryption_operations: int
    network_calls: int
    success: bool
    error_type: Optional[str] = None

def record_hook_performance(hook_name: str, metrics: PerformanceMetrics):
    """Record performance metrics for analysis"""
    # Update rolling averages
    update_performance_window(hook_name, metrics)
    
    # Check against targets
    if not meets_performance_targets(hook_name, metrics):
        trigger_performance_alert(hook_name, metrics)
    
    # Update long-term trends
    update_performance_trends(hook_name, metrics)
```

## 6. Error Handling and Recovery

### 6.1 Graceful Degradation Flow

```mermaid
graph TD
    A[Component Failure] --> B{Failure Type?}
    B -->|Learning Data| C[Use Memory Cache]
    B -->|Encryption| D[Use Plaintext Temporarily]  
    B -->|Network| E[Local-Only Mode]
    B -->|Agent| F[Hook-Only Mode]
    C --> G[Log Degradation]
    D --> G
    E --> G
    F --> G
    G --> H[Continue Operation]
    H --> I[Background Recovery]
    I --> J{Recovery Successful?}
    J -->|Yes| K[Restore Full Function]
    J -->|No| L[Maintain Degraded Mode]
```

This data flow architecture ensures that claude-sync operates efficiently and reliably across all components while maintaining strict performance targets and security guarantees.