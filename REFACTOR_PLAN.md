# REFACTOR_PLAN.md - Learning System Knowledge Accrual

## 🧠 **Learning System Architecture - Knowledge Accrual Focus**

The learning system is the core intelligence engine that continuously accumulates knowledge about hosts, commands, performance patterns, and user behaviors across the distributed computing environment.

## 📊 **Knowledge Categories & Accrual Mechanisms**

### 1. **Host Knowledge Accrual**

#### **System Capabilities Discovery**
```python
# Continuous capability learning
host_knowledge = {
    "hardware_profile": {
        "cpu_info": detect_cpu_capabilities(),
        "memory_total": get_memory_info(),  
        "gpu_devices": discover_gpu_hardware(),
        "storage_devices": analyze_storage_systems(),
        "network_interfaces": map_network_topology()
    },
    "software_environment": {
        "available_modules": scan_module_system(),
        "installed_packages": inventory_packages(),
        "schedulers": detect_job_schedulers(),
        "containers": find_container_systems(),
        "compilers": discover_toolchains()
    },
    "performance_characteristics": {
        "cpu_benchmarks": measure_compute_performance(),
        "io_throughput": test_storage_performance(),
        "network_bandwidth": measure_network_capacity(),
        "memory_bandwidth": test_memory_performance()
    }
}
```

#### **Dynamic Environment Learning**
- **File system topology**: Learn data locations, mount points, shared storage
- **Access patterns**: Track which paths are frequently accessed
- **Resource availability**: Monitor CPU, memory, GPU utilization over time
- **Network connectivity**: Map reachable hosts, connection quality, routing

### 2. **Command Pattern Knowledge Accrual**

Focus on your core workflow domains: **Bash, SSH, Tailscale, R/Rscript, Containers, Singularity, SLURM**

#### **Success Pattern Learning**
```python
# Learn what works well on each host
command_success_patterns = {
    # SLURM job patterns
    "sbatch_genomics": {
        "command_signature": "sbatch --mem=32G --cpus-per-task=8 run_analysis.sh",
        "success_rate": 0.92,
        "typical_duration": 3600,
        "optimal_flags": ["--mem=32G", "--cpus-per-task=8", "--partition=compute"],
        "failure_modes": ["out_of_memory", "time_limit", "node_unavailable"]
    },
    
    # Singularity container patterns
    "singularity_r_analysis": {
        "command_signature": "singularity exec --bind /data r-analysis.sif Rscript analysis.R",
        "success_rate": 0.88,
        "common_bind_mounts": ["/data", "/scratch", "/home"],
        "container_preferences": {"r-analysis.sif": 0.95, "rstudio.sif": 0.82}
    },
    
    # SSH + Tailscale patterns
    "tailscale_ssh_transfer": {
        "command_signature": "rsync -av data/ user@compute-node.tailnet:/scratch/",
        "success_rate": 0.96,
        "bandwidth_patterns": {"tailscale": "avg_50mbps", "direct_ssh": "avg_15mbps"},
        "preferred_routes": ["tailscale_direct", "ssh_jump_host"]
    },
    
    # R workflow patterns
    "r_data_processing": {
        "command_signature": "Rscript --vanilla process_data.R input.csv output.rds",
        "memory_usage": "avg_4GB_peak_8GB",
        "success_flags": ["--vanilla", "--max-mem-size=8G"],
        "common_failures": ["memory_exhausted", "package_missing", "data_corrupt"]
    }
}
```

#### **Adaptive Pattern Discovery**
```python
# Capture command execution context
command_execution_context = {
    "timestamp": 1704067200,
    "command": "sbatch --mem=64G --time=24:00:00 run_blast.sh",
    "host_context": {
        "hostname": "login-node-1",
        "tailscale_ip": "100.64.1.5",
        "slurm_partition": "gpu",
        "available_containers": ["blast-2.14.sif", "biotools.sif"]
    },
    "execution_result": {
        "exit_code": 0,
        "duration": 14400,  # 4 hours
        "job_id": "123456",
        "peak_memory": "48G",
        "cpu_efficiency": 0.89
    },
    "learned_insights": {
        "memory_overallocation": "requested_64G_used_48G",
        "time_efficiency": "completed_in_60%_of_requested_time",
        "pattern_match": "typical_blast_workload"
    }
}
```

#### **Domain-Specific Learning Focus**

**SLURM Learning:**
- Memory/CPU optimization patterns
- Queue time predictions by partition
- Job dependency patterns
- Resource efficiency tracking

**Container Learning:**
- Singularity bind mount patterns
- Container selection by task type
- Performance differences between containers
- Common container failures

**R/Rscript Learning:**
- Memory usage patterns by script type
- Package dependency resolution
- Data size vs processing time correlations
- Common R error patterns

**SSH/Tailscale Learning:**
- Connection reliability by route
- Bandwidth patterns (Tailscale vs direct SSH)
- File transfer optimization
- Authentication patterns

**Bash Command Learning:**
- Command flag effectiveness
- Pipeline optimization patterns
- File processing workflows
- Error recovery strategies

### 3. **User Behavior Knowledge Accrual**

#### **Workflow Pattern Recognition**
```python
# Learn user workflow sequences
workflow_knowledge = {
    "sequence_patterns": [
        {
            "pattern": ["ssh compute-node", "module load blast", "sbatch job.sh"],
            "frequency": 0.85,
            "success_rate": 0.92,
            "typical_timing": "morning_batch_submission"
        }
    ],
    "context_switches": {
        "genomics_workflow": ["data-prep-host", "compute-cluster", "results-host"],
        "ml_training": ["dev-machine", "gpu-cluster", "model-storage"]
    }
}
```

#### **Preference Learning**
- **Host preferences**: Which hosts user prefers for different task types
- **Tool preferences**: Preferred versions, configurations, parameters
- **Timing patterns**: When user typically runs different types of jobs
- **Error recovery**: How user typically resolves common problems

### 4. **Cross-Host Topology Knowledge Accrual**

#### **Tailscale-Centric Network Learning** (Encrypted)
```python
# Focus on Tailscale + SSH hybrid topology
network_topology = {
    "tailscale_mesh": {
        "primary_routes": {
            "compute-cluster": {
                "tailscale_ip": "100.64.1.10",
                "direct_latency": 12.5,
                "bandwidth": "avg_85mbps",
                "reliability": 0.98,
                "preferred_for": ["data_transfer", "interactive_sessions"]
            },
            "storage-nodes": {
                "tailscale_ip": "100.64.1.20-25",
                "aggregate_bandwidth": "200mbps",
                "best_for_large_transfers": True,
                "rsync_optimization": "--bwlimit=50000"
            }
        }
    },
    
    "ssh_fallback_routes": {
        "jump_hosts": {
            "primary_gateway": {
                "hostname": "gateway.institution.edu",
                "reliability": 0.95,
                "used_when": "tailscale_unavailable",
                "auth_method": "ssh_key"
            }
        }
    },
    
    "learned_patterns": {
        "file_transfer_routes": {
            "small_files_<100MB": "tailscale_direct",
            "large_files_>1GB": "tailscale_rsync_optimized", 
            "cluster_job_data": "shared_filesystem_preferred"
        },
        "connection_preferences": {
            "interactive_R_sessions": "tailscale_low_latency",
            "slurm_job_submission": "any_reliable_route",
            "container_pulls": "highest_bandwidth_available"
        }
    }
}
```

#### **SLURM Cluster Topology Learning**
```python
# Learn cluster resource patterns
slurm_cluster_knowledge = {
    "partition_characteristics": {
        "gpu": {
            "typical_queue_time": "15min_avg",
            "node_types": ["v100", "a100"],
            "memory_per_node": "256GB_avg",
            "best_for": ["deep_learning", "blast_large_db"]
        },
        "compute": {
            "typical_queue_time": "5min_avg", 
            "high_throughput": True,
            "best_for": ["r_analysis", "genomics_pipeline"]
        }
    },
    
    "storage_patterns": {
        "scratch_performance": "high_iops_temporary",
        "data_persistence": "/data_shared_permanent",
        "home_quotas": "limited_use_for_scripts_only"
    }
}
```

#### **Container Registry & Access Learning**
- **Singularity container locations**: Which hosts have which containers cached
- **Container build patterns**: Successful container recipes and configurations  
- **Bind mount strategies**: Optimal mount points for different workflow types
- **Performance characteristics**: Container overhead by host type

## 🔄 **Knowledge Accrual Mechanisms**

### 1. **Passive Command History Learning**
**What we capture from every command execution:**
```python
# PostToolUse hook captures this data automatically
command_learning_data = {
    "command": "sbatch --mem=32G run_analysis.sh", 
    "timestamp": 1704067200,
    "host_context": {
        "hostname": "login-node-1",
        "tailscale_status": "connected",
        "slurm_partition": "compute",  
        "available_memory": "512GB",
        "load_average": 2.3
    },
    "execution_outcome": {
        "exit_code": 0,
        "duration": 3600,
        "stdout_length": 1024,
        "stderr_patterns": []
    },
    "learned_abstractions": {
        "command_type": "slurm_submission",
        "resource_pattern": "medium_memory_job",
        "success_indicators": ["job_accepted", "normal_completion"]
    }
}
```

**Command pattern recognition:**
- **SLURM commands**: `sbatch`, `squeue`, `scancel` - track resource requests vs actual usage
- **R commands**: `Rscript`, `R CMD` - monitor memory patterns and package dependencies
- **Container commands**: `singularity exec/run` - learn bind mount patterns and container preferences
- **SSH/Tailscale**: `ssh user@host.tailnet`, `rsync via tailscale` - track connection reliability
- **File operations**: Large data transfers, genomics file processing workflows

### 2. **Active Discovery** 
- **Capability probing**: Periodically test for new capabilities
- **Performance benchmarking**: Run lightweight tests to assess performance
- **Connectivity scanning**: Map network topology and test connections
- **Software inventory**: Scan for new tools, modules, packages

### 3. **Collaborative Learning**
- **Cross-host knowledge sharing**: Sync insights between related hosts
- **Pattern aggregation**: Combine individual observations into general rules
- **Consensus building**: Resolve conflicts between different observations
- **Knowledge validation**: Verify learned patterns across multiple hosts

### 4. **Adaptive Schema Evolution**

**Problem:** Initial assumptions about learning patterns may prove incorrect as usage evolves.

**Solution:** NoSQL-style flexible schema that evolves with actual usage patterns.

```python
class AdaptiveLearningSchema:
    """Self-evolving knowledge schema based on observed patterns"""
    
    def __init__(self):
        self.schema_version = 1
        self.pattern_registry = defaultdict(dict)
        self.usage_frequency = Counter()
        self.evolution_triggers = {}
        
    def observe_command_pattern(self, command_data):
        """Learn from actual command execution patterns"""
        # Extract pattern signature
        pattern_sig = self._extract_pattern_signature(command_data)
        
        # Track usage frequency
        self.usage_frequency[pattern_sig] += 1
        
        # Discover new fields dynamically
        for key, value in command_data.items():
            if key not in self.pattern_registry[pattern_sig]:
                # New field discovered - add to schema
                self.pattern_registry[pattern_sig][key] = {
                    'type': type(value).__name__,
                    'first_seen': time.time(),
                    'frequency': 1,
                    'example_values': [value]
                }
            else:
                # Update existing field stats
                field_info = self.pattern_registry[pattern_sig][key]
                field_info['frequency'] += 1
                if value not in field_info['example_values']:
                    field_info['example_values'].append(value)
                    # Keep only recent examples
                    if len(field_info['example_values']) > 10:
                        field_info['example_values'] = field_info['example_values'][-10:]
    
    def evolve_schema_if_needed(self):
        """Trigger schema evolution based on usage patterns"""
        
        # Detect new dominant patterns
        top_patterns = self.usage_frequency.most_common(10)
        
        for pattern_sig, frequency in top_patterns:
            if frequency > 50:  # Significant usage
                pattern_data = self.pattern_registry[pattern_sig]
                
                # Check if this pattern suggests schema changes
                new_fields = [k for k, v in pattern_data.items() 
                             if v['frequency'] > frequency * 0.8]  # Field appears in 80%+ of cases
                
                if len(new_fields) > 3:  # Pattern has consistent structure
                    self._propose_schema_evolution(pattern_sig, new_fields)
    
    def _propose_schema_evolution(self, pattern_sig, consistent_fields):
        """Propose new schema elements based on discovered patterns"""
        
        # Example: Discovered consistent GPU usage patterns
        if 'gpu_usage' in consistent_fields and 'cuda_version' in consistent_fields:
            self._add_gpu_learning_category()
        
        # Example: Discovered consistent container patterns  
        if 'container_name' in consistent_fields and 'bind_mounts' in consistent_fields:
            self._enhance_container_learning()
            
        # Example: New workflow tools discovered
        if pattern_sig.startswith('nextflow_') or pattern_sig.startswith('snakemake_'):
            self._add_workflow_engine_support(pattern_sig)
    
    def _add_gpu_learning_category(self):
        """Add GPU-specific learning patterns"""
        # Dynamically add new learning category
        pass
    
    def get_current_schema(self):
        """Return current adaptive schema"""
        return {
            'version': self.schema_version,
            'discovered_patterns': dict(self.pattern_registry),
            'usage_stats': dict(self.usage_frequency),
            'evolution_history': self.evolution_triggers
        }
```

**Schema Evolution Examples:**

```python
# Week 1: System starts with basic assumptions
initial_patterns = {
    'slurm_job': ['command', 'memory', 'time', 'partition'],
    'r_script': ['command', 'memory_usage', 'duration']
}

# Week 4: Usage reveals new patterns
evolved_patterns = {
    'slurm_job': ['command', 'memory', 'time', 'partition', 'gpu_type', 'job_array_size'],
    'r_script': ['command', 'memory_usage', 'duration', 'package_dependencies', 'data_size'],
    'nextflow_pipeline': ['command', 'workflow_name', 'resume_option', 'container_engine'],  # NEW
    'jupyter_session': ['command', 'kernel_type', 'resource_usage', 'session_duration']      # NEW
}

# Week 8: Further evolution based on actual usage
specialized_patterns = {
    'gpu_slurm_job': {  # Split from general slurm_job
        'required_fields': ['gpu_count', 'cuda_version', 'memory_per_gpu'],
        'optimization_patterns': ['batch_size_tuning', 'memory_efficiency']
    },
    'bioinformatics_container': {  # Discovered domain-specific pattern
        'common_containers': ['blast', 'gatk', 'samtools'],
        'typical_bind_mounts': ['/data', '/scratch', '/reference_genomes'],
        'resource_patterns': ['io_intensive', 'memory_hungry', 'cpu_parallel']
    }
}
```

**Benefits of Adaptive Schema:**

1. **No upfront schema design needed** - Let usage patterns drive the structure
2. **Automatic discovery** - System learns what matters from actual commands  
3. **Evolution without breaking** - Old patterns remain valid while new ones emerge
4. **Usage-driven optimization** - Focus learning on what you actually use
5. **Future-proof** - Can adapt to new tools/workflows without code changes

**Evolution Triggers:**
- New command types used >50 times
- Consistent new fields in >80% of command executions
- Performance patterns that suggest new optimization categories
- User feedback indicating missing or incorrect learning categories

## 🏗️ **Knowledge Storage Architecture**

### **Dual-Security Model**

#### **Public Knowledge** (Unencrypted, Shareable)
```
learning/public/
├── command_optimizations.json    # grep→rg rules, etc.
├── safety_patterns.json          # Dangerous command detection
├── performance_heuristics.json   # General performance rules
└── tool_compatibility.json       # Software compatibility matrices
```

#### **Secure Knowledge** (Encrypted, Host-Specific)
```
learning/secure/
├── host_profiles.enc             # Detailed host capabilities
├── network_topology.enc          # Connection patterns & quality
├── user_patterns.enc             # Personal workflow preferences  
└── access_credentials.enc        # Authentication patterns
```

### **Knowledge Persistence**
- **Incremental updates**: Add new knowledge without rewriting everything
- **Atomic operations**: Ensure knowledge consistency during updates
- **Backup & recovery**: Protect against knowledge loss
- **Compression**: Efficient storage of large knowledge bases

### **Knowledge Indexing**
- **Fast lookups**: Index knowledge for quick retrieval
- **Pattern matching**: Efficient similarity search across patterns
- **Temporal indexing**: Track knowledge changes over time
- **Contextual search**: Find relevant knowledge based on current context

## 🎯 **Knowledge Quality Assurance**

### **Confidence Scoring**
- **Observation count**: More observations = higher confidence
- **Consistency**: Consistent patterns across time = higher confidence  
- **Validation**: Successfully applied knowledge = higher confidence
- **Recency**: Recent observations weighted more heavily

### **Knowledge Pruning**
- **Obsolete pattern detection**: Remove outdated or invalid patterns
- **Confidence thresholding**: Remove low-confidence knowledge
- **Storage optimization**: Compress or archive old knowledge
- **Conflict resolution**: Handle contradictory knowledge gracefully

### **Continuous Learning**
- **Pattern drift detection**: Notice when environments change
- **Adaptation triggers**: Update knowledge when patterns shift
- **Learning rate adjustment**: Learn faster in new environments
- **Forgetting mechanisms**: Gradually reduce weight of old patterns

## 📈 **Learning System Evolution**

### **Bootstrapping New Environments**
1. **Initial discovery phase**: Rapid capability and topology scanning
2. **Pattern establishment**: Build initial success/failure models
3. **Refinement phase**: Improve accuracy through continued observation
4. **Optimization phase**: Fine-tune recommendations based on experience

### **Knowledge Transfer**
- **Environment similarity**: Transfer knowledge between similar hosts
- **Pattern generalization**: Abstract specific patterns to general rules
- **Cross-validation**: Verify transferred knowledge works in new contexts
- **Selective transfer**: Only transfer relevant, high-confidence knowledge

This learning system focuses purely on knowledge accrual - continuously building a comprehensive, accurate, and adaptive understanding of the distributed computing environment. The knowledge usage (how this intelligence gets applied to help users) will be addressed in the next section.

---

## 🔐 **Simple Host Authorization & Key Management**

### **Current Problem**
Environment variables for encryption keys are:
- Visible in process lists
- Manual to distribute
- No rotation mechanism
- Hard to revoke access

### **Simple Solution: Hardware-Based Trust + Auto Key Rotation**

#### **Zero-Config Host Identity**
```python
class SimpleHostTrust:
    """Dead simple host trust based on hardware fingerprint"""
    
    def __init__(self):
        self.host_id = self._get_stable_host_id()
        self.trust_file = Path.home() / '.claude' / 'trusted_hosts'
        
    def _get_stable_host_id(self):
        """Generate stable ID from hardware that survives OS reinstalls"""
        # Combine stable hardware characteristics
        sources = [
            self._get_cpu_serial(),    # CPU identifier
            self._get_motherboard_id() # Motherboard UUID
        ]
        combined = ''.join(filter(None, sources))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def is_trusted_host(self, host_id):
        """Simple check: is this host in our trust list?"""
        if not self.trust_file.exists():
            return False
        with open(self.trust_file) as f:
            trusted = [line.strip() for line in f if line.strip()]
        return host_id in trusted
    
    def add_trusted_host(self, host_id):
        """Add host to trust list - that's it"""
        trusted = set()
        if self.trust_file.exists():
            with open(self.trust_file) as f:
                trusted = set(line.strip() for line in f if line.strip())
        
        trusted.add(host_id)
        self.trust_file.parent.mkdir(exist_ok=True)
        with open(self.trust_file, 'w') as f:
            for host in sorted(trusted):
                f.write(f"{host}\n")
```

#### **Auto-Rotating Encryption Keys**
```python
class SimpleKeyManager:
    """Simple key rotation without complex synchronization"""
    
    def __init__(self):
        self.key_dir = Path.home() / '.claude' / 'keys'
        self.key_dir.mkdir(parents=True, exist_ok=True)
        
    def get_current_key(self):
        """Get today's encryption key - auto-generates if needed"""
        today = datetime.now().strftime('%Y-%m-%d')
        key_file = self.key_dir / f"key_{today}.enc"
        
        if key_file.exists():
            return self._load_key(key_file)
        else:
            # Generate new key for today
            key = self._generate_key(today)
            self._save_key(key_file, key)
            # Clean up old keys (keep last 7 days)
            self._cleanup_old_keys()
            return key
    
    def _generate_key(self, date_str):
        """Generate deterministic key from host ID + date"""
        host_trust = SimpleHostTrust()
        seed_material = f"{host_trust.host_id}:{date_str}:claude-sync"
        
        # PBKDF2 for key derivation
        key = hashlib.pbkdf2_hmac(
            'sha256',
            seed_material.encode(),
            b'claude-learning-encryption',
            100000,
            32
        )
        return base64.urlsafe_b64encode(key)
    
    def _cleanup_old_keys(self):
        """Remove keys older than 7 days"""
        cutoff = datetime.now() - timedelta(days=7)
        for key_file in self.key_dir.glob("key_*.enc"):
            try:
                file_date = datetime.strptime(key_file.stem.split('_')[1], '%Y-%m-%d')
                if file_date < cutoff:
                    key_file.unlink()
            except:
                pass  # Skip malformed files
```

#### **Simple Authorization Workflow**
```bash
# On new host that wants access:
claude-sync request-access

# On authorized host:
claude-sync approve-host abc123def456

# List trusted hosts:
claude-sync list-hosts

# Revoke a host:
claude-sync revoke-host abc123def456
```

```python
def request_access():
    """Request access from another host"""
    trust = SimpleHostTrust()
    print(f"🔐 Host ID: {trust.host_id}")
    print("Run this on an authorized host:")
    print(f"    claude-sync approve-host {trust.host_id}")

def approve_host(host_id):
    """Approve a host for access"""
    trust = SimpleHostTrust()
    trust.add_trusted_host(host_id)
    print(f"✅ Host {host_id[:8]}... is now trusted")
    
    # Trigger key rotation to sync new access
    keys = SimpleKeyManager()
    keys.get_current_key()  # This will generate fresh keys

def list_hosts():
    """List all trusted hosts"""
    trust = SimpleHostTrust()
    if not trust.trust_file.exists():
        print("No trusted hosts")
        return
    
    with open(trust.trust_file) as f:
        hosts = [line.strip() for line in f if line.strip()]
    
    print(f"📋 {len(hosts)} trusted hosts:")
    for host in hosts:
        print(f"  {host[:8]}...")
```

### **How It Works**

1. **First time setup**: Host generates stable ID from hardware
2. **Request access**: New host shows its ID, user copies it
3. **Approve access**: Authorized host adds ID to trust list  
4. **Daily key rotation**: Keys auto-rotate based on date + host ID
5. **Revoke access**: Remove host ID from trust list, keys auto-rotate

### **Benefits**

#### **Secure**
- Hardware-based identity (survives OS reinstalls)
- Daily automatic key rotation
- Keys never stored in environment variables
- Simple trust list (easy to audit)

#### **Simple** 
- Binary trust: host is trusted or not
- No complex permissions or capabilities
- 3 commands: request, approve, revoke
- Zero configuration for daily use

#### **Resilient**
- Works offline (cached keys)
- Self-healing key rotation
- Survives system reinstalls
- No complex synchronization protocols

---

## 🎯 **Hook System Implementation - Inspired by claude-code-hooks-mastery**

Based on the excellent patterns from `disler/claude-code-hooks-mastery`, here's how we'll enhance our hook system:

### **Hook Lifecycle Integration**

```python
# hooks/intelligent-optimizer.py - Enhanced PreToolUse hook
#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# ///

import json
import sys
from pathlib import Path

def main():
    hook_input = json.loads(sys.stdin.read())
    
    # Extract command and context
    tool_name = hook_input.get('tool_name')
    command = hook_input.get('tool_input', {}).get('command', '')
    
    if tool_name != 'Bash':
        sys.exit(0)
    
    # Load learning data
    learning_data = load_secure_learning_data()
    
    # Apply intelligent optimization
    result = optimize_with_learning(command, learning_data, hook_input)
    
    if result:
        print(json.dumps(result))
    
    sys.exit(0)

def optimize_with_learning(command, learning_data, context):
    """Apply learned optimizations with sophisticated control"""
    
    # Get learned suggestions
    suggestions = get_learned_suggestions(command, learning_data)
    optimized_command = apply_optimizations(command, learning_data)
    safety_warnings = check_safety_patterns(command, learning_data)
    
    # Smart feedback based on confidence levels
    feedback_parts = []
    
    if optimized_command != command:
        confidence = calculate_optimization_confidence(command, learning_data)
        if confidence > 0.8:
            feedback_parts.append(f"🚀 **High-confidence optimization:**\n```bash\n{optimized_command}\n```")
        else:
            feedback_parts.append(f"💡 **Suggested optimization (confidence: {confidence:.0%}):**\n```bash\n{optimized_command}\n```")
    
    if safety_warnings:
        feedback_parts.append("⚠️ **Safety analysis:**\n" + "\n".join(f"  {w}" for w in safety_warnings))
    
    if suggestions:
        feedback_parts.append(suggestions)
    
    # Return control structure like claude-code-hooks-mastery
    if feedback_parts:
        return {
            'block': False,  # Never block, just suggest
            'message': "\n\n".join(feedback_parts)
        }
    
    return None
```

### **Post-Execution Learning Hook**

```python
# hooks/learning-collector.py - Enhanced PostToolUse hook  
#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# ///

import json
import sys
import time

def main():
    hook_input = json.loads(sys.stdin.read())
    
    # Collect execution data for learning
    command_data = extract_learning_data(hook_input)
    
    # Store with adaptive schema
    store_learning_data_adaptively(command_data)
    
    # Update success/failure patterns
    update_pattern_knowledge(command_data)
    
    # Never block or show messages in PostToolUse - just learn silently
    sys.exit(0)

def extract_learning_data(hook_input):
    """Extract rich learning data from execution context"""
    return {
        'command': hook_input.get('tool_input', {}).get('command', ''),
        'exit_code': hook_input.get('tool_output', {}).get('exit_code', 0),
        'duration': calculate_duration(hook_input),
        'host_context': get_host_context(),
        'tailscale_status': get_tailscale_status(),
        'slurm_context': get_slurm_context() if is_slurm_command() else None,
        'timestamp': time.time()
    }
```

### **Context Enhancement Hook**

```python
# hooks/context-enhancer.py - UserPromptSubmit hook
#!/usr/bin/env -S uv run  
# /// script
# requires-python = ">=3.10"
# ///

import json
import sys

def main():
    hook_input = json.loads(sys.stdin.read())
    
    user_prompt = hook_input.get('user_prompt', '')
    
    # Detect if user is asking for help with specific tools
    context_enhancement = detect_and_enhance_context(user_prompt)
    
    if context_enhancement:
        # Inject relevant context from learning data
        enhanced_prompt = f"{user_prompt}\n\n{context_enhancement}"
        
        result = {
            'block': False,
            'message': f"🧠 **Added context from learning data:**\n{context_enhancement}"
        }
        print(json.dumps(result))
    
    sys.exit(0)

def detect_and_enhance_context(prompt):
    """Enhance prompts with relevant learned context"""
    
    # Detect SLURM questions
    if any(word in prompt.lower() for word in ['sbatch', 'slurm', 'queue', 'partition']):
        slurm_context = get_learned_slurm_patterns()
        if slurm_context:
            return f"**Learned SLURM patterns on this cluster:**\n{slurm_context}"
    
    # Detect R/container questions  
    if any(word in prompt.lower() for word in ['rscript', 'singularity', 'container']):
        container_context = get_learned_container_patterns()
        if container_context:
            return f"**Common container patterns:**\n{container_context}"
    
    return None
```

### **Key Implementation Patterns from claude-code-hooks-mastery**

**1. JSON Control Structure:**
```python
# Never block unless absolutely necessary
result = {
    'block': False,  # True only for dangerous commands
    'message': 'Helpful suggestion or warning'
}
```

**2. UV Script Architecture:**
- Self-contained scripts with inline dependencies  
- Fast startup time
- Environment inheritance from Claude Code

**3. Lifecycle Event Specialization:**
- **UserPromptSubmit**: Context enhancement, learning data injection
- **PreToolUse**: Command optimization, safety checks
- **PostToolUse**: Silent learning data collection  
- **Stop**: Could be used for workflow completion tracking

**4. Sophisticated Flow Control:**
```python
# Confidence-based suggestions
if optimization_confidence > 0.9:
    # High confidence - show as recommendation
    feedback_type = "🚀 **Recommended:**"
elif optimization_confidence > 0.7:
    # Medium confidence - show as suggestion
    feedback_type = "💡 **Suggested:**"
else:
    # Low confidence - don't suggest
    return None
```

### **Integration with Our Learning System**

**Learning Data Flow:**
1. **UserPromptSubmit** → Inject relevant context from past patterns
2. **PreToolUse** → Apply learned optimizations and safety checks  
3. **PostToolUse** → Collect execution data for adaptive schema evolution
4. **Background Process** → Analyze patterns, evolve schema, sync across hosts

**Benefits from claude-code-hooks-mastery Patterns:**
- **Deterministic control** over Claude's behavior
- **Non-blocking suggestions** that enhance rather than interrupt workflow
- **Modular architecture** with specialized hooks for different purposes
- **Environment awareness** and context injection
- **Robust error handling** and graceful degradation

This gives us a solid foundation for implementing intelligent, learning-based hooks that enhance productivity without being intrusive.

---

---

## 🚀 **Claude-Sync Installation & Testing Strategy**

### **Installation Strategy Overview**

**Repository Location Flexibility:**
- claude-sync can live anywhere (currently `~/.claude/claude-sync`)
- Use symlinks to `~/.claude/` for activation
- Never modify user's existing `~/.claude/settings.json` directly

### **Clean Activation System**

```bash
# Activate claude-sync hooks globally
claude-sync activate --global    # Links to ~/.claude/settings.json

# Activate for current project only  
claude-sync activate --project   # Links to .claude/settings.json

# Status check
claude-sync status               # Shows active hooks and configuration

# Deactivate cleanly
claude-sync deactivate          # Remove symlinks, restore original settings
claude-sync deactivate --purge  # Also remove all learning data
```

### **Hook Installation via Symlinks**

```bash
# Claude Code looks for hooks in these locations
~/.claude/hooks/                 # Global hooks (our target)
.claude/hooks/                   # Project-specific hooks

# Our approach: symlink from claude-sync to Claude Code locations
~/.claude/hooks/intelligent-optimizer.py -> ~/.claude/claude-sync/hooks/intelligent-optimizer.py
~/.claude/hooks/learning-collector.py    -> ~/.claude/claude-sync/hooks/learning-collector.py
~/.claude/hooks/context-enhancer.py      -> ~/.claude/claude-sync/hooks/context-enhancer.py
```

### **Settings Management Strategy**

```json
// Template: ~/.claude/claude-sync/templates/settings.global.json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command", 
            "command": "~/.claude/hooks/context-enhancer.py"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/intelligent-optimizer.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command", 
            "command": "~/.claude/hooks/learning-collector.py"
          }
        ]
      }
    ]
  }
}
```

### **Activation Implementation**

```python
class ClaudeSyncActivator:
    """Manages clean activation/deactivation of claude-sync"""
    
    def __init__(self):
        self.claude_dir = Path.home() / '.claude'
        self.sync_dir = Path.home() / '.claude' / 'claude-sync'
        self.hooks_dir = self.claude_dir / 'hooks'
        self.backup_dir = self.sync_dir / 'backups'
        
    def activate_global(self):
        """Activate hooks globally for all Claude Code sessions"""
        # 1. Create hooks directory
        self.hooks_dir.mkdir(exist_ok=True)
        
        # 2. Backup existing settings
        self._backup_user_settings()
        
        # 3. Create hook symlinks
        self._create_hook_symlinks()
        
        # 4. Merge settings (JSON merge, not overwrite)
        self._merge_hook_settings('global')
        
        # 5. Verify activation
        return self._verify_activation()
    
    def activate_project(self):
        """Activate hooks for current project only"""
        project_claude_dir = Path.cwd() / '.claude'
        project_claude_dir.mkdir(exist_ok=True)
        
        # Similar process but for project-level settings
        
    def deactivate(self, purge_data=False):
        """Clean deactivation with optional data purging"""
        # 1. Remove hook symlinks
        # 2. Restore original settings
        # 3. Optionally purge learning data
        
    def _merge_hook_settings(self, scope):
        """Intelligently merge our hooks into existing settings"""
        # Load existing settings
        # Load our template
        # Merge without overwriting user's other settings
        # Write back merged settings
```

### **Testing Framework**

```bash
# Testing commands
claude-sync test-hooks           # Run hooks with mock data
claude-sync test-hooks --verbose # Show detailed hook execution
claude-sync activate --test-mode # Use separate test settings file
claude-sync activate --dry-run   # Show what would be changed

# Testing data
test-data/
├── mock-hook-inputs/           # Sample JSON inputs for each hook type
│   ├── user-prompt-submit.json
│   ├── pre-tool-use.json
│   └── post-tool-use.json
└── expected-outputs/           # Expected responses for validation
```

### **Status & Diagnostics**

```bash
claude-sync status               # Show activation status and active hooks
claude-sync diagnostics          # Run health checks on hooks and learning data
claude-sync logs                 # Show recent hook execution logs
claude-sync learning-stats       # Show learning data statistics
```

### **Enhanced Bootstrap.sh Commands**

```bash
# New bootstrap.sh commands
bootstrap.sh activate --global   # Activate globally
bootstrap.sh activate --project  # Activate for current project
bootstrap.sh deactivate          # Clean deactivation
bootstrap.sh test                # Run test suite
bootstrap.sh status              # Show detailed status
bootstrap.sh diagnostics         # Health check
```

### **Safety & Rollback**

**Backup Strategy:**
- Always backup user's original `~/.claude/settings.json`
- Store backups in `~/.claude/claude-sync/backups/` with timestamps
- Atomic operations - either fully activate or fully rollback

**Rollback Command:**
```bash
claude-sync rollback             # Restore to state before last activation
claude-sync rollback --timestamp # Restore to specific backup
```

### **Development Workflow**

```bash
# Safe development cycle
1. claude-sync activate --test-mode    # Isolated testing environment
2. # Make changes to hooks
3. claude-sync test-hooks              # Validate changes
4. claude-sync activate --global       # Deploy to real environment
5. claude-sync deactivate              # Clean removal when needed
```

This approach ensures:
- **Zero impact** on user's existing Claude Code setup
- **Easy activation/deactivation** for testing
- **Safe development environment** with isolated testing
- **Clean uninstall capability** with full rollback
- **Atomic operations** - no partial state corruption

---

---

## 🤖 **Knowledge Application via Claude Code Agents**

### **Hybrid Architecture: Hooks + Agents**

**Hooks = Real-time Learning & Quick Suggestions**
- Immediate command optimization 
- Safety warnings
- Context injection
- Pattern learning (silent)

**Agents = Deep Knowledge Analysis & Application**
- Comprehensive workflow review
- Learning pattern analysis
- Interactive troubleshooting
- Proactive optimization recommendations

### **Proposed Claude-Sync Agents**

#### **1. Learning Analyst Agent**
```markdown
---
name: learning-analyst
description: Analyzes accumulated learning data to identify optimization opportunities and workflow improvements
tools: Read, Grep, Glob, Bash
---

You are a Learning Analyst specialized in analyzing claude-sync's accumulated knowledge to provide insights and recommendations.

**Your Role:**
- Analyze learning data from ~/.claude/learning/ 
- Identify patterns in command success/failure rates
- Discover optimization opportunities
- Generate actionable recommendations

**Key Capabilities:**
- SLURM resource optimization analysis
- R/container workflow pattern recognition  
- Tailscale network performance insights
- Cross-host efficiency comparisons

**When Invoked:**
- User asks "How can I optimize my workflow?"
- Periodic analysis via `/analyze-learning`
- After significant learning data accumulation
- When performance issues are detected

**Output Format:**
Provide clear, actionable recommendations with confidence levels and supporting data.
```

#### **2. HPC Workflow Advisor Agent**
```markdown
---
name: hpc-advisor
description: Expert advisor for HPC, SLURM, and scientific computing workflows based on learned patterns
tools: Read, Grep, Glob, Bash
---

You are an HPC Workflow Advisor with deep knowledge of the user's learned patterns and cluster characteristics.

**Your Expertise:**
- SLURM job optimization based on historical success patterns
- Container selection and bind mount strategies
- R/genomics workflow best practices
- Tailscale network optimization for HPC environments

**Knowledge Sources:**
- Historical job success/failure patterns
- Resource utilization data
- Network performance metrics
- Container usage patterns

**When to Engage:**
- Complex SLURM job planning
- Workflow optimization requests
- Performance troubleshooting
- Cross-host data movement planning

**Response Style:**
Provide specific, learned-data-backed recommendations with confidence levels.
```

#### **3. Troubleshooting Detective Agent**
```markdown
---
name: troubleshooting-detective
description: Investigates failures and provides solutions based on learned error patterns and successful resolutions
tools: Read, Grep, Glob, Bash
---

You are a Troubleshooting Detective specialized in diagnosing issues using accumulated learning data.

**Investigation Approach:**
1. Analyze current error against historical failure patterns
2. Identify similar past failures and their resolutions
3. Check environmental factors (host, network, resources)
4. Provide step-by-step troubleshooting plan

**Knowledge Base:**
- Historical error patterns and solutions
- Host-specific quirks and workarounds
- Network connectivity issues and fixes
- Resource allocation problems and solutions

**When Called Upon:**
- Command failures that match learned patterns
- Performance degradation issues
- Network connectivity problems
- Resource allocation failures

**Output:**
Structured diagnosis with confidence-ranked solutions based on historical success rates.
```

### **Agent Integration with Learning System**

```python
# Integration pattern
class AgentKnowledgeInterface:
    """Provides learned knowledge to Claude Code agents"""
    
    def get_learning_summary_for_agent(self, agent_name, context=None):
        """Provide relevant learning data to agents"""
        
        if agent_name == "learning-analyst":
            return {
                "recent_patterns": self.get_recent_command_patterns(),
                "efficiency_metrics": self.calculate_workflow_efficiency(),
                "optimization_opportunities": self.identify_optimizations(),
                "success_rates": self.get_success_rate_trends()
            }
        
        elif agent_name == "hpc-advisor":
            return {
                "slurm_patterns": self.get_slurm_optimization_data(),
                "container_preferences": self.get_container_usage_patterns(),
                "network_performance": self.get_tailscale_metrics(),
                "resource_efficiency": self.get_resource_utilization_data()
            }
        
        elif agent_name == "troubleshooting-detective":
            return {
                "error_patterns": self.get_failure_pattern_database(),
                "resolution_history": self.get_successful_fixes(),
                "environmental_factors": self.get_host_specific_issues(),
                "similar_cases": self.find_similar_failures(context)
            }
```

### **Agent Invocation Patterns**

#### **Adaptive Threshold-Based Analysis**

```python
class InformationThresholdManager:
    """Triggers agent analysis based on accumulated information density"""
    
    def __init__(self):
        self.info_counters = {
            'new_commands': 0,        # Novel command patterns
            'failures': 0,            # Command failures
            'optimizations': 0,       # Successful optimizations applied
            'host_changes': 0,        # New hosts or environment changes
            'performance_shifts': 0   # Significant performance pattern changes
        }
        
        # Adaptive thresholds - adjust based on analysis effectiveness
        self.thresholds = {
            'learning_analyst': {
                'base_threshold': 50,     # bits of info to trigger analysis
                'weight_factors': {
                    'new_commands': 2.0,      # New patterns are valuable
                    'failures': 3.0,         # Failures need attention
                    'optimizations': 1.5,    # Success patterns matter
                    'performance_shifts': 4.0 # Performance changes are critical
                }
            },
            'hpc_advisor': {
                'base_threshold': 30,
                'weight_factors': {
                    'new_commands': 3.0,      # HPC commands are complex
                    'failures': 4.0,         # SLURM failures are expensive
                    'host_changes': 2.0,     # New hosts need optimization
                    'performance_shifts': 3.0
                }
            },
            'troubleshooting_detective': {
                'base_threshold': 15,     # Failures need quick attention
                'weight_factors': {
                    'failures': 5.0,         # Primary trigger
                    'new_commands': 1.0,     # Less relevant for troubleshooting
                    'performance_shifts': 3.0
                }
            }
        }
        
        self.analysis_effectiveness = {}  # Track how useful each analysis was
    
    def accumulate_information(self, info_type, significance=1.0):
        """Accumulate information and check if analysis should be triggered"""
        self.info_counters[info_type] += significance
        
        # Check each agent's threshold
        for agent_name, config in self.thresholds.items():
            if self._calculate_weighted_score(agent_name) >= config['base_threshold']:
                self._trigger_agent_analysis(agent_name)
                self._reset_counters_for_agent(agent_name)
    
    def _calculate_weighted_score(self, agent_name):
        """Calculate weighted information score for specific agent"""
        config = self.thresholds[agent_name]
        score = 0
        
        for info_type, count in self.info_counters.items():
            weight = config['weight_factors'].get(info_type, 1.0)
            score += count * weight
            
        return score
    
    def _trigger_agent_analysis(self, agent_name):
        """Trigger background agent analysis"""
        analysis_context = {
            'trigger_reason': f"Information threshold reached: {self._calculate_weighted_score(agent_name):.1f}",
            'accumulated_info': dict(self.info_counters),
            'priority': self._calculate_analysis_priority(agent_name)
        }
        
        # Queue background agent analysis
        queue_agent_analysis(agent_name, analysis_context)
    
    def adapt_thresholds_based_on_effectiveness(self, agent_name, effectiveness_score):
        """Adapt thresholds based on how useful the analysis was"""
        if effectiveness_score > 0.8:
            # Analysis was very useful - lower threshold slightly
            self.thresholds[agent_name]['base_threshold'] *= 0.9
        elif effectiveness_score < 0.3:
            # Analysis wasn't useful - raise threshold
            self.thresholds[agent_name]['base_threshold'] *= 1.2
        
        # Keep thresholds within reasonable bounds
        self.thresholds[agent_name]['base_threshold'] = max(10, min(200, 
            self.thresholds[agent_name]['base_threshold']))
```

#### **Information Accumulation Examples**

```python
# Examples of information "bits" that trigger analysis

# High-value information (triggers analysis sooner)
accumulate_information('failures', significance=3.0)           # SLURM job failed
accumulate_information('performance_shifts', significance=4.0) # 50% slower than usual
accumulate_information('host_changes', significance=2.0)       # New Tailscale host detected

# Medium-value information
accumulate_information('new_commands', significance=2.0)       # First time using Nextflow
accumulate_information('optimizations', significance=1.5)     # Hook suggestion accepted

# Low-value information (accumulates gradually)
accumulate_information('new_commands', significance=0.5)       # Slight variation of known command
accumulate_information('optimizations', significance=0.3)     # Minor flag suggestion
```

#### **Adaptive Threshold Examples**

```python
# Initial state - conservative thresholds
learning_analyst_threshold = 50 bits

# After several high-value analyses
# User finds agent insights very useful (effectiveness = 0.9)
learning_analyst_threshold = 50 * 0.9 = 45 bits  # More sensitive

# After low-value analysis  
# Agent analysis wasn't helpful (effectiveness = 0.2)
learning_analyst_threshold = 45 * 1.2 = 54 bits  # Less sensitive

# Different agents adapt independently
hpc_advisor_threshold = 30 bits        # Still sensitive to HPC issues
troubleshooting_detective = 18 bits    # Very sensitive to failures
```

#### **Proactive Analysis (Threshold-Triggered)**
```bash
# Automatically triggered when information thresholds are reached
# (No manual invocation needed)

Background: learning-analyst triggered (67.2 information bits accumulated)
  → 15 new SLURM patterns, 8 failures, 12 optimizations accepted
  → Analysis priority: HIGH (multiple failures in new patterns)

Background: hpc-advisor triggered (34.5 information bits accumulated)  
  → New Tailscale host detected, 5 container workflow changes
  → Analysis priority: MEDIUM (workflow adaptation needed)

Background: troubleshooting-detective triggered (16.8 information bits accumulated)
  → 3 similar failures in R memory allocation 
  → Analysis priority: URGENT (pattern suggests systematic issue)
```

#### **Reactive Support**
```bash
# After command failures
/troubleshooting-detective investigate-failure "sbatch job failed with exit code 1"

# For planning complex workflows
/hpc-advisor plan-genomics-pipeline "process 100GB FASTQ files"

# Learning insights
/learning-analyst explain-pattern "why do my R jobs fail on gpu partition?"
```

### **Agent-Hook Coordination with Information Thresholds**

**1. Hook Collects → Information Accumulates → Agent Triggers**
```python
# In PostToolUse hook - collect data and accumulate information
def handle_post_tool_use(command, hook_input):
    # Extract learning data
    learning_data = extract_learning_data(hook_input)
    
    # Store learning data
    store_learning_data_adaptively(learning_data)
    
    # Accumulate information for threshold system
    threshold_manager = get_global_threshold_manager()
    
    if learning_data['exit_code'] != 0:
        # Command failed - high significance for troubleshooting
        threshold_manager.accumulate_information('failures', significance=3.0)
        
        # Check if this is a new type of failure
        if is_novel_failure_pattern(learning_data):
            threshold_manager.accumulate_information('new_commands', significance=2.0)
    
    else:
        # Command succeeded
        if was_optimization_applied(learning_data):
            threshold_manager.accumulate_information('optimizations', significance=1.5)
        
        # Check for performance changes
        perf_change = detect_performance_shift(learning_data)
        if perf_change > 0.3:  # 30% change
            threshold_manager.accumulate_information('performance_shifts', 
                                                   significance=perf_change * 4.0)
    
    # Check for new command patterns
    if is_novel_command_pattern(learning_data):
        complexity = calculate_command_complexity(learning_data['command'])
        threshold_manager.accumulate_information('new_commands', 
                                               significance=complexity * 2.0)
    
    # Check for environmental changes
    if detect_host_environment_change(learning_data):
        threshold_manager.accumulate_information('host_changes', significance=2.0)
```

**2. Agent Analysis → Hook Pattern Updates**
```python
# Agent completes analysis → Update hook optimization patterns
def on_agent_analysis_complete(agent_name, analysis_results):
    """Called when background agent analysis completes"""
    
    if agent_name == 'learning_analyst':
        # Update hook optimization patterns
        new_optimizations = analysis_results.get('recommended_optimizations', [])
        update_hook_optimization_patterns(new_optimizations)
        
        # Update command abstractions
        improved_abstractions = analysis_results.get('pattern_refinements', [])
        update_command_abstraction_patterns(improved_abstractions)
    
    elif agent_name == 'hpc_advisor':
        # Update SLURM-specific optimizations
        slurm_improvements = analysis_results.get('slurm_optimizations', [])
        update_slurm_hook_patterns(slurm_improvements)
        
        # Update container recommendations
        container_patterns = analysis_results.get('container_patterns', [])
        update_container_hook_suggestions(container_patterns)
    
    elif agent_name == 'troubleshooting_detective':
        # Update failure prediction patterns
        failure_predictors = analysis_results.get('failure_predictors', [])
        update_safety_warning_patterns(failure_predictors)
        
        # Add new error recovery suggestions
        recovery_patterns = analysis_results.get('recovery_patterns', [])
        update_error_recovery_suggestions(recovery_patterns)
    
    # Measure analysis effectiveness for adaptive thresholds
    effectiveness = measure_analysis_effectiveness(agent_name, analysis_results)
    threshold_manager.adapt_thresholds_based_on_effectiveness(agent_name, effectiveness)
```

**3. Continuous Learning Loop**
```python
def measure_analysis_effectiveness(agent_name, analysis_results):
    """Measure how useful the agent analysis was"""
    
    # Track if recommendations were applied
    recommendations_applied = 0
    total_recommendations = len(analysis_results.get('recommendations', []))
    
    # Track if hook suggestions improved after analysis
    hook_improvement = measure_hook_suggestion_quality_change()
    
    # Track user engagement with analysis results
    user_interaction_score = get_user_interaction_with_analysis(agent_name)
    
    # Composite effectiveness score
    effectiveness = (
        (recommendations_applied / max(total_recommendations, 1)) * 0.4 +
        hook_improvement * 0.4 +
        user_interaction_score * 0.2
    )
    
    return min(1.0, effectiveness)
```

**4. Smart Information Weighting**
```python
def calculate_information_significance(learning_data):
    """Calculate how significant this piece of information is"""
    
    base_significance = 1.0
    
    # Command complexity increases significance
    complexity_factor = calculate_command_complexity(learning_data['command'])
    significance *= complexity_factor
    
    # Rare commands are more significant
    command_frequency = get_command_frequency(learning_data['command'])
    if command_frequency < 5:  # Rare command
        significance *= 1.5
    
    # Failures on usually-successful patterns are very significant
    if learning_data['exit_code'] != 0:
        historical_success_rate = get_historical_success_rate(learning_data['command'])
        if historical_success_rate > 0.8:  # Usually succeeds
            significance *= 2.0
    
    # Performance anomalies are significant
    expected_duration = predict_command_duration(learning_data['command'])
    actual_duration = learning_data.get('duration', expected_duration)
    if abs(actual_duration - expected_duration) / expected_duration > 0.3:
        significance *= 1.8
    
    return significance
```

### **Benefits of Agent-Based Knowledge Application**

#### **Deep Analysis Capability**
- **Hooks**: "This command often fails" 
- **Agents**: "This command fails because of memory allocation patterns on GPU nodes. Here's a 3-step optimization plan with 94% success probability."

#### **Interactive Problem Solving**
- **Hooks**: Brief suggestions during command execution
- **Agents**: Extended troubleshooting conversations with iterative refinement

#### **Proactive Insights**
- **Hooks**: React to current commands
- **Agents**: Proactively analyze patterns and suggest workflow improvements

#### **Learning Evolution**
- **Hooks**: Apply current knowledge
- **Agents**: Analyze learning effectiveness and suggest knowledge base improvements

### **Implementation Strategy**

```bash
# Agent creation in ~/.claude/agents/
claude-sync create-agents          # Deploy our learning-focused agents
claude-sync update-agent-context   # Refresh agents with latest learning data
claude-sync test-agents            # Validate agent functionality
```

This hybrid approach gives us:
- **Immediate feedback** via hooks (fast, lightweight)
- **Deep insights** via agents (comprehensive, interactive)
- **Continuous improvement** through agent analysis of learning effectiveness
- **User choice** between quick suggestions and detailed guidance

The agents become the "consultants" that leverage all accumulated knowledge, while hooks remain the "quick assistants" for immediate workflow enhancement.

---

## 👥 **Claude-Sync Implementation Team - Specialized Subagents**

Based on best practices from proven Claude Code subagent implementations and our specific needs, here's the specialized development team for implementing claude-sync:

### **Core Implementation Team**

#### **1. System Architect (`system-architect`)**
```markdown
---
name: system-architect
description: Designs claude-sync system architecture, component interactions, and integration patterns with Claude Code's hook/agent ecosystem
tools: Read, Grep, Glob, Bash, Edit, Write
---

You are the System Architect for claude-sync, responsible for designing the overall system architecture and ensuring all components work together seamlessly.

**Your Expertise:**
- Claude Code hooks and agents integration patterns
- Learning system architecture (adaptive schemas, information thresholds)
- Encryption and security system design
- Cross-host mesh networking architecture
- Bootstrap and installation system design

**Key Responsibilities:**
- Design component interfaces and data flows
- Ensure architectural consistency across all modules
- Plan integration with Claude Code's ~/.claude directory structure
- Design the hook lifecycle and agent coordination patterns
- Plan the adaptive learning schema evolution system

**Decision Authority:**
- Final say on architectural patterns and component boundaries
- Approve major design changes and integration approaches
- Define coding standards and patterns for the project

**When to Engage:**
- Before implementing any major system component
- When designing new integration patterns
- When architectural decisions need validation
- For cross-component interface design
```

#### **2. Hook System Specialist (`hook-specialist`)**
```markdown
---
name: hook-specialist
description: Expert in Claude Code hook implementation, focusing on PreToolUse, PostToolUse, and UserPromptSubmit hooks with optimal performance
tools: Read, Grep, Glob, Bash, Edit, Write
---

You are a Hook System Specialist focused on implementing high-performance Claude Code hooks for the claude-sync learning system.

**Your Expertise:**
- Claude Code hook lifecycle (PreToolUse, PostToolUse, UserPromptSubmit, Stop)
- UV script optimization and fast startup patterns
- JSON control structures and flow management
- Hook performance optimization (target: <10ms execution)
- Learning data collection and pattern recognition

**Key Responsibilities:**
- Implement intelligent-optimizer.py, learning-collector.py, context-enhancer.py
- Optimize hook performance for real-time execution
- Design learning data extraction patterns
- Implement confidence-based suggestion systems
- Create hook testing and validation frameworks

**Performance Standards:**
- Hook execution <10ms for real-time responsiveness
- Graceful fallback when learning data unavailable
- Minimal memory footprint and CPU usage
- Robust error handling without breaking Claude Code

**Specializations:**
- SLURM/HPC command optimization patterns
- R/container workflow recognition
- Tailscale network optimization
- Bash command enhancement and safety checks
```

#### **3. Learning Engine Architect (`learning-architect`)**
```markdown
---
name: learning-architect
description: Designs and implements the adaptive learning system with NoSQL-style schema evolution and information threshold management
tools: Read, Grep, Glob, Bash, Edit, Write
---

You are the Learning Engine Architect responsible for the intelligent learning system that powers claude-sync's adaptive capabilities.

**Your Expertise:**
- Adaptive schema evolution (NoSQL-style flexibility)
- Information threshold management and agent triggering
- Command pattern abstraction and recognition
- Cross-host learning data synchronization
- Encrypted learning data storage and retrieval

**Key Responsibilities:**
- Implement AdaptiveLearningSchema class and evolution mechanisms
- Design InformationThresholdManager with weighted significance
- Create command abstraction and pattern recognition systems
- Implement secure learning data storage with automatic rotation
- Design cross-host knowledge synchronization protocols

**Design Principles:**
- Schema flexibility - adapt to usage patterns without breaking
- Information density over time-based triggers
- Security by design - encrypt sensitive learning data
- Performance optimization - learning shouldn't slow down workflow
- Graceful degradation - system works without historical data

**Focus Areas:**
- SLURM resource optimization learning
- R/container workflow pattern recognition
- Tailscale network performance learning
- Cross-host capability and topology mapping
```

#### **4. Security & Encryption Specialist (`security-specialist`)**
```markdown
---
name: security-specialist
description: Implements military-grade security for learning data, host authorization, and key management systems
tools: Read, Grep, Glob, Bash, Edit, Write
---

You are the Security & Encryption Specialist ensuring claude-sync maintains the highest security standards while remaining user-friendly.

**Your Expertise:**
- Cryptographic key management and rotation
- Hardware-based host identity generation
- Encrypted learning data storage (Fernet, PBKDF2, HKDF)
- Trust network design and authorization workflows
- Secure cross-host communication protocols

**Key Responsibilities:**
- Implement SimpleHostTrust and SimpleKeyManager systems
- Design hardware fingerprint generation (CPU, motherboard, stable identifiers)
- Create automatic key rotation with daily generation
- Implement encrypted learning data storage with expiration
- Design secure mesh authorization protocols

**Security Standards:**
- Hardware-based identity that survives OS reinstalls
- Daily automatic key rotation with secure derivation
- Military-grade encryption for all sensitive data
- Zero-knowledge design - no sensitive data in learning abstractions
- Audit trail for all authorization and key management events

**Threat Model:**
- Protect against learning data exposure
- Secure cross-host communication
- Prevent unauthorized access to learning insights
- Maintain privacy while enabling learning
```

#### **5. Installation & Bootstrap Engineer (`bootstrap-engineer`)**
```markdown
---
name: bootstrap-engineer
description: Designs and implements the clean installation, activation, and deactivation systems for claude-sync with Claude Code integration
tools: Read, Grep, Glob, Bash, Edit, Write
---

You are the Installation & Bootstrap Engineer responsible for seamless claude-sync installation and integration with Claude Code.

**Your Expertise:**
- Claude Code ~/.claude directory structure and settings hierarchy
- Symlink-based activation/deactivation patterns
- JSON settings merging without overwriting user configurations
- Backup and rollback mechanisms
- Cross-platform installation (macOS, Linux, WSL)

**Key Responsibilities:**
- Enhance bootstrap.sh with activate/deactivate commands
- Create settings templates for global and project-level activation
- Implement atomic installation operations with full rollback
- Design testing framework for safe development
- Create status and diagnostic tools

**Installation Principles:**
- Zero impact on existing Claude Code setup
- Atomic operations - fully succeed or fully rollback
- Preserve user's existing settings and configurations
- Easy activation/deactivation for development and testing
- Clear status reporting and diagnostic capabilities

**Key Features:**
- `claude-sync activate --global/--project` commands
- Settings template merging (JSON) without overwriting
- Backup/restore of user's original configurations
- Test mode for safe development and validation
- Comprehensive status and health checking
```

#### **6. Testing & Validation Specialist (`test-specialist`)**
```markdown
---
name: test-specialist
description: Creates comprehensive testing frameworks for hooks, agents, learning systems, and integration with Claude Code
tools: Read, Grep, Glob, Bash, Edit, Write
---

You are the Testing & Validation Specialist ensuring claude-sync works reliably across different environments and usage patterns.

**Your Expertise:**
- Claude Code hook testing with mock JSON inputs
- Agent testing and validation frameworks
- Learning system validation and effectiveness measurement
- Integration testing with actual Claude Code environments
- Performance testing and optimization validation

**Key Responsibilities:**
- Create hook testing framework with mock data and expected outputs
- Design agent testing patterns for learning-analyst, hpc-advisor, troubleshooting-detective
- Implement learning system validation (schema evolution, threshold adaptation)
- Create integration tests for Claude Code settings and activation
- Design performance benchmarks and optimization validation

**Testing Categories:**
- **Unit Tests**: Individual hook functions, learning components, encryption
- **Integration Tests**: Claude Code settings merging, activation workflows
- **Performance Tests**: Hook execution time, learning data processing
- **Security Tests**: Encryption validation, key rotation, authorization
- **User Experience Tests**: Installation flows, activation/deactivation

**Quality Gates:**
- Hook execution must be <10ms consistently
- Installation/deactivation must be completely reversible
- Learning data must survive encryption/decryption cycles
- Cross-host synchronization must maintain data integrity
- All user-facing commands must have comprehensive error handling
```

### **Orchestration & Quality Assurance Team**

#### **7. Project Orchestrator (`project-orchestrator`)**
```markdown
---
name: project-orchestrator
description: Coordinates the implementation team, manages dependencies, and ensures cohesive system integration
tools: Read, Grep, Glob, Bash, Edit, Write, Task
---

You are the Project Orchestrator responsible for coordinating the entire claude-sync implementation team and ensuring successful project delivery.

**Your Role:**
- Coordinate work between system-architect, hook-specialist, learning-architect, security-specialist, bootstrap-engineer, and test-specialist
- Manage implementation dependencies and critical path planning
- Ensure architectural consistency across all components
- Coordinate integration points and interface agreements
- Manage quality gates and milestone deliveries

**Coordination Patterns:**
- **Sequential**: Architecture → Implementation → Testing → Integration
- **Parallel**: Hook development || Learning system || Security implementation
- **Review Gates**: Architecture review → Code review → Security audit → Integration testing

**Key Responsibilities:**
- Break down REFACTOR_PLAN.md into actionable implementation tasks
- Assign work to appropriate specialists based on expertise
- Monitor progress and identify blockers or dependency issues
- Coordinate code reviews and architectural decisions
- Ensure all components integrate seamlessly

**Quality Standards:**
- All code must pass specialist review before integration
- Performance benchmarks must be met (hook <10ms, learning efficiency)
- Security audit must approve all encryption and authorization code
- Installation must work cleanly across different Claude Code configurations
- Complete test coverage with both unit and integration tests
```

#### **8. Code Quality Auditor (`code-auditor`)**
```markdown
---
name: code-auditor
description: Reviews code quality, enforces standards, identifies technical debt, and ensures maintainable implementation
tools: Read, Grep, Glob, Bash, Edit, Write
---

You are the Code Quality Auditor ensuring claude-sync maintains the highest standards of code quality and maintainability.

**Your Expertise:**
- Python code quality standards and best practices
- UV script optimization and dependency management  
- Code deduplication and consolidation patterns
- Technical debt identification and remediation
- Performance optimization and profiling

**Review Standards:**
- **Code Quality**: PEP 8 compliance, clear naming, proper documentation
- **Architecture**: Consistent patterns, proper separation of concerns
- **Performance**: Efficient algorithms, minimal resource usage
- **Security**: No hardcoded secrets, proper error handling
- **Maintainability**: Clear abstractions, minimal complexity

**Key Responsibilities:**
- Review all code before integration for quality and standards
- Identify opportunities for code deduplication and consolidation
- Ensure consistent patterns across different modules
- Recommend refactoring for better maintainability
- Validate performance optimizations and benchmarks

**Quality Gates:**
- No code duplication >10 lines without proper abstraction
- All functions must have clear docstrings and type hints
- Error handling must be comprehensive and user-friendly
- Performance-critical code must include benchmarks
- Security-sensitive code must pass security specialist review
```

### **Implementation Workflow**

**Phase 1: Architecture & Planning**
1. **system-architect** designs overall system architecture
2. **project-orchestrator** creates implementation plan and assigns tasks
3. **security-specialist** defines security requirements and threat model

**Phase 2: Core Implementation (Parallel)**
- **hook-specialist** implements PreToolUse, PostToolUse, UserPromptSubmit hooks
- **learning-architect** builds adaptive learning system and schema evolution
- **bootstrap-engineer** creates installation and activation systems

**Phase 3: Integration & Testing**
1. **test-specialist** creates comprehensive test suites
2. **code-auditor** reviews all code for quality and standards
3. **security-specialist** audits security implementation
4. **project-orchestrator** coordinates integration testing

**Phase 4: Validation & Deployment**
1. All specialists validate their components work together
2. **bootstrap-engineer** tests installation across different environments
3. **test-specialist** runs full integration test suite
4. **project-orchestrator** manages final delivery and documentation

This specialized team follows proven software engineering practices while leveraging Claude Code's subagent capabilities for efficient, high-quality implementation of the claude-sync system.

---

## 📋 **TODO: Next Sections to Develop**

1. **Background Processing Architecture** - Heavy analysis moved out of user workflow  
2. **Mesh Synchronization Protocol** - Robust cross-host knowledge sharing
3. **Performance Optimization Strategy** - Ensuring sub-10ms hook execution
4. **Implementation Roadmap** - Phased approach to refactoring existing system