# Claude Code Sync - AI-Powered Development Intelligence

## Overview

Claude Code Sync is a sophisticated AI-powered hook system that provides intelligent command optimization and cross-host learning for Claude Code. The system learns from your development patterns to provide contextual suggestions, optimize commands, and enhance productivity across distributed computing environments.

## Architecture

### Core Components

#### 1. Hook System (`hooks/`)
- **bash-optimizer-enhanced.py**: Command optimization with AI learning
- **ssh-router-enhanced.py**: Intelligent host routing and topology learning  
- **resource-tracker.py**: Performance monitoring and resource optimization
- **background-sync.py**: Automatic updates and maintenance
- **process-sync.py**: Cross-process synchronization

#### 2. Learning Infrastructure (`learning/`)
- **encryption.py**: Military-grade encrypted storage for learning data
- **abstraction.py**: Security-focused data abstraction (zero-knowledge design)
- **mesh_discovery.py**: Tailscale + SSH peer discovery
- **mesh_sync.py**: Secure P2P learning synchronization


## Key Features

### 🚀 Command Optimization

**Automatic Command Upgrades**
- `grep` → `rg` (ripgrep)
- `find` → `fd`
- HPC-specific optimizations (SLURM, Singularity)
- Safety warnings for destructive operations

**AI Learning Patterns**
```python
# Example learning data structure (abstracted)
{
    'base_command': 'ssh',
    'flags': ['-J', '-o'],
    'file_types': ['.fastq', '.bam'],
    'path_types': ['data-path-a1b2', 'genomics-data-c3d4'],
    'complexity': 'medium',
    'success_patterns': ['host-type-compute-e5f6']
}
```

### 🧠 Intelligent Host Routing

**Topology Learning**
- Learns filesystem topology across hosts securely
- Routes commands to optimal hosts based on data location
- Bioinformatics workflow awareness
- Mixed Tailscale + SSH environment support

**Security Model**
- All learning data is abstracted (no sensitive paths/hostnames stored)
- Military-grade encryption with rotating keys
- Zero-knowledge design: useful to AI, useless to attackers

### 📊 Performance Intelligence

**Resource Pattern Learning**
```python
# Performance data structure
{
    'command_pattern': 'compute-intensive',
    'resource_usage': {
        'memory_peak': 'high',
        'cpu_usage': 'medium',
        'duration': 'long'
    },
    'optimal_hosts': ['gpu-host-x1y2', 'compute-host-z3w4'],
    'recommended_resources': {
        'memory': '32GB',
        'cores': 8,
        'walltime': '4:00:00'
    }
}
```


## Data Flow Architecture

### Hook Execution Flow

```
Claude Code Command
        ↓
PreToolUse Hooks → [bash-optimizer] → [ssh-router] → [resource-tracker]
        ↓                    ↓                ↓               ↓
   Optimization        Host Routing    Resource Planning  Learning
        ↓
Tool Execution (ssh, bash, etc.)
        ↓
PostToolUse Hooks → [resource-tracker] → [background-sync]
        ↓                    ↓                    ↓
   Performance        Learning Data         Updates
    Analysis           Storage
```

### Learning Data Pipeline

```
Raw Command/Data
        ↓
Security Abstraction (abstraction.py)
        ↓
Encrypted Storage (encryption.py)
        ↓
Cross-Host Sync (mesh_sync.py)
        ↓
AI Pattern Recognition
        ↓
Intelligent Suggestions
```

## Security & Privacy

### Zero-Knowledge Learning Design

**Data Abstraction Examples**
```python
# Real data (never stored)
hostname = "hpc-cluster-gpu01.university.edu"
path = "/data/genomics/patient_samples/sample_001.fastq"

# Abstracted data (what gets stored)
abstract_host = "gpu-host-a1b2"
abstract_path = "genomics-data-c3d4"
```

### Encryption Security
- **PBKDF2** key derivation with 100,000 iterations
- **Fernet** symmetric encryption (AES 128)
- **Host-specific entropy** for key generation
- **Automatic key rotation** (24-hour cycles)
- **Automatic expiration** of learning data

### Network Security
- **Tailscale mesh networking** preferred for peer communication
- **SSH fallback** with key-based authentication
- **No external API calls** - all learning is local/mesh
- **Rate-limited peer discovery** to prevent network abuse

## Installation & Configuration

### Quick Start
```bash
# One-liner install (recommended)
curl -sL https://raw.githubusercontent.com/drejom/claude-sync/main/bootstrap.sh | bash -s install

# With Claude Code installation
curl -sL https://raw.githubusercontent.com/drejom/claude-sync/main/bootstrap.sh | bash -s install --with-claude
```

### Manual Setup
```bash
# Clone repository
mkdir -p ~/.claude && cd ~/.claude
git clone https://github.com/drejom/claude-sync.git

# Run bootstrap installer
~/.claude/claude-sync/bootstrap.sh install
```

### Project Configuration
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude/claude-sync/hooks/bash-optimizer-enhanced.py"
          },
          {
            "type": "command", 
            "command": "$HOME/.claude/claude-sync/hooks/ssh-router-enhanced.py"
          }
        ]
      }
    ]
  }
}
```

## Learning System Deep Dive

### Command Pattern Recognition

The system learns from successful command patterns and builds an abstracted knowledge base:

**Pattern Categories**
- Command structure and flags
- File type associations
- Host capability matching
- Resource requirement patterns
- Workflow sequences

**Learning Triggers**
- Successful command executions
- Resource usage patterns
- Cross-host file operations
- Documentation updates
- Error pattern recognition

### Mesh Learning Network

**Peer Discovery**
```python
# Tailscale peer discovery
peers = {
    'peer_id': {
        'hostname': 'abstracted',
        'capabilities': ['gpu_computing', 'hpc_computing'],
        'connection_latency': 15.2,
        'preferred_method': 'tailscale'
    }
}
```

**Data Synchronization**
- Encrypted learning data exchange
- Capability-based routing
- Bandwidth-aware sync scheduling
- Conflict resolution for pattern disagreements

## Advanced Usage

### Statistics and Monitoring
```bash
# System status
~/.claude/claude-sync/bootstrap.sh status

# Interactive management
~/.claude/claude-sync/bootstrap.sh manage

# View learning statistics  
~/.claude/claude-sync/claude-sync-stats

# SSH learning data
~/.claude/claude-sync/claude-sync-stats ssh

# Bash optimization patterns
~/.claude/claude-sync/claude-sync-stats bash
```

### Custom Hook Development

**Hook Template**
```python
#!/usr/bin/env python3
import json
import sys

def main():
    hook_input = json.loads(sys.stdin.read())
    
    # Your logic here
    
    result = {
        'block': False,  # or True to block command
        'message': 'Your feedback message'
    }
    print(json.dumps(result))

if __name__ == '__main__':
    main()
```

### Learning Data Management

**Storage Locations**
- `~/.claude/learning/` - Encrypted learning data
- `~/.claude/ssh_topology.pkl` - Fallback topology data
- `~/.claude/settings.local.json` - Project configuration

**Data Cleanup**
```bash
# Remove learning data (regenerates automatically)
rm ~/.claude/learning/learning_*.enc
rm ~/.claude/ssh_topology_*.pkl
```

## Performance Characteristics

### Resource Usage
- **Memory**: ~10-20MB per active hook
- **CPU**: Minimal impact (~1-3% during learning)
- **Storage**: ~1-5MB encrypted learning data per host
- **Network**: Occasional background sync (KB/hour)

### Latency Impact
- **Command preprocessing**: <50ms typical
- **SSH routing decisions**: <100ms typical
- **Background learning**: Asynchronous, no blocking

## Use Cases

### High-Performance Computing
- **SLURM job optimization**: Learn optimal resource requests
- **Singularity workflow routing**: Route containers to capable hosts
- **Data locality optimization**: Move computation to data

### Bioinformatics Workflows
- **FASTQ/BAM file handling**: Intelligent host selection
- **Pipeline optimization**: Learn successful workflow patterns
- **Resource scaling**: Predict memory/time requirements

### Development Workflows
- **Cross-host development**: Seamless environment switching
- **Resource monitoring**: Performance-aware development

### Multi-Host Environments
- **Hybrid cloud/on-premise**: Unified learning across environments
- **Tailscale mesh networks**: Secure peer-to-peer learning
- **SSH infrastructure**: Fallback connectivity everywhere

## Troubleshooting

### Common Issues

**Installation issues**
```bash
# Check system status
~/.claude/claude-sync/bootstrap.sh status

# Fix stuck installations
~/.claude/claude-sync/bootstrap.sh fix-stuck

# Nuclear option (complete reinstall)
~/.claude/claude-sync/bootstrap.sh nuclear
```

**Learning data not syncing**
```bash
# Check peer connectivity
~/.claude/claude-sync/claude-sync-stats mesh

# Force sync
python3 ~/.claude/claude-sync/learning/mesh_sync.py --force
```

**Hooks not activating**
```bash
# Verify hook installation
ls -la ~/.claude/hooks/

# Check settings configuration
cat ./.claude/settings.local.json

# Update hooks
~/.claude/claude-sync/bootstrap.sh update
```

## Contributing

This system is designed for personal infrastructure learning. The architecture supports:

- **Custom hook development** in `hooks/` directory
- **Learning module extensions** in `learning/` directory  
- **Document agent customization** in `agents/` directory

### Development Guidelines

**Self-Contained Script Requirement**
All Python scripts MUST be self-contained uv scripts:
```python
#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "package>=version"
# ]
# requires-python = ">=3.10"
# ///
```

**Security First**
- Never log sensitive data (paths, hostnames, commands)
- Always use abstraction layer for learning data
- Encrypt all persistent learning storage
- Test with security review mindset

**Performance Conscious**
- Minimize hook execution time
- Use background threads for heavy operations
- Cache expensive computations
- Rate-limit network operations

**Privacy Preserving**
- Local-first learning architecture
- Opt-in mesh network participation
- User-controlled data retention
- Transparent abstraction algorithms

## Technical Specifications

### Dependencies
- **Python 3.10+** with modern async support
- **uv** - Modern Python package manager (auto-installed by bootstrap)
- **Self-contained scripts** - All Python scripts use uv inline dependency metadata

### Script Architecture
All Python scripts in the project are **self-contained uv scripts** with inline dependency specifications:

```python
#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "cryptography>=41.0.0",
#   "psutil>=5.9.0"
# ]
# requires-python = ">=3.10"
# ///
```

**Benefits:**
- **No global environment pollution** - Each script manages its own dependencies
- **Reproducible execution** - Dependencies are pinned per script
- **Simplified deployment** - No need for separate virtual environments
- **Version isolation** - Different scripts can use different dependency versions

### Compatibility
- **Claude Code**: Latest version with hooks support
- **Operating Systems**: Linux, macOS
- **Network**: Tailscale mesh (preferred), SSH (universal)
- **HPC Systems**: SLURM, PBS, SGE compatible

### File Structure
```
~/.claude/claude-sync/
├── hooks/                 # Hook implementations
├── learning/             # Learning infrastructure
├── templates/            # Configuration templates
├── bootstrap.sh          # Universal bootstrap script
├── claude-sync-stats     # Statistics viewer
└── README.md            # User documentation
```

## Future Roadmap

### Planned Enhancements
- **GPU resource optimization**: CUDA-aware host routing
- **Container orchestration**: Docker/Kubernetes integration
- **IDE integration**: VSCode/PyCharm hook support
- **Workflow automation**: Pipeline learning and suggestion
- **Security hardening**: Hardware security module support

### Research Areas
- **Federated learning**: Privacy-preserving cross-organization learning
- **Quantum-safe encryption**: Post-quantum cryptography migration
- **Edge computing**: IoT device learning integration
- **AI-driven refactoring**: Code improvement suggestions

---

*This system represents a new paradigm in AI-assisted development: learning from your patterns while respecting your privacy, optimizing your workflows while maintaining security, and enhancing your productivity while staying completely under your control.*