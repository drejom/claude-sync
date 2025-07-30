# Claude-Sync: AI-Powered Development Intelligence

Claude-Sync learns from your command patterns to provide intelligent optimization suggestions, safety warnings, and contextual assistance across distributed computing environments. The system learns successful patterns across hosts while maintaining military-grade privacy through encrypted, abstracted learning data.

## Quick Start

```bash
# Install and activate
curl -sL https://raw.githubusercontent.com/drejom/claude-sync/main/bootstrap.sh | bash -s install
~/.claude/claude-sync/bootstrap.sh activate --global

# Verify installation
~/.claude/claude-sync/bootstrap.sh status
```

## What You Get

**Intelligent Command Optimization:**
- `grep` → `rg` suggestions with confidence scoring
- SLURM resource optimization based on historical success
- Container workflow recommendations
- Safety warnings for dangerous operations

**Learning Intelligence:**
- Cross-host pattern sharing through encrypted P2P mesh
- Workflow sequence recognition and optimization
- Adaptive schema evolution based on your usage patterns
- Context-aware assistance for bioinformatics, HPC, and R workflows

**Zero Configuration Security:**
- Hardware-based host identity (survives OS reinstalls)
- Daily automatic key rotation with cleanup
- Military-grade encryption (Fernet + PBKDF2)
- Zero repository contamination guarantee

## Performance

- Hook execution: <10ms (real-time suggestions)
- Learning operations: <1ms overhead
- Memory usage: <50MB total
- Cross-platform: macOS, Linux, WSL

## Core Capabilities

### Command Optimization
- Automatic command upgrades: `grep` → `rg`, `find` → `fd`
- AI learns your successful command patterns and suggests improvements
- HPC-specific optimizations for SLURM, Singularity, genomics tools
- Safety warnings for destructive operations

### Intelligent Host Routing  
- Learns filesystem topology across your hosts securely
- Routes commands to optimal hosts based on data location and capabilities
- Bioinformatics workflow awareness (FASTQ, BAM, VCF file handling)
- Supports mixed Tailscale + SSH environments

### Performance Learning
- Monitors resource usage patterns (memory, CPU, execution time)
- Suggests optimal HPC resources based on workload history
- Cross-host performance data sharing with encrypted sync

### Documentation Intelligence
- Comprehensive documentation review system with specialized agents
- Ruthless editor that detects AI/inexperienced author patterns
- Context-aware analysis for R packages, READMEs, vignettes
- Example validation with syntax and runtime testing
- Audience optimization for user/developer/scientist content

### Background Maintenance
- Auto-updates hooks when improvements are available
- Documentation review before git pushes
- Secure cross-host learning data synchronization

## Commands

```bash
~/.claude/claude-sync/bootstrap.sh status         # Show activation status
~/.claude/claude-sync/bootstrap.sh test          # Run system tests
~/.claude/claude-sync/bootstrap.sh deactivate    # Clean removal
~/.claude/claude-sync/bootstrap.sh rollback      # Emergency restore
```

## Troubleshooting

**Installation issues:**
```bash
~/.claude/claude-sync/bootstrap.sh diagnostics --health-score
```

**Performance problems:**
```bash
~/.claude/claude-sync/bootstrap.sh test --performance
```

**Complete reset:**
```bash
~/.claude/claude-sync/bootstrap.sh deactivate --purge
```

## Architecture

Claude-Sync uses Claude Code hooks for real-time optimization and specialized agents for deep analysis:

- **PreToolUse**: Command optimization and safety warnings
- **PostToolUse**: Silent learning data collection
- **UserPromptSubmit**: Context injection from learned patterns

Learning data stays encrypted locally (`~/.claude/learning/*.enc`) and never touches the repository.

## 🎨 Examples

### Bash Optimization
```bash
# You type:
grep "function" *.R

# Hook suggests:
🚀 Optimized command: rg "function" *.R
```

### SSH Intelligence
```bash
# You type:
ls /data/projects/

# Hook suggests (after learning):
🧠 Learned pattern detected
Similar commands were previously run on:
  • ssh gemini-data1 'ls /data/projects/' (85% match)
```

## 🔧 Configuration

### Project Settings
Each project can customize which hooks to enable in `.claude/settings.local.json`:

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

### Global Settings
Hooks are installed globally but can be disabled per-project by not including them in the project's settings.

## 🧠 Learning System

The SSH router learns your infrastructure:
- **Filesystem topology**: Which paths exist on which hosts
- **Command patterns**: Which commands you run where
- **Usage frequency**: Weights suggestions by actual usage

The more you use SSH, the smarter the suggestions become!

## 🔒 Privacy & Security

- **Local learning**: All learning data stays on your machines
- **No external calls**: Hooks run entirely locally
- **Private repo**: Your configuration and usage patterns remain private

## 🛡️ Safety Features

- **Destructive operation warnings**: `rm -rf`, `chmod 777`, etc.
- **Remote sudo detection**: Alerts for elevated remote operations
- **Path validation**: Catches common path mistakes

## 🎯 Advanced Usage

### Custom Hook Development
Add your own hooks to the `hooks/` directory and they'll be synced across hosts.

### SSH Config Integration
The router automatically reads your `~/.ssh/config` for available hosts.

### Learning Data Management
- Check system status: `~/.claude/claude-sync/bootstrap.sh status`
- View all statistics: `~/.claude/claude-sync/claude-sync-stats`
- View SSH learning: `~/.claude/claude-sync/claude-sync-stats ssh`
- View bash optimization: `~/.claude/claude-sync/claude-sync-stats bash`
- Reset learning: `rm ~/.claude/ssh_topology_*.pkl ~/.claude/learning_*.json`

## 🤝 Contributing

This is a private repo for personal use. Fork and customize for your own needs!

## 📚 Related

- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [Claude Code Hooks Guide](https://docs.anthropic.com/en/docs/claude-code/hooks)

---

*🌟 Matrix-like superpowers for the command line!*