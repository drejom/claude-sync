# Claude Code Portable Hooks

AI-powered command optimization and intelligent host routing for Claude Code.

## Installation & Setup

**One-liner install + setup:**
```bash
mkdir -p ~/.claude && cd ~/.claude && git clone git@github.com:drejom/claude-hooks.git && ~/.claude/claude-hooks/update-claude.sh --update
```

**Or step by step:**
```bash
# Install
mkdir -p ~/.claude && cd ~/.claude && git clone git@github.com:drejom/claude-hooks.git

# Setup (auto-configures globally)
~/.claude/claude-hooks/update-claude.sh
```

## Dependencies

The system uses modern Python tooling:
- **uv** - Fast Python package manager (auto-installed)
- **Dependencies** - Automatically managed via `pyproject.toml`
- **Virtual environment** - Isolated Python environment per installation

All dependencies are handled automatically by the update script!

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

## Management

**Update hooks and Claude Code:**
```bash
~/.claude/claude-hooks/update-claude.sh --update
```

**Fix stuck installations:**
```bash
~/.claude/claude-hooks/update-claude.sh --nuclear
```

**Manual per-project setup** (if needed):
```bash
mkdir -p .claude && cp ~/.claude/claude-hooks/templates/settings.local.json ./.claude/
```

## 🛠️ How It Works

### Hook System
- **PreToolUse hooks**: Analyze commands before execution
- **Learning database**: Builds knowledge from your SSH usage
- **Smart suggestions**: Context-aware recommendations

### File Structure
```
~/.claude/
├── hooks-repo/          # Git repository (this repo)
├── hooks/               # Symlinks to active hooks
├── project-template.json # Template for new projects
└── ssh_topology.pkl     # Learned filesystem topology
```

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
            "command": "$HOME/.claude/claude-hooks/hooks/bash-optimizer-enhanced.py"
          },
          {
            "type": "command",
            "command": "$HOME/.claude/claude-hooks/hooks/ssh-router-enhanced.py"
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
- View all statistics: `~/.claude/claude-hooks/claude-hooks-stats`
- View SSH learning: `~/.claude/claude-hooks/claude-hooks-stats ssh`
- View bash optimization: `~/.claude/claude-hooks/claude-hooks-stats bash`
- Reset learning: `rm ~/.claude/ssh_topology_*.pkl ~/.claude/learning_*.json`

## 🤝 Contributing

This is a private repo for personal use. Fork and customize for your own needs!

## 📚 Related

- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [Claude Code Hooks Guide](https://docs.anthropic.com/en/docs/claude-code/hooks)

---

*🌟 Matrix-like superpowers for the command line!*