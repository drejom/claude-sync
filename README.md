# Claude Code Portable Hooks

🚀 **Matrix-like bash and SSH superpowers for Claude Code across all your hosts**

## ⚡ One-Liner Install

```bash
curl -sL https://raw.githubusercontent.com/drejom/claude-hooks/main/install.sh | bash
```

## 🎯 What You Get

### 🔧 **Bash Optimization Superpowers**
- **Smart upgrades**: `grep` → `rg`, `find` → `fd`, `cat` → `bat`
- **Safety warnings**: Dangerous operations get friendly warnings
- **Context awareness**: Auto-adds useful flags based on environment

### 🧠 **SSH Intelligence**
- **Self-learning filesystem topology**: Learns which paths exist on which hosts
- **Intelligent routing**: Suggests the right host for your commands
- **Context switching**: Seamlessly handles local vs remote operations

### 🌍 **Portable Across Hosts**
- **Synced configuration**: Same superpowers on every machine
- **Easy updates**: `update-claude-hooks` keeps everything current
- **Project templates**: One command to activate in new projects

## 🚀 Quick Start

### 1. Install (one-time per host)
```bash
curl -sL https://raw.githubusercontent.com/drejom/claude-hooks/main/install.sh | bash
```

### 2. Activate in any project
```bash
cd your-project
cp ~/.claude/project-template.json ./.claude/settings.local.json
```

### 3. Start using Claude Code with superpowers! 🎉

## 🔄 Keeping Updated

Update hooks across all hosts:
```bash
update-claude-hooks
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
            "command": "$HOME/.claude/hooks/bash-optimizer.py"
          },
          {
            "type": "command",
            "command": "$HOME/.claude/hooks/ssh-router.py"
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
- View learning data: `python3 ~/.claude/hooks/ssh-router.py --stats`
- Reset learning: `rm ~/.claude/ssh_topology.pkl`

## 🤝 Contributing

This is a private repo for personal use. Fork and customize for your own needs!

## 📚 Related

- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [Claude Code Hooks Guide](https://docs.anthropic.com/en/docs/claude-code/hooks)

---

*🌟 Matrix-like superpowers for the command line!*