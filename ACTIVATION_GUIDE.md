# Claude-Sync Activation & Management Guide

## Quick Start

The new activation system provides clean, atomic operations for managing claude-sync hooks with full rollback capabilities.

### Essential Commands

```bash
# Install claude-sync (first time)
curl -sL https://raw.githubusercontent.com/drejom/claude-sync/main/bootstrap.sh | bash -s install

# Activate globally (all Claude Code sessions)
~/.claude/claude-sync/bootstrap.sh activate --global

# Check status
~/.claude/claude-sync/bootstrap.sh status

# Deactivate cleanly
~/.claude/claude-sync/bootstrap.sh deactivate
```

## Activation System Overview

The activation system uses **symlinks** and **settings merging** to provide:

- ✅ **Zero impact** on existing Claude Code setup
- ✅ **Atomic operations** - fully succeed or fully rollback  
- ✅ **Clean deactivation** with complete removal
- ✅ **Safe development** with test mode and dry-run
- ✅ **Automatic backups** with rollback capability

## Commands Reference

### Activation Commands

#### Global Activation
```bash
bootstrap.sh activate --global [--dry-run] [--test-mode]
```

**What it does:**
- Creates symlinks in `~/.claude/hooks/` pointing to claude-sync hooks
- Backs up existing `~/.claude/settings.json`
- Merges claude-sync hooks into global settings
- Verifies all symlinks are working

**Hooks activated globally:**
- `background-sync.py` - Background maintenance
- `bash-optimizer-enhanced.py` - Command optimization
- `ssh-router-enhanced.py` - Intelligent host routing
- `resource-tracker.py` - Performance monitoring

#### Project Activation
```bash
bootstrap.sh activate --project [--dry-run] [--test-mode]
```

**What it does:**
- Creates `.claude/` directory in current project
- Creates symlinks in `./.claude/hooks/`
- Creates/merges `./.claude/settings.json`
- Only affects current project

### Deactivation Commands

#### Clean Deactivation
```bash
bootstrap.sh deactivate [--purge] [--dry-run]
```

**What it does:**
- Removes all claude-sync hook symlinks
- Restores original settings from backup
- Cleans up project hooks across all directories
- Optionally purges learning data (with `--purge`)

### Status & Diagnostics

#### System Status
```bash
bootstrap.sh status
```

**Shows:**
- Current activation state (global/project/none)
- List of active hooks with verification
- Learning data size and system metrics
- Warnings and errors

#### Health Diagnostics
```bash
bootstrap.sh diagnostics
```

**Checks:**
- Hook symlink integrity
- Settings file validity
- Learning system health
- Installation completeness
- Performance metrics

#### Test Suite
```bash
bootstrap.sh test
```

**Tests:**
- All hook files for syntax errors
- Settings templates for JSON validity
- Activation manager functionality
- Overall system readiness

### Recovery Commands

#### Rollback
```bash
bootstrap.sh rollback
```

**What it does:**
- Lists available backups with timestamps
- Restores settings from most recent backup
- Emergency recovery for corrupted configurations

## Architecture Details

### Symlink Strategy

Instead of copying files or modifying paths, claude-sync uses symlinks:

```
~/.claude/hooks/bash-optimizer-enhanced.py → ~/.claude/claude-sync/hooks/bash-optimizer-enhanced.py
~/.claude/hooks/ssh-router-enhanced.py     → ~/.claude/claude-sync/hooks/ssh-router-enhanced.py
~/.claude/hooks/resource-tracker.py       → ~/.claude/claude-sync/hooks/resource-tracker.py
```

**Benefits:**
- Hooks always run latest version
- Updates propagate immediately
- Clean uninstall removes all traces
- No file duplication

### Settings Merging

The system intelligently merges hook configurations:

**Before (user's settings):**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Git",
        "hooks": [{"type": "command", "command": "my-git-hook.py"}]
      }
    ]
  }
}
```

**After (merged settings):**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Git", 
        "hooks": [{"type": "command", "command": "my-git-hook.py"}]
      },
      {
        "matcher": "Bash",
        "hooks": [
          {"type": "command", "command": "$HOME/.claude/hooks/bash-optimizer-enhanced.py"},
          {"type": "command", "command": "$HOME/.claude/hooks/ssh-router-enhanced.py"}
        ]
      }
    ]
  }
}
```

**Key features:**
- Preserves existing user hooks
- Avoids duplicate entries
- Validates merged JSON structure
- Creates timestamped backups

### Backup System

Every activation creates timestamped backups:

```
~/.claude/claude-sync/backups/
├── settings_backup_1753863294.json    # Before global activation
├── settings_backup_1753863301.json    # Before project activation  
└── learning_data_1753863308/          # Before purge (optional)
```

## Development Workflow

### Safe Development Cycle

```bash
# 1. Test in dry-run mode
bootstrap.sh activate --global --dry-run

# 2. Run test suite
bootstrap.sh test

# 3. Activate for real
bootstrap.sh activate --global

# 4. Check everything works
bootstrap.sh status
bootstrap.sh diagnostics

# 5. Make changes and test
# ... edit hooks ...
bootstrap.sh test

# 6. Emergency rollback if needed
bootstrap.sh rollback
```

### Testing Features

#### Dry Run Mode
```bash
bootstrap.sh activate --global --dry-run
bootstrap.sh deactivate --dry-run
```

Shows exactly what would be changed without making modifications.

#### Test Mode (Future)
```bash
bootstrap.sh activate --global --test-mode
```

Uses isolated test environment for safe development.

## Integration with Claude Code

### Hook Execution Flow

```
Claude Code Command
        ↓
Settings: ~/.claude/settings.json
        ↓
Hooks: ~/.claude/hooks/bash-optimizer-enhanced.py (symlink)
        ↓
Actual Hook: ~/.claude/claude-sync/hooks/bash-optimizer-enhanced.py
        ↓
Learning Data: ~/.claude/claude-sync/learning/
```

### Settings Hierarchy

Claude Code loads settings in this order:
1. **Enterprise settings** (if applicable)
2. **CLI settings** (global)  ← Our global activation
3. **Local settings** (project) ← Our project activation
4. **Shared settings**
5. **User settings**

## Troubleshooting

### Common Issues

#### "Hooks not activating"
```bash
bootstrap.sh status          # Check activation state
bootstrap.sh diagnostics     # Check for issues
bootstrap.sh test           # Verify hook files
```

#### "Settings corrupted"
```bash
bootstrap.sh rollback       # Restore from backup
```

#### "Symlinks broken"
```bash
bootstrap.sh deactivate     # Clean removal
bootstrap.sh activate --global  # Fresh activation
```

#### "Learning data too large"
```bash
bootstrap.sh deactivate --purge  # Remove learning data
```

### Emergency Recovery

If something goes wrong:

1. **Immediate rollback:**
   ```bash
   bootstrap.sh rollback
   ```

2. **Clean slate:**
   ```bash
   bootstrap.sh deactivate
   rm -rf ~/.claude/hooks/
   ```

3. **Nuclear option:**
   ```bash
   bootstrap.sh nuclear
   ```

## Performance Targets

The activation system is designed for speed:

- **Activation time:** < 5 seconds
- **Status check:** < 1 second  
- **Hook execution overhead:** < 10ms per hook
- **Memory footprint:** < 20MB total
- **Storage footprint:** < 5MB (plus learning data)

## Security Model

### Symlink Security

- Symlinks always point to claude-sync directory
- No external or dangerous targets allowed
- Verification checks prevent symlink attacks

### Backup Security

- Backups stored within claude-sync directory
- Timestamped to prevent overwrites
- Automatic cleanup of old backups

### Learning Data Security

- All learning data encrypted at rest
- Zero-knowledge abstractions 
- User controls retention and purging

## What's Next

The activation system provides the foundation for:

- **Hot-reload:** Update hooks without reactivation
- **Selective activation:** Choose specific hooks
- **Profile management:** Different hook sets for different workflows
- **Cross-host sync:** Activate configurations across machines
- **IDE integration:** Activate from VSCode/PyCharm

---

*This activation system represents a new standard for AI development tools: powerful when you need it, invisible when you don't, and always respectful of your existing setup.*