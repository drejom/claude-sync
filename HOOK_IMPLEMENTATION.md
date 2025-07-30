# Claude-Sync Hook Implementation

## Overview

This document describes the implementation of the three critical claude-sync hooks that provide intelligent command optimization and learning data collection with real-time performance targets.

## Implemented Hooks

### 1. intelligent-optimizer.py (PreToolUse Hook)

**Purpose:** Real-time command optimization with AI learning integration
**Performance Target:** <10ms execution time
**Actual Performance:** ~50-60ms (limited by UV startup overhead)

**Key Features:**
- **Safety-first design:** Critical safety warnings for destructive commands
- **Fast optimizations:** Precomputed pattern matching for common tools
- **Domain awareness:** SLURM, R, and container-specific suggestions
- **Confidence-based suggestions:** High-confidence recommendations highlighted
- **Never blocks execution:** Only provides helpful suggestions

**Example Output:**
```bash
echo '{"tool_name": "Bash", "tool_input": {"command": "grep -r pattern /data"}}' | ./intelligent-optimizer.py
```
```json
{
  "block": false,
  "message": "🚀 **High-confidence optimization:**\n```bash\nrg -r pattern /data\n```"
}
```

**Optimization Patterns:**
- `grep` → `rg` (ripgrep) for faster text search
- `find` → `fd` for faster file search  
- `cat` → `bat` for enhanced file viewing
- SLURM job resource allocation suggestions
- R script reproducibility flags
- Container bind mount recommendations

**Safety Patterns:**
- Filesystem destruction detection (`rm -rf /`, `mkfs`)
- Direct disk operations (`dd if=`)
- Device file writing (`> /dev/`)
- Privilege escalation warnings

### 2. learning-collector.py (PostToolUse Hook)

**Purpose:** Silent learning data collection and pattern analysis
**Performance Target:** <50ms execution time
**Actual Performance:** ~50-70ms (limited by UV startup overhead)

**Key Features:**
- **Silent operation:** Never shows messages to user
- **Structured data collection:** Command execution metrics and context
- **Failure tracking:** Monitors failure patterns for threshold analysis
- **Daily file rotation:** Efficient storage with automatic cleanup
- **Schema evolution triggers:** Adaptive learning system integration

**Learning Data Structure:**
```json
{
  "command": "sbatch --mem=32G run_analysis.sh",
  "exit_code": 0,
  "duration_ms": 2500,
  "timestamp": 1753863055.212561,
  "success": true
}
```

**Storage Locations:**
- `~/.claude/learning/commands_YYYY-MM-DD.jsonl` - Daily command logs
- `~/.claude/learning/recent_failures.txt` - Failure tracking
- Trigger files for background analysis processes

**Data Collection:**
- Command execution success/failure rates
- Performance metrics and anomaly detection
- Context abstraction for privacy-safe learning
- Cross-host synchronization preparation

### 3. context-enhancer.py (UserPromptSubmit Hook)

**Purpose:** Intelligent context injection from learning data
**Performance Target:** <20ms execution time
**Actual Performance:** ~55-75ms (limited by UV startup overhead)

**Key Features:**
- **Multi-domain detection:** SLURM, containers, R, networking, performance
- **Contextual responses:** Precomputed advice for common scenarios
- **Learning integration:** Future support for historical pattern injection
- **Responsive design:** Fast keyword-based pattern matching
- **Never blocks prompts:** Only enhances with relevant context

**Context Detection Keywords:**
- **SLURM:** sbatch, slurm, queue, partition, hpc
- **Containers:** singularity, container, docker, .sif
- **R:** rscript, r analysis, rstudio
- **Networking:** ssh, remote, tailscale, connection
- **Performance:** slow, performance, optimize, faster
- **Errors:** error, failed, debug, troubleshoot

**Example Output:**
```bash
echo '{"user_prompt": "How do I optimize my SLURM jobs?"}' | ./context-enhancer.py
```
```json
{
  "block": false,
  "message": "🧠 **Added context from learning data:**\n\n**SLURM optimization tips:**\n- Add resource allocation: `--mem=16G --cpus-per-task=4`\n- Set time limit: `--time=4:00:00`\n- Use appropriate partition: `--partition=compute` or `--partition=gpu`"
}
```

## Performance Analysis

### Performance Targets vs. Actual

| Hook | Target | Actual Avg | P95 | Status |
|------|--------|------------|-----|---------|
| intelligent-optimizer.py | <10ms | ~50ms | ~110ms | ⚠️ UV overhead |
| learning-collector.py | <50ms | ~55ms | ~170ms | 📊 Near target |
| context-enhancer.py | <20ms | ~55ms | ~135ms | ⚠️ UV overhead |

### Performance Bottlenecks

**Primary Bottleneck: UV Script Startup**
- UV adds ~40-50ms startup overhead for dependency resolution
- Native Python execution: ~20-30ms
- Trade-off: UV provides dependency isolation as required by architecture

**Optimization Strategies Applied:**
1. **Minimal imports:** Reduced import overhead to essential modules only
2. **Precomputed patterns:** Fast lookup tables instead of complex logic
3. **Early exit conditions:** Fast paths for non-applicable tools
4. **Limited functionality:** Focused on high-impact suggestions only
5. **Error resilience:** Silent failures prevent cascading delays

### Performance Comparison

```bash
# UV Script (with dependency isolation)
time ./intelligent-optimizer.py < input.json  # ~50ms

# Native Python (without isolation)  
time ./intelligent-optimizer-native.py < input.json  # ~20ms
```

## Architecture Compliance

### Interface Adherence

✅ **HookInterface Compliance**
- All hooks implement proper JSON input/output
- Return structured HookResult format
- Handle errors gracefully without breaking Claude Code

✅ **Hook-Specific Interfaces**
- **PreToolUseHookInterface:** Provides analysis and suggestions
- **PostToolUseHookInterface:** Extracts learning data and updates patterns  
- **UserPromptSubmitHookInterface:** Detects context needs and enhances prompts

✅ **Performance Monitoring**
- Built-in execution time tracking
- Performance warnings for threshold violations
- Graceful degradation under load

### Security & Privacy

✅ **Data Abstraction**
- Commands truncated to prevent sensitive data logging
- File paths abstracted in learning patterns
- Context hashed for privacy-safe analysis

✅ **Error Handling**
- Silent failures prevent information disclosure
- No sensitive data in error messages
- Robust exception handling throughout

## Integration Points

### Learning System Integration

**Data Flow:**
1. **intelligent-optimizer.py** → Loads cached learning patterns for suggestions
2. **learning-collector.py** → Stores execution data in structured format
3. **context-enhancer.py** → Reads learning data for context enhancement

**Storage Schema:**
```bash
~/.claude/learning/
├── commands_YYYY-MM-DD.jsonl    # Daily execution logs
├── recent_failures.txt          # Failure tracking  
├── context_cache.json          # Cached context data
└── *.flag                      # Analysis trigger files
```

### Claude Code Integration

**Hook Registration:**
```json
{
  "hooks": {
    "PreToolUse": [
      {"matcher": "Bash", "hooks": [{"type": "command", "command": "~/.claude/claude-sync/hooks/intelligent-optimizer.py"}]}
    ],
    "PostToolUse": [
      {"matcher": "Bash", "hooks": [{"type": "command", "command": "~/.claude/claude-sync/hooks/learning-collector.py"}]}
    ],
    "UserPromptSubmit": [
      {"hooks": [{"type": "command", "command": "~/.claude/claude-sync/hooks/context-enhancer.py"}]}
    ]
  }
}
```

## Testing & Validation

### Performance Testing
```bash
./test_performance.py  # Comprehensive performance benchmarks
```

### Interface Validation
```bash
./validate_interfaces.py  # Interface compliance testing
```

### Functional Testing
```bash
# Test optimization suggestions
echo '{"tool_name": "Bash", "tool_input": {"command": "grep pattern file"}}' | ./intelligent-optimizer.py

# Test learning data collection
echo '{"tool_name": "Bash", "tool_input": {"command": "test"}, "tool_output": {"exit_code": 0, "duration_ms": 100}}' | ./learning-collector.py

# Test context enhancement
echo '{"user_prompt": "How do I debug SLURM issues?"}' | ./context-enhancer.py
```

## Future Enhancements

### Performance Optimizations
1. **Pre-compiled bytecode:** Reduce Python startup overhead
2. **Shared memory cache:** Cross-invocation pattern sharing
3. **Background processes:** Move heavy operations to background threads
4. **Native extensions:** C extensions for critical path operations

### Learning System Enhancements
1. **Adaptive schema evolution:** Dynamic pattern recognition
2. **Cross-host synchronization:** Mesh learning network
3. **Agent integration:** Threshold-based expert system triggering
4. **Performance prediction:** Historical execution time modeling

### Advanced Features
1. **Context-aware suggestions:** More sophisticated pattern matching
2. **User preference learning:** Personalized optimization strategies
3. **Workflow automation:** Multi-step command sequence optimization
4. **Resource prediction:** Intelligent SLURM resource allocation

## Troubleshooting

### Common Issues

**Hook not executing:**
```bash
# Check hook exists and is executable
ls -la ~/.claude/claude-sync/hooks/
chmod +x ~/.claude/claude-sync/hooks/*.py
```

**Performance degradation:**
```bash
# Check for disk space issues
df -h ~/.claude/learning/

# Rotate old learning data
find ~/.claude/learning/ -name "commands_*.jsonl" -mtime +30 -delete
```

**Learning data corruption:**
```bash
# Reset learning data
rm ~/.claude/learning/*.jsonl ~/.claude/learning/*.json
```

### Debug Mode
```bash
# Enable debug output
export CLAUDE_SYNC_DEBUG=1
./intelligent-optimizer.py < test_input.json
```

## Conclusion

The claude-sync hook implementation provides a robust foundation for intelligent command optimization and learning data collection. While UV startup overhead prevents achieving the most aggressive performance targets, the hooks deliver significant value through:

- **Real-time safety warnings** preventing dangerous operations
- **Intelligent optimizations** improving command performance
- **Silent learning** building knowledge for future improvements
- **Contextual assistance** enhancing user productivity

The architecture is designed for extensibility, allowing future enhancements while maintaining backward compatibility and performance characteristics suitable for real-time Claude Code integration.