#!/bin/bash
# Activate claude-sync implementation team agents

set -e

AGENTS_SOURCE="$HOME/.claude/claude-sync/agents"
AGENTS_TARGET="$HOME/.claude/agents"

echo "🤖 Activating claude-sync implementation team agents..."

# Create target directory
mkdir -p "$AGENTS_TARGET"

# Copy agents to active location
cp "$AGENTS_SOURCE"/*.md "$AGENTS_TARGET/"

echo "✅ Activated 8 implementation team agents:"
echo "   - system-architect"
echo "   - hook-specialist" 
echo "   - learning-architect"
echo "   - security-specialist"
echo "   - bootstrap-engineer"
echo "   - test-specialist"
echo "   - project-orchestrator"
echo "   - code-auditor"
echo ""
echo "🚀 Ready to implement claude-sync! Use /agents to see available agents."