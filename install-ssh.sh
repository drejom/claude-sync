#!/bin/bash
# Claude Code Hooks - SSH Installation Script
# For private repositories using SSH keys

set -e

echo "🚀 Installing Claude Code Hooks (SSH version)..."

# Check if SSH key exists
if [ ! -f ~/.ssh/id_rsa ] && [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "❌ No SSH key found. Please set up SSH keys with GitHub first."
    echo "   Run: ssh-keygen -t ed25519 -C \"your_email@domain.com\""
    echo "   Then add the public key to your GitHub account."
    exit 1
fi

# Test SSH connection to GitHub
echo "🔑 Testing SSH connection to GitHub..."
if ! ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "❌ SSH connection to GitHub failed."
    echo "   Make sure your SSH key is added to your GitHub account."
    exit 1
fi

# Create hooks directory
mkdir -p ~/.claude/hooks
cd ~/.claude

# Clone or update repository
if [ -d "claude-hooks" ]; then
    echo "📥 Updating existing installation..."
    cd claude-hooks
    git pull origin main
    cd ..
else
    echo "📥 Cloning claude-hooks repository..."
    git clone git@github.com:drejom/claude-hooks.git
fi

# Install hooks
echo "⚙️  Installing hooks..."
cp claude-hooks/hooks/*.py hooks/ 2>/dev/null || true
cp claude-hooks/agents/*.py hooks/ 2>/dev/null || true
cp claude-hooks/learning/*.py hooks/ 2>/dev/null || true

# Make all Python files executable
find hooks/ -name "*.py" -exec chmod +x {} \;

# Install settings template
if [ ! -f "settings.local.json" ]; then
    echo "📋 Installing settings template..."
    cp claude-hooks/templates/settings.local.json .
else
    echo "⚠️  settings.local.json already exists, not overwriting"
    echo "   New template available at: claude-hooks/templates/settings.local.json"
fi

# Add update script to PATH
SCRIPT_DIR="$HOME/.claude"
if [[ ":$PATH:" != *":$SCRIPT_DIR:"* ]]; then
    echo "📍 Adding update script to PATH..."
    echo 'export PATH="$HOME/.claude:$PATH"' >> ~/.bashrc
    cp claude-hooks/update.sh ~/.claude/update-claude-hooks
    chmod +x ~/.claude/update-claude-hooks
fi

echo ""
echo "✅ Claude Code Hooks installed successfully!"
echo ""
echo "📚 Next steps:"
echo "   1. Activate in any project: cp ~/.claude/settings.local.json ./.claude/"
echo "   2. Update hooks anytime: update-claude-hooks"
echo "   3. Check README: cat ~/.claude/claude-hooks/README.md"
echo ""
echo "🔧 Hooks installed:"
find ~/.claude/hooks -name "*.py" -exec basename {} \; | sort

echo ""
echo "🎯 Ready for AI-powered development across your infrastructure!"