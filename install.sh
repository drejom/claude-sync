#!/bin/bash
# Claude Hooks - Portable Installation Script
# Usage: curl -sL https://raw.githubusercontent.com/drejom/claude-hooks/main/install.sh | bash

set -e

REPO_URL="https://github.com/drejom/claude-hooks.git"
HOOKS_REPO_DIR="$HOME/.claude/hooks-repo"
HOOKS_DIR="$HOME/.claude/hooks"

echo "🚀 Installing Claude Code Portable Hooks..."
echo "   Repo: $REPO_URL"
echo ""

# Create .claude directory if it doesn't exist
mkdir -p "$HOME/.claude"

# Clone or update the hooks repository
if [ -d "$HOOKS_REPO_DIR" ]; then
    echo "📦 Updating existing hooks repository..."
    cd "$HOOKS_REPO_DIR"
    git pull origin main
else
    echo "📥 Cloning hooks repository..."
    git clone "$REPO_URL" "$HOOKS_REPO_DIR"
fi

# Create hooks directory and symlinks
echo "🔗 Setting up hooks symlinks..."
mkdir -p "$HOOKS_DIR"

# Remove old files and create symlinks
cd "$HOOKS_REPO_DIR/hooks"
for hook_file in *.py; do
    if [ -f "$hook_file" ]; then
        target="$HOOKS_DIR/$hook_file"
        # Remove existing file/symlink
        [ -e "$target" ] && rm "$target"
        # Create symlink
        ln -s "$HOOKS_REPO_DIR/hooks/$hook_file" "$target"
        echo "  ✅ Linked $hook_file"
    fi
done

# Set up project template
echo "📝 Installing project template..."
cp "$HOOKS_REPO_DIR/templates/settings.local.json" "$HOME/.claude/project-template.json"

# Add update alias to shell profile
echo "🔄 Setting up update alias..."
SHELL_RC=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

if [ -n "$SHELL_RC" ]; then
    ALIAS_LINE='alias update-claude-hooks="cd ~/.claude/hooks-repo && git pull origin main"'
    if ! grep -q "update-claude-hooks" "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo "# Claude Code Hooks" >> "$SHELL_RC"
        echo "$ALIAS_LINE" >> "$SHELL_RC"
        echo "  ✅ Added update alias to $SHELL_RC"
    else
        echo "  ⏭️  Update alias already exists in $SHELL_RC"
    fi
fi

echo ""
echo "🎯 Installation complete!"
echo ""
echo "📋 To activate hooks in any project:"
echo "   cp ~/.claude/project-template.json ./.claude/settings.local.json"
echo ""
echo "🔄 To update hooks across all hosts:"
echo "   update-claude-hooks"
echo ""
echo "📖 Documentation: $REPO_URL"
echo ""
echo "🌟 Matrix-like bash & SSH superpowers ready!"