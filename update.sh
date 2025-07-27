#!/bin/bash
# Claude Hooks - Update Script
# Updates hooks on existing installations

set -e

HOOKS_REPO_DIR="$HOME/.claude/hooks-repo"
HOOKS_DIR="$HOME/.claude/hooks"

echo "🔄 Updating Claude Code Hooks..."

# Check if hooks repo exists
if [ ! -d "$HOOKS_REPO_DIR" ]; then
    echo "❌ Hooks repository not found at $HOOKS_REPO_DIR"
    echo "   Run the install script first: curl -sL https://raw.githubusercontent.com/drejom/claude-hooks/main/install.sh | bash"
    exit 1
fi

# Pull latest changes
echo "📥 Pulling latest changes..."
cd "$HOOKS_REPO_DIR"
OLD_COMMIT=$(git rev-parse HEAD)
git pull origin main
NEW_COMMIT=$(git rev-parse HEAD)

# Check if there were updates
if [ "$OLD_COMMIT" = "$NEW_COMMIT" ]; then
    echo "✅ Already up to date!"
    exit 0
fi

# Show what changed
echo ""
echo "📋 Changes in this update:"
git log --oneline "$OLD_COMMIT..$NEW_COMMIT"
echo ""

# Update symlinks in case new hooks were added
echo "🔗 Updating hook symlinks..."
mkdir -p "$HOOKS_DIR"

cd "$HOOKS_REPO_DIR/hooks"
for hook_file in *.py; do
    if [ -f "$hook_file" ]; then
        target="$HOOKS_DIR/$hook_file"
        if [ ! -L "$target" ] || [ ! -e "$target" ]; then
            # Remove and recreate symlink if it's broken or doesn't exist
            [ -e "$target" ] && rm "$target"
            ln -s "$HOOKS_REPO_DIR/hooks/$hook_file" "$target"
            echo "  ✅ Updated $hook_file"
        fi
    fi
done

# Update project template
if [ -f "$HOOKS_REPO_DIR/templates/settings.local.json" ]; then
    cp "$HOOKS_REPO_DIR/templates/settings.local.json" "$HOME/.claude/project-template.json"
    echo "📝 Updated project template"
fi

echo ""
echo "🎉 Update complete!"
echo "🚀 Your hooks are now running the latest version"