#!/bin/bash
# Robust Claude Code + Hooks Update Script
# Handles installations, updates, and stuck situations

set -e

CLAUDE_HOOKS_DIR="$HOME/.claude/claude-hooks"
CLAUDE_BIN="$(which claude 2>/dev/null || echo "")"

show_help() {
    cat << EOF
Claude Code + Hooks Update Manager

Usage:
  $0 [OPTION]

Options:
  --help              Show this help
  --status            Show current installation status
  --update            Update hooks and check Claude Code
  --install-claude    Install/reinstall Claude Code
  --nuclear           Nuclear option: completely reinstall everything
  --fix-stuck         Fix stuck Claude Code updates
  
Examples:
  $0                  # Interactive mode
  $0 --update         # Quick update
  $0 --nuclear        # Start fresh when things are broken
EOF
}

check_status() {
    echo "🔍 Checking current status..."
    
    # Check Claude Code
    if command -v claude >/dev/null 2>&1; then
        CLAUDE_VERSION=$(claude --version 2>/dev/null || echo "unknown")
        echo "✅ Claude Code: $CLAUDE_VERSION"
    else
        echo "❌ Claude Code: Not installed"
    fi
    
    # Check hooks
    if [ -d "$CLAUDE_HOOKS_DIR" ]; then
        cd "$CLAUDE_HOOKS_DIR"
        HOOKS_STATUS=$(git status --porcelain 2>/dev/null || echo "dirty")
        HOOKS_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        echo "✅ Hooks: $HOOKS_COMMIT $([ -z "$HOOKS_STATUS" ] && echo "(clean)" || echo "(dirty)")"
        echo "   Location: $CLAUDE_HOOKS_DIR"
        echo "   Files: $(find hooks/ agents/ learning/ -name "*.py" 2>/dev/null | wc -l) Python files"
    else
        echo "❌ Hooks: Not installed"
    fi
    
    # Check active projects
    ACTIVE_PROJECTS=$(find $HOME -name ".claude" -type d 2>/dev/null | wc -l)
    echo "📁 Active projects: $ACTIVE_PROJECTS with .claude directories"
}

update_hooks() {
    echo "📥 Updating Claude hooks..."
    
    if [ ! -d "$CLAUDE_HOOKS_DIR" ]; then
        echo "Installing hooks for first time..."
        mkdir -p ~/.claude
        cd ~/.claude
        git clone git@github.com:drejom/claude-hooks.git
    else
        echo "Updating existing hooks..."
        cd "$CLAUDE_HOOKS_DIR"
        
        # Handle dirty working directory
        if [ -n "$(git status --porcelain)" ]; then
            echo "⚠️  Working directory is dirty. Stashing changes..."
            git stash push -m "Auto-stash before update $(date)"
        fi
        
        # Pull latest changes
        git pull origin main
        echo "✅ Hooks updated successfully"
    fi
}

install_claude() {
    echo "🚀 Installing/updating Claude Code..."
    
    if command -v curl >/dev/null 2>&1; then
        echo "📥 Downloading Claude Code installer..."
        # Try user-space install first, fallback to system install
        if curl -sL https://claude.ai/install.sh | bash; then
            echo "✅ Claude Code installed in user space"
        elif curl -sL https://claude.ai/install.sh | sudo bash; then
            echo "✅ Claude Code installed system-wide"
        else
            echo "❌ Installation failed. Trying manual approaches..."
            echo ""
            echo "You can try:"
            echo "1. Install manually: https://docs.anthropic.com/en/docs/claude-code"
            echo "2. Check if you have access to install packages"
            echo "3. Contact your system administrator"
            exit 1
        fi
    else
        echo "❌ curl not found. Please install curl first:"
        echo "   # Ubuntu/Debian: sudo apt install curl"
        echo "   # CentOS/RHEL:   sudo yum install curl"
        echo "   # macOS:         brew install curl"
        exit 1
    fi
}

fix_stuck_update() {
    echo "🔧 Fixing stuck Claude Code update..."
    
    # Kill any running Claude processes
    echo "Stopping Claude processes..."
    pkill -f claude 2>/dev/null || true
    
    # Clear any locks
    sudo rm -f /usr/local/bin/.claude-lock 2>/dev/null || true
    sudo rm -f /tmp/claude-* 2>/dev/null || true
    
    # Clear cache
    rm -rf ~/.cache/claude 2>/dev/null || true
    rm -rf ~/.config/claude/cache 2>/dev/null || true
    
    echo "🧹 Cleared processes and cache"
    
    # Reinstall
    install_claude
}

nuclear_option() {
    echo "☢️  NUCLEAR OPTION: Complete reinstallation"
    echo ""
    echo "This will:"
    echo "  • Remove Claude Code completely (user & system)"
    echo "  • Remove all hooks"
    echo "  • Reinstall everything from scratch"
    echo "  • Preserve your project .claude/settings files"
    echo ""
    read -p "Are you absolutely sure? Type 'NUCLEAR' to confirm: " confirm
    
    if [ "$confirm" != "NUCLEAR" ]; then
        echo "Cancelled. Wise choice."
        exit 0
    fi
    
    echo "💣 Starting nuclear reinstall..."
    
    # Remove Claude Code from common locations
    if [ -n "$CLAUDE_BIN" ]; then
        rm -f "$CLAUDE_BIN" 2>/dev/null || sudo rm -f "$CLAUDE_BIN" 2>/dev/null || true
    fi
    
    # Remove from common installation paths
    rm -rf ~/.local/bin/claude 2>/dev/null || true
    sudo rm -f /usr/local/bin/claude 2>/dev/null || true
    sudo rm -rf /usr/local/lib/claude 2>/dev/null || true
    
    # Remove config and cache
    rm -rf ~/.config/claude 2>/dev/null || true
    rm -rf ~/.cache/claude 2>/dev/null || true
    rm -rf ~/.local/share/claude 2>/dev/null || true
    
    # Remove hooks
    rm -rf "$CLAUDE_HOOKS_DIR" 2>/dev/null || true
    
    # Clean processes
    pkill -f claude 2>/dev/null || true
    
    echo "🧹 Everything nuked. Reinstalling..."
    
    # Reinstall everything
    install_claude
    update_hooks
    
    echo "🎯 Nuclear reinstall complete!"
}

interactive_mode() {
    echo "🎛️  Claude Code + Hooks Manager"
    echo ""
    check_status
    echo ""
    echo "What would you like to do?"
    echo "1) Update hooks and Claude Code"
    echo "2) Install/reinstall Claude Code only"
    echo "3) Fix stuck Claude Code update"
    echo "4) Nuclear option (reinstall everything)"
    echo "5) Just check status again"
    echo "6) Exit"
    echo ""
    read -p "Choose (1-6): " choice
    
    case $choice in
        1)
            update_hooks
            install_claude
            ;;
        2)
            install_claude
            ;;
        3)
            fix_stuck_update
            ;;
        4)
            nuclear_option
            ;;
        5)
            check_status
            ;;
        6)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice"
            exit 1
            ;;
    esac
}

# Main script logic
case "${1:-}" in
    --help|-h)
        show_help
        ;;
    --status)
        check_status
        ;;
    --update)
        update_hooks
        install_claude
        ;;
    --install-claude)
        install_claude
        ;;
    --nuclear)
        nuclear_option
        ;;
    --fix-stuck)
        fix_stuck_update
        ;;
    "")
        interactive_mode
        ;;
    *)
        echo "❌ Unknown option: $1"
        show_help
        exit 1
        ;;
esac

echo ""
echo "✅ All done! Current status:"
check_status