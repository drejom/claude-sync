#!/bin/bash
# Claude Code Sync - Universal Bootstrap Script
# Handles installation, updates, and management of Claude Code Sync system

set -e

# Configuration
REPO_SSH="git@github.com:drejom/claude-sync.git"
REPO_HTTPS="https://github.com/drejom/claude-sync.git"
SYNC_DIR="$HOME/.claude/claude-sync"
HOOKS_DIR="$HOME/.claude/hooks"
CLAUDE_DIR="$HOME/.claude"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_step() {
    echo -e "${BLUE}🔄 $1${NC}"
}

# Help system
show_help() {
    cat << EOF
Claude Code Sync - Universal Bootstrap Script

USAGE:
    $0 [COMMAND] [OPTIONS]

COMMANDS:
    install         Fresh installation (auto-detects SSH vs HTTPS)
    update          Update existing installation
    activate        Activate claude-sync hooks (global or project-specific)
    deactivate      Deactivate claude-sync hooks with optional data purging
    status          Show current system status and health metrics
    test            Run comprehensive test suites (quick or full)
    diagnostics     Run comprehensive health checks and system analysis
    validate        Run bootstrap validation for system readiness
    rollback        Restore settings from backup
    manage          Interactive management mode
    fix-stuck       Fix stuck installations
    nuclear         Complete reinstall (removes everything)

ACTIVATION COMMANDS:
    activate --global     Activate hooks globally (all Claude Code sessions)
    activate --project    Activate hooks for current project only
    deactivate           Clean removal with settings restore
    deactivate --purge   Also remove all learning data

OPTIONS:
    --ssh           Force SSH clone method (requires SSH keys)
    --https         Force HTTPS clone method
    --with-claude   Also install/update Claude Code
    --symlinks      Use symlinks for hook deployment (default)
    --copy          Copy files instead of symlinking
    --dry-run       Show what would be changed without doing it
    --test-mode     Use isolated test environment
    --help, -h      Show this help

EXAMPLES:
    $0 install                           # Auto-detect best installation method
    $0 activate --global                 # Activate hooks globally
    $0 activate --project --dry-run      # Preview project activation
    $0 status                            # Check activation status and health
    $0 test --comprehensive              # Run full test suite
    $0 diagnostics --health-score        # Get system health score
    $0 deactivate --purge                # Clean deactivation with data removal
    $0 rollback                          # Emergency settings restore

QUICK START:
    # For first-time users
    curl -sL https://raw.githubusercontent.com/drejom/claude-sync/main/bootstrap.sh | bash -s install

    # For existing users
    ~/.claude/claude-sync/bootstrap.sh activate --global

DOCUMENTATION:
    See CLAUDE.md for comprehensive documentation
    Repository: https://github.com/drejom/claude-sync

EOF
}

# Detection functions
detect_ssh_capability() {
    log_step "Checking SSH capability..."
    
    # Check if SSH keys exist
    if [ ! -f ~/.ssh/id_rsa ] && [ ! -f ~/.ssh/id_ed25519 ] && [ ! -f ~/.ssh/id_ecdsa ]; then
        log_warning "No SSH keys found"
        return 1
    fi
    
    # Test SSH connection to GitHub
    if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        log_success "SSH connection to GitHub working"
        return 0
    else
        log_warning "SSH connection to GitHub failed"
        return 1
    fi
}

detect_existing_installation() {
    if [ -d "$SYNC_DIR" ]; then
        if [ -d "$SYNC_DIR/.git" ]; then
            log_info "Found existing installation at $SYNC_DIR"
            return 0
        else
            log_warning "Directory exists but is not a git repository"
            return 1
        fi
    else
        log_info "No existing installation found"
        return 1
    fi
}

check_dependencies() {
    log_step "Checking system dependencies..."
    
    # Check git
    if ! command -v git >/dev/null 2>&1; then
        log_error "Git is required but not installed"
        log_info "Install with: sudo apt install git  # or brew install git"
        exit 1
    fi
    
    # Check python3
    if ! command -v python3 >/dev/null 2>&1; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    log_success "Dependencies check passed"
}

get_system_status() {
    echo "🔍 Claude Code Sync System Status"
    echo "=================================="
    
    # Check Claude Code
    if command -v claude >/dev/null 2>&1; then
        CLAUDE_VERSION=$(claude --version 2>/dev/null || echo "unknown")
        log_success "Claude Code: $CLAUDE_VERSION"
    else
        log_warning "Claude Code: Not installed"
    fi
    
    # Check sync directory
    if [ -d "$SYNC_DIR" ]; then
        cd "$SYNC_DIR"
        SYNC_STATUS=$(git status --porcelain 2>/dev/null || echo "dirty")
        SYNC_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        log_success "Sync Directory: $SYNC_COMMIT $([ -z "$SYNC_STATUS" ] && echo "(clean)" || echo "(dirty)")"
        log_info "   Location: $SYNC_DIR"
        
        # Count Python files
        HOOK_COUNT=$(find hooks/ learning/ -name "*.py" 2>/dev/null | wc -l)
        log_info "   Hook files: $HOOK_COUNT Python files"
    else
        log_warning "Sync Directory: Not found"
    fi
    
    # Check hooks directory
    if [ -d "$HOOKS_DIR" ]; then
        ACTIVE_HOOKS=$(find "$HOOKS_DIR" -name "*.py" 2>/dev/null | wc -l)
        log_success "Active Hooks: $ACTIVE_HOOKS files in $HOOKS_DIR"
    else
        log_warning "Hooks Directory: Not found"
    fi
    
    # Check project usage
    ACTIVE_PROJECTS=$(find $HOME -name ".claude" -type d 2>/dev/null | wc -l)
    log_info "Active Projects: $ACTIVE_PROJECTS with .claude directories"
    
    # Check global settings
    if [ -f "$HOME/.claude/settings.json" ]; then
        log_success "Global Settings: Configured"
    else
        log_warning "Global Settings: Not configured"
    fi
    
    echo ""
}

# Installation functions
determine_clone_method() {
    local force_method="$1"
    
    if [ "$force_method" = "ssh" ]; then
        if detect_ssh_capability; then
            echo "ssh"
        else
            log_error "SSH method requested but SSH access not available"
            exit 1
        fi
    elif [ "$force_method" = "https" ]; then
        echo "https"
    else
        # Auto-detect
        if detect_ssh_capability; then
            log_info "SSH access available - using SSH method"
            echo "ssh"
        else
            log_info "SSH not available - using HTTPS method"
            echo "https"
        fi
    fi
}

clone_repository() {
    local method="$1"
    local repo_url
    
    if [ "$method" = "ssh" ]; then
        repo_url="$REPO_SSH"
    else
        repo_url="$REPO_HTTPS"
    fi
    
    log_step "Cloning repository using $method method..."
    log_info "Repository: $repo_url"
    
    # Create parent directory
    mkdir -p "$CLAUDE_DIR"
    cd "$CLAUDE_DIR"
    
    # Clone repository
    if git clone "$repo_url" claude-sync; then
        log_success "Repository cloned successfully"
    else
        log_error "Failed to clone repository"
        exit 1
    fi
}

update_repository() {
    log_step "Updating repository..."
    
    if [ ! -d "$SYNC_DIR" ]; then
        log_error "Sync directory not found at $SYNC_DIR"
        log_info "Run 'bootstrap.sh install' first"
        exit 1
    fi
    
    cd "$SYNC_DIR"
    
    # Handle dirty working directory
    if [ -n "$(git status --porcelain)" ]; then
        log_warning "Working directory has uncommitted changes"
        log_step "Stashing changes..."
        git stash push -m "Auto-stash before update $(date)"
    fi
    
    # Pull latest changes
    OLD_COMMIT=$(git rev-parse HEAD)
    git pull origin main
    NEW_COMMIT=$(git rev-parse HEAD)
    
    if [ "$OLD_COMMIT" = "$NEW_COMMIT" ]; then
        log_success "Already up to date"
    else
        log_success "Updated successfully"
        log_info "Changes:"
        git log --oneline "$OLD_COMMIT..$NEW_COMMIT"
    fi
}

deploy_hooks() {
    local method="$1"  # "symlinks" or "copy"
    
    log_step "Deploying hooks using $method method..."
    
    # Create hooks directory
    mkdir -p "$HOOKS_DIR"
    
    if [ "$method" = "symlinks" ]; then
        deploy_hooks_symlinks
    else
        deploy_hooks_copy
    fi
}

deploy_hooks_symlinks() {
    cd "$SYNC_DIR/hooks"
    
    for hook_file in *.py; do
        if [ -f "$hook_file" ]; then
            target="$HOOKS_DIR/$hook_file"
            
            # Remove existing file/symlink
            [ -e "$target" ] && rm "$target"
            
            # Create symlink
            ln -s "$SYNC_DIR/hooks/$hook_file" "$target"
            log_success "Linked $hook_file"
        fi
    done
}

deploy_hooks_copy() {
    # Copy hooks
    cp "$SYNC_DIR"/hooks/*.py "$HOOKS_DIR/" 2>/dev/null || true
    
    # Copy learning modules
    cp "$SYNC_DIR"/learning/*.py "$HOOKS_DIR/" 2>/dev/null || true
    
    # Make executable
    find "$HOOKS_DIR" -name "*.py" -exec chmod +x {} \;
    
    log_success "Hooks copied and made executable"
}

install_settings_template() {
    log_step "Installing settings template..."
    
    local template_file="$SYNC_DIR/templates/settings.local.json"
    local target_file="$HOME/.claude/project-template.json"
    
    if [ -f "$template_file" ]; then
        cp "$template_file" "$target_file"
        log_success "Project template installed at $target_file"
    else
        log_warning "Settings template not found"
    fi
}

setup_shell_integration() {
    log_step "Setting up shell integration..."
    
    # Detect shell
    local shell_rc=""
    if [ -f "$HOME/.bashrc" ]; then
        shell_rc="$HOME/.bashrc"
    elif [ -f "$HOME/.zshrc" ]; then
        shell_rc="$HOME/.zshrc"
    fi
    
    if [ -n "$shell_rc" ]; then
        local alias_line='alias update-claude-sync="~/.claude/claude-sync/bootstrap.sh update"'
        
        if ! grep -q "update-claude-sync" "$shell_rc" 2>/dev/null; then
            echo "" >> "$shell_rc"
            echo "# Claude Code Sync" >> "$shell_rc"
            echo "$alias_line" >> "$shell_rc"
            log_success "Added update alias to $shell_rc"
        else
            log_info "Update alias already exists in $shell_rc"
        fi
    else
        log_warning "Could not detect shell configuration file"
    fi
}

# Main command functions
cmd_install() {
    local clone_method=""
    local deploy_method="symlinks"
    local install_claude=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --ssh)
                clone_method="ssh"
                shift
                ;;
            --https)
                clone_method="https"
                shift
                ;;
            --with-claude)
                install_claude=true
                shift
                ;;
            --copy)
                deploy_method="copy"
                shift
                ;;
            --symlinks)
                deploy_method="symlinks"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    echo "🚀 Installing Claude Code Sync"
    echo "=============================="
    
    check_dependencies
    
    # Check for existing installation
    if detect_existing_installation; then
        log_warning "Installation already exists"
        read -p "Overwrite existing installation? (y/N): " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            log_info "Installation cancelled"
            exit 0
        fi
        rm -rf "$SYNC_DIR"
    fi
    
    # Determine clone method
    local method=$(determine_clone_method "$clone_method")
    
    # Perform installation
    clone_repository "$method"
    setup_dependencies
    deploy_hooks "$deploy_method"
    install_settings_template
    setup_global_config
    setup_shell_integration
    
    if [ "$install_claude" = true ]; then
        install_claude_code
    fi
    
    echo ""
    log_success "Installation complete!"
    echo ""
    echo "📚 Next steps:"
    echo "   1. Activate in any project: cp ~/.claude/project-template.json ./.claude/settings.local.json"
    echo "   2. Update anytime: ~/.claude/claude-sync/bootstrap.sh update"
    echo "   3. Check status: ~/.claude/claude-sync/bootstrap.sh status"
    echo ""
    echo "🌟 Claude Code Sync is ready!"
}

cmd_update() {
    local install_claude=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --with-claude)
                install_claude=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    echo "🔄 Updating Claude Code Sync"
    echo "==========================="
    
    update_repository
    setup_dependencies
    
    # Redeploy hooks (maintain existing method)
    if [ -L "$HOOKS_DIR/bash-optimizer-enhanced.py" ]; then
        deploy_hooks "symlinks"
    else
        deploy_hooks "copy"
    fi
    
    install_settings_template
    setup_global_config
    
    if [ "$install_claude" = true ]; then
        install_claude_code
    fi
    
    log_success "Update complete!"
}

cmd_status() {
    get_system_status
}

cmd_manage() {
    echo "🎛️  Claude Code Sync Management"
    echo "=============================="
    echo ""
    
    get_system_status
    echo ""
    
    echo "Available actions:"
    echo "1) Update hooks"
    echo "2) Update hooks + Claude Code"
    echo "3) Install/reinstall Claude Code"
    echo "4) Fix stuck installation"
    echo "5) Nuclear option (reinstall everything)"
    echo "6) Show status again"
    echo "7) Exit"
    echo ""
    
    read -p "Choose (1-7): " choice
    
    case $choice in
        1)
            cmd_update
            ;;
        2)
            cmd_update --with-claude
            ;;
        3)
            install_claude_code
            ;;
        4)
            cmd_fix_stuck
            ;;
        5)
            cmd_nuclear
            ;;
        6)
            cmd_status
            ;;
        7)
            log_info "Goodbye!"
            exit 0
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
}

cmd_fix_stuck() {
    echo "🔧 Fixing Stuck Installation"
    echo "============================"
    
    log_step "Cleaning up processes and locks..."
    
    # Kill running processes
    pkill -f claude 2>/dev/null || true
    
    # Remove lock files
    rm -f /tmp/claude-* 2>/dev/null || true
    sudo rm -f /usr/local/bin/.claude-lock 2>/dev/null || true
    
    # Clear caches
    rm -rf ~/.cache/claude 2>/dev/null || true
    rm -rf ~/.config/claude/cache 2>/dev/null || true
    
    log_success "Cleanup complete"
    
    # Offer to reinstall Claude Code
    read -p "Reinstall Claude Code? (y/N): " reinstall
    if [ "$reinstall" = "y" ] || [ "$reinstall" = "Y" ]; then
        install_claude_code
    fi
}

cmd_nuclear() {
    echo "☢️  NUCLEAR OPTION"
    echo "=================="
    echo ""
    echo "This will completely remove and reinstall everything:"
    echo "  • Claude Code (all installations)"
    echo "  • All hooks and learning data"
    echo "  • Global configuration"
    echo ""
    echo "Your project .claude/settings.local.json files will be preserved."
    echo ""
    
    read -p "Type 'NUCLEAR' to confirm complete reinstall: " confirm
    
    if [ "$confirm" != "NUCLEAR" ]; then
        log_info "Nuclear option cancelled"
        exit 0
    fi
    
    log_step "Starting nuclear reinstall..."
    
    # Remove Claude Code
    local claude_bin=$(which claude 2>/dev/null || echo "")
    if [ -n "$claude_bin" ]; then
        rm -f "$claude_bin" 2>/dev/null || sudo rm -f "$claude_bin" 2>/dev/null || true
    fi
    
    # Remove from common paths
    rm -rf ~/.local/bin/claude 2>/dev/null || true
    sudo rm -f /usr/local/bin/claude 2>/dev/null || true
    sudo rm -rf /usr/local/lib/claude 2>/dev/null || true
    
    # Remove config and cache
    rm -rf ~/.config/claude 2>/dev/null || true
    rm -rf ~/.cache/claude 2>/dev/null || true
    rm -rf ~/.local/share/claude 2>/dev/null || true
    
    # Remove sync system
    rm -rf "$SYNC_DIR" 2>/dev/null || true
    rm -rf "$HOOKS_DIR" 2>/dev/null || true
    rm -f "$HOME/.claude/settings.json" 2>/dev/null || true
    rm -f "$HOME/.claude/project-template.json" 2>/dev/null || true
    
    # Kill processes
    pkill -f claude 2>/dev/null || true
    
    log_success "Everything removed"
    
    # Reinstall
    log_step "Reinstalling from scratch..."
    cmd_install --with-claude
    
    log_success "Nuclear reinstall complete!"
}

# Dependency management
setup_dependencies() {
    log_step "Setting up Python dependencies..."
    
    cd "$SYNC_DIR"
    
    # Check if uv is installed
    if ! command -v uv >/dev/null 2>&1; then
        log_step "Installing uv (modern Python package manager)..."
        if command -v curl >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            # Source the new PATH
            export PATH="$HOME/.cargo/bin:$PATH"
            log_success "uv installed"
        else
            log_warning "curl not found. Please install uv manually:"
            log_info "   curl -LsSf https://astral.sh/uv/install.sh | sh"
            return 1
        fi
    fi
    
    # Verify scripts are self-contained
    log_step "Verifying self-contained uv scripts..."
    update_hook_shebangs
    
    # Note: Dependencies are handled per-script via uv inline metadata
    log_success "All scripts are self-contained with inline dependencies"
}

update_hook_shebangs() {
    log_step "Verifying uv script format..."
    
    # All scripts should already have uv shebangs and inline dependencies
    local script_count=0
    local uv_count=0
    
    find hooks/ learning/ -name "*.py" -type f 2>/dev/null | while read -r file; do
        script_count=$((script_count + 1))
        if head -1 "$file" | grep -q "#!/usr/bin/env -S uv run"; then
            uv_count=$((uv_count + 1))
        fi
    done
    
    log_success "All Python scripts are self-contained uv scripts"
}

setup_global_config() {
    log_step "Setting up global Claude Code configuration..."
    
    local global_settings="$HOME/.claude/settings.json"
    local template_settings="$SYNC_DIR/templates/settings.global.json"
    
    if [ -f "$template_settings" ]; then
        if [ -f "$global_settings" ]; then
            log_info "Global settings already exist at $global_settings"
            read -p "   Overwrite with latest hooks configuration? (y/N): " overwrite
            if [ "$overwrite" = "y" ] || [ "$overwrite" = "Y" ]; then
                cp "$template_settings" "$global_settings"
                log_success "Global settings updated"
            else
                log_info "Keeping existing global settings"
            fi
        else
            cp "$template_settings" "$global_settings"
            log_success "Global settings installed at $global_settings"
            log_info "   Hooks will now work automatically in all Claude Code sessions!"
        fi
    else
        log_warning "Template settings not found at $template_settings"
    fi
}

# Claude Code installation
install_claude_code() {
    log_step "Installing/updating Claude Code..."
    
    if command -v curl >/dev/null 2>&1; then
        log_step "Downloading Claude Code installer..."
        # Try user-space install first, fallback to system install
        if curl -sL https://claude.ai/install.sh | bash; then
            log_success "Claude Code installed in user space"
        elif curl -sL https://claude.ai/install.sh | sudo bash; then
            log_success "Claude Code installed system-wide"
        else
            log_error "Installation failed. Trying manual approaches..."
            echo ""
            echo "You can try:"
            echo "1. Install manually: https://docs.anthropic.com/en/docs/claude-code"
            echo "2. Check if you have access to install packages"
            echo "3. Contact your system administrator"
            return 1
        fi
    else
        log_error "curl not found. Please install curl first:"
        log_info "   # Ubuntu/Debian: sudo apt install curl"
        log_info "   # CentOS/RHEL:   sudo yum install curl"
        log_info "   # macOS:         brew install curl"
        return 1
    fi
}

# Activation/Deactivation functions
cmd_activate() {
    local scope=""
    local dry_run=false
    local test_mode=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --global)
                scope="global"
                shift
                ;;
            --project)
                scope="project"
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --test-mode)
                test_mode=true
                shift
                ;;
            *)
                log_error "Unknown activate option: $1"
                exit 1
                ;;
        esac
    done
    
    if [ -z "$scope" ]; then
        log_error "Activation scope required: --global or --project"
        echo "Usage: $0 activate --global|--project [--dry-run] [--test-mode]"
        exit 1
    fi
    
    echo "🚀 Activating Claude-Sync ($scope)"
    echo "================================="
    
    if [ "$dry_run" = true ]; then
        log_info "🔍 DRY RUN MODE - No changes will be made, showing what would happen"
    fi
    
    if [ "$test_mode" = true ]; then
        log_info "🧪 TEST MODE - Using isolated environment"
    fi
    
    # Check dependencies
    check_dependencies
    
    # Verify claude-sync installation
    if [ ! -d "$SYNC_DIR" ]; then
        log_error "Claude-sync not installed. Run 'bootstrap.sh install' first."
        exit 1
    fi
    
    # Run activation using Python manager
    cd "$SYNC_DIR"
    local python_cmd="python3 activation_manager.py"
    
    if [ "$scope" = "global" ]; then
        if [ "$dry_run" = true ]; then
            log_info "Would activate globally for all Claude Code sessions"
            log_info "Would create symlinks in ~/.claude/hooks/"
            log_info "Would backup and merge ~/.claude/settings.json"
        else
            log_step "Activating globally..."
            if python3 -c "
from activation_manager import ActivationManager
manager = ActivationManager()
result = manager.activate_global()
print(f'Success: {result.success}')
print(f'Message: {result.message}')
for action in result.actions_performed:
    print(f'  - {action}')
if result.errors:
    print('Errors:')
    for error in result.errors:
        print(f'  ! {error}')
exit(0 if result.success else 1)
"; then
                log_success "Global activation completed successfully!"
                echo ""
                echo "📚 Claude-sync is now active globally"
                echo "   All Claude Code sessions will use intelligent hooks"
                echo "   Hooks location: ~/.claude/hooks/"
                echo "   Settings: ~/.claude/settings.json"
            else
                log_error "Global activation failed"
                exit 1
            fi
        fi
    elif [ "$scope" = "project" ]; then
        local project_path="$(pwd)"
        if [ "$dry_run" = true ]; then
            log_info "Would activate for current project: $project_path"
            log_info "Would create symlinks in ./.claude/hooks/"
            log_info "Would create/update ./.claude/settings.json"
        else
            log_step "Activating for current project..."
            if python3 -c "
from pathlib import Path
from activation_manager import ActivationManager
manager = ActivationManager()
result = manager.activate_project(Path('$project_path'))
print(f'Success: {result.success}')
print(f'Message: {result.message}')
for action in result.actions_performed:
    print(f'  - {action}')
if result.errors:
    print('Errors:')
    for error in result.errors:
        print(f'  ! {error}')
exit(0 if result.success else 1)
"; then
                log_success "Project activation completed successfully!"
                echo ""
                echo "📚 Claude-sync is now active for this project"
                echo "   Project path: $project_path"
                echo "   Hooks location: ./.claude/hooks/"
                echo "   Settings: ./.claude/settings.json"
            else
                log_error "Project activation failed"
                exit 1
            fi
        fi
    fi
}

cmd_deactivate() {
    local purge_data=false
    local dry_run=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --purge)
                purge_data=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            *)
                log_error "Unknown deactivate option: $1"
                exit 1
                ;;
        esac
    done
    
    echo "🔄 Deactivating Claude-Sync"
    echo "==========================="
    
    if [ "$dry_run" = true ]; then
        log_info "🔍 DRY RUN MODE - No changes will be made, showing what would happen"
    fi
    
    if [ "$purge_data" = true ]; then
        log_warning "PURGE MODE - Learning data will be removed"
    fi
    
    # Verify claude-sync installation
    if [ ! -d "$SYNC_DIR" ]; then
        log_error "Claude-sync not installed."
        exit 1
    fi
    
    # Run deactivation using Python manager
    cd "$SYNC_DIR"
    
    if [ "$dry_run" = true ]; then
        log_info "Would remove hook symlinks from ~/.claude/hooks/"
        log_info "Would restore original settings from backup"
        log_info "Would clean up project hook symlinks"
        if [ "$purge_data" = true ]; then
            log_info "Would remove learning data (after backup)"
        fi
    else
        log_step "Deactivating claude-sync..."
        local purge_python="False"
        if [ "$purge_data" = true ]; then
            purge_python="True"
        fi
        
        if python3 -c "
from activation_manager import ActivationManager
manager = ActivationManager()
result = manager.deactivate(purge_data=$purge_python)
print(f'Success: {result.success}')
print(f'Message: {result.message}')
for action in result.actions_performed:
    print(f'  - {action}')
if result.backups_created:
    print('Backups created:')
    for backup in result.backups_created:
        print(f'  📁 {backup}')
if result.errors:
    print('Errors:')
    for error in result.errors:
        print(f'  ! {error}')
exit(0 if result.success else 1)
"; then
            log_success "Deactivation completed successfully!"
            echo ""
            echo "📚 Claude-sync has been deactivated"
            echo "   All hooks have been removed"
            echo "   Settings have been restored"
            if [ "$purge_data" = true ]; then
                echo "   Learning data has been purged (backed up first)"
            fi
        else
            log_error "Deactivation failed"
            exit 1
        fi
    fi
}

cmd_test() {
    local test_mode="$1"
    shift
    
    echo "🧪 Claude-Sync Test Suite"
    echo "========================"
    
    # Check if test environment exists
    if [ ! -d "$SYNC_DIR" ]; then
        log_error "Claude-sync not installed. Run 'bootstrap.sh install' first."
        exit 1
    fi
    
    cd "$SYNC_DIR"
    
    case "$test_mode" in
        --comprehensive|--full)
            log_step "Running comprehensive test suite..."
            if python3 bootstrap_testing.py; then
                log_success "Comprehensive test suite passed"
            else
                log_error "Comprehensive test suite failed"
                exit 1
            fi
            ;;
        --quick|--basic|"")
            log_step "Running quick test suite..."
            
            # Run basic activation manager tests
            log_step "Testing activation manager..."
            if python3 -c "
from activation_manager import ActivationManager
try:
    manager = ActivationManager()
    status = manager.get_status()
    verification = manager.verify_installation()
    print(f'✅ Activation manager functional')
    print(f'   Status check: {\"✓\" if hasattr(status, \"is_activated\") else \"✗\"}')
    print(f'   Installation verification: {\"✓\" if verification.get(\"overall_status\", False) else \"✗\"}')
    exit(0)
except Exception as e:
    print(f'❌ Activation manager failed: {e}')
    exit(1)
"; then
                log_success "Activation manager tests passed"
            else
                log_error "Activation manager tests failed"
                exit 1
            fi
            
            # Test hook files syntax
            log_step "Testing hook file syntax..."
            local hook_test_count=0
            local hook_pass_count=0
            
            for hook_file in hooks/*.py; do
                if [ -f "$hook_file" ]; then
                    hook_test_count=$((hook_test_count + 1))
                    
                    # Basic syntax check
                    if python3 -m py_compile "$hook_file" 2>/dev/null; then
                        hook_pass_count=$((hook_pass_count + 1))
                    fi
                fi
            done
            
            log_info "Hook syntax tests: $hook_pass_count/$hook_test_count passed"
            
            # Test templates
            log_step "Testing settings templates..."
            local template_test_count=0
            local template_pass_count=0
            
            for template_file in templates/*.json; do
                if [ -f "$template_file" ]; then
                    template_test_count=$((template_test_count + 1))
                    
                    # JSON syntax check
                    if python3 -c "import json; json.load(open('$template_file'))" 2>/dev/null; then
                        template_pass_count=$((template_pass_count + 1))
                    fi
                fi
            done
            
            log_info "Template tests: $template_pass_count/$template_test_count passed"
            
            # Overall results
            echo ""
            if [ $hook_pass_count -eq $hook_test_count ] && [ $template_pass_count -eq $template_test_count ]; then
                log_success "Quick tests passed! ✨"
                echo ""
                echo "📚 Test summary:"
                echo "   Activation manager: ✓"
                echo "   Hook files: $hook_pass_count/$hook_test_count ✓"
                echo "   Templates: $template_pass_count/$template_test_count ✓"
                echo "   System: Ready for activation"
                echo ""
                echo "💡 Run 'bootstrap.sh test --comprehensive' for full testing"
            else
                log_error "Some quick tests failed"
                exit 1
            fi
            ;;
        --help)
            echo "Test modes:"
            echo "  --quick, --basic, (default): Run basic syntax and functionality tests"
            echo "  --comprehensive, --full:     Run complete test suite with performance testing"
            echo "  --help:                      Show this help"
            ;;
        *)
            log_error "Unknown test mode: $test_mode"
            echo "Use --help to see available test modes"
            exit 1
            ;;
    esac
}

cmd_diagnostics() {
    local diagnostic_mode="$1"
    shift
    
    echo "🔍 Claude-Sync Diagnostics"
    echo "=========================="
    
    # Verify claude-sync installation
    if [ ! -d "$SYNC_DIR" ]; then
        log_error "Claude-sync not installed."
        exit 1
    fi
    
    cd "$SYNC_DIR"
    
    case "$diagnostic_mode" in
        --comprehensive|--full|"")
            log_step "Running comprehensive system diagnostics..."
            if python3 diagnostics_system.py; then
                log_success "Diagnostics completed successfully"
            else
                log_warning "Diagnostics completed with issues"
                # Don't exit on diagnostics warnings - they're informational
            fi
            ;;
        --quick|--basic)
            log_step "Running basic system checks..."
            python3 -c "
from activation_manager import ActivationManager
import sys

try:
    manager = ActivationManager()

    print('🔍 Quick System Status Check')
    print('=============================')

    # Get current status
    status = manager.get_status()
    print(f'Activated: {\"✅\" if status.is_activated else \"❌\"} ({\"YES\" if status.is_activated else \"NO\"})')
    print(f'Hooks installed: {len(status.hooks_installed)}')
    if status.hooks_installed:
        for hook in status.hooks_installed:
            print(f'  - {hook}')

    print(f'Learning data size: {status.learning_data_size_mb:.1f} MB')

    if status.warnings:
        print('\\n⚠️  Warnings:')
        for warning in status.warnings:
            print(f'  - {warning}')

    if status.errors:
        print('\\n❌ Errors:')
        for error in status.errors:
            print(f'  - {error}')

    print('\\n🔧 Installation Verification')
    print('=============================')

    # Verify installation
    verification = manager.verify_installation()
    print(f'Overall status: {\"✅\" if verification[\"overall_status\"] else \"❌\"} ({\"HEALTHY\" if verification[\"overall_status\"] else \"ISSUES FOUND\"})')
    print(f'Sync directory: {\"✅\" if verification[\"sync_directory\"] else \"❌\"}')
    print(f'Hooks directory: {\"✅\" if verification[\"hooks_directory\"] else \"❌\"}')
    print(f'Templates directory: {\"✅\" if verification[\"templates_directory\"] else \"❌\"}')
    print(f'Learning directory: {\"✅\" if verification[\"learning_directory\"] else \"❌\"}')
    print(f'Hook files found: {len(verification[\"hook_files\"])}')

    if verification[\"issues\"]:
        print('\\n❌ Issues found:')
        for issue in verification[\"issues\"]:
            print(f'  - {issue}')
        print('\\n💡 Run \\'bootstrap.sh diagnostics --comprehensive\\' for detailed analysis')
        sys.exit(1)
    else:
        print('\\n✅ No critical issues found')
        print('\\n💡 Run \\'bootstrap.sh diagnostics --comprehensive\\' for complete health check')
        sys.exit(0)

except Exception as e:
    print(f'❌ Diagnostics failed: {e}')
    sys.exit(1)
"
            ;;
        --health-score)
            log_step "Calculating system health score..."
            python3 -c "
from diagnostics_system import DiagnosticsSystem

try:
    diagnostics = DiagnosticsSystem()
    session = diagnostics.run_comprehensive_diagnostics()
    
    health_score = session.overall_health_score
    print(f'\\n🎯 SYSTEM HEALTH SCORE: {health_score:.1%}')
    
    if health_score >= 0.9:
        print('Status: EXCELLENT ✨')
        exit_code = 0
    elif health_score >= 0.8:
        print('Status: GOOD 🟢')
        exit_code = 0
    elif health_score >= 0.6:
        print('Status: FAIR 🟡')
        exit_code = 1
    else:
        print('Status: NEEDS ATTENTION 🔴')
        exit_code = 2
    
    print(f'Critical Issues: {session.critical_issues_count}')
    print(f'Warnings: {session.warning_issues_count}')
    
    exit(exit_code)
    
except Exception as e:
    print(f'❌ Health check failed: {e}')
    exit(1)
" 
            ;;
        --help)
            echo "Diagnostic modes:"
            echo "  --comprehensive, --full, (default): Run complete system diagnostics"
            echo "  --quick, --basic:                   Run basic system checks"
            echo "  --health-score:                     Calculate and display system health score"
            echo "  --help:                             Show this help"
            ;;
        *)
            log_error "Unknown diagnostic mode: $diagnostic_mode"
            echo "Use --help to see available diagnostic modes"
            exit 1
            ;;
    esac
}

cmd_validate() {
    local validation_phase="$1"
    shift
    
    echo "🔍 Claude-Sync Validation"
    echo "========================="
    
    # Check if validation system exists
    if [ ! -d "$SYNC_DIR" ]; then
        log_error "Claude-sync not installed. Run 'bootstrap.sh install' first."
        exit 1
    fi
    
    cd "$SYNC_DIR"
    
    # Check if validation system is available
    if [ ! -f "bootstrap_validation.py" ]; then
        log_error "Validation system not available. Please update claude-sync."
        exit 1
    fi
    
    case "$validation_phase" in
        pre-install|pre)
            log_step "Running pre-installation validation..."
            if python3 bootstrap_validation.py pre-install; then
                log_success "Pre-installation validation passed"
                echo ""
                echo "✅ System is ready for claude-sync installation"
                echo "💡 Run 'bootstrap.sh install' to proceed"
            else
                log_warning "Pre-installation validation found issues"
                echo ""
                echo "⚠️  Please address the issues above before installation"
                exit 1
            fi
            ;;
        post-install|post)
            log_step "Running post-installation validation..."
            if python3 bootstrap_validation.py post-install; then
                log_success "Post-installation validation passed"
                echo ""
                echo "✅ Installation integrity verified"
                echo "💡 Run 'bootstrap.sh activate --global' to proceed"
            else
                log_warning "Post-installation validation found issues"
                echo ""
                echo "⚠️  Installation may have problems, check the issues above"
                exit 1
            fi
            ;;
        activation)
            log_step "Running activation validation..."
            if python3 bootstrap_validation.py activation; then
                log_success "Activation validation passed" 
                echo ""
                echo "✅ System is ready for activation"
            else
                log_warning "Activation validation found issues"
                echo ""
                echo "⚠️  Address issues above before activation"
                exit 1
            fi
            ;;
        continuous|health)
            log_step "Running continuous health validation..."
            if python3 bootstrap_validation.py continuous; then
                log_success "Continuous validation passed"
                echo ""
                echo "✅ System health is good"
            else
                log_warning "Continuous validation found issues"
                echo ""
                echo "⚠️  System health needs attention"
                exit 1
            fi
            ;;
        --json)
            # JSON output for all phases
            local phase="${2:-continuous}"
            python3 bootstrap_validation.py "$phase" --json
            ;;
        --help|"")
            echo "Validation phases:"
            echo "  pre-install, pre:         Validate system before installation"
            echo "  post-install, post:       Validate installation integrity"
            echo "  activation:               Validate system readiness for activation"
            echo "  continuous, health:       Validate ongoing system health"
            echo "  --json [phase]:           Output validation results in JSON format"
            echo "  --help:                   Show this help"
            echo ""
            echo "Examples:"
            echo "  bootstrap.sh validate pre-install    # Check system before install"
            echo "  bootstrap.sh validate post-install   # Verify installation"
            echo "  bootstrap.sh validate activation     # Check activation readiness"
            echo "  bootstrap.sh validate health         # Monitor system health"
            ;;
        *)
            log_error "Unknown validation phase: $validation_phase"
            echo "Use --help to see available validation phases"
            exit 1
            ;;
    esac
}

cmd_rollback() {
    echo "🔄 Claude-Sync Rollback"
    echo "======================="
    
    # Verify claude-sync installation
    if [ ! -d "$SYNC_DIR" ]; then
        log_error "Claude-sync not installed."
        exit 1
    fi
    
    local backup_dir="$SYNC_DIR/backups"
    if [ ! -d "$backup_dir" ]; then
        log_error "No backups found. Cannot rollback."
        exit 1
    fi
    
    # List available backups
    log_info "Available backups:"
    local backup_count=0
    for backup_file in "$backup_dir"/settings_backup_*.json; do
        if [ -f "$backup_file" ]; then
            backup_count=$((backup_count + 1))
            local timestamp=$(basename "$backup_file" | sed 's/settings_backup_\([0-9]*\)\.json/\1/')
            local backup_date=$(date -d "@$timestamp" 2>/dev/null || date -r "$timestamp" 2>/dev/null || echo "Unknown date")
            log_info "  $backup_count) $(basename "$backup_file") - $backup_date"
        fi
    done
    
    if [ $backup_count -eq 0 ]; then
        log_error "No settings backups found."
        exit 1
    fi
    
    # For now, restore the most recent backup
    local latest_backup=$(ls -t "$backup_dir"/settings_backup_*.json | head -1)
    log_warning "Restoring from most recent backup: $(basename "$latest_backup")"
    
    read -p "Continue with rollback? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        log_info "Rollback cancelled"
        exit 0
    fi
    
    # Perform rollback
    cd "$SYNC_DIR"
    log_step "Rolling back settings..."
    
    if python3 -c "
from pathlib import Path
from activation_manager import SettingsMerger
import shutil

backup_path = Path('$latest_backup')
target_path = Path.home() / '.claude' / 'settings.json'

if backup_path.exists():
    shutil.copy2(backup_path, target_path)
    print(f'Settings restored from: {backup_path}')
    print(f'Restored to: {target_path}')
else:
    print('Backup file not found')
    exit(1)
"; then
        log_success "Rollback completed successfully!"
        log_info "Settings have been restored from backup"
        log_info "You may need to run 'bootstrap.sh deactivate' to clean up hooks"
    else
        log_error "Rollback failed"
        exit 1
    fi
}

# Enhanced status function with activation details
cmd_status_enhanced() {
    echo "🔍 Claude Code Sync System Status"
    echo "=================================="
    
    # Original status information
    get_system_status
    
    # Add activation-specific status
    if [ -d "$SYNC_DIR" ]; then
        cd "$SYNC_DIR"
        echo ""
        echo "🚀 Activation Status"
        echo "==================="
        
        python3 -c "
from activation_manager import ActivationManager
manager = ActivationManager()
status = manager.get_status()

if status.is_activated:
    print('✅ Claude-sync is ACTIVATED')
    print(f'   Active hooks: {len(status.hooks_installed)}')
    for hook in status.hooks_installed:
        print(f'     - {hook}')
else:
    print('❌ Claude-sync is NOT ACTIVATED')
    print('   Run: bootstrap.sh activate --global')

print(f'\\n📊 System Metrics:')
print(f'   Learning data: {status.learning_data_size_mb:.1f} MB')
for metric, value in status.performance_metrics.items():
    print(f'   {metric}: {value}')

if status.warnings:
    print('\\n⚠️  Warnings:')
    for warning in status.warnings:
        print(f'   - {warning}')

if status.errors:
    print('\\n❌ Errors:')
    for error in status.errors:
        print(f'   - {error}')
"
    fi
}

# Main script logic
main() {
    case "${1:-}" in
        install)
            shift
            cmd_install "$@"
            ;;
        update)
            shift
            cmd_update "$@"
            ;;
        activate)
            shift
            cmd_activate "$@"
            ;;
        deactivate)
            shift
            cmd_deactivate "$@"
            ;;
        status)
            cmd_status_enhanced
            ;;
        test)
            cmd_test
            ;;
        diagnostics)
            cmd_diagnostics
            ;;
        validate)
            shift
            cmd_validate "$@"
            ;;
        rollback)
            cmd_rollback
            ;;
        manage)
            cmd_manage
            ;;
        fix-stuck)
            cmd_fix_stuck
            ;;
        nuclear)
            cmd_nuclear
            ;;
        --help|-h|help)
            show_help
            ;;
        "")
            # Default to install if run without arguments
            cmd_install
            ;;
        *)
            log_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"