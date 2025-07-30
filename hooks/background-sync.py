#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# ///
"""
Background Sync Hook
Auto-magic background synchronization of hooks and learning data.
"""

import json
import sys
import time
import subprocess
import threading
from pathlib import Path
import hashlib
import os

# Try to import learning infrastructure
HOOKS_DIR = Path(__file__).parent.parent / 'learning'

def import_learning_modules():
    """Import learning modules with fallback"""
    try:
        sys.path.insert(0, str(HOOKS_DIR))
        from encryption import get_secure_storage
        from mesh_sync import get_mesh_sync
        return get_secure_storage(), get_mesh_sync(), True
    except ImportError:
        return None, None, False

STORAGE, MESH_SYNC, SYNC_AVAILABLE = import_learning_modules()

class BackgroundSync:
    """Manages background synchronization"""
    
    def __init__(self):
        self.sync_enabled = True
        self.last_hook_check = 0
        self.last_learning_sync = 0
        self.hook_check_interval = 86400  # 24 hours
        self.learning_sync_interval = 3600  # 1 hour
        
    def should_check_hooks(self):
        """Check if we should check for hook updates"""
        current_time = time.time()
        return (current_time - self.last_hook_check) > self.hook_check_interval
    
    def should_sync_learning(self):
        """Check if we should sync learning data"""
        current_time = time.time()
        return (current_time - self.last_learning_sync) > self.learning_sync_interval
    
    def run_background_sync(self):
        """Main background sync orchestration"""
        try:
            # Check for hook updates (daily)
            if self.should_check_hooks():
                self._check_hook_updates()
                self.last_hook_check = time.time()
            
            # Sync learning data (hourly)  
            if SYNC_AVAILABLE and self.should_sync_learning():
                self._sync_learning_data()
                self.last_learning_sync = time.time()
                
        except Exception:
            # Background sync should never break normal operation
            pass
    
    def _check_hook_updates(self):
        """Check for hook updates from GitHub"""
        try:
            hooks_repo_dir = Path.home() / '.claude' / 'claude-sync'
            
            if not hooks_repo_dir.exists():
                return
            
            # Quick git fetch to check for updates
            result = subprocess.run(
                ['git', 'fetch', 'origin', 'main'],
                cwd=hooks_repo_dir,
                capture_output=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return
            
            # Check if updates are available
            status_result = subprocess.run(
                ['git', 'status', '-uno', '--porcelain=v1'],
                cwd=hooks_repo_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Check for commits ahead/behind
            behind_result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD..origin/main'],
                cwd=hooks_repo_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if behind_result.returncode == 0:
                commits_behind = int(behind_result.stdout.strip() or '0')
                
                if commits_behind > 0:
                    # Updates available - perform smart update
                    self._perform_smart_update(hooks_repo_dir, commits_behind)
                    
        except Exception:
            pass
    
    def _perform_smart_update(self, repo_dir, commits_behind):
        """Perform smart update with user notification"""
        try:
            # Get changelog
            changelog = self._get_update_changelog(repo_dir, commits_behind)
            
            # Determine update urgency
            urgency = self._assess_update_urgency(changelog)
            
            if urgency == 'critical':
                # Auto-update critical fixes
                self._auto_update_hooks(repo_dir)
                self._notify_user("🚨 Critical hook updates applied automatically")
                
            elif urgency == 'major':
                # Notify about major updates but don't auto-apply
                self._notify_user(f"🔄 Major hook updates available ({commits_behind} commits)\n" +
                                f"Run 'update-claude-sync' to apply:\n{changelog}")
                
            elif urgency == 'minor':
                # Silent update for minor improvements
                if commits_behind <= 2:  # Only auto-update small changes
                    self._auto_update_hooks(repo_dir)
                
        except Exception:
            pass
    
    def _get_update_changelog(self, repo_dir, commits_behind):
        """Get changelog for pending updates"""
        try:
            result = subprocess.run(
                ['git', 'log', '--oneline', f'HEAD..origin/main', f'-{min(commits_behind, 5)}'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
                
        except Exception:
            pass
        
        return ""
    
    def _assess_update_urgency(self, changelog):
        """Assess urgency of updates"""
        changelog_lower = changelog.lower()
        
        # Critical updates
        if any(word in changelog_lower for word in ['security', 'critical', 'urgent', 'fix']):
            return 'critical'
        
        # Major updates
        if any(word in changelog_lower for word in ['feature', 'new', 'major', 'breaking']):
            return 'major'
        
        # Minor updates
        if any(word in changelog_lower for word in ['improve', 'enhance', 'optimize', 'update']):
            return 'minor'
        
        return 'minor'
    
    def _auto_update_hooks(self, repo_dir):
        """Automatically update hooks"""
        try:
            # Pull updates
            subprocess.run(
                ['git', 'pull', 'origin', 'main'],
                cwd=repo_dir,
                capture_output=True,
                timeout=30
            )
            
            # Update symlinks (run the update script)
            update_script = repo_dir / 'update.sh'
            if update_script.exists():
                subprocess.run(
                    [str(update_script)],
                    capture_output=True,
                    timeout=30
                )
                
        except Exception:
            pass
    
    def _sync_learning_data(self):
        """Sync learning data across hosts"""
        if MESH_SYNC:
            # Run mesh sync in background thread to avoid blocking
            sync_thread = threading.Thread(
                target=MESH_SYNC.sync_learning_data,
                daemon=True
            )
            sync_thread.start()
    
    def _notify_user(self, message):
        """Notify user about updates (non-intrusive)"""
        try:
            # Write to a log file that can be checked
            log_file = Path.home() / '.claude' / 'sync.log'
            log_file.parent.mkdir(exist_ok=True)
            
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(log_file, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
            
            # Optional: Desktop notification if available
            try:
                subprocess.run(
                    ['notify-send', 'Claude Hooks', message],
                    capture_output=True,
                    timeout=5
                )
            except:
                pass
                
        except Exception:
            pass

def main():
    hook_input = json.loads(sys.stdin.read())
    
    if hook_input.get('tool_name') != 'Bash':
        sys.exit(0)
    
    # Only run background sync on first command of the day
    if should_run_background_sync():
        sync = BackgroundSync()
        # Run sync in background thread to avoid delaying command
        sync_thread = threading.Thread(
            target=sync.run_background_sync,
            daemon=True
        )
        sync_thread.start()
    
    sys.exit(0)

def should_run_background_sync():
    """Check if we should run background sync now"""
    try:
        # Check if we've already synced today
        sync_marker = Path.home() / '.claude' / 'last_sync'
        current_date = time.strftime('%Y-%m-%d')
        
        if sync_marker.exists():
            last_sync_date = sync_marker.read_text().strip()
            if last_sync_date == current_date:
                return False  # Already synced today
        
        # Update sync marker
        sync_marker.parent.mkdir(exist_ok=True)
        sync_marker.write_text(current_date)
        
        return True
        
    except Exception:
        return False

if __name__ == '__main__':
    main()