#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "typing-extensions>=4.0.0"
# ]
# ///
"""
Claude-Sync Activation Manager

Manages clean activation/deactivation of claude-sync with atomic operations
and comprehensive rollback capabilities.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from interfaces import ActivationManagerInterface, ActivationResult, SystemState


@dataclass
class BackupInfo:
    """Information about a backup file"""
    path: Path
    timestamp: float
    original_path: Path
    backup_type: str  # 'settings', 'hooks', 'config'


class SettingsMerger:
    """Handles JSON settings merging without overwriting user configurations"""
    
    @staticmethod
    def backup_user_settings(settings_path: Path, backup_dir: Path) -> Path:
        """Create timestamped backup of user settings"""
        if not settings_path.exists():
            return None
            
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        backup_path = backup_dir / f"settings_backup_{timestamp}.json"
        
        shutil.copy2(settings_path, backup_path)
        return backup_path
    
    @staticmethod
    def merge_hook_settings(user_settings: Dict[str, Any], claude_sync_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Merge claude-sync hooks into existing user settings without overwriting"""
        merged = user_settings.copy()
        
        # Ensure hooks section exists
        if "hooks" not in merged:
            merged["hooks"] = {}
        
        # Merge each hook type (PreToolUse, PostToolUse, UserPromptSubmit)
        for hook_type, hook_configs in claude_sync_settings.get("hooks", {}).items():
            if hook_type not in merged["hooks"]:
                merged["hooks"][hook_type] = []
            
            # Add claude-sync hooks, avoiding duplicates
            existing_commands = set()
            for existing_hook in merged["hooks"][hook_type]:
                for hook in existing_hook.get("hooks", []):
                    if hook.get("type") == "command":
                        existing_commands.add(hook["command"])
            
            # Add new claude-sync hooks
            for hook_config in hook_configs:
                for hook in hook_config.get("hooks", []):
                    if hook.get("type") == "command" and hook["command"] not in existing_commands:
                        # Find or create matcher group
                        matcher = hook_config.get("matcher", ".*")
                        matcher_group = None
                        for existing_group in merged["hooks"][hook_type]:
                            if existing_group.get("matcher") == matcher:
                                matcher_group = existing_group
                                break
                        
                        if matcher_group is None:
                            matcher_group = {"matcher": matcher, "hooks": []}
                            merged["hooks"][hook_type].append(matcher_group)
                        
                        matcher_group["hooks"].append(hook)
                        existing_commands.add(hook["command"])
        
        # Merge permissions if they exist in claude-sync settings
        if "permissions" in claude_sync_settings:
            if "permissions" not in merged:
                merged["permissions"] = {}
            
            # Merge allow permissions
            if "allow" in claude_sync_settings["permissions"]:
                if "allow" not in merged["permissions"]:
                    merged["permissions"]["allow"] = []
                
                for permission in claude_sync_settings["permissions"]["allow"]:
                    if permission not in merged["permissions"]["allow"]:
                        merged["permissions"]["allow"].append(permission)
        
        return merged
    
    @staticmethod
    def validate_merged_settings(settings: Dict[str, Any]) -> bool:
        """Validate merged settings structure"""
        try:
            # Check required structure
            if "hooks" in settings:
                for hook_type, hook_list in settings["hooks"].items():
                    if not isinstance(hook_list, list):
                        return False
                    for hook_config in hook_list:
                        if not isinstance(hook_config, dict):
                            return False
                        if "matcher" not in hook_config or "hooks" not in hook_config:
                            return False
                        if not isinstance(hook_config["hooks"], list):
                            return False
            return True
        except Exception:
            return False


class SymlinkManager:
    """Manages hook symlinks for activation/deactivation"""
    
    @staticmethod
    def create_hook_symlinks(source_dir: Path, target_dir: Path) -> List[Path]:
        """Create symlinks for all hooks, return created symlinks"""
        target_dir.mkdir(parents=True, exist_ok=True)
        created_symlinks = []
        
        # Find all Python hook files
        for hook_file in source_dir.glob("*.py"):
            target_path = target_dir / hook_file.name
            
            # Remove existing file/symlink
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()
            
            # Create symlink
            target_path.symlink_to(hook_file.absolute())
            created_symlinks.append(target_path)
        
        return created_symlinks
    
    @staticmethod
    def remove_hook_symlinks(target_dir: Path, source_dir: Path) -> List[Path]:
        """Remove claude-sync symlinks, return removed paths"""
        removed_symlinks = []
        
        if not target_dir.exists():
            return removed_symlinks
        
        # Find all symlinks that point to claude-sync hooks
        for potential_symlink in target_dir.glob("*.py"):
            if potential_symlink.is_symlink():
                target = potential_symlink.resolve()
                if target.parent == source_dir.resolve():
                    potential_symlink.unlink()
                    removed_symlinks.append(potential_symlink)
        
        return removed_symlinks
    
    @staticmethod
    def verify_symlinks(target_dir: Path, source_dir: Path) -> Dict[str, bool]:
        """Verify all expected symlinks exist and are valid"""
        verification_results = {}
        
        if not target_dir.exists():
            return {"directory_exists": False}
        
        # Check each hook file
        for hook_file in source_dir.glob("*.py"):
            symlink_path = target_dir / hook_file.name
            verification_results[hook_file.name] = (
                symlink_path.exists() and 
                symlink_path.is_symlink() and 
                symlink_path.resolve() == hook_file.resolve()
            )
        
        return verification_results


class ActivationManager(ActivationManagerInterface):
    """Main activation manager implementing the interface"""
    
    def __init__(self):
        self.claude_dir = Path.home() / '.claude'
        self.sync_dir = Path.home() / '.claude' / 'claude-sync'
        self.hooks_dir = self.claude_dir / 'hooks'
        self.backup_dir = self.sync_dir / 'backups'
        self.settings_merger = SettingsMerger()
        self.symlink_manager = SymlinkManager()
    
    def activate_global(self) -> ActivationResult:
        """Activate claude-sync globally for all Claude Code sessions"""
        actions_performed = []
        backups_created = []
        errors = []
        
        try:
            # 1. Create hooks directory
            self.hooks_dir.mkdir(parents=True, exist_ok=True)
            actions_performed.append(f"Created hooks directory: {self.hooks_dir}")
            
            # 2. Backup existing settings
            global_settings_path = self.claude_dir / 'settings.json'
            if global_settings_path.exists():
                backup_path = self.settings_merger.backup_user_settings(
                    global_settings_path, self.backup_dir
                )
                if backup_path:
                    backups_created.append(backup_path)
                    actions_performed.append(f"Backed up settings to: {backup_path}")
            
            # 3. Create hook symlinks
            hooks_source_dir = self.sync_dir / 'hooks'
            created_symlinks = self.symlink_manager.create_hook_symlinks(
                hooks_source_dir, self.hooks_dir
            )
            actions_performed.append(f"Created {len(created_symlinks)} hook symlinks")
            
            # 4. Merge settings
            template_path = self.sync_dir / 'templates' / 'settings.global.json'
            if template_path.exists():
                with open(template_path) as f:
                    claude_sync_settings = json.load(f)
                
                # Update template to use symlinks instead of direct paths
                claude_sync_settings = self._update_settings_for_symlinks(claude_sync_settings)
                
                # Load existing user settings or create empty
                user_settings = {}
                if global_settings_path.exists():
                    with open(global_settings_path) as f:
                        user_settings = json.load(f)
                
                # Merge settings
                merged_settings = self.settings_merger.merge_hook_settings(
                    user_settings, claude_sync_settings
                )
                
                # Validate merged settings
                if not self.settings_merger.validate_merged_settings(merged_settings):
                    errors.append("Merged settings validation failed")
                    return ActivationResult(
                        success=False,
                        message="Settings validation failed",
                        actions_performed=actions_performed,
                        backups_created=backups_created,
                        errors=errors,
                        rollback_required=True
                    )
                
                # Write merged settings
                with open(global_settings_path, 'w') as f:
                    json.dump(merged_settings, f, indent=2)
                actions_performed.append("Updated global settings with claude-sync hooks")
            else:
                errors.append(f"Template not found: {template_path}")
            
            # 5. Verify activation
            verification_results = self.symlink_manager.verify_symlinks(
                self.hooks_dir, hooks_source_dir
            )
            
            failed_verifications = [
                name for name, success in verification_results.items() 
                if not success
            ]
            
            if failed_verifications:
                errors.append(f"Symlink verification failed for: {failed_verifications}")
            
            success = len(errors) == 0
            message = "Global activation completed successfully" if success else "Global activation completed with errors"
            
            return ActivationResult(
                success=success,
                message=message,
                actions_performed=actions_performed,
                backups_created=backups_created,
                errors=errors,
                rollback_required=not success
            )
            
        except Exception as e:
            errors.append(f"Unexpected error during activation: {str(e)}")
            return ActivationResult(
                success=False,
                message=f"Activation failed: {str(e)}",
                actions_performed=actions_performed,
                backups_created=backups_created,
                errors=errors,
                rollback_required=True
            )
    
    def activate_project(self, project_path: Path) -> ActivationResult:
        """Activate claude-sync for specific project"""
        actions_performed = []
        backups_created = []
        errors = []
        
        try:
            project_claude_dir = project_path / '.claude'
            project_hooks_dir = project_claude_dir / 'hooks'
            project_settings_path = project_claude_dir / 'settings.json'
            
            # 1. Create project .claude directory
            project_claude_dir.mkdir(parents=True, exist_ok=True)
            actions_performed.append(f"Created project directory: {project_claude_dir}")
            
            # 2. Backup existing project settings
            if project_settings_path.exists():
                backup_path = self.settings_merger.backup_user_settings(
                    project_settings_path, self.backup_dir
                )
                if backup_path:
                    backups_created.append(backup_path)
                    actions_performed.append(f"Backed up project settings to: {backup_path}")
            
            # 3. Create project hook symlinks
            hooks_source_dir = self.sync_dir / 'hooks'
            created_symlinks = self.symlink_manager.create_hook_symlinks(
                hooks_source_dir, project_hooks_dir
            )
            actions_performed.append(f"Created {len(created_symlinks)} project hook symlinks")
            
            # 4. Merge project settings
            template_path = self.sync_dir / 'templates' / 'settings.local.json'
            if template_path.exists():
                with open(template_path) as f:
                    claude_sync_settings = json.load(f)
                
                # Update template to use project symlinks
                claude_sync_settings = self._update_settings_for_project_symlinks(claude_sync_settings)
                
                # Load existing project settings or create empty
                user_settings = {}
                if project_settings_path.exists():
                    with open(project_settings_path) as f:
                        user_settings = json.load(f)
                
                # Merge settings
                merged_settings = self.settings_merger.merge_hook_settings(
                    user_settings, claude_sync_settings
                )
                
                # Validate and write
                if self.settings_merger.validate_merged_settings(merged_settings):
                    with open(project_settings_path, 'w') as f:
                        json.dump(merged_settings, f, indent=2)
                    actions_performed.append("Updated project settings with claude-sync hooks")
                else:
                    errors.append("Project settings validation failed")
            else:
                errors.append(f"Project template not found: {template_path}")
            
            success = len(errors) == 0
            message = "Project activation completed successfully" if success else "Project activation completed with errors"
            
            return ActivationResult(
                success=success,
                message=message,
                actions_performed=actions_performed,
                backups_created=backups_created,
                errors=errors,
                rollback_required=not success
            )
            
        except Exception as e:
            errors.append(f"Unexpected error during project activation: {str(e)}")
            return ActivationResult(
                success=False,
                message=f"Project activation failed: {str(e)}",
                actions_performed=actions_performed,
                backups_created=backups_created,
                errors=errors,
                rollback_required=True
            )
    
    def deactivate(self, purge_data: bool = False) -> ActivationResult:
        """Deactivate claude-sync and optionally purge learning data"""
        actions_performed = []
        backups_created = []
        errors = []
        
        try:
            hooks_source_dir = self.sync_dir / 'hooks'
            
            # 1. Remove global hook symlinks
            removed_global = self.symlink_manager.remove_hook_symlinks(
                self.hooks_dir, hooks_source_dir
            )
            if removed_global:
                actions_performed.append(f"Removed {len(removed_global)} global hook symlinks")
            
            # 2. Restore global settings from latest backup
            global_settings_path = self.claude_dir / 'settings.json'
            if global_settings_path.exists():
                backup_path = self._find_latest_backup('settings')
                if backup_path and backup_path.exists():
                    shutil.copy2(backup_path, global_settings_path)
                    actions_performed.append(f"Restored global settings from: {backup_path}")
                else:
                    # Remove claude-sync hooks manually
                    self._remove_claude_sync_from_settings(global_settings_path)
                    actions_performed.append("Manually removed claude-sync hooks from global settings")
            
            # 3. Clean up project hooks (find all .claude directories)
            project_count = 0
            for claude_dir in Path.home().rglob('.claude'):
                if claude_dir.is_dir() and claude_dir != self.claude_dir:
                    project_hooks_dir = claude_dir / 'hooks'
                    removed_project = self.symlink_manager.remove_hook_symlinks(
                        project_hooks_dir, hooks_source_dir
                    )
                    if removed_project:
                        project_count += 1
            
            if project_count > 0:
                actions_performed.append(f"Cleaned up hooks from {project_count} projects")
            
            # 4. Optionally purge learning data
            if purge_data:
                learning_dir = self.sync_dir / 'learning'
                if learning_dir.exists():
                    # Backup learning data before purging
                    backup_learning_path = self.backup_dir / f"learning_data_{int(time.time())}"
                    shutil.copytree(learning_dir, backup_learning_path, ignore_dangling_symlinks=True)
                    backups_created.append(backup_learning_path)
                    
                    # Remove learning data files (keep Python modules)
                    for data_file in learning_dir.glob("*.enc"):
                        data_file.unlink()
                    for data_file in learning_dir.glob("*.pkl"):
                        data_file.unlink()
                    
                    actions_performed.append("Purged learning data (backed up first)")
            
            success = len(errors) == 0
            message = "Deactivation completed successfully" if success else "Deactivation completed with errors"
            
            return ActivationResult(
                success=success,
                message=message,
                actions_performed=actions_performed,
                backups_created=backups_created,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Unexpected error during deactivation: {str(e)}")
            return ActivationResult(
                success=False,
                message=f"Deactivation failed: {str(e)}",
                actions_performed=actions_performed,
                backups_created=backups_created,
                errors=errors
            )
    
    def get_status(self) -> SystemState:
        """Get current activation status"""
        try:
            # Check if hooks are installed
            hooks_source_dir = self.sync_dir / 'hooks'
            verification_results = self.symlink_manager.verify_symlinks(
                self.hooks_dir, hooks_source_dir
            )
            
            hooks_installed = [
                name for name, success in verification_results.items() 
                if success and name != "directory_exists"
            ]
            
            # Check learning data size
            learning_data_size = 0
            learning_dir = self.sync_dir / 'learning'
            if learning_dir.exists():
                for data_file in learning_dir.glob("*.enc"):
                    learning_data_size += data_file.stat().st_size
                for data_file in learning_dir.glob("*.pkl"):
                    learning_data_size += data_file.stat().st_size
            
            learning_data_size_mb = learning_data_size / (1024 * 1024)
            
            # Check key rotation status
            key_files = list((self.sync_dir / 'keys').glob("*.key")) if (self.sync_dir / 'keys').exists() else []
            last_key_rotation = max([f.stat().st_mtime for f in key_files]) if key_files else 0
            
            # Basic performance metrics
            performance_metrics = {
                "hooks_count": len(hooks_installed),
                "learning_data_mb": learning_data_size_mb,
                "backup_count": len(list(self.backup_dir.glob("*"))) if self.backup_dir.exists() else 0
            }
            
            # Check for issues
            errors = []
            warnings = []
            
            if not hooks_installed:
                warnings.append("No hooks are currently active")
            
            if learning_data_size_mb > 50:  # More than 50MB
                warnings.append(f"Learning data size is large: {learning_data_size_mb:.1f}MB")
            
            is_activated = len(hooks_installed) > 0
            
            return SystemState(
                is_activated=is_activated,
                hooks_installed=hooks_installed,
                learning_data_size_mb=learning_data_size_mb,
                last_key_rotation=last_key_rotation,
                trusted_hosts_count=0,  # TODO: Implement when mesh sync is ready
                performance_metrics=performance_metrics,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            return SystemState(
                is_activated=False,
                hooks_installed=[],
                learning_data_size_mb=0,
                last_key_rotation=0,
                trusted_hosts_count=0,
                performance_metrics={},
                errors=[f"Status check error: {str(e)}"],
                warnings=[]
            )
    
    def verify_installation(self) -> Dict[str, Any]:
        """Verify system is properly installed and configured"""
        verification = {
            "sync_directory": self.sync_dir.exists(),
            "hooks_directory": (self.sync_dir / 'hooks').exists(),
            "templates_directory": (self.sync_dir / 'templates').exists(),
            "learning_directory": (self.sync_dir / 'learning').exists(),
            "global_template": (self.sync_dir / 'templates' / 'settings.global.json').exists(),
            "local_template": (self.sync_dir / 'templates' / 'settings.local.json').exists(),
            "hook_files": [],
            "issues": []
        }
        
        # Check hook files
        hooks_dir = self.sync_dir / 'hooks'
        if hooks_dir.exists():
            verification["hook_files"] = [f.name for f in hooks_dir.glob("*.py")]
        
        # Check for common issues
        if not verification["sync_directory"]:
            verification["issues"].append("claude-sync directory not found")
        
        if not verification["hooks_directory"]:
            verification["issues"].append("hooks directory not found")
        
        if len(verification["hook_files"]) == 0:
            verification["issues"].append("No hook files found")
        
        verification["overall_status"] = len(verification["issues"]) == 0
        
        return verification
    
    def _update_settings_for_symlinks(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update settings template to use symlinks in ~/.claude/hooks/"""
        updated_settings = json.loads(json.dumps(settings))  # Deep copy
        
        if "hooks" in updated_settings:
            for hook_type, hook_configs in updated_settings["hooks"].items():
                for hook_config in hook_configs:
                    for hook in hook_config.get("hooks", []):
                        if hook.get("type") == "command":
                            original_path = hook["command"]
                            if "claude-sync/hooks/" in original_path:
                                # Replace with symlink path
                                hook_filename = Path(original_path).name
                                hook["command"] = f"$HOME/.claude/hooks/{hook_filename}"
        
        return updated_settings
    
    def _update_settings_for_project_symlinks(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update settings template to use project symlinks in .claude/hooks/"""
        updated_settings = json.loads(json.dumps(settings))  # Deep copy
        
        if "hooks" in updated_settings:
            for hook_type, hook_configs in updated_settings["hooks"].items():
                for hook_config in hook_configs:
                    for hook in hook_config.get("hooks", []):
                        if hook.get("type") == "command":
                            original_path = hook["command"]
                            if "claude-sync/hooks/" in original_path:
                                # Replace with project symlink path
                                hook_filename = Path(original_path).name
                                hook["command"] = f"./.claude/hooks/{hook_filename}"
        
        return updated_settings
    
    def _find_latest_backup(self, backup_type: str) -> Optional[Path]:
        """Find the most recent backup of specified type"""
        if not self.backup_dir.exists():
            return None
        
        pattern = f"{backup_type}_backup_*.json"
        backup_files = list(self.backup_dir.glob(pattern))
        
        if not backup_files:
            return None
        
        # Sort by modification time, newest first
        backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return backup_files[0]
    
    def _remove_claude_sync_from_settings(self, settings_path: Path):
        """Remove claude-sync hooks from settings file manually"""
        try:
            with open(settings_path) as f:
                settings = json.load(f)
            
            if "hooks" in settings:
                for hook_type, hook_configs in settings["hooks"].items():
                    # Filter out claude-sync hooks
                    filtered_configs = []
                    for hook_config in hook_configs:
                        filtered_hooks = []
                        for hook in hook_config.get("hooks", []):
                            if hook.get("type") == "command":
                                command = hook["command"]
                                if "claude-sync" not in command and ".claude/hooks/" not in command:
                                    filtered_hooks.append(hook)
                        
                        if filtered_hooks:
                            hook_config["hooks"] = filtered_hooks
                            filtered_configs.append(hook_config)
                    
                    settings["hooks"][hook_type] = filtered_configs
            
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not clean settings file: {e}")


if __name__ == "__main__":
    # Test the activation manager
    manager = ActivationManager()
    
    print("Claude-Sync Activation Manager")
    print("==============================")
    
    # Get current status
    status = manager.get_status()
    print(f"Currently activated: {status.is_activated}")
    print(f"Hooks installed: {len(status.hooks_installed)}")
    if status.hooks_installed:
        for hook in status.hooks_installed:
            print(f"  - {hook}")
    
    # Verify installation
    verification = manager.verify_installation()
    print(f"\nInstallation verification: {'✓' if verification['overall_status'] else '✗'}")
    if verification["issues"]:
        print("Issues found:")
        for issue in verification["issues"]:
            print(f"  - {issue}")