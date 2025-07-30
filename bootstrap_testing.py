#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "psutil>=5.9.0",
#   "typing-extensions>=4.0.0"
# ]
# ///
"""
Bootstrap Testing Framework

Comprehensive testing system for claude-sync bootstrap operations:
- Installation verification
- Activation/deactivation testing
- Settings template validation
- Hook integration testing
- Performance benchmarking
- Cross-platform compatibility
"""

import json
import os
import sys
import time
import tempfile
import shutil
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from interfaces import ActivationResult, SystemState, PerformanceTargets
from activation_manager import ActivationManager, SettingsMerger, SymlinkManager

# ============================================================================
# Bootstrap Test Results
# ============================================================================

@dataclass
class BootstrapTestResult:
    """Result of a bootstrap operation test"""
    test_name: str
    success: bool
    execution_time_ms: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def meets_performance_target(self, target_ms: float = 1000) -> bool:
        """Check if test meets performance target"""
        return self.execution_time_ms <= target_ms

@dataclass
class BootstrapTestSession:
    """Complete bootstrap testing session"""
    session_id: str
    start_time: float
    end_time: float
    platform_info: Dict[str, str]
    test_results: List[BootstrapTestResult] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    @property
    def success_rate(self) -> float:
        if not self.test_results:
            return 0.0
        passed = sum(1 for r in self.test_results if r.success)
        return passed / len(self.test_results)
    
    @property
    def total_tests(self) -> int:
        return len(self.test_results)
    
    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.test_results if r.success)
    
    @property
    def failed_tests(self) -> int:
        return self.total_tests - self.passed_tests

# ============================================================================
# Bootstrap Test Framework
# ============================================================================

class BootstrapTestFramework:
    """Main bootstrap testing framework"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.temp_dir: Optional[Path] = None
        self.original_home = Path.home()
        self.test_home: Optional[Path] = None
        
        # Platform detection
        self.platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'python_version': platform.python_version()
        }
    
    def create_test_environment(self) -> Path:
        """Create isolated test environment mimicking user's home"""
        if self.temp_dir:
            return self.temp_dir
        
        self.temp_dir = Path(tempfile.mkdtemp(prefix="claude_sync_bootstrap_test_"))
        self.test_home = self.temp_dir / "test_home"
        self.test_home.mkdir()
        
        # Create Claude directory structure
        claude_dir = self.test_home / ".claude"
        claude_dir.mkdir()
        
        # Copy source code to test environment
        test_sync_dir = claude_dir / "claude-sync"
        shutil.copytree(self.project_root, test_sync_dir, 
                       ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git', 'test_*'))
        
        return self.temp_dir
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            self.test_home = None
    
    def run_test(self, test_func, test_name: str) -> BootstrapTestResult:
        """Execute a single bootstrap test"""
        start_time = time.perf_counter()
        
        try:
            result = test_func()
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            
            if isinstance(result, tuple):
                success, message = result[0], result[1]
                details = result[2] if len(result) > 2 else {}
                errors = result[3] if len(result) > 3 else []
                warnings = result[4] if len(result) > 4 else []
            elif isinstance(result, bool):
                success, message = result, "Test completed"
                details, errors, warnings = {}, [], []
            else:
                success, message = True, "Test completed successfully"
                details, errors, warnings = {}, [], []
            
            return BootstrapTestResult(
                test_name=test_name,
                success=success,
                execution_time_ms=execution_time_ms,
                message=message,
                details=details,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            return BootstrapTestResult(
                test_name=test_name,
                success=False,
                execution_time_ms=execution_time_ms,
                message=f"Test failed with exception: {str(e)}",
                errors=[str(e)]
            )
    
    def run_all_tests(self) -> BootstrapTestSession:
        """Run comprehensive bootstrap test suite"""
        session_id = f"bootstrap_test_{int(time.time())}"
        start_time = time.time()
        
        print(f"🚀 Starting Bootstrap Test Suite: {session_id}")
        print(f"🖥️  Platform: {self.platform_info['system']} {self.platform_info['release']}")
        print(f"🐍 Python: {self.platform_info['python_version']}")
        print("=" * 80)
        
        # Create test environment
        try:
            self.create_test_environment()
            print(f"🔧 Test environment created: {self.temp_dir}")
            
            # Define test suite
            test_suite = [
                (self.test_activation_manager_initialization, "activation_manager_init"),
                (self.test_settings_merger_functionality, "settings_merger"),
                (self.test_symlink_manager_operations, "symlink_manager"),
                (self.test_global_activation_workflow, "global_activation"),
                (self.test_project_activation_workflow, "project_activation"),
                (self.test_deactivation_workflow, "deactivation"),
                (self.test_backup_and_restore, "backup_restore"),
                (self.test_template_validation, "template_validation"),
                (self.test_cross_platform_compatibility, "cross_platform"),
                (self.test_performance_benchmarks, "performance_benchmarks"),
                (self.test_error_handling, "error_handling"),
                (self.test_bootstrap_script_integration, "bootstrap_script")
            ]
            
            # Run tests
            results = []
            for test_func, test_name in test_suite:
                print(f"🧪 Running {test_name}...", end=" ")
                result = self.run_test(test_func, test_name)
                results.append(result)
                
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"{status} ({result.execution_time_ms:.1f}ms)")
                
                if not result.success:
                    print(f"   └─ {result.message}")
                    for error in result.errors:
                        print(f"      ❌ {error}")
                
                if result.warnings:
                    for warning in result.warnings:
                        print(f"      ⚠️ {warning}")
        
        finally:
            self.cleanup_test_environment()
        
        end_time = time.time()
        
        session = BootstrapTestSession(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            platform_info=self.platform_info,
            test_results=results
        )
        
        self.print_session_summary(session)
        return session
    
    def print_session_summary(self, session: BootstrapTestSession):
        """Print comprehensive test session summary"""
        print("\n" + "=" * 80)
        print(f"📋 BOOTSTRAP TEST SUMMARY: {session.session_id}")
        print("=" * 80)
        
        print(f"⏱️  Duration: {session.duration_ms:.0f}ms")
        print(f"📊 Tests: {session.total_tests} total | {session.passed_tests} passed | {session.failed_tests} failed")
        print(f"✅ Success Rate: {session.success_rate:.1%}")
        
        # Performance analysis
        if session.test_results:
            avg_time = sum(r.execution_time_ms for r in session.test_results) / len(session.test_results)
            max_time = max(r.execution_time_ms for r in session.test_results)
            print(f"⚡ Performance: Avg {avg_time:.1f}ms | Max {max_time:.1f}ms")
            
            # Check performance targets
            slow_tests = [r for r in session.test_results if not r.meets_performance_target()]
            if slow_tests:
                print(f"⚠️  Slow Tests: {len(slow_tests)} exceeded 1000ms target")
        
        # Failed tests
        failed_tests = [r for r in session.test_results if not r.success]
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  • {test.test_name}: {test.message}")
        
        # Overall verdict
        if session.success_rate >= 1.0:
            print(f"\n🎉 ALL TESTS PASSED! Bootstrap system is ready for production.")
        elif session.success_rate >= 0.8:
            print(f"\n⚠️  MOSTLY SUCCESSFUL. {session.failed_tests} issues need attention.")
        else:
            print(f"\n❌ SIGNIFICANT ISSUES FOUND. {session.failed_tests} critical failures.")
        
        print("=" * 80)
    
    # ========================================================================
    # Individual Test Methods
    # ========================================================================
    
    def test_activation_manager_initialization(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test ActivationManager initialization and basic functionality"""
        try:
            # Change to test home for initialization
            os.environ['HOME'] = str(self.test_home)
            
            manager = ActivationManager()
            
            # Verify paths are set correctly
            expected_claude_dir = self.test_home / '.claude'
            expected_sync_dir = expected_claude_dir / 'claude-sync'
            
            success = (
                manager.claude_dir == expected_claude_dir and
                manager.sync_dir == expected_sync_dir and
                hasattr(manager, 'settings_merger') and
                hasattr(manager, 'symlink_manager')
            )
            
            details = {
                'claude_dir': str(manager.claude_dir),
                'sync_dir': str(manager.sync_dir),
                'has_settings_merger': hasattr(manager, 'settings_merger'),
                'has_symlink_manager': hasattr(manager, 'symlink_manager')
            }
            
            message = "ActivationManager initialized successfully" if success else "ActivationManager initialization failed"
            return success, message, details
            
        except Exception as e:
            return False, f"Initialization failed: {str(e)}", {'error': str(e)}
        finally:
            # Restore original home
            os.environ['HOME'] = str(self.original_home)
    
    def test_settings_merger_functionality(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test SettingsMerger capabilities"""
        try:
            merger = SettingsMerger()
            
            # Test data
            user_settings = {
                "permissions": {"allow": ["Read(*)", "Write(*)"]},
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Read",
                            "hooks": [{"type": "command", "command": "/path/to/user/hook.py"}]
                        }
                    ]
                }
            }
            
            claude_sync_settings = {
                "permissions": {"allow": ["Bash(*)"]},
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {"type": "command", "command": "$HOME/.claude/hooks/bash-optimizer.py"},
                                {"type": "command", "command": "$HOME/.claude/hooks/ssh-router.py"}
                            ]
                        }
                    ]
                }
            }
            
            # Test merging
            merged = merger.merge_hook_settings(user_settings, claude_sync_settings)
            
            # Validate results
            success = (
                "permissions" in merged and
                "allow" in merged["permissions"] and
                len(merged["permissions"]["allow"]) == 3 and  # User's 2 + claude-sync's 1
                "hooks" in merged and
                "PreToolUse" in merged["hooks"] and
                len(merged["hooks"]["PreToolUse"]) == 2  # User's Read matcher + claude-sync's Bash matcher
            )
            
            # Test validation
            is_valid = merger.validate_merged_settings(merged)
            success = success and is_valid
            
            details = {
                'merged_permissions_count': len(merged.get("permissions", {}).get("allow", [])),
                'merged_hook_matchers': len(merged.get("hooks", {}).get("PreToolUse", [])),
                'validation_passed': is_valid
            }
            
            message = "Settings merger works correctly" if success else "Settings merger failed validation"
            return success, message, details
            
        except Exception as e:
            return False, f"Settings merger test failed: {str(e)}", {'error': str(e)}
    
    def test_symlink_manager_operations(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test SymlinkManager symlink operations"""
        try:
            manager = SymlinkManager()
            
            # Create test source directory with hook files
            source_dir = self.temp_dir / "source_hooks"
            source_dir.mkdir()
            
            # Create test hook files
            test_hooks = ["test_hook1.py", "test_hook2.py", "test_hook3.py"]
            for hook_name in test_hooks:
                hook_file = source_dir / hook_name
                hook_file.write_text(f"#!/usr/bin/env python3\n# Test hook: {hook_name}\n")
                hook_file.chmod(0o755)
            
            # Create target directory
            target_dir = self.temp_dir / "target_hooks"
            target_dir.mkdir()
            
            # Test symlink creation
            created_symlinks = manager.create_hook_symlinks(source_dir, target_dir)
            
            # Verify symlinks
            verification_results = manager.verify_symlinks(target_dir, source_dir)
            
            # Test symlink removal
            removed_symlinks = manager.remove_hook_symlinks(target_dir, source_dir)
            
            # Validate results
            success = (
                len(created_symlinks) == len(test_hooks) and
                all(verification_results.get(hook) for hook in test_hooks) and
                len(removed_symlinks) == len(test_hooks)
            )
            
            details = {
                'created_symlinks': len(created_symlinks),
                'expected_symlinks': len(test_hooks),
                'verification_passed': all(verification_results.get(hook, False) for hook in test_hooks),
                'removed_symlinks': len(removed_symlinks)
            }
            
            message = "Symlink manager operations successful" if success else "Symlink manager operations failed"
            return success, message, details
            
        except Exception as e:
            return False, f"Symlink manager test failed: {str(e)}", {'error': str(e)}
    
    def test_global_activation_workflow(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test complete global activation workflow"""
        try:
            # Set up test environment
            os.environ['HOME'] = str(self.test_home)
            
            manager = ActivationManager()
            
            # Create templates directory and files
            templates_dir = manager.sync_dir / 'templates'
            templates_dir.mkdir(parents=True, exist_ok=True)
            
            # Create global template
            global_template = {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {"type": "command", "command": "$HOME/.claude/hooks/test-hook.py"}
                            ]
                        }
                    ]
                }
            }
            
            (templates_dir / 'settings.global.json').write_text(json.dumps(global_template, indent=2))
            
            # Create hooks directory with test hook
            hooks_dir = manager.sync_dir / 'hooks'
            hooks_dir.mkdir(parents=True, exist_ok=True)
            (hooks_dir / 'test-hook.py').write_text("#!/usr/bin/env python3\nprint('test hook')\n")
            
            # Test activation
            result = manager.activate_global()
            
            # Verify activation
            status = manager.get_status()
            
            success = (
                result.success and
                len(result.errors) == 0 and
                status.is_activated and
                len(status.hooks_installed) > 0
            )
            
            details = {
                'activation_success': result.success,
                'activation_errors': result.errors,
                'actions_performed': result.actions_performed,
                'hooks_installed': status.hooks_installed,
                'is_activated': status.is_activated
            }
            
            message = "Global activation workflow successful" if success else f"Global activation failed: {result.message}"
            return success, message, details
            
        except Exception as e:
            return False, f"Global activation test failed: {str(e)}", {'error': str(e)}
        finally:
            os.environ['HOME'] = str(self.original_home)
    
    def test_project_activation_workflow(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test project-specific activation workflow"""
        try:
            os.environ['HOME'] = str(self.test_home)
            
            manager = ActivationManager()
            
            # Create templates and hooks as in global test
            templates_dir = manager.sync_dir / 'templates'
            templates_dir.mkdir(parents=True, exist_ok=True)
            
            local_template = {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {"type": "command", "command": "./.claude/hooks/project-hook.py"}
                            ]
                        }
                    ]
                }
            }
            
            (templates_dir / 'settings.local.json').write_text(json.dumps(local_template, indent=2))
            
            hooks_dir = manager.sync_dir / 'hooks'
            hooks_dir.mkdir(parents=True, exist_ok=True)
            (hooks_dir / 'project-hook.py').write_text("#!/usr/bin/env python3\nprint('project hook')\n")
            
            # Create test project
            test_project = self.temp_dir / "test_project"
            test_project.mkdir()
            
            # Test project activation
            result = manager.activate_project(test_project)
            
            # Verify project activation
            project_claude_dir = test_project / '.claude'
            project_hooks_dir = project_claude_dir / 'hooks'
            project_settings = project_claude_dir / 'settings.json'
            
            success = (
                result.success and
                len(result.errors) == 0 and
                project_claude_dir.exists() and
                project_hooks_dir.exists() and
                project_settings.exists() and
                any(hook_file.is_symlink() for hook_file in project_hooks_dir.glob("*.py"))
            )
            
            details = {
                'activation_success': result.success,
                'activation_errors': result.errors,
                'project_dir_created': project_claude_dir.exists(),
                'hooks_dir_created': project_hooks_dir.exists(),
                'settings_created': project_settings.exists(),
                'symlinks_created': sum(1 for f in project_hooks_dir.glob("*.py") if f.is_symlink())
            }
            
            message = "Project activation workflow successful" if success else f"Project activation failed: {result.message}"
            return success, message, details
            
        except Exception as e:
            return False, f"Project activation test failed: {str(e)}", {'error': str(e)}
        finally:
            os.environ['HOME'] = str(self.original_home)
    
    def test_deactivation_workflow(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test deactivation workflow"""
        try:
            os.environ['HOME'] = str(self.test_home)
            
            manager = ActivationManager()
            
            # Set up for activation first (simplified)
            templates_dir = manager.sync_dir / 'templates'
            templates_dir.mkdir(parents=True, exist_ok=True)
            
            template = {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "$HOME/.claude/hooks/test.py"}]}]}}
            (templates_dir / 'settings.global.json').write_text(json.dumps(template))
            
            hooks_dir = manager.sync_dir / 'hooks'
            hooks_dir.mkdir(parents=True, exist_ok=True)
            (hooks_dir / 'test.py').write_text("#!/usr/bin/env python3\npass\n")
            
            # Activate first
            activate_result = manager.activate_global()
            
            if not activate_result.success:
                return False, "Could not activate for deactivation test", {'activation_failed': True}
            
            # Test deactivation
            deactivate_result = manager.deactivate()
            
            # Verify deactivation
            status = manager.get_status()
            
            success = (
                deactivate_result.success and
                len(deactivate_result.errors) == 0 and
                not status.is_activated and
                len(status.hooks_installed) == 0
            )
            
            details = {
                'deactivation_success': deactivate_result.success,
                'deactivation_errors': deactivate_result.errors,
                'actions_performed': deactivate_result.actions_performed,
                'is_deactivated': not status.is_activated,
                'hooks_removed': len(status.hooks_installed) == 0
            }
            
            message = "Deactivation workflow successful" if success else f"Deactivation failed: {deactivate_result.message}"
            return success, message, details
            
        except Exception as e:
            return False, f"Deactivation test failed: {str(e)}", {'error': str(e)}
        finally:
            os.environ['HOME'] = str(self.original_home)
    
    def test_backup_and_restore(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test backup and restore functionality"""
        try:
            merger = SettingsMerger()
            
            # Create test settings file
            test_settings_dir = self.temp_dir / "settings_test"
            test_settings_dir.mkdir()
            test_settings_file = test_settings_dir / "settings.json"
            
            original_settings = {"test": "original_value", "permissions": {"allow": ["Read(*)"]}}
            test_settings_file.write_text(json.dumps(original_settings, indent=2))
            
            # Create backup directory
            backup_dir = self.temp_dir / "backups"
            
            # Test backup creation
            backup_path = merger.backup_user_settings(test_settings_file, backup_dir)
            
            # Modify original file
            modified_settings = {"test": "modified_value", "permissions": {"allow": ["Write(*)"]}}
            test_settings_file.write_text(json.dumps(modified_settings, indent=2))
            
            # Test restore
            if backup_path:
                shutil.copy2(backup_path, test_settings_file)
                
                # Verify restore
                with open(test_settings_file) as f:
                    restored_settings = json.load(f)
                
                restore_success = restored_settings == original_settings
            else:
                restore_success = False
            
            success = (
                backup_path is not None and
                backup_path.exists() and
                restore_success
            )
            
            details = {
                'backup_created': backup_path is not None,
                'backup_exists': backup_path.exists() if backup_path else False,
                'restore_successful': restore_success
            }
            
            message = "Backup and restore successful" if success else "Backup and restore failed"
            return success, message, details
            
        except Exception as e:
            return False, f"Backup/restore test failed: {str(e)}", {'error': str(e)}
    
    def test_template_validation(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test settings template validation"""
        try:
            # Test existing templates
            templates_dir = self.project_root / 'templates'
            
            results = {}
            errors = []
            
            # Test each template
            template_files = ['settings.global.json', 'settings.local.json']
            
            for template_file in template_files:
                template_path = templates_dir / template_file
                if template_path.exists():
                    try:
                        with open(template_path) as f:
                            template_data = json.load(f)
                        
                        # Validate structure
                        has_hooks = 'hooks' in template_data
                        has_permissions = 'permissions' in template_data
                        
                        if has_hooks and isinstance(template_data['hooks'], dict):
                            hook_types = template_data['hooks'].keys()
                            valid_hook_structure = all(
                                isinstance(template_data['hooks'][hook_type], list)
                                for hook_type in hook_types
                            )
                        else:
                            valid_hook_structure = False
                        
                        results[template_file] = {
                            'exists': True,
                            'valid_json': True,
                            'has_hooks': has_hooks,
                            'has_permissions': has_permissions,
                            'valid_structure': valid_hook_structure
                        }
                        
                    except json.JSONDecodeError as e:
                        results[template_file] = {'exists': True, 'valid_json': False, 'error': str(e)}
                        errors.append(f"{template_file}: Invalid JSON - {str(e)}")
                else:
                    results[template_file] = {'exists': False}
                    errors.append(f"{template_file}: Template file missing")
            
            success = (
                all(results[t].get('exists', False) for t in template_files) and
                all(results[t].get('valid_json', False) for t in template_files) and
                all(results[t].get('valid_structure', False) for t in template_files)
            )
            
            details = {'template_results': results}
            
            message = "Template validation successful" if success else f"Template validation failed: {errors}"
            return success, message, details, errors
            
        except Exception as e:
            return False, f"Template validation test failed: {str(e)}", {'error': str(e)}
    
    def test_cross_platform_compatibility(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test cross-platform compatibility"""
        try:
            platform_system = platform.system()
            
            # Test path handling
            test_paths = [
                Path.home() / '.claude',
                Path('/tmp') / 'test' if platform_system != 'Windows' else Path('C:\\temp') / 'test',
                Path.cwd() / '.claude' / 'settings.json'
            ]
            
            path_results = {}
            for i, test_path in enumerate(test_paths):
                try:
                    # Test path operations
                    path_str = str(test_path)
                    path_absolute = test_path.absolute()
                    path_exists = test_path.exists()
                    
                    path_results[f'path_{i}'] = {
                        'string_conversion': bool(path_str),
                        'absolute_conversion': bool(str(path_absolute)),
                        'exists_check': isinstance(path_exists, bool)
                    }
                except Exception as e:
                    path_results[f'path_{i}'] = {'error': str(e)}
            
            # Test platform-specific features
            platform_features = {
                'symlink_support': hasattr(os, 'symlink'),
                'chmod_support': hasattr(os, 'chmod'),
                'home_detection': bool(str(Path.home())),
                'temp_dir_access': os.access(tempfile.gettempdir(), os.W_OK)
            }
            
            success = (
                all(all(r.values()) if isinstance(r, dict) and 'error' not in r else False 
                    for r in path_results.values()) and
                platform_features['symlink_support'] and
                platform_features['home_detection'] and
                platform_features['temp_dir_access']
            )
            
            details = {
                'platform': platform_system,
                'path_results': path_results,
                'platform_features': platform_features
            }
            
            warnings = []
            if not platform_features['chmod_support']:
                warnings.append("chmod not available on this platform")
            
            message = f"Cross-platform compatibility verified for {platform_system}" if success else "Cross-platform issues detected"
            return success, message, details, [], warnings
            
        except Exception as e:
            return False, f"Cross-platform test failed: {str(e)}", {'error': str(e)}
    
    def test_performance_benchmarks(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test system performance against targets"""
        try:
            os.environ['HOME'] = str(self.test_home)
            
            # Benchmark activation manager operations
            manager = ActivationManager()
            
            # Setup for benchmarking
            templates_dir = manager.sync_dir / 'templates'
            templates_dir.mkdir(parents=True, exist_ok=True)
            
            template = {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "$HOME/.claude/hooks/perf-test.py"}]}]}}
            (templates_dir / 'settings.global.json').write_text(json.dumps(template))
            
            hooks_dir = manager.sync_dir / 'hooks'
            hooks_dir.mkdir(parents=True, exist_ok=True)
            (hooks_dir / 'perf-test.py').write_text("#!/usr/bin/env python3\npass\n")
            
            benchmarks = {}
            
            # Benchmark initialization
            start_time = time.perf_counter()
            test_manager = ActivationManager()
            init_time = (time.perf_counter() - start_time) * 1000
            benchmarks['initialization_ms'] = init_time
            
            # Benchmark status check
            start_time = time.perf_counter()
            status = manager.get_status()
            status_time = (time.perf_counter() - start_time) * 1000
            benchmarks['status_check_ms'] = status_time
            
            # Benchmark settings merger
            merger = SettingsMerger()
            user_settings = {"hooks": {"PreToolUse": []}}
            claude_settings = {"hooks": {"PreToolUse": [{"matcher": "Test", "hooks": []}]}}
            
            start_time = time.perf_counter()
            merged = merger.merge_hook_settings(user_settings, claude_settings)
            merge_time = (time.perf_counter() - start_time) * 1000
            benchmarks['settings_merge_ms'] = merge_time
            
            # Check against performance targets (define reasonable targets for bootstrap operations)
            targets = {
                'initialization_ms': 100,    # ActivationManager init should be fast
                'status_check_ms': 50,       # Status check should be very fast
                'settings_merge_ms': 10      # Settings merge should be nearly instant
            }
            
            performance_results = {}
            for operation, time_ms in benchmarks.items():
                target = targets.get(operation, 100)
                meets_target = time_ms <= target
                performance_results[operation] = {
                    'time_ms': time_ms,
                    'target_ms': target,
                    'meets_target': meets_target
                }
            
            success = all(r['meets_target'] for r in performance_results.values())
            
            details = {
                'benchmarks': benchmarks,
                'performance_results': performance_results,
                'overall_performance': 'good' if success else 'needs_optimization'
            }
            
            message = "Performance benchmarks passed" if success else "Performance targets not met"
            return success, message, details
            
        except Exception as e:
            return False, f"Performance benchmark test failed: {str(e)}", {'error': str(e)}
        finally:
            os.environ['HOME'] = str(self.original_home)
    
    def test_error_handling(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test error handling and recovery"""
        try:
            os.environ['HOME'] = str(self.test_home)
            
            manager = ActivationManager()
            
            error_scenarios = {}
            
            # Test activation without templates
            try:
                result = manager.activate_global()
                error_scenarios['missing_template'] = {
                    'handled_gracefully': not result.success and len(result.errors) > 0,
                    'error_message': result.message
                }
            except Exception as e:
                error_scenarios['missing_template'] = {
                    'handled_gracefully': False,
                    'exception': str(e)
                }
            
            # Test activation with invalid template
            templates_dir = manager.sync_dir / 'templates'
            templates_dir.mkdir(parents=True, exist_ok=True)
            (templates_dir / 'settings.global.json').write_text('{"invalid": json}')  # Invalid JSON
            
            try:
                result = manager.activate_global()
                error_scenarios['invalid_template'] = {
                    'handled_gracefully': not result.success and len(result.errors) > 0,
                    'error_message': result.message
                }
            except Exception as e:
                error_scenarios['invalid_template'] = {
                    'handled_gracefully': False,
                    'exception': str(e)
                }
            
            # Test deactivation when not activated
            try:
                result = manager.deactivate()
                error_scenarios['deactivate_not_activated'] = {
                    'handled_gracefully': True,  # Should succeed gracefully
                    'result_success': result.success
                }
            except Exception as e:
                error_scenarios['deactivate_not_activated'] = {
                    'handled_gracefully': False,
                    'exception': str(e)
                }
            
            # Evaluate error handling
            success = all(
                scenario.get('handled_gracefully', False) 
                for scenario in error_scenarios.values()
            )
            
            details = {'error_scenarios': error_scenarios}
            
            message = "Error handling works correctly" if success else "Error handling needs improvement"
            return success, message, details
            
        except Exception as e:
            return False, f"Error handling test failed: {str(e)}", {'error': str(e)}
        finally:
            os.environ['HOME'] = str(self.original_home)
    
    def test_bootstrap_script_integration(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test integration with bootstrap.sh script"""
        try:
            bootstrap_script = self.project_root / 'bootstrap.sh'
            
            integration_results = {}
            
            # Test script exists and is executable
            integration_results['script_exists'] = bootstrap_script.exists()
            integration_results['script_executable'] = os.access(bootstrap_script, os.X_OK)
            
            # Test script can be invoked (basic syntax check)
            try:
                result = subprocess.run([str(bootstrap_script), '--help'], 
                                      capture_output=True, text=True, timeout=10)
                integration_results['script_help'] = {
                    'exit_code': result.returncode,
                    'has_output': len(result.stdout) > 0,
                    'no_errors': len(result.stderr) == 0 or 'error' not in result.stderr.lower()
                }
            except subprocess.TimeoutExpired:
                integration_results['script_help'] = {'timeout': True}
            except Exception as e:
                integration_results['script_help'] = {'error': str(e)}
            
            # Check for key commands in script
            script_content = bootstrap_script.read_text()
            expected_commands = ['activate', 'deactivate', 'test', 'diagnostics', 'rollback']
            integration_results['has_commands'] = {
                cmd: cmd in script_content for cmd in expected_commands
            }
            
            success = (
                integration_results['script_exists'] and
                integration_results['script_executable'] and
                integration_results.get('script_help', {}).get('exit_code') == 0 and
                all(integration_results['has_commands'].values())
            )
            
            details = integration_results
            
            message = "Bootstrap script integration verified" if success else "Bootstrap script integration issues found"
            return success, message, details
            
        except Exception as e:
            return False, f"Bootstrap script integration test failed: {str(e)}", {'error': str(e)}


if __name__ == "__main__":
    # Run bootstrap testing framework
    framework = BootstrapTestFramework()
    session = framework.run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if session.success_rate == 1.0 else 1 if session.success_rate >= 0.8 else 2
    print(f"\n🏁 Bootstrap testing complete. Exit code: {exit_code}")
    exit(exit_code)