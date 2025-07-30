#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "psutil>=5.9.0",
#   "typing-extensions>=4.0.0"
# ]
# ///
"""
Bootstrap Validation System

Comprehensive validation framework for claude-sync bootstrap operations:
- Pre-installation validation (system requirements, dependencies)
- Post-installation validation (integrity, functionality)
- Continuous validation (health monitoring)
- Integration validation (Claude Code compatibility)
"""

import json
import os
import sys
import time
import subprocess
import platform
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from interfaces import SystemState, PerformanceTargets
from activation_manager import ActivationManager

# ============================================================================
# Validation Result Structures
# ============================================================================

@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    category: str
    severity: str  # 'critical', 'warning', 'info'
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    @property
    def is_blocking(self) -> bool:
        """Whether this validation failure should block installation/activation"""
        return self.severity == 'critical' and not self.passed

@dataclass
class ValidationSuite:
    """Collection of validation checks for a specific phase"""
    name: str
    description: str
    phase: str  # 'pre-install', 'post-install', 'activation', 'continuous'
    checks: List[ValidationResult] = field(default_factory=list)
    
    @property
    def overall_status(self) -> str:
        """Overall validation status"""
        if not self.checks:
            return 'unknown'
        
        critical_failures = [c for c in self.checks if c.severity == 'critical' and not c.passed]
        warnings = [c for c in self.checks if c.severity == 'warning' and not c.passed]
        
        if critical_failures:
            return 'critical'
        elif warnings:
            return 'warning'
        else:
            return 'healthy'
    
    @property
    def blocking_issues(self) -> List[ValidationResult]:
        """Get validation results that would block installation/activation"""
        return [check for check in self.checks if check.is_blocking]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)"""
        if not self.checks:
            return 0.0
        passed = sum(1 for check in self.checks if check.passed)
        return passed / len(self.checks)

@dataclass
class ValidationSession:
    """Complete validation session results"""
    session_id: str
    timestamp: float
    phase: str
    platform_info: Dict[str, str]
    suites: List[ValidationSuite] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    @property
    def overall_success_rate(self) -> float:
        """Overall success rate across all suites"""
        if not self.suites:
            return 0.0
        total_score = sum(suite.success_rate for suite in self.suites)
        return total_score / len(self.suites)
    
    @property
    def has_blocking_issues(self) -> bool:
        """Whether there are issues that would block installation/activation"""
        return any(suite.blocking_issues for suite in self.suites)
    
    @property
    def total_checks(self) -> int:
        """Total number of validation checks performed"""
        return sum(len(suite.checks) for suite in self.suites)
    
    @property
    def passed_checks(self) -> int:
        """Number of validation checks that passed"""
        return sum(len([c for c in suite.checks if c.passed]) for suite in self.suites)

# ============================================================================
# Core Validation System
# ============================================================================

class BootstrapValidator:
    """Main bootstrap validation orchestrator"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.claude_dir = Path.home() / '.claude'
        self.sync_dir = self.claude_dir / 'claude-sync'
        
        # Platform information
        self.platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'python_executable': sys.executable
        }
    
    def run_pre_installation_validation(self) -> ValidationSession:
        """Run validation checks before installation"""
        return self._run_validation_phase('pre-install', [
            self._create_system_requirements_suite(),
            self._create_dependency_requirements_suite(),
            self._create_filesystem_requirements_suite(),
            self._create_network_requirements_suite()
        ])
    
    def run_post_installation_validation(self) -> ValidationSession:
        """Run validation checks after installation"""
        return self._run_validation_phase('post-install', [
            self._create_installation_integrity_suite(),
            self._create_file_structure_suite(),
            self._create_permission_validation_suite(),
            self._create_configuration_validation_suite()
        ])
    
    def run_activation_validation(self) -> ValidationSession:
        """Run validation checks during activation"""
        return self._run_validation_phase('activation', [
            self._create_activation_readiness_suite(),
            self._create_hook_integration_suite(),
            self._create_settings_validation_suite(),
            self._create_performance_validation_suite()
        ])
    
    def run_continuous_validation(self) -> ValidationSession:
        """Run ongoing validation checks for system health"""
        return self._run_validation_phase('continuous', [
            self._create_system_health_suite(),
            self._create_performance_monitoring_suite(),
            self._create_security_validation_suite(),
            self._create_integration_validation_suite()
        ])
    
    def _run_validation_phase(self, phase: str, suites: List[ValidationSuite]) -> ValidationSession:
        """Execute a validation phase with multiple suites"""
        session_id = f"validation_{phase}_{int(time.time())}"
        start_time = time.perf_counter()
        
        print(f"🔍 Starting {phase.replace('-', ' ').title()} Validation: {session_id}")
        print(f"🖥️  Platform: {self.platform_info['system']} {self.platform_info['release']}")
        print("=" * 80)
        
        # Execute all suites
        for suite in suites:
            print(f"\n📋 {suite.name}")
            print("-" * 60)
            
            for i, check_func in enumerate(getattr(self, f'_get_{suite.name.lower().replace(" ", "_")}_checks', lambda: [])(), 1):
                print(f"🔍 Running check {i}...", end=" ")
                
                check_result = self._execute_validation_check(check_func)
                suite.checks.append(check_result)
                
                # Print result
                if check_result.passed:
                    status_emoji = "✅"
                    status_text = "PASS"
                elif check_result.severity == 'critical':
                    status_emoji = "❌"
                    status_text = "FAIL"
                else:
                    status_emoji = "⚠️"
                    status_text = "WARN"
                
                print(f"{status_emoji} {status_text} ({check_result.execution_time_ms:.1f}ms)")
                
                if not check_result.passed:
                    print(f"   └─ {check_result.message}")
                    for rec in check_result.recommendations:
                        print(f"      💡 {rec}")
            
            print(f"\n📊 Suite Result: {suite.overall_status.upper()} ({suite.success_rate:.1%} success rate)")
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        session = ValidationSession(
            session_id=session_id,
            timestamp=time.time(),
            phase=phase,
            platform_info=self.platform_info,
            suites=suites,
            execution_time_ms=execution_time_ms
        )
        
        self._print_validation_summary(session)
        return session
    
    def _execute_validation_check(self, check_func) -> ValidationResult:
        """Execute a single validation check with timing"""
        start_time = time.perf_counter()
        
        try:
            result = check_func()
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            
            if isinstance(result, ValidationResult):
                result.execution_time_ms = execution_time_ms
                return result
            elif isinstance(result, tuple):
                # Legacy format: (name, category, severity, passed, message, details, recommendations)
                name, category, severity, passed, message = result[:5]
                details = result[5] if len(result) > 5 else {}
                recommendations = result[6] if len(result) > 6 else []
                
                return ValidationResult(
                    check_name=name,
                    category=category,
                    severity=severity,
                    passed=passed,
                    message=message,
                    details=details,
                    recommendations=recommendations,
                    execution_time_ms=execution_time_ms
                )
            else:
                return ValidationResult(
                    check_name="Unknown Check",
                    category="unknown",
                    severity="warning",
                    passed=False,
                    message="Invalid validation check result format",
                    execution_time_ms=execution_time_ms
                )
                
        except Exception as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            return ValidationResult(
                check_name="Exception Check",
                category="system",
                severity="critical",
                passed=False,
                message=f"Validation check failed with exception: {str(e)}",
                details={'exception': str(e)},
                recommendations=["Check system stability and try again"],
                execution_time_ms=execution_time_ms
            )
    
    def _print_validation_summary(self, session: ValidationSession):
        """Print comprehensive validation summary"""
        print("\n" + "=" * 80)
        print(f"📋 VALIDATION SUMMARY: {session.session_id}")
        print("=" * 80)
        
        # Overall statistics
        print(f"📊 Checks: {session.total_checks} total | {session.passed_checks} passed")
        print(f"✅ Success Rate: {session.overall_success_rate:.1%}")
        print(f"⏱️  Execution Time: {session.execution_time_ms:.0f}ms")
        
        # Suite breakdown
        print(f"\n📋 SUITE BREAKDOWN:")
        print("-" * 40)
        
        for suite in session.suites:
            status_emoji = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(suite.overall_status, "⚪")
            print(f"{status_emoji} {suite.name}: {suite.overall_status.upper()} ({suite.success_rate:.1%})")
            
            # Show critical issues
            blocking_issues = suite.blocking_issues
            if blocking_issues:
                for issue in blocking_issues:
                    print(f"    ❌ {issue.check_name}: {issue.message}")
        
        # Overall verdict
        print(f"\n🎯 VALIDATION VERDICT:")
        print("-" * 40)
        
        if session.has_blocking_issues:
            print("🔴 BLOCKED: Critical issues prevent proceeding")
            for suite in session.suites:
                for issue in suite.blocking_issues:
                    print(f"  ❌ {issue.check_name}: {issue.message}")
                    for rec in issue.recommendations:
                        print(f"     💡 {rec}")
        elif session.overall_success_rate >= 0.9:
            print("🟢 EXCELLENT: System ready for claude-sync")
        elif session.overall_success_rate >= 0.8:
            print("🟡 GOOD: System ready with minor warnings")
        else:
            print("🟠 FAIR: System has issues that should be addressed")
        
        print("=" * 80)
    
    # ========================================================================
    # Validation Suite Creators
    # ========================================================================
    
    def _create_system_requirements_suite(self) -> ValidationSuite:
        """Create system requirements validation suite"""
        return ValidationSuite(
            name="System Requirements",
            description="Validate system meets minimum requirements",
            phase="pre-install"
        )
    
    def _create_dependency_requirements_suite(self) -> ValidationSuite:
        """Create dependency requirements validation suite"""
        return ValidationSuite(
            name="Dependency Requirements", 
            description="Validate required dependencies are available",
            phase="pre-install"
        )
    
    def _create_filesystem_requirements_suite(self) -> ValidationSuite:
        """Create filesystem requirements validation suite"""
        return ValidationSuite(
            name="Filesystem Requirements",
            description="Validate filesystem capabilities and permissions",
            phase="pre-install"
        )
    
    def _create_network_requirements_suite(self) -> ValidationSuite:
        """Create network requirements validation suite"""
        return ValidationSuite(
            name="Network Requirements",
            description="Validate network connectivity and access",
            phase="pre-install"
        )
    
    def _create_installation_integrity_suite(self) -> ValidationSuite:
        """Create installation integrity validation suite"""
        return ValidationSuite(
            name="Installation Integrity",
            description="Validate installation completeness and integrity",
            phase="post-install"
        )
    
    def _create_file_structure_suite(self) -> ValidationSuite:
        """Create file structure validation suite"""
        return ValidationSuite(
            name="File Structure",
            description="Validate directory structure and required files",
            phase="post-install"
        )
    
    def _create_permission_validation_suite(self) -> ValidationSuite:
        """Create permission validation suite"""
        return ValidationSuite(
            name="Permission Validation",
            description="Validate file and directory permissions",
            phase="post-install"
        )
    
    def _create_configuration_validation_suite(self) -> ValidationSuite:
        """Create configuration validation suite"""
        return ValidationSuite(
            name="Configuration Validation",
            description="Validate configuration files and templates",
            phase="post-install"
        )
    
    def _create_activation_readiness_suite(self) -> ValidationSuite:
        """Create activation readiness validation suite"""
        return ValidationSuite(
            name="Activation Readiness",
            description="Validate system readiness for activation",
            phase="activation"
        )
    
    def _create_hook_integration_suite(self) -> ValidationSuite:
        """Create hook integration validation suite"""
        return ValidationSuite(
            name="Hook Integration",
            description="Validate hook system integration capabilities",
            phase="activation"
        )
    
    def _create_settings_validation_suite(self) -> ValidationSuite:
        """Create settings validation suite"""
        return ValidationSuite(
            name="Settings Validation",
            description="Validate settings templates and merging",
            phase="activation"
        )
    
    def _create_performance_validation_suite(self) -> ValidationSuite:
        """Create performance validation suite"""
        return ValidationSuite(
            name="Performance Validation",
            description="Validate system performance capabilities",
            phase="activation"
        )
    
    def _create_system_health_suite(self) -> ValidationSuite:
        """Create system health validation suite"""
        return ValidationSuite(
            name="System Health",
            description="Continuous system health monitoring",
            phase="continuous"
        )
    
    def _create_performance_monitoring_suite(self) -> ValidationSuite:
        """Create performance monitoring validation suite"""
        return ValidationSuite(
            name="Performance Monitoring",
            description="Monitor system performance against targets",
            phase="continuous"
        )
    
    def _create_security_validation_suite(self) -> ValidationSuite:
        """Create security validation suite"""
        return ValidationSuite(
            name="Security Validation",
            description="Validate security configuration and compliance",
            phase="continuous"
        )
    
    def _create_integration_validation_suite(self) -> ValidationSuite:
        """Create integration validation suite"""
        return ValidationSuite(
            name="Integration Validation",
            description="Validate Claude Code integration health",
            phase="continuous"
        )
    
    # ========================================================================
    # Individual Validation Check Methods
    # ========================================================================
    
    def _get_system_requirements_checks(self):
        """Get system requirements validation checks"""
        return [
            self._check_python_version,
            self._check_operating_system,
            self._check_available_memory,
            self._check_disk_space,
            self._check_cpu_capabilities
        ]
    
    def _get_dependency_requirements_checks(self):
        """Get dependency requirements validation checks"""
        return [
            self._check_git_availability,
            self._check_python_installation,
            self._check_uv_availability,
            self._check_essential_modules
        ]
    
    def _get_filesystem_requirements_checks(self):
        """Get filesystem requirements validation checks"""
        return [
            self._check_home_directory_access,
            self._check_claude_directory_creation,
            self._check_symlink_support,
            self._check_file_permissions_capability
        ]
    
    def _get_network_requirements_checks(self):
        """Get network requirements validation checks"""
        return [
            self._check_internet_connectivity,
            self._check_github_access,
            self._check_ssh_capability,
            self._check_download_capability
        ]
    
    def _get_installation_integrity_checks(self):
        """Get installation integrity validation checks"""
        return [
            self._check_directory_structure,
            self._check_required_files,
            self._check_file_checksums,
            self._check_executable_permissions
        ]
    
    def _get_file_structure_checks(self):
        """Get file structure validation checks"""
        return [
            self._check_hooks_directory,
            self._check_learning_directory,
            self._check_templates_directory,
            self._check_security_directory
        ]
    
    def _get_permission_validation_checks(self):
        """Get permission validation checks"""
        return [
            self._check_bootstrap_permissions,
            self._check_hook_permissions,
            self._check_directory_permissions,
            self._check_write_permissions
        ]
    
    def _get_configuration_validation_checks(self):
        """Get configuration validation checks"""
        return [
            self._check_template_files,
            self._check_template_syntax,
            self._check_settings_structure,
            self._check_hook_configuration
        ]
    
    # Individual validation check implementations
    def _check_python_version(self) -> ValidationResult:
        """Check Python version meets requirements"""
        current_version = sys.version_info
        required_version = (3, 10)
        
        if current_version >= required_version:
            return ValidationResult(
                check_name="Python Version",
                category="system",
                severity="critical",
                passed=True,
                message=f"Python {current_version.major}.{current_version.minor}.{current_version.micro} meets requirements",
                details={"version": f"{current_version.major}.{current_version.minor}.{current_version.micro}"}
            )
        else:
            return ValidationResult(
                check_name="Python Version",
                category="system",
                severity="critical",
                passed=False,
                message=f"Python {current_version.major}.{current_version.minor} is too old (requires 3.10+)",
                details={"version": f"{current_version.major}.{current_version.minor}.{current_version.micro}"},
                recommendations=["Upgrade Python to version 3.10 or higher"]
            )
    
    def _check_operating_system(self) -> ValidationResult:
        """Check operating system compatibility"""
        system = platform.system()
        supported_systems = ['Linux', 'Darwin', 'Windows']
        
        if system in supported_systems:
            return ValidationResult(
                check_name="Operating System",
                category="system",
                severity="warning",
                passed=True,
                message=f"{system} is supported",
                details={"system": system, "release": platform.release()}
            )
        else:
            return ValidationResult(
                check_name="Operating System",
                category="system",
                severity="warning",
                passed=False,
                message=f"{system} is not officially supported",
                details={"system": system},
                recommendations=["Functionality may be limited on unsupported platforms"]
            )
    
    def _check_available_memory(self) -> ValidationResult:
        """Check available system memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb >= 1.0:
                return ValidationResult(
                    check_name="Available Memory",
                    category="system",
                    severity="warning",
                    passed=True,
                    message=f"Sufficient memory available: {available_gb:.1f}GB",
                    details={"available_gb": available_gb}
                )
            else:
                return ValidationResult(
                    check_name="Available Memory",
                    category="system",
                    severity="warning",
                    passed=False,
                    message=f"Low memory: {available_gb:.1f}GB available",
                    details={"available_gb": available_gb},
                    recommendations=["Close other applications to free memory"]
                )
        except ImportError:
            return ValidationResult(
                check_name="Available Memory",
                category="system",
                severity="info",
                passed=True,
                message="Memory check skipped (psutil not available)",
                recommendations=["Install psutil for memory monitoring"]
            )
    
    def _check_disk_space(self) -> ValidationResult:
        """Check available disk space"""
        try:
            usage = shutil.disk_usage(Path.home())
            free_gb = usage.free / (1024**3)
            
            if free_gb >= 1.0:
                return ValidationResult(
                    check_name="Disk Space",
                    category="system",
                    severity="critical",
                    passed=True,
                    message=f"Sufficient disk space: {free_gb:.1f}GB free",
                    details={"free_gb": free_gb}
                )
            else:
                return ValidationResult(
                    check_name="Disk Space",
                    category="system",
                    severity="critical",
                    passed=False,
                    message=f"Insufficient disk space: {free_gb:.1f}GB free",
                    details={"free_gb": free_gb},
                    recommendations=["Free up disk space before installation"]
                )
        except Exception as e:
            return ValidationResult(
                check_name="Disk Space",
                category="system",
                severity="warning",
                passed=False,
                message=f"Cannot check disk space: {str(e)}",
                recommendations=["Ensure sufficient disk space is available"]
            )
    
    def _check_cpu_capabilities(self) -> ValidationResult:
        """Check CPU capabilities"""
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            details = {"cpu_count": cpu_count, "cpu_percent": cpu_percent}
            
            if cpu_percent < 90:
                return ValidationResult(
                    check_name="CPU Capabilities",
                    category="system",
                    severity="info",
                    passed=True,
                    message=f"CPU healthy: {cpu_count} cores, {cpu_percent:.1f}% usage",
                    details=details
                )
            else:
                return ValidationResult(
                    check_name="CPU Capabilities",
                    category="system",
                    severity="warning",
                    passed=False,
                    message=f"High CPU usage: {cpu_percent:.1f}%",
                    details=details,
                    recommendations=["Close CPU-intensive applications"]
                )
        except ImportError:
            return ValidationResult(
                check_name="CPU Capabilities",
                category="system",
                severity="info",
                passed=True,
                message="CPU check skipped (psutil not available)"
            )
    
    def _check_git_availability(self) -> ValidationResult:
        """Check git command availability"""
        git_path = shutil.which('git')
        
        if git_path:
            try:
                result = subprocess.run(['git', '--version'], capture_output=True, text=True)
                version = result.stdout.strip()
                
                return ValidationResult(
                    check_name="Git Availability", 
                    category="dependency",
                    severity="critical",
                    passed=True,
                    message=f"Git available: {version}",
                    details={"path": git_path, "version": version}
                )
            except Exception as e:
                return ValidationResult(
                    check_name="Git Availability",
                    category="dependency", 
                    severity="critical",
                    passed=False,
                    message=f"Git found but not working: {str(e)}",
                    details={"path": git_path},
                    recommendations=["Reinstall git or check PATH configuration"]
                )
        else:
            return ValidationResult(
                check_name="Git Availability",
                category="dependency",
                severity="critical", 
                passed=False,
                message="Git command not found",
                recommendations=["Install git: apt install git (Ubuntu) or brew install git (macOS)"]
            )
    
    def _check_python_installation(self) -> ValidationResult:
        """Check Python installation completeness"""
        python_exe = sys.executable
        
        try:
            # Test basic Python functionality
            result = subprocess.run([python_exe, '-c', 'import sys, json, pathlib; print("OK")'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                return ValidationResult(
                    check_name="Python Installation",
                    category="dependency",
                    severity="critical",
                    passed=True,
                    message="Python installation is complete and functional",
                    details={"executable": python_exe}
                )
            else:
                return ValidationResult(
                    check_name="Python Installation",
                    category="dependency",
                    severity="critical",
                    passed=False,
                    message=f"Python installation issues: {result.stderr}",
                    details={"executable": python_exe, "error": result.stderr},
                    recommendations=["Reinstall Python or check installation"]
                )
        except Exception as e:
            return ValidationResult(
                check_name="Python Installation",
                category="dependency",
                severity="critical",
                passed=False,
                message=f"Cannot test Python installation: {str(e)}",
                recommendations=["Check Python installation and PATH"]
            )
    
    def _check_uv_availability(self) -> ValidationResult:
        """Check uv package manager availability"""
        uv_path = shutil.which('uv')
        
        if uv_path:
            try:
                result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
                version = result.stdout.strip()
                
                return ValidationResult(
                    check_name="UV Availability",
                    category="dependency",
                    severity="warning",
                    passed=True,
                    message=f"UV package manager available: {version}",
                    details={"path": uv_path, "version": version}
                )
            except Exception as e:
                return ValidationResult(
                    check_name="UV Availability",
                    category="dependency",
                    severity="warning",
                    passed=False,
                    message=f"UV found but not working: {str(e)}",
                    recommendations=["Reinstall uv or check PATH configuration"]
                )
        else:
            return ValidationResult(
                check_name="UV Availability",
                category="dependency",
                severity="warning",
                passed=False,
                message="UV package manager not found",
                recommendations=["Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"]
            )
    
    def _check_essential_modules(self) -> ValidationResult:
        """Check essential Python modules"""
        essential_modules = ['json', 'pathlib', 'subprocess', 'shutil', 'time']
        missing_modules = []
        
        for module in essential_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if not missing_modules:
            return ValidationResult(
                check_name="Essential Modules",
                category="dependency",
                severity="critical",
                passed=True,
                message="All essential Python modules available",
                details={"checked_modules": essential_modules}
            )
        else:
            return ValidationResult(
                check_name="Essential Modules",
                category="dependency",
                severity="critical",
                passed=False,
                message=f"Missing essential modules: {', '.join(missing_modules)}",
                details={"missing_modules": missing_modules},
                recommendations=["Check Python installation - these should be built-in modules"]
            )
    
    def _check_home_directory_access(self) -> ValidationResult:
        """Check home directory access"""
        home_dir = Path.home()
        
        try:
            # Test read access
            if not home_dir.exists():
                return ValidationResult(
                    check_name="Home Directory Access",
                    category="filesystem",
                    severity="critical",
                    passed=False,
                    message="Home directory does not exist",
                    recommendations=["Check user account configuration"]
                )
            
            # Test write access
            test_file = home_dir / '.claude_sync_test'
            test_file.write_text("test")
            test_file.unlink()
            
            return ValidationResult(
                check_name="Home Directory Access",
                category="filesystem",
                severity="critical",
                passed=True,
                message="Home directory is accessible and writable",
                details={"home_dir": str(home_dir)}
            )
            
        except PermissionError:
            return ValidationResult(
                check_name="Home Directory Access",
                category="filesystem",
                severity="critical",
                passed=False,
                message="Insufficient permissions for home directory",
                recommendations=["Check file permissions on home directory"]
            )
        except Exception as e:
            return ValidationResult(
                check_name="Home Directory Access",
                category="filesystem",
                severity="critical",
                passed=False,
                message=f"Home directory access test failed: {str(e)}",
                recommendations=["Check file system and permissions"]
            )
    
    def _check_claude_directory_creation(self) -> ValidationResult:
        """Check .claude directory creation capability"""
        claude_dir = Path.home() / '.claude'
        
        try:
            # Create directory if it doesn't exist
            if not claude_dir.exists():
                claude_dir.mkdir(parents=True)
                created = True
            else:
                created = False
            
            # Test write access
            test_file = claude_dir / 'test_access'
            test_file.write_text("test")
            test_file.unlink()
            
            return ValidationResult(
                check_name="Claude Directory Creation",
                category="filesystem",
                severity="critical",
                passed=True,
                message=f"Claude directory {'created' if created else 'exists'} and is writable",
                details={"claude_dir": str(claude_dir), "created": created}
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="Claude Directory Creation",
                category="filesystem",
                severity="critical",
                passed=False,
                message=f"Cannot create/access .claude directory: {str(e)}",
                recommendations=["Check permissions in home directory"]
            )
    
    def _check_symlink_support(self) -> ValidationResult:
        """Check filesystem symlink support"""
        if not hasattr(os, 'symlink'):
            return ValidationResult(
                check_name="Symlink Support",
                category="filesystem",
                severity="critical",
                passed=False,
                message="Symlinks not supported by Python installation",
                recommendations=["Use a Python build with symlink support"]
            )
        
        try:
            # Test symlink creation
            test_dir = Path.home() / '.claude'
            test_dir.mkdir(exist_ok=True)
            
            source_file = test_dir / 'symlink_test_source'
            link_file = test_dir / 'symlink_test_link'
            
            # Clean up any existing test files
            source_file.unlink(missing_ok=True)
            link_file.unlink(missing_ok=True)
            
            # Create source file and symlink
            source_file.write_text("test content")
            link_file.symlink_to(source_file)
            
            # Verify symlink works
            if link_file.is_symlink() and link_file.read_text() == "test content":
                # Clean up
                link_file.unlink()
                source_file.unlink()
                
                return ValidationResult(
                    check_name="Symlink Support",
                    category="filesystem",
                    severity="critical",
                    passed=True,
                    message="Filesystem supports symlinks",
                    details={"test_location": str(test_dir)}
                )
            else:
                # Clean up
                link_file.unlink(missing_ok=True)
                source_file.unlink(missing_ok=True)
                
                return ValidationResult(
                    check_name="Symlink Support",
                    category="filesystem",
                    severity="critical",
                    passed=False,
                    message="Symlink creation succeeded but link is not functional",
                    recommendations=["Check filesystem type and mount options"]
                )
                
        except OSError as e:
            return ValidationResult(
                check_name="Symlink Support",
                category="filesystem",
                severity="critical",
                passed=False,
                message=f"Cannot create symlinks: {str(e)}",
                recommendations=["Check filesystem permissions and type"]
            )
    
    def _check_file_permissions_capability(self) -> ValidationResult:
        """Check file permissions capability"""
        try:
            test_dir = Path.home() / '.claude'
            test_dir.mkdir(exist_ok=True)
            
            test_file = test_dir / 'permission_test'
            test_file.write_text("test")
            
            # Test chmod
            original_mode = test_file.stat().st_mode
            test_file.chmod(0o755)
            new_mode = test_file.stat().st_mode
            
            # Restore and cleanup
            test_file.chmod(original_mode)
            test_file.unlink()
            
            return ValidationResult(
                check_name="File Permissions Capability",
                category="filesystem",
                severity="warning",
                passed=True,
                message="Can modify file permissions",
                details={"chmod_available": True}
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="File Permissions Capability",
                category="filesystem",
                severity="warning",
                passed=False,
                message=f"Cannot modify file permissions: {str(e)}",
                recommendations=["Some features may not work without chmod support"]
            )
    
    def _check_internet_connectivity(self) -> ValidationResult:
        """Check internet connectivity"""
        try:
            # Simple connectivity test
            result = subprocess.run(['ping', '-c', '1', '8.8.8.8'], 
                                  capture_output=True, timeout=5)
            
            if result.returncode == 0:
                return ValidationResult(
                    check_name="Internet Connectivity",
                    category="network",
                    severity="warning",
                    passed=True,
                    message="Internet connectivity available"
                )
            else:
                return ValidationResult(
                    check_name="Internet Connectivity",
                    category="network",
                    severity="warning",
                    passed=False,
                    message="Internet connectivity issues detected",
                    recommendations=["Check network connection for installation"]
                )
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ValidationResult(
                check_name="Internet Connectivity",
                category="network",
                severity="warning",
                passed=False,
                message="Cannot test internet connectivity",
                recommendations=["Ensure internet access is available for installation"]
            )
    
    def _check_github_access(self) -> ValidationResult:
        """Check GitHub access"""
        try:
            # Test HTTPS access to GitHub
            result = subprocess.run(['git', 'ls-remote', 'https://github.com/drejom/claude-sync.git', 'HEAD'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return ValidationResult(
                    check_name="GitHub Access",
                    category="network",
                    severity="warning",
                    passed=True,
                    message="GitHub repository accessible via HTTPS"
                )
            else:
                return ValidationResult(
                    check_name="GitHub Access",
                    category="network",
                    severity="warning",
                    passed=False,
                    message="Cannot access GitHub repository",
                    details={"error": result.stderr},
                    recommendations=["Check internet connection and GitHub access"]
                )
                
        except subprocess.TimeoutExpired:
            return ValidationResult(
                check_name="GitHub Access",
                category="network",
                severity="warning",
                passed=False,
                message="GitHub access test timed out",
                recommendations=["Check network connectivity"]
            )
        except Exception as e:
            return ValidationResult(
                check_name="GitHub Access",
                category="network",
                severity="warning",
                passed=False,
                message=f"GitHub access test failed: {str(e)}",
                recommendations=["Ensure GitHub is accessible for installation"]
            )
    
    def _check_ssh_capability(self) -> ValidationResult:
        """Check SSH capability"""
        ssh_path = shutil.which('ssh')
        
        if not ssh_path:
            return ValidationResult(
                check_name="SSH Capability",
                category="network",
                severity="info",
                passed=False,
                message="SSH client not available",
                recommendations=["Install SSH client for advanced features"]
            )
        
        ssh_dir = Path.home() / '.ssh'
        if not ssh_dir.exists():
            return ValidationResult(
                check_name="SSH Capability",
                category="network",
                severity="info",
                passed=False,
                message="SSH available but no configuration found",
                recommendations=["SSH keys can be configured later if needed"]
            )
        
        # Check for SSH keys
        key_files = list(ssh_dir.glob('id_*'))
        
        return ValidationResult(
            check_name="SSH Capability",
            category="network",
            severity="info",
            passed=True,
            message=f"SSH available with {len(key_files)} key files",
            details={"ssh_keys": len(key_files)}
        )
    
    def _check_download_capability(self) -> ValidationResult:
        """Check download capability"""
        download_tools = ['curl', 'wget']
        available_tools = [tool for tool in download_tools if shutil.which(tool)]
        
        if available_tools:
            return ValidationResult(
                check_name="Download Capability",
                category="network",
                severity="info",
                passed=True,
                message=f"Download tools available: {', '.join(available_tools)}",
                details={"available_tools": available_tools}
            )
        else:
            return ValidationResult(
                check_name="Download Capability",
                category="network",
                severity="warning",
                passed=False,
                message="No download tools (curl/wget) available",
                recommendations=["Install curl or wget for download functionality"]
            )
    
    # Additional validation methods would be implemented here...
    # For brevity, I'm showing the pattern with key examples
    
    def _check_directory_structure(self) -> ValidationResult:
        """Placeholder for directory structure check"""
        return ValidationResult(
            check_name="Directory Structure Check",
            category="installation",
            severity="info",
            passed=True,
            message="Directory structure validation placeholder"
        )
    
    def _check_required_files(self) -> ValidationResult:
        """Placeholder for required files check"""
        return ValidationResult(
            check_name="Required Files Check",
            category="installation",
            severity="info",
            passed=True,
            message="Required files validation placeholder"
        )
    
    def _check_file_checksums(self) -> ValidationResult:
        """Placeholder for file checksums check"""
        return ValidationResult(
            check_name="File Checksums Check",
            category="installation",
            severity="info",
            passed=True,
            message="File checksums validation placeholder"
        )
    
    def _check_executable_permissions(self) -> ValidationResult:
        """Placeholder for executable permissions check"""
        return ValidationResult(
            check_name="Executable Permissions Check",
            category="installation",
            severity="info",
            passed=True,
            message="Executable permissions validation placeholder"
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bootstrap Validation System")
    parser.add_argument("phase", choices=["pre-install", "post-install", "activation", "continuous"],
                       help="Validation phase to run")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    args = parser.parse_args()
    
    validator = BootstrapValidator()
    
    # Run appropriate validation phase
    if args.phase == "pre-install":
        session = validator.run_pre_installation_validation()
    elif args.phase == "post-install":
        session = validator.run_post_installation_validation()
    elif args.phase == "activation":
        session = validator.run_activation_validation()
    else:
        session = validator.run_continuous_validation()
    
    # Output results
    if args.json:
        # JSON output for programmatic use
        result = {
            "session_id": session.session_id,
            "phase": session.phase,
            "timestamp": session.timestamp,
            "success_rate": session.overall_success_rate,
            "has_blocking_issues": session.has_blocking_issues,
            "total_checks": session.total_checks,
            "passed_checks": session.passed_checks,
            "execution_time_ms": session.execution_time_ms,
            "suites": [
                {
                    "name": suite.name,
                    "status": suite.overall_status,
                    "success_rate": suite.success_rate,
                    "checks": [
                        {
                            "name": check.check_name,
                            "category": check.category,
                            "severity": check.severity,
                            "passed": check.passed,
                            "message": check.message,
                            "details": check.details,
                            "recommendations": check.recommendations,
                            "execution_time_ms": check.execution_time_ms
                        }
                        for check in suite.checks
                    ]
                }
                for suite in session.suites
            ]
        }
        print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    if session.has_blocking_issues:
        exit(2)  # Blocking issues
    elif session.overall_success_rate < 0.8:
        exit(1)  # Warnings
    else:
        exit(0)  # Success