#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "psutil>=5.9.0",
#   "typing-extensions>=4.0.0"
# ]
# ///
"""
Claude-Sync Diagnostics System

Comprehensive system health checks and diagnostics for claude-sync:
- Installation integrity verification
- Hook system health monitoring
- Performance analysis and bottleneck detection
- Security system validation
- Learning data health assessment
- Network connectivity diagnostics
- Cross-platform compatibility checks
"""

import json
import os
import sys
import time
import platform
import subprocess
import psutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from interfaces import SystemState, PerformanceTargets
from activation_manager import ActivationManager

# ============================================================================
# Diagnostic Result Structures
# ============================================================================

@dataclass
class DiagnosticResult:
    """Result of a diagnostic check"""
    check_name: str
    status: str  # 'healthy', 'warning', 'critical', 'unknown'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    @property
    def is_healthy(self) -> bool:
        return self.status == 'healthy'
    
    @property
    def needs_attention(self) -> bool:
        return self.status in ['warning', 'critical']

@dataclass
class DiagnosticCategory:
    """Collection of related diagnostic checks"""
    name: str
    description: str
    checks: List[DiagnosticResult] = field(default_factory=list)
    
    @property
    def overall_status(self) -> str:
        """Determine overall category status"""
        if not self.checks:
            return 'unknown'
        
        statuses = [check.status for check in self.checks]
        
        if 'critical' in statuses:
            return 'critical'
        elif 'warning' in statuses:
            return 'warning'
        elif all(status == 'healthy' for status in statuses):
            return 'healthy'
        else:
            return 'unknown'
    
    @property
    def health_score(self) -> float:
        """Calculate health score (0.0 to 1.0)"""
        if not self.checks:
            return 0.0
        
        status_scores = {'healthy': 1.0, 'warning': 0.5, 'critical': 0.0, 'unknown': 0.3}
        total_score = sum(status_scores.get(check.status, 0.0) for check in self.checks)
        return total_score / len(self.checks)

@dataclass
class DiagnosticSession:
    """Complete diagnostic session results"""
    session_id: str
    timestamp: float
    platform_info: Dict[str, str]
    categories: List[DiagnosticCategory] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    @property
    def overall_health_score(self) -> float:
        """Calculate overall system health score"""
        if not self.categories:
            return 0.0
        
        total_score = sum(cat.health_score for cat in self.categories)
        return total_score / len(self.categories)
    
    @property
    def critical_issues_count(self) -> int:
        """Count critical issues across all categories"""
        return sum(
            len([check for check in cat.checks if check.status == 'critical'])
            for cat in self.categories
        )
    
    @property
    def warning_issues_count(self) -> int:
        """Count warning issues across all categories"""
        return sum(
            len([check for check in cat.checks if check.status == 'warning'])
            for cat in self.categories
        )

# ============================================================================
# Core Diagnostics System
# ============================================================================

class DiagnosticsSystem:
    """Main diagnostics orchestrator"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.claude_dir = Path.home() / '.claude'
        self.sync_dir = self.claude_dir / 'claude-sync'
        
        # System information
        self.platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'python_executable': sys.executable
        }
    
    def run_comprehensive_diagnostics(self) -> DiagnosticSession:
        """Run all diagnostic categories"""
        session_id = f"diagnostics_{int(time.time())}"
        start_time = time.perf_counter()
        
        print(f"🔍 Starting Comprehensive Diagnostics: {session_id}")
        print(f"🖥️  Platform: {self.platform_info['system']} {self.platform_info['release']}")
        print(f"🐍 Python: {self.platform_info['python_version']}")
        print("=" * 80)
        
        # Define diagnostic categories
        categories = [
            self.diagnose_installation_integrity(),
            self.diagnose_hook_system_health(),
            self.diagnose_activation_system(),
            self.diagnose_template_system(),
            self.diagnose_learning_infrastructure(),
            self.diagnose_security_system(),
            self.diagnose_performance_metrics(),
            self.diagnose_cross_platform_compatibility(),
            self.diagnose_network_connectivity(),
            self.diagnose_file_system_health()
        ]
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        session = DiagnosticSession(
            session_id=session_id,
            timestamp=time.time(),
            platform_info=self.platform_info,
            categories=categories,
            execution_time_ms=execution_time_ms
        )
        
        self.print_diagnostic_summary(session)
        return session
    
    def print_diagnostic_summary(self, session: DiagnosticSession):
        """Print comprehensive diagnostic summary"""
        print("\n" + "=" * 80)
        print(f"📋 DIAGNOSTIC SUMMARY: {session.session_id}")
        print("=" * 80)
        
        # Overall health
        health_score = session.overall_health_score
        health_emoji = "🟢" if health_score >= 0.8 else "🟡" if health_score >= 0.6 else "🔴"
        print(f"{health_emoji} Overall Health Score: {health_score:.1%}")
        print(f"⏱️  Total Execution Time: {session.execution_time_ms:.0f}ms")
        print(f"❌ Critical Issues: {session.critical_issues_count}")
        print(f"⚠️  Warnings: {session.warning_issues_count}")
        
        print(f"\n📊 CATEGORY BREAKDOWN:")
        print("-" * 40)
        
        for category in session.categories:
            status_emoji = self._get_status_emoji(category.overall_status)
            print(f"{status_emoji} {category.name}: {category.overall_status.upper()} ({category.health_score:.1%})")
            
            # Show issues for categories with problems
            if category.overall_status != 'healthy':
                for check in category.checks:
                    if check.needs_attention:
                        issue_emoji = "❌" if check.status == 'critical' else "⚠️"
                        print(f"    {issue_emoji} {check.check_name}: {check.message}")
        
        # Recommendations
        all_recommendations = []
        for category in session.categories:
            for check in category.checks:
                all_recommendations.extend(check.recommendations)
        
        if all_recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            print("-" * 40)
            for i, recommendation in enumerate(set(all_recommendations), 1):
                print(f"{i}. {recommendation}")
        
        # Overall verdict
        print(f"\n🎯 VERDICT:")
        print("-" * 40)
        if health_score >= 0.9:
            print("✅ EXCELLENT: System is in optimal condition")
        elif health_score >= 0.8:
            print("🟢 GOOD: System is healthy with minor optimizations possible")
        elif health_score >= 0.6:
            print("🟡 FAIR: System needs attention to resolve warnings")
        elif health_score >= 0.4:
            print("🟠 POOR: System has significant issues requiring immediate attention")
        else:
            print("🔴 CRITICAL: System has major problems and may not function correctly")
        
        print("=" * 80)
    
    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for status"""
        return {
            'healthy': '🟢',
            'warning': '🟡',
            'critical': '🔴',
            'unknown': '⚪'
        }.get(status, '⚪')
    
    def _time_check(self, func) -> Tuple[Any, float]:
        """Execute function and return result with execution time"""
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        return result, execution_time_ms
    
    # ========================================================================
    # Diagnostic Categories
    # ========================================================================
    
    def diagnose_installation_integrity(self) -> DiagnosticCategory:
        """Diagnose installation integrity"""
        category = DiagnosticCategory(
            name="Installation Integrity",
            description="Verify claude-sync installation completeness and integrity"
        )
        
        print(f"🔍 Checking installation integrity...")
        
        # Check directory structure
        result, exec_time = self._time_check(self._check_directory_structure)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check required files
        result, exec_time = self._time_check(self._check_required_files)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check file permissions
        result, exec_time = self._time_check(self._check_file_permissions)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check Python dependencies
        result, exec_time = self._time_check(self._check_python_dependencies)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        print(f"   └─ Installation integrity: {category.overall_status}")
        return category
    
    def diagnose_hook_system_health(self) -> DiagnosticCategory:
        """Diagnose hook system health"""
        category = DiagnosticCategory(
            name="Hook System Health",
            description="Verify hook files and execution capabilities"
        )
        
        print(f"🔍 Checking hook system health...")
        
        # Check hook files
        result, exec_time = self._time_check(self._check_hook_files)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check hook syntax
        result, exec_time = self._time_check(self._check_hook_syntax)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check hook performance
        result, exec_time = self._time_check(self._check_hook_performance)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        print(f"   └─ Hook system health: {category.overall_status}")
        return category
    
    def diagnose_activation_system(self) -> DiagnosticCategory:
        """Diagnose activation system"""
        category = DiagnosticCategory(
            name="Activation System",
            description="Verify activation/deactivation functionality"
        )
        
        print(f"🔍 Checking activation system...")
        
        # Check activation manager
        result, exec_time = self._time_check(self._check_activation_manager)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check current activation status
        result, exec_time = self._time_check(self._check_activation_status)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check backup system
        result, exec_time = self._time_check(self._check_backup_system)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        print(f"   └─ Activation system: {category.overall_status}")
        return category
    
    def diagnose_template_system(self) -> DiagnosticCategory:
        """Diagnose template system"""
        category = DiagnosticCategory(
            name="Template System",
            description="Verify settings templates and merging logic"
        )
        
        print(f"🔍 Checking template system...")
        
        # Check template files
        result, exec_time = self._time_check(self._check_template_files)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check template syntax
        result, exec_time = self._time_check(self._check_template_syntax)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check template merging
        result, exec_time = self._time_check(self._check_template_merging)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        print(f"   └─ Template system: {category.overall_status}")
        return category
    
    def diagnose_learning_infrastructure(self) -> DiagnosticCategory:
        """Diagnose learning infrastructure"""
        category = DiagnosticCategory(
            name="Learning Infrastructure",
            description="Verify learning data storage and management"
        )
        
        print(f"🔍 Checking learning infrastructure...")
        
        # Check learning directory
        result, exec_time = self._time_check(self._check_learning_directory)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check learning data integrity
        result, exec_time = self._time_check(self._check_learning_data_integrity)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check learning data size
        result, exec_time = self._time_check(self._check_learning_data_size)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        print(f"   └─ Learning infrastructure: {category.overall_status}")
        return category
    
    def diagnose_security_system(self) -> DiagnosticCategory:
        """Diagnose security system"""
        category = DiagnosticCategory(
            name="Security System",
            description="Verify encryption and security components"
        )
        
        print(f"🔍 Checking security system...")
        
        # Check security directory
        result, exec_time = self._time_check(self._check_security_directory)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check encryption modules
        result, exec_time = self._time_check(self._check_encryption_modules)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check file permissions security
        result, exec_time = self._time_check(self._check_security_permissions)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        print(f"   └─ Security system: {category.overall_status}")
        return category
    
    def diagnose_performance_metrics(self) -> DiagnosticCategory:
        """Diagnose performance metrics"""
        category = DiagnosticCategory(
            name="Performance Metrics",
            description="Analyze system performance and bottlenecks"
        )
        
        print(f"🔍 Checking performance metrics...")
        
        # Check system resources
        result, exec_time = self._time_check(self._check_system_resources)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check disk usage
        result, exec_time = self._time_check(self._check_disk_usage)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check performance targets
        result, exec_time = self._time_check(self._check_performance_targets)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        print(f"   └─ Performance metrics: {category.overall_status}")
        return category
    
    def diagnose_cross_platform_compatibility(self) -> DiagnosticCategory:
        """Diagnose cross-platform compatibility"""
        category = DiagnosticCategory(
            name="Cross-Platform Compatibility",
            description="Verify platform-specific functionality"
        )
        
        print(f"🔍 Checking cross-platform compatibility...")
        
        # Check platform support
        result, exec_time = self._time_check(self._check_platform_support)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check path handling
        result, exec_time = self._time_check(self._check_path_handling)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check command availability
        result, exec_time = self._time_check(self._check_command_availability)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        print(f"   └─ Cross-platform compatibility: {category.overall_status}")
        return category
    
    def diagnose_network_connectivity(self) -> DiagnosticCategory:
        """Diagnose network connectivity (for mesh sync)"""
        category = DiagnosticCategory(
            name="Network Connectivity",
            description="Verify network connectivity for mesh synchronization"
        )
        
        print(f"🔍 Checking network connectivity...")
        
        # Check internet connectivity
        result, exec_time = self._time_check(self._check_internet_connectivity)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check local network
        result, exec_time = self._time_check(self._check_local_network)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check SSH connectivity
        result, exec_time = self._time_check(self._check_ssh_connectivity)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        print(f"   └─ Network connectivity: {category.overall_status}")
        return category
    
    def diagnose_file_system_health(self) -> DiagnosticCategory:
        """Diagnose file system health"""
        category = DiagnosticCategory(
            name="File System Health",
            description="Verify file system integrity and permissions"
        )
        
        print(f"🔍 Checking file system health...")
        
        # Check disk space
        result, exec_time = self._time_check(self._check_disk_space)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check file system permissions
        result, exec_time = self._time_check(self._check_filesystem_permissions)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        # Check symlink health
        result, exec_time = self._time_check(self._check_symlink_health)
        result.execution_time_ms = exec_time
        category.checks.append(result)
        
        print(f"   └─ File system health: {category.overall_status}")
        return category
    
    # ========================================================================
    # Individual Check Methods
    # ========================================================================
    
    def _check_directory_structure(self) -> DiagnosticResult:
        """Check required directory structure"""
        required_dirs = [
            self.sync_dir,
            self.sync_dir / 'hooks',
            self.sync_dir / 'learning',
            self.sync_dir / 'templates',
            self.sync_dir / 'security'
        ]
        
        missing_dirs = [d for d in required_dirs if not d.exists()]
        
        if not missing_dirs:
            return DiagnosticResult(
                check_name="Directory Structure",
                status="healthy",
                message="All required directories present",
                details={"required_dirs": [str(d) for d in required_dirs]}
            )
        else:
            return DiagnosticResult(
                check_name="Directory Structure",
                status="critical",
                message=f"Missing {len(missing_dirs)} required directories",
                details={"missing_dirs": [str(d) for d in missing_dirs]},
                recommendations=["Run bootstrap.sh install to create missing directories"]
            )
    
    def _check_required_files(self) -> DiagnosticResult:
        """Check required files are present"""
        required_files = [
            self.sync_dir / 'bootstrap.sh',
            self.sync_dir / 'activation_manager.py',
            self.sync_dir / 'interfaces.py',
            self.sync_dir / 'templates' / 'settings.global.json',
            self.sync_dir / 'templates' / 'settings.local.json'
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if not missing_files:
            return DiagnosticResult(
                check_name="Required Files",
                status="healthy",
                message="All required files present",
                details={"file_count": len(required_files)}
            )
        else:
            return DiagnosticResult(
                check_name="Required Files",
                status="critical",
                message=f"Missing {len(missing_files)} required files",
                details={"missing_files": [str(f) for f in missing_files]},
                recommendations=["Run bootstrap.sh install to restore missing files"]
            )
    
    def _check_file_permissions(self) -> DiagnosticResult:
        """Check file permissions"""
        bootstrap_script = self.sync_dir / 'bootstrap.sh'
        
        issues = []
        if bootstrap_script.exists():
            if not os.access(bootstrap_script, os.X_OK):
                issues.append("bootstrap.sh is not executable")
        
        hook_files = list((self.sync_dir / 'hooks').glob('*.py'))
        non_executable_hooks = [f for f in hook_files if not os.access(f, os.X_OK)]
        
        if non_executable_hooks:
            issues.append(f"{len(non_executable_hooks)} hook files are not executable")
        
        if not issues:
            return DiagnosticResult(
                check_name="File Permissions",
                status="healthy",
                message="File permissions are correct",
                details={"checked_files": len(hook_files) + 1}
            )
        else:
            return DiagnosticResult(
                check_name="File Permissions",
                status="warning",
                message=f"Found {len(issues)} permission issues",
                details={"issues": issues},
                recommendations=["Run chmod +x on the affected files"]
            )
    
    def _check_python_dependencies(self) -> DiagnosticResult:
        """Check Python dependencies"""
        try:
            # Test key imports
            import json  # Built-in
            import pathlib  # Built-in
            
            # Test that uv is available
            result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
            uv_available = result.returncode == 0
            
            if uv_available:
                return DiagnosticResult(
                    check_name="Python Dependencies",
                    status="healthy",
                    message="All Python dependencies available",
                    details={"uv_version": result.stdout.strip()}
                )
            else:
                return DiagnosticResult(
                    check_name="Python Dependencies",
                    status="critical",
                    message="uv package manager not available",
                    recommendations=["Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"]
                )
                
        except Exception as e:
            return DiagnosticResult(
                check_name="Python Dependencies",
                status="critical",
                message=f"Dependency check failed: {str(e)}",
                recommendations=["Check Python installation and PATH"]
            )
    
    def _check_hook_files(self) -> DiagnosticResult:
        """Check hook files"""
        hooks_dir = self.sync_dir / 'hooks'
        
        if not hooks_dir.exists():
            return DiagnosticResult(
                check_name="Hook Files",
                status="critical",
                message="Hooks directory does not exist"
            )
        
        hook_files = list(hooks_dir.glob('*.py'))
        
        if len(hook_files) == 0:
            return DiagnosticResult(
                check_name="Hook Files",
                status="critical",
                message="No hook files found",
                recommendations=["Run bootstrap.sh install to restore hook files"]
            )
        
        # Check for key hooks
        expected_hooks = ['bash-optimizer-enhanced.py', 'ssh-router-enhanced.py', 'resource-tracker.py']
        missing_hooks = [hook for hook in expected_hooks if not (hooks_dir / hook).exists()]
        
        if missing_hooks:
            return DiagnosticResult(
                check_name="Hook Files",
                status="warning",
                message=f"Missing {len(missing_hooks)} expected hook files",
                details={"missing_hooks": missing_hooks, "total_hooks": len(hook_files)},
                recommendations=["Run bootstrap.sh update to restore missing hooks"]
            )
        else:
            return DiagnosticResult(
                check_name="Hook Files",
                status="healthy",
                message=f"Found {len(hook_files)} hook files",
                details={"hook_count": len(hook_files)}
            )
    
    def _check_hook_syntax(self) -> DiagnosticResult:
        """Check hook syntax"""
        hooks_dir = self.sync_dir / 'hooks'
        hook_files = list(hooks_dir.glob('*.py'))
        
        syntax_errors = []
        
        for hook_file in hook_files:
            try:
                subprocess.run([sys.executable, '-m', 'py_compile', str(hook_file)], 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError:
                syntax_errors.append(hook_file.name)
        
        if not syntax_errors:
            return DiagnosticResult(
                check_name="Hook Syntax",
                status="healthy",
                message=f"All {len(hook_files)} hook files have valid syntax",
                details={"checked_files": len(hook_files)}
            )
        else:
            return DiagnosticResult(
                check_name="Hook Syntax",
                status="critical",
                message=f"{len(syntax_errors)} hook files have syntax errors",
                details={"syntax_errors": syntax_errors},
                recommendations=["Fix syntax errors in the affected hook files"]
            )
    
    def _check_hook_performance(self) -> DiagnosticResult:
        """Check hook performance expectations"""
        # This is a basic check - in practice, we'd want to actually execute hooks
        hooks_dir = self.sync_dir / 'hooks'
        hook_files = list(hooks_dir.glob('*.py'))
        
        large_hooks = []
        for hook_file in hook_files:
            size_kb = hook_file.stat().st_size / 1024
            if size_kb > 100:  # Arbitrarily large for a hook
                large_hooks.append((hook_file.name, size_kb))
        
        if not large_hooks:
            return DiagnosticResult(
                check_name="Hook Performance Profile",
                status="healthy",
                message="Hook files are appropriately sized",
                details={"average_size_kb": sum(f.stat().st_size for f in hook_files) / len(hook_files) / 1024 if hook_files else 0}
            )
        else:
            return DiagnosticResult(
                check_name="Hook Performance Profile",
                status="warning",
                message=f"{len(large_hooks)} hook files are unusually large",
                details={"large_hooks": large_hooks},
                recommendations=["Review large hook files for optimization opportunities"]
            )
    
    def _check_activation_manager(self) -> DiagnosticResult:
        """Check activation manager functionality"""
        try:
            manager = ActivationManager()
            
            # Test basic operations
            status = manager.get_status()
            verification = manager.verify_installation()
            
            return DiagnosticResult(
                check_name="Activation Manager",
                status="healthy",
                message="Activation manager is functional",
                details={
                    "is_activated": status.is_activated,
                    "installation_verified": verification.get("overall_status", False)
                }
            )
            
        except Exception as e:
            return DiagnosticResult(
                check_name="Activation Manager",
                status="critical",
                message=f"Activation manager failed: {str(e)}",
                recommendations=["Check activation_manager.py for errors"]
            )
    
    def _check_activation_status(self) -> DiagnosticResult:
        """Check current activation status"""
        try:
            manager = ActivationManager()
            status = manager.get_status()
            
            if status.is_activated:
                return DiagnosticResult(
                    check_name="Activation Status",
                    status="healthy",
                    message=f"Claude-sync is activated with {len(status.hooks_installed)} hooks",
                    details={
                        "hooks_installed": status.hooks_installed,
                        "learning_data_mb": status.learning_data_size_mb
                    }
                )
            else:
                return DiagnosticResult(
                    check_name="Activation Status",
                    status="warning",
                    message="Claude-sync is not currently activated",
                    recommendations=["Run bootstrap.sh activate --global to activate"]
                )
                
        except Exception as e:
            return DiagnosticResult(
                check_name="Activation Status",
                status="critical",
                message=f"Cannot determine activation status: {str(e)}"
            )
    
    def _check_backup_system(self) -> DiagnosticResult:
        """Check backup system"""
        backups_dir = self.sync_dir / 'backups'
        
        if not backups_dir.exists():
            return DiagnosticResult(
                check_name="Backup System",
                status="warning",
                message="No backups directory found",
                recommendations=["Backups will be created automatically during activation"]
            )
        
        backup_files = list(backups_dir.glob('*.json'))
        
        if backup_files:
            # Check backup age
            recent_backups = [
                f for f in backup_files 
                if (time.time() - f.stat().st_mtime) < (7 * 24 * 3600)  # Within 7 days
            ]
            
            return DiagnosticResult(
                check_name="Backup System",
                status="healthy",
                message=f"Found {len(backup_files)} backups ({len(recent_backups)} recent)",
                details={"total_backups": len(backup_files), "recent_backups": len(recent_backups)}
            )
        else:
            return DiagnosticResult(
                check_name="Backup System",
                status="warning",
                message="No backup files found",
                recommendations=["Backups will be created during activation/deactivation"]
            )
    
    def _check_template_files(self) -> DiagnosticResult:
        """Check template files"""
        templates_dir = self.sync_dir / 'templates'
        expected_templates = ['settings.global.json', 'settings.local.json']
        
        missing_templates = []
        for template in expected_templates:
            if not (templates_dir / template).exists():
                missing_templates.append(template)
        
        if not missing_templates:
            return DiagnosticResult(
                check_name="Template Files",
                status="healthy",
                message="All template files present",
                details={"template_count": len(expected_templates)}
            )
        else:
            return DiagnosticResult(
                check_name="Template Files",
                status="critical",
                message=f"Missing {len(missing_templates)} template files",
                details={"missing_templates": missing_templates},
                recommendations=["Run bootstrap.sh install to restore templates"]
            )
    
    def _check_template_syntax(self) -> DiagnosticResult:
        """Check template syntax"""
        templates_dir = self.sync_dir / 'templates'
        template_files = list(templates_dir.glob('*.json'))
        
        syntax_errors = []
        for template_file in template_files:
            try:
                with open(template_file) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                syntax_errors.append((template_file.name, str(e)))
        
        if not syntax_errors:
            return DiagnosticResult(
                check_name="Template Syntax",
                status="healthy",
                message=f"All {len(template_files)} templates have valid JSON",
                details={"checked_templates": len(template_files)}
            )
        else:
            return DiagnosticResult(
                check_name="Template Syntax",
                status="critical",
                message=f"{len(syntax_errors)} templates have JSON errors",
                details={"syntax_errors": syntax_errors},
                recommendations=["Fix JSON syntax errors in template files"]
            )
    
    def _check_template_merging(self) -> DiagnosticResult:
        """Check template merging logic"""
        try:
            from activation_manager import SettingsMerger
            
            merger = SettingsMerger()
            
            # Test with sample data
            user_settings = {"permissions": {"allow": ["Read(*)"]}}
            claude_settings = {"permissions": {"allow": ["Bash(*)"]}}
            
            merged = merger.merge_hook_settings(user_settings, claude_settings)
            is_valid = merger.validate_merged_settings(merged)
            
            if is_valid:
                return DiagnosticResult(
                    check_name="Template Merging",
                    status="healthy",
                    message="Template merging logic works correctly",
                    details={"test_merge_successful": True}
                )
            else:
                return DiagnosticResult(
                    check_name="Template Merging",
                    status="critical",
                    message="Template merging produces invalid results",
                    recommendations=["Check SettingsMerger implementation"]
                )
                
        except Exception as e:
            return DiagnosticResult(
                check_name="Template Merging",
                status="critical",
                message=f"Template merging test failed: {str(e)}",
                recommendations=["Check SettingsMerger class for errors"]
            )
    
    def _check_learning_directory(self) -> DiagnosticResult:
        """Check learning directory"""
        learning_dir = self.sync_dir / 'learning'
        
        if not learning_dir.exists():
            return DiagnosticResult(
                check_name="Learning Directory",
                status="warning",
                message="Learning directory does not exist",
                recommendations=["Learning directory will be created automatically"]
            )
        
        python_files = list(learning_dir.glob('*.py'))
        data_files = list(learning_dir.glob('*.enc')) + list(learning_dir.glob('*.pkl'))
        
        return DiagnosticResult(
            check_name="Learning Directory",
            status="healthy",
            message=f"Learning directory contains {len(python_files)} modules, {len(data_files)} data files",
            details={"module_count": len(python_files), "data_file_count": len(data_files)}
        )
    
    def _check_learning_data_integrity(self) -> DiagnosticResult:
        """Check learning data integrity"""
        learning_dir = self.sync_dir / 'learning'
        
        if not learning_dir.exists():
            return DiagnosticResult(
                check_name="Learning Data Integrity",
                status="healthy",
                message="No learning data to check (clean state)"
            )
        
        # Check for corrupted files (basic check)
        data_files = list(learning_dir.glob('*.enc')) + list(learning_dir.glob('*.pkl'))
        corrupted_files = []
        
        for data_file in data_files:
            try:
                # Basic read test
                with open(data_file, 'rb') as f:
                    f.read(100)  # Read first 100 bytes
            except Exception:
                corrupted_files.append(data_file.name)
        
        if not corrupted_files:
            return DiagnosticResult(
                check_name="Learning Data Integrity",
                status="healthy",
                message=f"All {len(data_files)} data files appear intact",
                details={"data_file_count": len(data_files)}
            )
        else:
            return DiagnosticResult(
                check_name="Learning Data Integrity",
                status="warning",
                message=f"{len(corrupted_files)} data files may be corrupted",
                details={"corrupted_files": corrupted_files},
                recommendations=["Consider removing corrupted files (they will be regenerated)"]
            )
    
    def _check_learning_data_size(self) -> DiagnosticResult:
        """Check learning data size"""
        learning_dir = self.sync_dir / 'learning'
        
        if not learning_dir.exists():
            return DiagnosticResult(
                check_name="Learning Data Size",
                status="healthy",
                message="No learning data (0 MB)"
            )
        
        total_size = 0
        for data_file in learning_dir.rglob('*'):
            if data_file.is_file():
                total_size += data_file.stat().st_size
        
        size_mb = total_size / (1024 * 1024)
        
        if size_mb < PerformanceTargets.MAX_LEARNING_DATA_MB:
            return DiagnosticResult(
                check_name="Learning Data Size",
                status="healthy",
                message=f"Learning data size: {size_mb:.1f} MB (within limits)",
                details={"size_mb": size_mb, "limit_mb": PerformanceTargets.MAX_LEARNING_DATA_MB}
            )
        else:
            return DiagnosticResult(
                check_name="Learning Data Size",
                status="warning",
                message=f"Learning data size: {size_mb:.1f} MB (exceeds {PerformanceTargets.MAX_LEARNING_DATA_MB} MB limit)",
                details={"size_mb": size_mb, "limit_mb": PerformanceTargets.MAX_LEARNING_DATA_MB},
                recommendations=["Consider running learning data cleanup"]
            )
    
    def _check_security_directory(self) -> DiagnosticResult:
        """Check security directory"""
        security_dir = self.sync_dir / 'security'
        
        if not security_dir.exists():
            return DiagnosticResult(
                check_name="Security Directory",
                status="warning",
                message="Security directory does not exist",
                recommendations=["Security modules will be available when needed"]
            )
        
        security_files = list(security_dir.glob('*.py'))
        
        return DiagnosticResult(
            check_name="Security Directory",
            status="healthy",
            message=f"Security directory contains {len(security_files)} modules",
            details={"module_count": len(security_files)}
        )
    
    def _check_encryption_modules(self) -> DiagnosticResult:
        """Check encryption modules"""
        try:
            # Test that we can import cryptography
            import cryptography
            return DiagnosticResult(
                check_name="Encryption Modules",
                status="healthy",
                message="Cryptography modules available",
                details={"cryptography_version": cryptography.__version__}
            )
        except ImportError:
            return DiagnosticResult(
                check_name="Encryption Modules",
                status="warning",
                message="Cryptography not available (will be installed when needed)",
                recommendations=["Encryption will be available via uv when required"]
            )
    
    def _check_security_permissions(self) -> DiagnosticResult:
        """Check security-related file permissions"""
        # Check that sensitive directories have appropriate permissions
        
        secure_dirs = [self.sync_dir / 'learning', self.sync_dir / 'security']
        permission_issues = []
        
        for secure_dir in secure_dirs:
            if secure_dir.exists():
                stat_info = secure_dir.stat()
                # Check if directory is world-readable (basic security check)
                if stat_info.st_mode & 0o004:  # World readable
                    permission_issues.append(f"{secure_dir.name} is world-readable")
        
        if not permission_issues:
            return DiagnosticResult(
                check_name="Security Permissions",
                status="healthy",
                message="Security-related permissions are appropriate"
            )
        else:
            return DiagnosticResult(
                check_name="Security Permissions",
                status="warning",
                message=f"Found {len(permission_issues)} permission concerns",
                details={"issues": permission_issues},
                recommendations=["Consider restricting directory permissions with chmod"]
            )
    
    def _check_system_resources(self) -> DiagnosticResult:
        """Check system resources"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        issues = []
        if memory_available_gb < 1.0:  # Less than 1GB available
            issues.append(f"Low available memory: {memory_available_gb:.1f}GB")
        
        if cpu_percent > 90:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if not issues:
            return DiagnosticResult(
                check_name="System Resources",
                status="healthy",
                message=f"System resources healthy (RAM: {memory_available_gb:.1f}GB available, CPU: {cpu_percent:.1f}%)",
                details={"memory_gb": memory_gb, "available_gb": memory_available_gb, "cpu_percent": cpu_percent}
            )
        else:
            return DiagnosticResult(
                check_name="System Resources",
                status="warning",
                message=f"System resource concerns: {', '.join(issues)}",
                details={"memory_gb": memory_gb, "available_gb": memory_available_gb, "cpu_percent": cpu_percent},
                recommendations=["Consider closing other applications or upgrading system resources"]
            )
    
    def _check_disk_usage(self) -> DiagnosticResult:
        """Check disk usage"""
        claude_dir_usage = shutil.disk_usage(self.claude_dir)
        
        total_gb = claude_dir_usage.total / (1024**3)
        free_gb = claude_dir_usage.free / (1024**3)
        used_percent = ((claude_dir_usage.total - claude_dir_usage.free) / claude_dir_usage.total) * 100
        
        if free_gb < 1.0:  # Less than 1GB free
            return DiagnosticResult(
                check_name="Disk Usage",
                status="critical",
                message=f"Very low disk space: {free_gb:.1f}GB free ({used_percent:.1f}% used)",
                details={"total_gb": total_gb, "free_gb": free_gb, "used_percent": used_percent},
                recommendations=["Free up disk space immediately"]
            )
        elif free_gb < 5.0:  # Less than 5GB free
            return DiagnosticResult(
                check_name="Disk Usage",
                status="warning",
                message=f"Low disk space: {free_gb:.1f}GB free ({used_percent:.1f}% used)",
                details={"total_gb": total_gb, "free_gb": free_gb, "used_percent": used_percent},
                recommendations=["Consider freeing up disk space"]
            )
        else:
            return DiagnosticResult(
                check_name="Disk Usage",
                status="healthy",
                message=f"Disk space healthy: {free_gb:.1f}GB free ({used_percent:.1f}% used)",
                details={"total_gb": total_gb, "free_gb": free_gb, "used_percent": used_percent}
            )
    
    def _check_performance_targets(self) -> DiagnosticResult:
        """Check if system can meet performance targets"""
        # Simple performance test - measure basic operations
        
        # Test file I/O
        test_file = self.sync_dir / '.diagnostic_test'
        start_time = time.perf_counter()
        test_file.write_text("test data")
        content = test_file.read_text()
        test_file.unlink()
        io_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Test JSON operations
        test_data = {"hooks": {"PreToolUse": [{"matcher": "test", "hooks": []}]}}
        start_time = time.perf_counter()
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        json_time_ms = (time.perf_counter() - start_time) * 1000
        
        details = {
            "file_io_ms": io_time_ms,
            "json_ops_ms": json_time_ms,
            "io_target_ms": 10,
            "json_target_ms": 1
        }
        
        if io_time_ms <= 10 and json_time_ms <= 1:
            return DiagnosticResult(
                check_name="Performance Targets",
                status="healthy",
                message="System meets performance targets",
                details=details
            )
        else:
            return DiagnosticResult(
                check_name="Performance Targets",
                status="warning",
                message="System may not meet optimal performance targets",
                details=details,
                recommendations=["System performance may impact hook execution times"]
            )
    
    def _check_platform_support(self) -> DiagnosticResult:
        """Check platform support"""
        supported_platforms = ['Linux', 'Darwin', 'Windows']  # macOS is Darwin
        current_platform = platform.system()
        
        if current_platform in supported_platforms:
            return DiagnosticResult(
                check_name="Platform Support",
                status="healthy",
                message=f"Platform {current_platform} is supported",
                details={"platform": current_platform, "supported": True}
            )
        else:
            return DiagnosticResult(
                check_name="Platform Support",
                status="warning",
                message=f"Platform {current_platform} is not officially tested",
                details={"platform": current_platform, "supported": False},
                recommendations=["Functionality may be limited on untested platforms"]
            )
    
    def _check_path_handling(self) -> DiagnosticResult:
        """Check path handling"""
        try:
            # Test various path operations
            home_path = Path.home()
            claude_path = home_path / '.claude'
            relative_path = Path('.') / 'test'
            
            # Test path operations
            home_str = str(home_path)
            claude_exists = claude_path.exists()
            relative_absolute = relative_path.absolute()
            
            return DiagnosticResult(
                check_name="Path Handling",
                status="healthy",
                message="Path operations work correctly",
                details={
                    "home_detected": bool(home_str),
                    "path_exists_check": isinstance(claude_exists, bool),
                    "absolute_conversion": bool(str(relative_absolute))
                }
            )
            
        except Exception as e:
            return DiagnosticResult(
                check_name="Path Handling",
                status="critical",
                message=f"Path operations failed: {str(e)}",
                recommendations=["Check Python pathlib support"]
            )
    
    def _check_command_availability(self) -> DiagnosticResult:
        """Check command availability"""
        required_commands = ['python3', 'git']
        optional_commands = ['ssh', 'curl']
        
        missing_required = []
        missing_optional = []
        
        for cmd in required_commands:
            if not shutil.which(cmd):
                missing_required.append(cmd)
        
        for cmd in optional_commands:
            if not shutil.which(cmd):
                missing_optional.append(cmd)
        
        if missing_required:
            return DiagnosticResult(
                check_name="Command Availability",
                status="critical",
                message=f"Missing required commands: {', '.join(missing_required)}",
                details={"missing_required": missing_required, "missing_optional": missing_optional},
                recommendations=[f"Install missing commands: {', '.join(missing_required)}"]
            )
        elif missing_optional:
            return DiagnosticResult(
                check_name="Command Availability",
                status="warning",
                message=f"Missing optional commands: {', '.join(missing_optional)}",
                details={"missing_optional": missing_optional},
                recommendations=[f"Consider installing: {', '.join(missing_optional)}"]
            )
        else:
            return DiagnosticResult(
                check_name="Command Availability",
                status="healthy",
                message="All required and optional commands available"
            )
    
    def _check_internet_connectivity(self) -> DiagnosticResult:
        """Check internet connectivity"""
        try:
            # Simple connectivity test
            result = subprocess.run(['ping', '-c', '1', '8.8.8.8'], 
                                  capture_output=True, timeout=5)
            
            if result.returncode == 0:
                return DiagnosticResult(
                    check_name="Internet Connectivity",
                    status="healthy",
                    message="Internet connectivity available"
                )
            else:
                return DiagnosticResult(
                    check_name="Internet Connectivity",
                    status="warning",
                    message="Internet connectivity issues detected",
                    recommendations=["Some features may not work without internet"]
                )
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return DiagnosticResult(
                check_name="Internet Connectivity",
                status="warning",
                message="Cannot test internet connectivity",
                recommendations=["Ping command not available or connectivity issues"]
            )
    
    def _check_local_network(self) -> DiagnosticResult:
        """Check local network configuration"""
        try:
            import socket
            
            # Get local hostname and IP
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            return DiagnosticResult(
                check_name="Local Network",
                status="healthy",
                message=f"Local network configured (IP: {local_ip})",
                details={"hostname": hostname, "local_ip": local_ip}
            )
            
        except Exception as e:
            return DiagnosticResult(
                check_name="Local Network",
                status="warning",
                message=f"Local network configuration issues: {str(e)}",
                recommendations=["Check network configuration"]
            )
    
    def _check_ssh_connectivity(self) -> DiagnosticResult:
        """Check SSH connectivity capability"""
        ssh_available = shutil.which('ssh') is not None
        ssh_dir = Path.home() / '.ssh'
        
        if not ssh_available:
            return DiagnosticResult(
                check_name="SSH Connectivity",
                status="warning",
                message="SSH command not available",
                recommendations=["Install SSH client for remote connectivity features"]
            )
        
        if not ssh_dir.exists():
            return DiagnosticResult(
                check_name="SSH Connectivity",
                status="warning",
                message="No SSH configuration directory",
                recommendations=["SSH keys can be generated when needed"]
            )
        
        # Check for SSH keys
        key_files = list(ssh_dir.glob('id_*'))
        
        if key_files:
            return DiagnosticResult(
                check_name="SSH Connectivity",
                status="healthy",
                message=f"SSH available with {len(key_files)} key files",
                details={"ssh_keys": len(key_files)}
            )
        else:
            return DiagnosticResult(
                check_name="SSH Connectivity",
                status="warning",
                message="SSH available but no keys found",
                recommendations=["Generate SSH keys for remote host connectivity"]
            )
    
    def _check_disk_space(self) -> DiagnosticResult:
        """Check available disk space"""
        usage = shutil.disk_usage(self.claude_dir)
        free_gb = usage.free / (1024**3)
        
        if free_gb >= 5.0:
            return DiagnosticResult(
                check_name="Disk Space",
                status="healthy",
                message=f"Sufficient disk space: {free_gb:.1f}GB free"
            )
        elif free_gb >= 1.0:
            return DiagnosticResult(
                check_name="Disk Space",
                status="warning",
                message=f"Low disk space: {free_gb:.1f}GB free",
                recommendations=["Consider freeing up disk space"]
            )
        else:
            return DiagnosticResult(
                check_name="Disk Space",
                status="critical",
                message=f"Very low disk space: {free_gb:.1f}GB free",
                recommendations=["Free up disk space immediately"]
            )
    
    def _check_filesystem_permissions(self) -> DiagnosticResult:
        """Check filesystem permissions"""
        test_file = self.claude_dir / '.permission_test'
        
        try:
            # Test write permissions
            test_file.write_text("test")
            
            # Test read permissions
            content = test_file.read_text()
            
            # Test delete permissions
            test_file.unlink()
            
            return DiagnosticResult(
                check_name="Filesystem Permissions",
                status="healthy",
                message="Filesystem permissions are correct"
            )
            
        except PermissionError:
            return DiagnosticResult(
                check_name="Filesystem Permissions",
                status="critical",
                message="Insufficient filesystem permissions",
                recommendations=["Check file permissions on ~/.claude directory"]
            )
        except Exception as e:
            return DiagnosticResult(
                check_name="Filesystem Permissions",
                status="warning",
                message=f"Permission test failed: {str(e)}"
            )
    
    def _check_symlink_health(self) -> DiagnosticResult:
        """Check symlink health"""
        hooks_dir = self.claude_dir / 'hooks'
        
        if not hooks_dir.exists():
            return DiagnosticResult(
                check_name="Symlink Health",
                status="healthy",
                message="No symlinks to check (not activated)"
            )
        
        symlinks = [f for f in hooks_dir.glob('*') if f.is_symlink()]
        broken_symlinks = []
        
        for symlink in symlinks:
            try:
                # Test if symlink target exists
                target = symlink.resolve()
                if not target.exists():
                    broken_symlinks.append(symlink.name)
            except Exception:
                broken_symlinks.append(symlink.name)
        
        if not broken_symlinks:
            return DiagnosticResult(
                check_name="Symlink Health",
                status="healthy",
                message=f"All {len(symlinks)} symlinks are healthy",
                details={"symlink_count": len(symlinks)}
            )
        else:
            return DiagnosticResult(
                check_name="Symlink Health",
                status="warning",
                message=f"{len(broken_symlinks)} broken symlinks found",
                details={"broken_symlinks": broken_symlinks},
                recommendations=["Run bootstrap.sh activate to fix broken symlinks"]
            )


if __name__ == "__main__":
    # Run comprehensive diagnostics
    diagnostics = DiagnosticsSystem()
    session = diagnostics.run_comprehensive_diagnostics()
    
    # Exit with appropriate code based on health
    if session.overall_health_score >= 0.8:
        exit_code = 0  # Healthy
    elif session.overall_health_score >= 0.6:
        exit_code = 1  # Warnings
    else:
        exit_code = 2  # Critical issues
    
    print(f"\n🏁 Diagnostics complete. Health Score: {session.overall_health_score:.1%} (Exit code: {exit_code})")
    exit(exit_code)