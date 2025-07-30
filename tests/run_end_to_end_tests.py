#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-asyncio>=0.21.0",
#   "pytest-benchmark>=4.0.0",
#   "psutil>=5.9.0",
#   "cryptography>=41.0.0",
#   "typing-extensions>=4.0.0",
#   "rich>=12.0.0"
# ]
# ///
"""
Claude-Sync End-to-End Test Runner

Comprehensive test runner that validates the entire claude-sync system with
rich output formatting and detailed reporting.

Usage:
    python run_end_to_end_tests.py [options]
    
Options:
    --quick         Run quick validation tests only
    --full          Run comprehensive full-system tests
    --benchmark     Include performance benchmarking
    --report        Generate detailed HTML report
    --verbose       Enable verbose output
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available, using basic output formatting")

from tests.test_end_to_end import EndToEndTestFramework, EndToEndTestConfig

@dataclass
class TestRunnerConfig:
    """Configuration for test runner"""
    test_mode: str = "full"  # "quick", "full", "benchmark"
    generate_report: bool = False
    verbose: bool = False
    output_file: Optional[Path] = None
    
    # Test parameters
    hook_limit_ms: int = 10
    memory_limit_mb: int = 50
    concurrent_hooks: int = 10
    stress_duration_s: int = 30

class EndToEndTestRunner:
    """Enhanced test runner with rich output and reporting"""
    
    def __init__(self, config: TestRunnerConfig):
        self.config = config
        self.console = Console() if RICH_AVAILABLE else None
        self.results: Dict[str, Any] = {}
        
    def run_tests(self) -> bool:
        """Run end-to-end tests with enhanced output"""
        if self.console:
            self._run_tests_with_rich()
        else:
            self._run_tests_basic()
        
        return self.results.get('overall_success', False)
    
    def _run_tests_with_rich(self) -> None:
        """Run tests with rich formatting"""
        # Display header
        self.console.print(Panel.fit(
            "[bold blue]Claude-Sync End-to-End Integration Tests[/bold blue]\n"
            f"Mode: {self.config.test_mode.upper()} | "
            f"Hook Limit: {self.config.hook_limit_ms}ms | "
            f"Memory Limit: {self.config.memory_limit_mb}MB",
            border_style="blue"
        ))
        
        # Configure test framework
        e2e_config = EndToEndTestConfig(
            hook_execution_limit_ms=self.config.hook_limit_ms,
            memory_limit_mb=self.config.memory_limit_mb,
            concurrent_hooks=self.config.concurrent_hooks,
            stress_test_duration_s=self.config.stress_duration_s
        )
        
        if self.config.test_mode == "quick":
            e2e_config.concurrent_hooks = 3
            e2e_config.stress_test_duration_s = 5
            e2e_config.workflow_repetitions = 1
        elif self.config.test_mode == "benchmark":
            e2e_config.concurrent_hooks = 20
            e2e_config.stress_test_duration_s = 60
            e2e_config.workflow_repetitions = 10
        
        # Run tests with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Running comprehensive end-to-end tests...", total=None)
            
            framework = EndToEndTestFramework(e2e_config)
            self.results = framework.run_all_end_to_end_tests()
            
            progress.update(task, completed=100, description="Tests completed!")
        
        # Display results
        self._display_results_rich()
        
        # Generate report if requested
        if self.config.generate_report:
            self._generate_html_report()
    
    def _run_tests_basic(self) -> None:
        """Run tests with basic output"""
        print("Claude-Sync End-to-End Integration Tests")
        print("=" * 50)
        print(f"Mode: {self.config.test_mode.upper()}")
        print(f"Hook Limit: {self.config.hook_limit_ms}ms")
        print(f"Memory Limit: {self.config.memory_limit_mb}MB")
        print("=" * 50)
        
        # Configure and run tests
        e2e_config = EndToEndTestConfig(
            hook_execution_limit_ms=self.config.hook_limit_ms,
            memory_limit_mb=self.config.memory_limit_mb,
            concurrent_hooks=self.config.concurrent_hooks,
            stress_test_duration_s=self.config.stress_test_duration_s
        )
        
        framework = EndToEndTestFramework(e2e_config)
        self.results = framework.run_all_end_to_end_tests()
        
        # Display basic results
        self._display_results_basic()
    
    def _display_results_rich(self) -> None:
        """Display results with rich formatting"""
        if not self.console:
            return
        
        # Overall status
        overall_success = self.results.get('overall_success', False)
        status_color = "green" if overall_success else "red"
        status_text = "✅ ALL SYSTEMS GO" if overall_success else "❌ QUALITY GATES FAILED"
        
        self.console.print(f"\n[bold {status_color}]{status_text}[/bold {status_color}]")
        
        # Summary table
        summary_table = Table(title="Test Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        summary_table.add_column("Status", style="bold")
        
        success_rate = self.results.get('success_rate', 0)
        passed_tests = self.results.get('passed_tests', 0)
        total_tests = self.results.get('total_tests', 0)
        session_duration = self.results.get('session_duration_ms', 0) / 1000
        
        summary_table.add_row(
            "Success Rate",
            f"{success_rate:.1%}",
            "✅" if success_rate >= 0.95 else "❌"
        )
        summary_table.add_row(
            "Tests Passed",
            f"{passed_tests}/{total_tests}",
            "✅" if passed_tests == total_tests else "❌"
        )
        summary_table.add_row(
            "Duration",
            f"{session_duration:.1f}s",
            "✅" if session_duration < 300 else "⚠️"
        )
        
        self.console.print(summary_table)
        
        # Performance metrics table
        perf_metrics = self.results.get('performance_metrics', {})
        if perf_metrics:
            perf_table = Table(title="Performance Metrics", box=box.ROUNDED)
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Average", style="white")
            perf_table.add_column("Maximum", style="white")
            perf_table.add_column("Target", style="yellow")
            perf_table.add_column("Status", style="bold")
            
            avg_exec = perf_metrics.get('avg_execution_time_ms', 0)
            max_exec = perf_metrics.get('max_execution_time_ms', 0)
            avg_mem = perf_metrics.get('avg_memory_usage_mb', 0)
            max_mem = perf_metrics.get('max_memory_usage_mb', 0)
            
            perf_table.add_row(
                "Execution Time",
                f"{avg_exec:.1f}ms",
                f"{max_exec:.1f}ms",
                f"{self.config.hook_limit_ms}ms",
                "✅" if max_exec <= self.config.hook_limit_ms * 5 else "❌"
            )
            perf_table.add_row(
                "Memory Usage",
                f"{avg_mem:.1f}MB",
                f"{max_mem:.1f}MB",
                f"{self.config.memory_limit_mb}MB",
                "✅" if max_mem <= self.config.memory_limit_mb else "❌"
            )
            
            self.console.print(perf_table)
        
        # Quality gates table
        quality_gates = self.results.get('quality_gates', {})
        if quality_gates:
            gates_table = Table(title="Quality Gates", box=box.ROUNDED)
            gates_table.add_column("Gate", style="cyan")
            gates_table.add_column("Status", style="bold")
            gates_table.add_column("Description", style="white")
            
            gate_descriptions = {
                'hook_performance': 'Hook execution within performance limits',
                'memory_usage': 'Memory usage within acceptable bounds',
                'success_rate': 'Test success rate meets minimum threshold',
                'performance_violations': 'Performance violations within tolerance'
            }
            
            for gate, passed in quality_gates.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                description = gate_descriptions.get(gate, gate.replace('_', ' ').title())
                gates_table.add_row(gate.replace('_', ' ').title(), status, description)
            
            self.console.print(gates_table)
        
        # Failed tests details
        detailed_results = self.results.get('detailed_results', {})
        failed_tests = {name: result for name, result in detailed_results.items() 
                       if not result.get('passed', True)}
        
        if failed_tests:
            self.console.print(f"\n[bold red]Failed Tests Details:[/bold red]")
            for test_name, result in failed_tests.items():
                message = result.get('message', 'Unknown error')
                self.console.print(f"❌ {test_name}: {message}")
    
    def _display_results_basic(self) -> None:
        """Display results with basic formatting"""
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        overall_success = self.results.get('overall_success', False)
        status_text = "ALL SYSTEMS GO" if overall_success else "QUALITY GATES FAILED"
        print(f"Overall Status: {status_text}")
        
        success_rate = self.results.get('success_rate', 0)
        passed_tests = self.results.get('passed_tests', 0)
        total_tests = self.results.get('total_tests', 0)
        session_duration = self.results.get('session_duration_ms', 0) / 1000
        
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Duration: {session_duration:.1f}s")
        
        # Performance metrics
        perf_metrics = self.results.get('performance_metrics', {})
        if perf_metrics:
            print("\nPerformance Metrics:")
            print(f"  Average Execution Time: {perf_metrics.get('avg_execution_time_ms', 0):.1f}ms")
            print(f"  Maximum Execution Time: {perf_metrics.get('max_execution_time_ms', 0):.1f}ms")
            print(f"  Average Memory Usage: {perf_metrics.get('avg_memory_usage_mb', 0):.1f}MB")
            print(f"  Maximum Memory Usage: {perf_metrics.get('max_memory_usage_mb', 0):.1f}MB")
        
        # Quality gates
        quality_gates = self.results.get('quality_gates', {})
        if quality_gates:
            print("\nQuality Gates:")
            for gate, passed in quality_gates.items():
                status = "PASS" if passed else "FAIL"
                print(f"  {gate.replace('_', ' ').title()}: {status}")
        
        print("=" * 60)
    
    def _generate_html_report(self) -> None:
        """Generate detailed HTML report"""
        if not self.config.output_file:
            self.config.output_file = Path("claude_sync_test_report.html")
        
        html_content = self._create_html_report()
        
        with open(self.config.output_file, 'w') as f:
            f.write(html_content)
        
        if self.console:
            self.console.print(f"📄 HTML report generated: {self.config.output_file}")
        else:
            print(f"HTML report generated: {self.config.output_file}")
    
    def _create_html_report(self) -> str:
        """Create HTML report content"""
        overall_success = self.results.get('overall_success', False)
        status_color = "#28a745" if overall_success else "#dc3545"
        status_text = "ALL SYSTEMS GO" if overall_success else "QUALITY GATES FAILED"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Claude-Sync End-to-End Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: {status_color}; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .table {{ border-collapse: collapse; width: 100%; }}
        .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .table th {{ background-color: #f2f2f2; }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        .warn {{ color: #ffc107; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Claude-Sync End-to-End Test Report</h1>
        <h2>{status_text}</h2>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h3>Test Summary</h3>
        <table class="table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Success Rate</td><td>{self.results.get('success_rate', 0):.1%}</td></tr>
            <tr><td>Tests Passed</td><td>{self.results.get('passed_tests', 0)}/{self.results.get('total_tests', 0)}</td></tr>
            <tr><td>Duration</td><td>{self.results.get('session_duration_ms', 0) / 1000:.1f}s</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h3>Quality Gates</h3>
        <table class="table">
            <tr><th>Gate</th><th>Status</th></tr>
"""
        
        quality_gates = self.results.get('quality_gates', {})
        for gate, passed in quality_gates.items():
            status_class = "pass" if passed else "fail"
            status_text = "PASS" if passed else "FAIL"
            html += f'<tr><td>{gate.replace("_", " ").title()}</td><td class="{status_class}">{status_text}</td></tr>'
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h3>Performance Metrics</h3>
        <table class="table">
            <tr><th>Metric</th><th>Average</th><th>Maximum</th></tr>
"""
        
        perf_metrics = self.results.get('performance_metrics', {})
        html += f"""
            <tr><td>Execution Time</td><td>{perf_metrics.get('avg_execution_time_ms', 0):.1f}ms</td><td>{perf_metrics.get('max_execution_time_ms', 0):.1f}ms</td></tr>
            <tr><td>Memory Usage</td><td>{perf_metrics.get('avg_memory_usage_mb', 0):.1f}MB</td><td>{perf_metrics.get('max_memory_usage_mb', 0):.1f}MB</td></tr>
        """
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h3>Detailed Test Results</h3>
        <table class="table">
            <tr><th>Test Name</th><th>Status</th><th>Execution Time</th><th>Memory Used</th><th>Message</th></tr>
"""
        
        detailed_results = self.results.get('detailed_results', {})
        for test_name, result in detailed_results.items():
            passed = result.get('passed', False)
            status_class = "pass" if passed else "fail"
            status_text = "PASS" if passed else "FAIL"
            
            html += f"""
            <tr>
                <td>{test_name.replace('_', ' ').title()}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{result.get('execution_time_ms', 0):.1f}ms</td>
                <td>{result.get('memory_used_mb', 0):.1f}MB</td>
                <td>{result.get('message', 'N/A')}</td>
            </tr>
            """
        
        html += """
        </table>
    </div>
</body>
</html>
"""
        
        return html

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Claude-Sync End-to-End Test Runner')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick validation tests only')
    parser.add_argument('--full', action='store_true', 
                       help='Run comprehensive full-system tests (default)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Include performance benchmarking')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed HTML report')
    parser.add_argument('--output', type=str,
                       help='Output file for HTML report')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    # Performance configuration
    parser.add_argument('--hook-limit-ms', type=int, default=10,
                       help='Hook execution time limit in milliseconds')
    parser.add_argument('--memory-limit-mb', type=int, default=50,
                       help='Memory usage limit in MB')
    parser.add_argument('--concurrent-hooks', type=int, default=10,
                       help='Number of concurrent hooks for stress testing')
    parser.add_argument('--stress-duration-s', type=int, default=30,
                       help='Stress test duration in seconds')
    
    args = parser.parse_args()
    
    # Determine test mode
    test_mode = "full"  # default
    if args.quick:
        test_mode = "quick"
    elif args.benchmark:
        test_mode = "benchmark"
    
    # Create configuration
    config = TestRunnerConfig(
        test_mode=test_mode,
        generate_report=args.report,
        verbose=args.verbose,
        output_file=Path(args.output) if args.output else None,
        hook_limit_ms=args.hook_limit_ms,
        memory_limit_mb=args.memory_limit_mb,
        concurrent_hooks=args.concurrent_hooks,
        stress_duration_s=args.stress_duration_s
    )
    
    # Run tests
    runner = EndToEndTestRunner(config)
    success = runner.run_tests()
    
    # Save results to JSON for CI/CD integration
    results_file = Path("test_results.json")
    with open(results_file, 'w') as f:
        json.dump(runner.results, f, indent=2, default=str)
    
    if config.verbose or not RICH_AVAILABLE:
        print(f"Results saved to: {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()