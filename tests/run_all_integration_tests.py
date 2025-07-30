#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pytest>=7.0.0"
# ]
# ///
"""
Comprehensive Integration Test Runner for Claude-Sync

Executes all integration tests and provides comprehensive reporting.
This is the main entry point for validating the entire claude-sync system.
"""

import json
import time
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

def run_test_script(script_path: Path, description: str) -> Tuple[bool, Dict[str, Any]]:
    """Run a test script and parse results"""
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = (time.time() - start_time) * 1000
        success = result.returncode == 0
        
        return success, {
            'script': script_path.name,
            'description': description,
            'success': success,
            'duration_ms': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        duration = (time.time() - start_time) * 1000
        return False, {
            'script': script_path.name,
            'description': description,  
            'success': False,
            'duration_ms': duration,
            'stdout': '',
            'stderr': 'Test timed out after 5 minutes',
            'returncode': -1
        }
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        return False, {
            'script': script_path.name,
            'description': description,
            'success': False,
            'duration_ms': duration,
            'stdout': '',
            'stderr': f'Exception: {str(e)}',
            'returncode': -1
        }

def extract_test_stats(output: str) -> Dict[str, Any]:
    """Extract test statistics from output"""
    stats = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'success_rate': 0.0,
        'total_time_ms': 0.0
    }
    
    lines = output.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Look for test summary patterns
        if 'total |' in line and 'passed |' in line:
            # Parse "Tests: X total | Y passed | Z failed"
            parts = line.split('|')
            for part in parts:
                part = part.strip()
                if 'total' in part:
                    try:
                        stats['total_tests'] = int(part.split()[1])
                    except (IndexError, ValueError):
                        pass
                elif 'passed' in part:
                    try:
                        stats['passed_tests'] = int(part.split()[0])
                    except (IndexError, ValueError):
                        pass
                elif 'failed' in part:
                    try:
                        stats['failed_tests'] = int(part.split()[0])
                    except (IndexError, ValueError):
                        pass
        
        elif 'Success Rate:' in line:
            try:
                rate_str = line.split('Success Rate:')[1].strip().rstrip('%')
                stats['success_rate'] = float(rate_str)
            except (IndexError, ValueError):
                pass
        
        elif 'Total Time:' in line:
            try:
                time_str = line.split('Total Time:')[1].strip().rstrip('ms')
                stats['total_time_ms'] = float(time_str)
            except (IndexError, ValueError):
                pass
    
    return stats

def print_test_summary(results: List[Dict[str, Any]]) -> None:
    """Print comprehensive test summary"""
    print("\n" + "=" * 100)
    print("🎯 CLAUDE-SYNC COMPREHENSIVE INTEGRATION TEST SUMMARY")
    print("=" * 100)
    
    total_scripts = len(results)
    successful_scripts = sum(1 for r in results if r['success'])
    total_duration = sum(r['duration_ms'] for r in results)
    
    print(f"📊 Test Scripts: {total_scripts} total | {successful_scripts} passed | {total_scripts - successful_scripts} failed")
    print(f"⏱️  Total Duration: {total_duration:.1f}ms ({total_duration/1000:.1f}s)")
    print(f"✅ Script Success Rate: {(successful_scripts/total_scripts)*100:.1f}%")
    
    # Aggregate test statistics
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    print(f"\n📋 Detailed Results:")
    print("-" * 100)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = result['duration_ms']
        
        print(f"{status} {result['description']:<40} ({duration:>6.1f}ms)")
        
        # Extract and aggregate test stats
        stats = extract_test_stats(result['stdout'])
        if stats['total_tests'] > 0:
            total_tests += stats['total_tests']
            total_passed += stats['passed_tests']
            total_failed += stats['failed_tests']
            
            print(f"     └─ {stats['total_tests']} tests | {stats['passed_tests']} passed | {stats['success_rate']:.1f}% success")
        
        # Show errors for failed scripts
        if not result['success'] and result['stderr']:
            error_lines = result['stderr'].split('\n')[:3]  # First 3 lines
            for line in error_lines:
                if line.strip():
                    print(f"     └─ Error: {line.strip()}")
    
    # Overall test statistics
    if total_tests > 0:
        overall_success_rate = (total_passed / total_tests) * 100
        print(f"\n📈 Aggregate Test Statistics:")
        print(f"   Total Individual Tests: {total_tests}")
        print(f"   Individual Tests Passed: {total_passed}")
        print(f"   Individual Tests Failed: {total_failed}")
        print(f"   Overall Success Rate: {overall_success_rate:.1f}%")
    
    print("\n" + "=" * 100)
    
    # Final assessment
    script_threshold = 80  # 80% of scripts must pass
    test_threshold = 75   # 75% of individual tests must pass
    
    script_success_rate = (successful_scripts / total_scripts) * 100
    individual_success_rate = (total_passed / max(total_tests, 1)) * 100
    
    if script_success_rate >= script_threshold and individual_success_rate >= test_threshold:
        print("🎉 CLAUDE-SYNC INTEGRATION TESTING: PASSED")
        print(f"   ✅ Script Success Rate: {script_success_rate:.1f}% (≥{script_threshold}%)")
        print(f"   ✅ Individual Test Success Rate: {individual_success_rate:.1f}% (≥{test_threshold}%)")
        print("   🚀 System is ready for deployment!")
    else:
        print("❌ CLAUDE-SYNC INTEGRATION TESTING: FAILED")
        if script_success_rate < script_threshold:
            print(f"   ❌ Script Success Rate: {script_success_rate:.1f}% (<{script_threshold}%)")
        if individual_success_rate < test_threshold:
            print(f"   ❌ Individual Test Success Rate: {individual_success_rate:.1f}% (<{test_threshold}%)")
        print("   ⚠️  System needs attention before deployment")
    
    print("=" * 100)

def main():
    """Run all integration tests"""
    print("🚀 Starting Claude-Sync Comprehensive Integration Testing")
    print("=" * 100)
    
    test_dir = Path(__file__).parent
    
    # Define test scripts to run
    test_scripts = [
        (test_dir / "test_integration_basic.py", "Basic Integration Tests"),
        (test_dir / "test_integration_complete.py", "Complete Integration Tests"),
    ]
    
    # Add advanced tests if they exist
    advanced_scripts = [
        (test_dir / "test_end_to_end.py", "End-to-End Integration Tests"),
        (test_dir / "test_performance.py", "Performance Integration Tests"),
        (test_dir / "test_security_validation.py", "Security Integration Tests")
    ]
    
    for script_path, description in advanced_scripts:
        if script_path.exists():
            test_scripts.append((script_path, description))
        else:
            print(f"ℹ️  Optional test not found: {description} ({script_path.name})")
    
    print(f"📋 Running {len(test_scripts)} test suites...")
    print("-" * 100)
    
    # Run all tests
    results = []
    start_time = time.time()
    
    for i, (script_path, description) in enumerate(test_scripts, 1):
        print(f"🔄 [{i}/{len(test_scripts)}] Running {description}...")
        
        success, result = run_test_script(script_path, description)
        results.append(result)
        
        if success:
            print(f"   ✅ {description} completed successfully ({result['duration_ms']:.1f}ms)")
        else:
            print(f"   ❌ {description} failed ({result['duration_ms']:.1f}ms)")
            if result['stderr']:
                print(f"   └─ {result['stderr'].split(chr(10))[0]}")  # First error line
    
    total_duration = time.time() - start_time
    
    print(f"\n✨ All test suites completed in {total_duration:.1f}s")
    
    # Print comprehensive summary
    print_test_summary(results)
    
    # Determine exit code
    successful_scripts = sum(1 for r in results if r['success'])
    success_rate = (successful_scripts / len(results)) * 100
    
    if success_rate >= 80:
        return 0  # Success
    else:
        return 1  # Failure

if __name__ == "__main__":
    exit(main())