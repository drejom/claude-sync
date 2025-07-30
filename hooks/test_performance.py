#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# ///
"""
Performance test for claude-sync hooks

Tests that all hooks meet their performance targets:
- intelligent-optimizer.py: <10ms
- learning-collector.py: <50ms  
- context-enhancer.py: <20ms
"""

import json
import subprocess
import time
import statistics
from pathlib import Path

def test_hook_performance(hook_path: Path, test_input: dict, target_ms: int, iterations: int = 10) -> dict:
    """Test hook performance over multiple iterations"""
    durations = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                [str(hook_path)],
                input=json.dumps(test_input),
                text=True,
                capture_output=True,
                timeout=1.0  # 1 second timeout
            )
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            durations.append(duration_ms)
            
        except subprocess.TimeoutExpired:
            durations.append(1000)  # 1 second = timeout
    
    return {
        'hook_name': hook_path.stem,
        'target_ms': target_ms,
        'avg_ms': statistics.mean(durations),
        'median_ms': statistics.median(durations),
        'p95_ms': sorted(durations)[int(0.95 * len(durations))],
        'max_ms': max(durations),
        'min_ms': min(durations),
        'meets_target': all(d <= target_ms for d in durations),
        'success_rate': sum(1 for d in durations if d < 1000) / len(durations)
    }

def main():
    """Run performance tests on all hooks"""
    hooks_dir = Path(__file__).parent
    
    # Test cases for each hook
    test_cases = [
        {
            'hook': hooks_dir / 'intelligent-optimizer.py',
            'input': {
                'tool_name': 'Bash',
                'tool_input': {'command': 'find /data -name "*.txt" | grep pattern'},
                'context': {'working_directory': '/home/user'}
            },
            'target_ms': 10
        },
        {
            'hook': hooks_dir / 'learning-collector.py', 
            'input': {
                'tool_name': 'Bash',
                'tool_input': {'command': 'sbatch --mem=32G run_analysis.sh'},
                'tool_output': {'exit_code': 0, 'duration_ms': 2500},
                'context': {'timestamp': time.time(), 'working_directory': '/scratch'}
            },
            'target_ms': 50
        },
        {
            'hook': hooks_dir / 'context-enhancer.py',
            'input': {
                'user_prompt': 'How can I optimize my R scripts for SLURM? They keep running out of memory.'
            },
            'target_ms': 20
        }
    ]
    
    print("Claude-Sync Hook Performance Tests")
    print("=" * 50)
    
    all_passed = True
    
    for test_case in test_cases:
        hook_path = test_case['hook']
        test_input = test_case['input']
        target_ms = test_case['target_ms']
        
        print(f"\n🧪 Testing {hook_path.name}")
        print(f"Target: <{target_ms}ms")
        
        if not hook_path.exists():
            print(f"❌ Hook not found: {hook_path}")
            all_passed = False
            continue
        
        # Run performance test
        results = test_hook_performance(hook_path, test_input, target_ms)
        
        # Print results
        print(f"Average: {results['avg_ms']:.1f}ms")
        print(f"Median: {results['median_ms']:.1f}ms") 
        print(f"95th percentile: {results['p95_ms']:.1f}ms")
        print(f"Range: {results['min_ms']:.1f}ms - {results['max_ms']:.1f}ms")
        print(f"Success rate: {results['success_rate']:.1%}")
        
        if results['meets_target']:
            print(f"✅ PASS - All executions under {target_ms}ms")
        else:
            print(f"❌ FAIL - Some executions exceeded {target_ms}ms target")
            all_passed = False
            
        if results['p95_ms'] <= target_ms:
            print(f"✅ P95 target met ({results['p95_ms']:.1f}ms <= {target_ms}ms)")
        else:
            print(f"⚠️  P95 target missed ({results['p95_ms']:.1f}ms > {target_ms}ms)")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL HOOKS PASS PERFORMANCE TARGETS")
    else:
        print("❌ Some hooks failed performance targets")
    
    print(f"\nPerformance Summary:")
    print(f"- All hooks should execute in real-time (<100ms)")
    print(f"- PreToolUse hooks must be especially fast (<10ms)")
    print(f"- PostToolUse hooks can be slightly slower (<50ms)")
    print(f"- UserPromptSubmit hooks should be responsive (<20ms)")

if __name__ == '__main__':
    main()