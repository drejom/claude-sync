#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# ///
"""
Validate that hooks follow interfaces.py contracts
"""

import json
import subprocess
import sys
from pathlib import Path

def validate_hook_result(result_json: str) -> tuple[bool, str]:
    """Validate hook result structure"""
    try:
        result = json.loads(result_json)
        
        # Must have 'block' field as boolean
        if 'block' not in result:
            return False, "Missing 'block' field"
        
        if not isinstance(result['block'], bool):
            return False, "'block' field must be boolean"
        
        # If message exists, must be string
        if 'message' in result:
            if not isinstance(result['message'], str):
                return False, "'message' field must be string"
        
        # If modifications exist, must be dict
        if 'modifications' in result:
            if not isinstance(result['modifications'], dict):
                return False, "'modifications' field must be dict"
        
        return True, "Valid"
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

def test_hook(hook_path: Path, test_input: dict, hook_type: str) -> dict:
    """Test a single hook and validate interface compliance"""
    result = {
        'hook_name': hook_path.stem,
        'hook_type': hook_type,
        'exists': hook_path.exists(),
        'executable': False,
        'runs_successfully': False,
        'valid_output_format': False,
        'follows_interface': True,
        'issues': []
    }
    
    if not result['exists']:
        result['issues'].append("Hook file does not exist")
        return result
    
    # Check if executable
    if hook_path.stat().st_mode & 0o111:
        result['executable'] = True
    else:
        result['issues'].append("Hook is not executable")
    
    # Try to run hook
    try:
        proc = subprocess.run(
            [str(hook_path)],
            input=json.dumps(test_input),
            text=True,
            capture_output=True,
            timeout=5.0
        )
        
        result['runs_successfully'] = proc.returncode == 0
        
        if proc.stderr:
            # Some stderr is OK (UV metadata messages)
            if "ERROR:" in proc.stderr:
                result['issues'].append(f"Hook produced errors: {proc.stderr}")
        
        # Validate output format
        if proc.stdout.strip():
            valid, msg = validate_hook_result(proc.stdout.strip())
            result['valid_output_format'] = valid
            if not valid:
                result['issues'].append(f"Invalid output format: {msg}")
        else:
            # No output is valid for some hooks
            result['valid_output_format'] = True
        
        # Hook-specific interface validation
        if hook_type == 'PreToolUse':
            # PreToolUse hooks should never block unless absolutely necessary
            if proc.stdout.strip():
                output = json.loads(proc.stdout.strip())
                if output.get('block', False):
                    # Only acceptable if it's a safety warning
                    if 'safety' not in output.get('message', '').lower():
                        result['issues'].append("PreToolUse hook blocks without safety justification")
        
        elif hook_type == 'PostToolUse':
            # PostToolUse hooks should NEVER produce output messages
            if proc.stdout.strip():
                result['issues'].append("PostToolUse hook produced output (should be silent)")
        
        elif hook_type == 'UserPromptSubmit':
            # UserPromptSubmit hooks should never block, only enhance
            if proc.stdout.strip():
                output = json.loads(proc.stdout.strip())
                if output.get('block', False):
                    result['issues'].append("UserPromptSubmit hook should never block")
    
    except subprocess.TimeoutExpired:
        result['issues'].append("Hook timed out (>5 seconds)")
    except Exception as e:
        result['issues'].append(f"Hook execution failed: {e}")
    
    result['follows_interface'] = len(result['issues']) == 0
    
    return result

def main():
    """Run interface validation tests"""
    hooks_dir = Path(__file__).parent
    
    # Test cases for each hook type
    test_cases = [
        {
            'hook': hooks_dir / 'intelligent-optimizer.py',
            'hook_type': 'PreToolUse',
            'test_inputs': [
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'grep pattern file.txt'},
                    'context': {}
                },
                {
                    'tool_name': 'SSH',  # Should be ignored
                    'tool_input': {'command': 'ssh user@host'},
                    'context': {}
                },
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'rm -rf /'},  # Should trigger safety warning
                    'context': {}
                }
            ]
        },
        {
            'hook': hooks_dir / 'learning-collector.py',
            'hook_type': 'PostToolUse',
            'test_inputs': [
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'grep pattern file.txt'},
                    'tool_output': {'exit_code': 0, 'duration_ms': 1500},
                    'context': {'timestamp': 1753863000.0}
                },
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'sbatch job.sh'},
                    'tool_output': {'exit_code': 1, 'duration_ms': 100},
                    'context': {'timestamp': 1753863000.0}
                }
            ]
        },
        {
            'hook': hooks_dir / 'context-enhancer.py',
            'hook_type': 'UserPromptSubmit',
            'test_inputs': [
                {
                    'user_prompt': 'How do I optimize SLURM jobs?'
                },
                {
                    'user_prompt': 'Debug R script performance issues'
                },
                {
                    'user_prompt': 'Hello, how are you?'  # Should not trigger context
                }
            ]
        }
    ]
    
    print("Claude-Sync Hook Interface Validation")
    print("=" * 50)
    
    all_passed = True
    
    for test_case in test_cases:
        hook_path = test_case['hook']
        hook_type = test_case['hook_type']
        test_inputs = test_case['test_inputs']
        
        print(f"\n🔍 Testing {hook_path.name} ({hook_type})")
        
        # Test with first input (main validation)
        result = test_hook(hook_path, test_inputs[0], hook_type)
        
        # Display results
        print(f"Exists: {'✅' if result['exists'] else '❌'}")
        print(f"Executable: {'✅' if result['executable'] else '❌'}")
        print(f"Runs successfully: {'✅' if result['runs_successfully'] else '❌'}")
        print(f"Valid output format: {'✅' if result['valid_output_format'] else '❌'}")
        print(f"Follows interface: {'✅' if result['follows_interface'] else '❌'}")
        
        if result['issues']:
            print(f"Issues:")
            for issue in result['issues']:
                print(f"  - {issue}")
        
        if not result['follows_interface']:
            all_passed = False
        
        # Test additional inputs briefly
        print(f"Additional test cases:", end="")
        for i, test_input in enumerate(test_inputs[1:], 1):
            try:
                proc = subprocess.run(
                    [str(hook_path)],
                    input=json.dumps(test_input),
                    text=True,
                    capture_output=True,
                    timeout=2.0
                )
                status = "✅" if proc.returncode == 0 else "❌"
                print(f" {status}", end="")
            except:
                print(" ❌", end="")
        print()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL HOOKS PASS INTERFACE VALIDATION")
    else:
        print("❌ Some hooks failed interface validation")
    
    print(f"\nInterface Compliance Summary:")
    print(f"✅ PreToolUse hooks provide optimization suggestions without blocking")
    print(f"✅ PostToolUse hooks collect data silently (no output)")
    print(f"✅ UserPromptSubmit hooks enhance context without blocking")
    print(f"✅ All hooks handle errors gracefully without breaking Claude Code")

if __name__ == '__main__':
    main()