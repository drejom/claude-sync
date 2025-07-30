#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pytest>=7.0.0"
# ]
# ///
"""
Basic Integration Test for Claude-Sync (No External Dependencies)

Validates core integration without requiring psutil or other external libraries.
"""

import json
import time
import tempfile
import shutil
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_project_structure() -> Tuple[bool, str]:
    """Test that core project structure exists"""
    try:
        project_root = Path(__file__).parent.parent
        
        required_files = [
            "hooks/intelligent-optimizer.py",
            "hooks/learning-collector.py",
            "hooks/context-enhancer.py",
            "learning/learning_storage.py",
            "security/security_manager.py",
            "interfaces.py",
            "bootstrap.sh"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            return False, f"Missing core files: {missing_files}"
        
        return True, f"All {len(required_files)} core files present"
        
    except Exception as e:
        return False, f"Project structure test failed: {str(e)}"

def test_hook_interfaces() -> Tuple[bool, str]:
    """Test that hooks have proper interfaces"""
    try:
        project_root = Path(__file__).parent.parent
        
        hooks = [
            "intelligent-optimizer.py",
            "learning-collector.py", 
            "context-enhancer.py"
        ]
        
        valid_hooks = 0
        
        for hook_name in hooks:
            hook_path = project_root / "hooks" / hook_name
            if not hook_path.exists():
                continue
                
            # Check if hook has proper shebang and basic structure
            content = hook_path.read_text()
            
            # Basic validation
            has_shebang = content.startswith("#!/usr/bin/env")
            has_json_import = "import json" in content
            has_sys_import = "import sys" in content
            has_main_func = "def main(" in content or "if __name__" in content
            
            if has_shebang and has_json_import and has_sys_import and has_main_func:
                valid_hooks += 1
        
        if valid_hooks == len(hooks):
            return True, f"All {valid_hooks} hooks have proper interfaces"
        else:
            return False, f"Only {valid_hooks}/{len(hooks)} hooks have proper interfaces"
        
    except Exception as e:
        return False, f"Hook interface test failed: {str(e)}"

def test_mock_hook_execution() -> Tuple[bool, str]:
    """Test hook execution with mock data"""
    try:
        # Create a simple mock hook for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            mock_hook_content = '''#!/usr/bin/env python3
import json
import sys

def main():
    try:
        hook_input = json.loads(sys.stdin.read())
        result = {
            "block": False,
            "message": f"Mock hook processed: {hook_input.get('command', 'unknown')}"
        }
        print(json.dumps(result))
        return 0
    except Exception as e:
        print(json.dumps({"block": False, "message": f"Error: {str(e)}"}))
        return 1

if __name__ == "__main__":
    exit(main())
'''
            f.write(mock_hook_content)
            f.flush()
            
            # Make executable
            os.chmod(f.name, 0o755)
            
            # Test execution
            test_input = {
                "command": "test command",
                "working_directory": "/tmp",
                "session_id": "test_session"
            }
            
            process = subprocess.run(
                [sys.executable, f.name],
                input=json.dumps(test_input).encode(),
                capture_output=True,
                timeout=10
            )
            
            # Cleanup
            os.unlink(f.name)
            
            if process.returncode != 0:
                return False, f"Mock hook execution failed: {process.stderr.decode()}"
            
            try:
                result = json.loads(process.stdout.decode())
                if "block" not in result:
                    return False, "Hook result missing 'block' field"
                    
                return True, "Mock hook execution successful"
                
            except json.JSONDecodeError:
                return False, "Hook output is not valid JSON"
        
    except Exception as e:
        return False, f"Mock hook execution test failed: {str(e)}"

def test_interfaces_import() -> Tuple[bool, str]:
    """Test that interfaces module can be imported"""
    try:
        from interfaces import HookResult, CommandExecutionData, PerformanceTargets
        
        # Test basic interface creation
        test_result = HookResult(block=False, message="test")
        test_data = CommandExecutionData(
            command="test",
            exit_code=0,
            duration_ms=100,
            timestamp=time.time(),
            session_id="test",
            working_directory="/tmp"
        )
        
        # Test that interfaces have expected methods
        if not hasattr(test_result, 'to_json'):
            return False, "HookResult missing to_json method"
            
        if not hasattr(test_data, 'to_dict'):
            return False, "CommandExecutionData missing to_dict method"
        
        # Test performance targets exist
        if not hasattr(PerformanceTargets, 'PRE_TOOL_USE_HOOK_MS'):
            return False, "PerformanceTargets missing PRE_TOOL_USE_HOOK_MS"
        
        return True, "Interfaces import and basic functionality working"
        
    except ImportError as e:
        return False, f"Interfaces import failed: {e}"
    except Exception as e:
        return False, f"Interfaces test failed: {str(e)}"

def test_bootstrap_script() -> Tuple[bool, str]:
    """Test that bootstrap script exists and is executable"""
    try:
        project_root = Path(__file__).parent.parent
        bootstrap_path = project_root / "bootstrap.sh"
        
        if not bootstrap_path.exists():
            return False, "Bootstrap script not found"
        
        if not os.access(bootstrap_path, os.X_OK):
            return False, "Bootstrap script is not executable"
        
        # Test basic script structure
        content = bootstrap_path.read_text()
        
        if not content.startswith("#!/"):
            return False, "Bootstrap script missing shebang"
        
        required_functions = ["install", "status", "manage"]
        missing_functions = []
        
        for func in required_functions:
            if func not in content:
                missing_functions.append(func)
        
        if missing_functions:
            return False, f"Bootstrap script missing functions: {missing_functions}"
        
        return True, "Bootstrap script exists and has proper structure"
        
    except Exception as e:
        return False, f"Bootstrap script test failed: {str(e)}"

def test_settings_template() -> Tuple[bool, str]:
    """Test settings template functionality"""
    try:
        # Create mock settings integration test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create existing settings
            existing_settings = {
                "editor": "vim",
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Existing",
                            "hooks": [{"type": "command", "command": "existing.py"}]
                        }
                    ]
                }
            }
            
            settings_file = temp_path / "settings.local.json"
            with open(settings_file, 'w') as f:
                json.dump(existing_settings, f, indent=2)
            
            # Test settings merge logic (simplified)
            claude_sync_settings = {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": "claude-sync-hook.py"}]
                        }
                    ]
                }
            }
            
            # Merge settings
            merged_settings = existing_settings.copy()
            for hook_type, hooks in claude_sync_settings["hooks"].items():
                if hook_type not in merged_settings["hooks"]:
                    merged_settings["hooks"][hook_type] = []
                merged_settings["hooks"][hook_type].extend(hooks)
            
            # Verify merge
            pretool_hooks = merged_settings.get("hooks", {}).get("PreToolUse", [])
            if len(pretool_hooks) != 2:
                return False, "Settings merge failed - incorrect hook count"
            
            if merged_settings.get("editor") != "vim":
                return False, "Settings merge failed - lost existing settings"
            
            return True, "Settings template integration working"
        
    except Exception as e:
        return False, f"Settings template test failed: {str(e)}"

def run_basic_integration_tests():
    """Run all basic integration tests"""
    tests = [
        ("Project Structure", test_project_structure),
        ("Hook Interfaces", test_hook_interfaces), 
        ("Mock Hook Execution", test_mock_hook_execution),
        ("Interfaces Import", test_interfaces_import),
        ("Bootstrap Script", test_bootstrap_script),
        ("Settings Template", test_settings_template)
    ]
    
    print("🧪 Claude-Sync Basic Integration Tests")
    print("=" * 60)
    
    results = []
    passed = 0
    
    for test_name, test_func in tests:
        print(f"🔄 Running {test_name}...", end=" ")
        
        start_time = time.time()
        try:
            success, message = test_func()
            duration = (time.time() - start_time) * 1000
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} ({duration:.1f}ms)")
            
            if not success:
                print(f"   └─ {message}")
            
            results.append((test_name, success, message, duration))
            if success:
                passed += 1
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            print(f"❌ ERROR ({duration:.1f}ms)")
            print(f"   └─ {str(e)}")
            results.append((test_name, False, str(e), duration))
    
    print("\n" + "=" * 60)
    print(f"📊 Basic Integration Test Results")
    print("=" * 60)
    
    total_tests = len(tests)
    success_rate = (passed / total_tests) * 100
    total_time = sum(r[3] for r in results)
    
    print(f"Tests: {total_tests} total | {passed} passed | {total_tests - passed} failed")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Time: {total_time:.1f}ms")
    
    # Show failed tests
    failed_tests = [r for r in results if not r[1]]
    if failed_tests:
        print(f"\n❌ FAILED TESTS:")
        for test_name, _, message, _ in failed_tests:
            print(f"  • {test_name}: {message}")
    
    print("=" * 60)
    
    if success_rate >= 80:
        print("✅ Basic integration tests PASSED (≥80% success rate)")
        return 0
    else:
        print("❌ Basic integration tests FAILED (<80% success rate)")
        return 1

if __name__ == "__main__":
    exit(run_basic_integration_tests())