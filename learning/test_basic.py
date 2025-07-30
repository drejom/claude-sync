#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0",
#   "psutil>=5.9.0"
# ]
# ///

import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from interfaces import CommandExecutionData
from learning.learning_engine import LearningEngine

# Create test learning engine
print("Creating learning engine...")
engine = LearningEngine()

# Test basic command processing
print("Testing command processing...")
cmd_data = CommandExecutionData(
    command='grep pattern file.txt',
    exit_code=0,
    duration_ms=500,
    timestamp=time.time(),
    session_id='test',
    working_directory='/data'
)

result = engine.process_command_execution(cmd_data)
print('Command processed:', result['processed'])
print('Analysis category:', result.get('analysis', {}).get('category', 'unknown'))

# Test command suggestions
print("Testing command suggestions...")
suggestions = engine.get_command_suggestions('grep pattern *.txt')
print('Optimization opportunities:', len(suggestions.get('optimization_opportunities', [])))
print('Safety warnings:', len(suggestions.get('safety_warnings', [])))

# Test system status
print("Testing system status...")
status = engine.get_system_status()
print('System enabled:', status['enabled'])
print('Commands processed:', status['operation_stats']['commands_processed'])

engine.shutdown()
print('✅ Basic learning system test passed!')