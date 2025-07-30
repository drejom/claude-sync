#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0",
#   "psutil>=5.9.0"
# ]
# ///
"""
Claude-Sync Learning System - Adaptive AI-powered development intelligence

This package provides the core learning system for claude-sync, implementing
adaptive schema evolution, information threshold management, and intelligent
pattern recognition for command optimization and cross-host learning.

Key Components:
- AdaptiveLearningSchema: NoSQL-style schema that evolves with usage patterns
- InformationThresholdManager: Intelligent agent triggering based on information density
- LearningStorage: Encrypted persistence with fast pattern lookups
- AdvancedCommandAbstractor: Semantic command analysis and pattern recognition
- PerformanceMonitor: Real-time performance tracking and optimization
- LearningEngine: Main integration layer coordinating all components

Usage:
    from learning import create_learning_engine
    engine = create_learning_engine()
    
    # Process command execution
    result = engine.process_command_execution(command_data)
    
    # Get optimization suggestions
    suggestions = engine.get_command_suggestions("grep pattern file.txt")
    
    # Get learning context for prompts
    context = engine.get_learning_context_for_prompt("Help optimize my workflow")

Based on REFACTOR_PLAN.md sections 310-1655 (Adaptive Learning System)
"""

# Import main components for easy access
from .learning_engine import LearningEngine, create_learning_engine
from .adaptive_schema import AdaptiveLearningSchema
from .threshold_manager import InformationThresholdManager, ThresholdEvent
from .learning_storage import LearningStorage
from .command_abstractor import AdvancedCommandAbstractor, CommandAnalysis
from .performance_monitor import PerformanceMonitor, PerformanceAlert
from .abstraction import SecureAbstractor
from .encryption import SecureLearningStorage

# Import from parent interfaces for convenience
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from interfaces import (
    CommandExecutionData,
    OptimizationPattern,
    LearningPattern,
    HookResult,
    PerformanceTargets,
    InformationTypes,
    AgentNames
)

# Version information
__version__ = "1.0.0"
__author__ = "Claude-Sync Learning Team"
__description__ = "Adaptive AI-powered development intelligence for claude-sync"

# Package exports
__all__ = [
    # Main interfaces
    'create_learning_engine',
    'LearningEngine',
    
    # Core components
    'AdaptiveLearningSchema',
    'InformationThresholdManager',
    'LearningStorage', 
    'AdvancedCommandAbstractor',
    'PerformanceMonitor',
    'SecureAbstractor',
    'SecureLearningStorage',
    
    # Data structures
    'CommandExecutionData',
    'OptimizationPattern',
    'LearningPattern',
    'CommandAnalysis',
    'ThresholdEvent',
    'PerformanceAlert',
    
    # Constants
    'PerformanceTargets',
    'InformationTypes', 
    'AgentNames',
    
    # Utilities
    'get_learning_system_status',
    'validate_learning_configuration'
]

# Utility functions
def get_learning_system_status(storage_dir: Path = None) -> dict:
    """Get comprehensive learning system status without full initialization"""
    try:
        from pathlib import Path
        import json
        
        storage_dir = storage_dir or Path.home() / '.claude' / 'learning'
        
        # Check if learning data exists
        schema_file = storage_dir / 'adaptive_schema.json'
        threshold_file = storage_dir / 'threshold_manager_state.json'
        cache_file = storage_dir / 'learning_cache.json'
        
        status = {
            'learning_data_exists': False,
            'components_initialized': {
                'adaptive_schema': schema_file.exists(),
                'threshold_manager': threshold_file.exists(),
                'learning_cache': cache_file.exists()
            },
            'storage_directory': str(storage_dir),
            'total_size_mb': 0.0
        }
        
        # Calculate total storage size
        if storage_dir.exists():
            total_size = sum(f.stat().st_size for f in storage_dir.rglob('*') if f.is_file())
            status['total_size_mb'] = total_size / (1024 * 1024)
            status['learning_data_exists'] = total_size > 0
        
        # Get basic statistics if data exists
        if schema_file.exists():
            try:
                with open(schema_file, 'r') as f:
                    schema_data = json.load(f)
                status['schema_version'] = schema_data.get('version', 1)
                status['pattern_count'] = len(schema_data.get('pattern_registry', {}))
            except:
                pass
        
        return status
        
    except Exception as e:
        return {'error': str(e)}

def validate_learning_configuration(config: dict) -> dict:
    """Validate learning system configuration"""
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Check required configuration options
        if 'enabled' in config and not isinstance(config['enabled'], bool):
            validation_result['errors'].append("'enabled' must be a boolean")
        
        # Check storage directory
        if 'storage_dir' in config:
            try:
                storage_path = Path(config['storage_dir'])
                if not storage_path.parent.exists():
                    validation_result['errors'].append(f"Parent directory does not exist: {storage_path.parent}")
            except Exception as e:
                validation_result['errors'].append(f"Invalid storage_dir path: {e}")
        
        # Check performance targets
        if 'performance_targets' in config:
            targets = config['performance_targets']
            if not isinstance(targets, dict):
                validation_result['errors'].append("'performance_targets' must be a dictionary")
            else:
                for target_name, target_value in targets.items():
                    if not isinstance(target_value, (int, float)) or target_value <= 0:
                        validation_result['errors'].append(f"Invalid performance target {target_name}: must be positive number")
        
        # Set validation result
        validation_result['valid'] = len(validation_result['errors']) == 0
        
        return validation_result
        
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Configuration validation failed: {e}"],
            'warnings': []
        }

# Module-level initialization check
def _check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import cryptography
    except ImportError:
        missing_deps.append('cryptography>=41.0.0')
    
    try:
        import psutil
    except ImportError:
        missing_deps.append('psutil>=5.9.0')
    
    if missing_deps:
        import warnings
        warnings.warn(
            f"Missing dependencies for claude-sync learning system: {', '.join(missing_deps)}. "
            "Install with: uv add " + " ".join(missing_deps),
            ImportWarning
        )

# Check dependencies on import
_check_dependencies()

# Example usage documentation
if __name__ == "__main__":
    print(__doc__)
    print(f"\nVersion: {__version__}")
    print(f"Components: {len(__all__)} exported")
    
    # Show system status
    status = get_learning_system_status()
    print(f"Learning system status: {status}")
    
    # Example configuration validation
    example_config = {
        'enabled': True,
        'storage_dir': '/tmp/test_learning',
        'performance_targets': {
            'learning_operation': 100,
            'pattern_lookup': 1
        }
    }
    
    validation = validate_learning_configuration(example_config)
    print(f"Configuration validation: {'Valid' if validation['valid'] else 'Invalid'}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")