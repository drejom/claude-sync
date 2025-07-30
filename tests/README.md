# Claude-Sync Testing Framework

Comprehensive testing suite for validating all integrated components of the claude-sync system.

## Overview

This testing framework provides comprehensive validation of the claude-sync system across multiple dimensions:

- **Unit Tests**: Individual component functionality validation
- **Integration Tests**: Component interaction and data flow validation  
- **Performance Tests**: Benchmarking against defined performance targets
- **Security Tests**: Encryption, authentication, and attack resistance validation
- **Mock Data Generation**: Realistic test data for consistent testing

## Quick Start

### Run All Tests
```bash
# Run comprehensive test suite
cd ~/.claude/claude-sync/tests
python run_all_tests.py

# Quick test mode (faster execution)
python run_all_tests.py --quick

# Run specific test categories
python run_all_tests.py --categories unit integration performance
```

### Run Individual Test Categories
```bash
# Unit tests only
python test_unit_components.py

# Integration tests only  
python test_integration.py

# Performance benchmarking
python test_performance.py

# Security validation
python test_security_validation.py
```

## Test Framework Architecture

### Core Components

#### `test_framework.py`
Core testing infrastructure providing:
- Test execution orchestration
- Performance monitoring and benchmarking
- Test environment isolation
- Comprehensive result reporting

#### `mock_data_generators.py`
Realistic test data generation:
- Claude Code hook input simulation
- HPC/bioinformatics command patterns
- Learning data generation
- Error scenario simulation

#### `run_all_tests.py`
Comprehensive test runner with:
- Parallel test execution
- Quality gate evaluation
- Performance analysis
- Detailed reporting

### Test Categories

#### Unit Tests (`test_unit_components.py`)

**Hook System Tests:**
- `intelligent-optimizer.py` (PreToolUse hook)
- `learning-collector.py` (PostToolUse hook)  
- `context-enhancer.py` (UserPromptSubmit hook)
- Performance target validation
- Error handling verification

**Learning System Tests:**
- Learning storage operations
- Adaptive schema evolution
- Information threshold management

**Security System Tests:**
- Encryption/decryption cycles
- Hardware identity generation
- Key rotation functionality

**Bootstrap System Tests:**
- Activation/deactivation workflows
- Settings file management
- Symlink creation/cleanup

#### Integration Tests (`test_integration.py`)

**Component Integration:**
- Hook ↔ Learning system data flow
- Security ↔ Learning encrypted storage
- Bootstrap ↔ Hook deployment
- Cross-component interface validation

**Workflow Integration:**
- Bioinformatics workflow simulation
- HPC troubleshooting scenarios
- Learning data accumulation
- Context enhancement workflows

#### Performance Tests (`test_performance.py`)

**Performance Targets:**
- PreToolUse hooks: ≤10ms execution
- PostToolUse hooks: ≤50ms execution  
- UserPromptSubmit hooks: ≤20ms execution
- Learning operations: ≤100ms
- Encryption operations: ≤5ms
- Total system memory: ≤100MB

**Benchmarking:**
- Hook execution time measurement
- Memory usage profiling
- Concurrent operation testing
- Performance regression detection

#### Security Tests (`test_security_validation.py`)

**Security Validation:**
- Encryption/decryption integrity
- Key rotation security
- Hardware identity stability
- Host authorization security
- Data abstraction privacy protection
- Attack scenario simulation

## Performance Targets

The testing framework validates against these production-ready performance targets:

| Component | Target | Validation |
|-----------|--------|------------|
| PreToolUse Hook | ≤10ms | 95th percentile execution time |
| PostToolUse Hook | ≤50ms | 95th percentile execution time |
| UserPromptSubmit Hook | ≤20ms | 95th percentile execution time |
| Learning Operations | ≤100ms | Storage/retrieval operations |
| Encryption Operations | ≤5ms | Encrypt/decrypt cycles |
| Key Rotation | ≤1000ms | Daily rotation process |
| System Memory | ≤100MB | Total system footprint |
| Hook Memory | ≤10MB | Per-hook memory usage |

## Mock Data Generation

### Hook Input Patterns

The framework generates realistic Claude Code hook inputs for:

**HPC Commands:**
```json
{
  "tool_name": "Bash",
  "tool_input": {
    "command": "sbatch --partition=compute --time=04:00:00 --mem=16GB analysis.sh"
  },
  "context": {
    "working_directory": "/scratch/analysis_run_001",
    "host_info": {
      "abstract_host_id": "compute_node-1234",
      "host_type": "compute_node",
      "capabilities": ["high_cpu", "slurm_worker", "singularity"]
    }
  }
}
```

**Bioinformatics Workflows:**
```json
{
  "tool_input": {
    "command": "singularity exec --bind /data:/data biotools.sif Rscript analysis.R"
  },
  "tool_output": {
    "exit_code": 0,
    "duration_ms": 15000,
    "stdout": "Analysis complete. Results saved to output/"
  }
}
```

**User Prompts:**
```json
{
  "user_prompt": "How do I optimize my SLURM jobs for better queue times?",
  "context": {
    "recent_commands": ["sbatch job.sh", "squeue -u $USER"],
    "context_hints": ["slurm_patterns", "performance_insights"]
  }
}
```

## Test Environment Isolation

The framework creates isolated test environments to prevent interference:

```
test_environment/
├── .claude/
│   ├── learning/        # Isolated learning data
│   ├── security/        # Test security keys
│   └── hooks/          # Test hook symlinks
├── test_project/       # Mock project structure
└── backups/           # Test backup files
```

Environment variables ensure test isolation:
- `CLAUDE_SYNC_TEST_MODE=1`
- `CLAUDE_SYNC_DATA_DIR=/path/to/test/env`
- `PYTHONPATH` includes project root

## Quality Gates

The framework enforces quality gates for production readiness:

### Overall Quality Requirements
- **Minimum Success Rate**: 80% across all tests
- **Critical Categories**: Unit, Integration, Security must achieve ≥85%
- **Performance Violations**: ≤10% of tests may exceed performance targets
- **Security Tests**: ≥90% success rate required

### Category-Specific Requirements
- **Unit Tests**: ≥85% (critical system components)
- **Integration Tests**: ≥70% (complex component interactions)
- **Performance Tests**: ≥70% (system performance variations)
- **Security Tests**: ≥90% (security is non-negotiable)

## Continuous Integration

The framework supports CI/CD integration:

```bash
# CI-friendly execution
python run_all_tests.py \
  --categories unit integration performance security \
  --quick \
  --output ./ci_results \
  --sequential

# Exit codes
# 0: All quality gates passed
# 1: Quality gates failed
```

## Test Data and Patterns

### Realistic Command Patterns

**SLURM Commands:**
- Job submission with various resource requests
- Queue monitoring and job management
- Array jobs and dependency handling

**Container Workflows:**
- Singularity execution with bind mounts
- Multi-stage container pipelines
- GPU-accelerated container jobs

**Data Processing:**
- Large-scale genomics data processing
- R statistical analysis workflows
- SSH and network operations

### Error Scenarios

**Hook Failures:**
- Invalid input handling
- Timeout scenarios
- Resource exhaustion
- Component unavailability

**Integration Failures:**
- Network partitions
- Key rotation failures
- Learning data corruption
- Settings conflicts

## Extending the Framework

### Adding New Tests

1. **Create Test Class:**
```python
class NewFeatureTests:
    def __init__(self, test_env: TestEnvironment):
        self.test_env = test_env
    
    def test_new_functionality(self) -> Tuple[bool, str]:
        # Test implementation
        return True, "Test passed"
```

2. **Register Test Suite:**
```python
def create_test_suites(test_env):
    new_suite = TestSuite(
        name="new_feature_tests",
        tests=[NewFeatureTests(test_env).test_new_functionality]
    )
    return [new_suite]
```

### Custom Mock Data

```python
class CustomMockGenerator(MockDataGenerator):
    def generate_custom_pattern(self):
        return {
            "custom_field": "custom_value",
            "timestamp": time.time()
        }
```

## Troubleshooting

### Common Issues

**Tests Failing Due to Missing Components:**
```bash
# Ensure all modules are available
python -c "import sys; sys.path.insert(0, '.'); from interfaces import *"
```

**Performance Test Variations:**
- System load affects performance measurements
- Run tests on idle system for consistent results
- Use `--quick` mode for faster feedback

**Test Environment Cleanup:**
```bash
# Manual cleanup if needed
rm -rf test_results/
find /tmp -name "claude_sync_test_*" -type d -exec rm -rf {} +
```

### Debug Mode

```bash
# Verbose output
python run_all_tests.py --verbose

# Single test debugging
python test_unit_components.py
```

## Contributing

When adding new tests:

1. Follow existing patterns and naming conventions
2. Include both positive and negative test cases
3. Add appropriate mock data generators
4. Validate performance impact
5. Update this README with new test descriptions

## Report Generation

Test reports are generated in JSON format:

```json
{
  "session_id": "comprehensive_test_1234567890",
  "overall_success_rate": 0.85,
  "quality_gates_passed": true,
  "category_results": [
    {
      "category": "unit",
      "success_rate": 0.90,
      "total_tests": 20,
      "performance_violations": 0
    }
  ],
  "recommendations": [
    "✅ All tests passed quality gates - system ready for production deployment"
  ]
}
```

## Integration with Claude Code

The testing framework validates integration with Claude Code:

- **Hook Interface Compliance**: Validates JSON input/output format
- **Performance Requirements**: Ensures hooks meet Claude Code timing requirements  
- **Error Handling**: Verifies graceful failure modes
- **Settings Integration**: Tests merge with existing Claude Code settings

This comprehensive testing framework ensures claude-sync integrates reliably with Claude Code while maintaining high performance and security standards.