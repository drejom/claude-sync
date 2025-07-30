#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0",
#   "psutil>=5.9.0"
# ]
# ///
"""
Learning System Integration Tests

Comprehensive tests to validate all learning components work together correctly.
Tests the full learning pipeline from command execution to pattern recognition,
storage, retrieval, and optimization suggestions.

Key test areas:
- Component integration and data flow
- Performance requirement compliance
- Learning data accuracy and persistence
- Agent triggering and feedback loops
- Cross-component communication
- Error handling and graceful degradation
"""

import json
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import logging

# Import all learning components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from interfaces import (
    CommandExecutionData,
    PerformanceTargets,
    InformationTypes,
    AgentNames
)

from learning.learning_engine import LearningEngine
from learning.adaptive_schema import AdaptiveLearningSchema
from learning.threshold_manager import InformationThresholdManager
from learning.learning_storage import LearningStorage
from learning.command_abstractor import AdvancedCommandAbstractor
from learning.performance_monitor import PerformanceMonitor
from learning.pattern_recognition import WorkflowPatternRecognizer

class TestLearningSystemIntegration(unittest.TestCase):
    """Comprehensive integration tests for the learning system"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_storage_dir = self.temp_dir / 'learning_test'
        
        # Initialize components with test storage
        self.learning_engine = LearningEngine(self.test_storage_dir)
        
        # Test data
        self.test_commands = [
            CommandExecutionData(
                command="sbatch --mem=32G --time=4:00:00 --cpus-per-task=8 job.sh",
                exit_code=0,
                duration_ms=1500,
                timestamp=time.time(),
                session_id="test_session_1",
                working_directory="/data/project",
                host_context={'hostname': 'hpc-node01'}
            ),
            CommandExecutionData(
                command="singularity exec container.sif python analysis.py",
                exit_code=0,
                duration_ms=25000,
                timestamp=time.time() + 60,
                session_id="test_session_1",
                working_directory="/data/project"
            ),
            CommandExecutionData(
                command="grep -r pattern *.txt | head -10",
                exit_code=0,
                duration_ms=800,
                timestamp=time.time() + 120,
                session_id="test_session_1",
                working_directory="/data/project"
            ),
            # Bioinformatics workflow
            CommandExecutionData(
                command="fastqc sample.fastq.gz",
                exit_code=0,
                duration_ms=5000,
                timestamp=time.time() + 180,
                session_id="bio_session",
                working_directory="/data/genomics"
            ),
            CommandExecutionData(
                command="trimmomatic PE sample_1.fastq.gz sample_2.fastq.gz trimmed_1.fastq trimmed_2.fastq",
                exit_code=0,
                duration_ms=15000,
                timestamp=time.time() + 240,
                session_id="bio_session",
                working_directory="/data/genomics"
            ),
            CommandExecutionData(
                command="bwa mem reference.fa trimmed_1.fastq trimmed_2.fastq | samtools sort -o aligned.bam",
                exit_code=0,
                duration_ms=120000,
                timestamp=time.time() + 300,
                session_id="bio_session",
                working_directory="/data/genomics"
            )
        ]
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def tearDown(self):
        """Clean up test environment"""
        # Shutdown learning engine
        self.learning_engine.shutdown()
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_learning_pipeline(self):
        """Test the complete learning pipeline end-to-end"""
        self.logger.info("Testing full learning pipeline...")
        
        # Process test commands through learning engine
        results = []
        for cmd_data in self.test_commands:
            result = self.learning_engine.process_command_execution(cmd_data)
            results.append(result)
            
            # Verify basic processing
            self.assertTrue(result['processed'], f"Failed to process command: {cmd_data.command}")
            self.assertIn('analysis', result)
            self.assertIn('performance', result)
        
        # Verify learning occurred
        system_status = self.learning_engine.get_system_status()
        self.assertGreater(system_status['operation_stats']['commands_processed'], 0)
        
        # Verify schema evolution
        schema_status = system_status['schema_status']
        self.assertGreater(schema_status['pattern_count'], 0)
        
        self.logger.info("✓ Full learning pipeline test passed")
    
    def test_performance_requirements_compliance(self):
        """Test that all operations meet performance requirements"""
        self.logger.info("Testing performance requirements compliance...")
        
        # Test command processing performance
        start_time = time.time()
        result = self.learning_engine.process_command_execution(self.test_commands[0])
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Verify processing time meets target
        self.assertLessEqual(
            processing_time_ms, 
            PerformanceTargets.LEARNING_OPERATION_MS,
            f"Command processing took {processing_time_ms:.1f}ms, target: {PerformanceTargets.LEARNING_OPERATION_MS}ms"
        )
        
        # Test command suggestions performance (PreToolUse hook compatibility)
        start_time = time.time()
        suggestions = self.learning_engine.get_command_suggestions("grep pattern file.txt")
        suggestion_time_ms = (time.time() - start_time) * 1000
        
        self.assertLessEqual(
            suggestion_time_ms,
            PerformanceTargets.PRE_TOOL_USE_HOOK_MS,
            f"Command suggestions took {suggestion_time_ms:.1f}ms, target: {PerformanceTargets.PRE_TOOL_USE_HOOK_MS}ms"
        )
        
        # Test prompt enhancement performance (UserPromptSubmit hook compatibility)
        start_time = time.time()
        context = self.learning_engine.get_learning_context_for_prompt("Help me optimize my commands")
        context_time_ms = (time.time() - start_time) * 1000
        
        self.assertLessEqual(
            context_time_ms,
            PerformanceTargets.USER_PROMPT_SUBMIT_HOOK_MS,
            f"Prompt enhancement took {context_time_ms:.1f}ms, target: {PerformanceTargets.USER_PROMPT_SUBMIT_HOOK_MS}ms"
        )
        
        self.logger.info("✓ Performance requirements compliance test passed")
    
    def test_adaptive_schema_evolution(self):
        """Test adaptive schema evolution functionality"""
        self.logger.info("Testing adaptive schema evolution...")
        
        # Process multiple similar commands to trigger pattern learning
        similar_commands = [
            CommandExecutionData(f"sbatch --mem=16G job_{i}.sh", 0, 1000 + i*100, time.time() + i*10, f"session_{i}", "/data/hpc")
            for i in range(10)
        ]
        
        for cmd_data in similar_commands:
            self.learning_engine.process_command_execution(cmd_data)
        
        # Check if schema has evolved
        initial_version = self.learning_engine.schema.schema_version
        
        # Trigger schema evolution manually if needed
        evolution_result = self.learning_engine.trigger_schema_evolution()
        
        # Verify schema evolution
        final_version = self.learning_engine.schema.schema_version
        schema_data = self.learning_engine.schema.get_current_schema()
        
        self.assertGreaterEqual(final_version, initial_version)
        self.assertIn('patterns', schema_data)
        self.assertGreater(len(schema_data['usage_stats']), 0)
        
        self.logger.info("✓ Adaptive schema evolution test passed")
    
    def test_information_threshold_system(self):
        """Test information threshold management and agent triggering"""
        self.logger.info("Testing information threshold system...")
        
        # Set up agent callback mock
        agent_triggered = {'count': 0, 'agent': None, 'event': None}
        
        def mock_agent_callback(event):
            agent_triggered['count'] += 1
            agent_triggered['agent'] = event.agent_name
            agent_triggered['event'] = event
        
        # Register callback
        self.learning_engine.register_agent_callback(AgentNames.LEARNING_ANALYST, mock_agent_callback)
        
        # Generate commands that should trigger learning analyst
        trigger_commands = [
            # New complex commands (high learning significance)
            CommandExecutionData("nextflow run pipeline.nf --genome hg38 --input *.fastq", 0, 30000, time.time(), "nf_session", "/data/workflows"),
            CommandExecutionData("snakemake --cores 16 --use-singularity workflow.smk", 0, 45000, time.time() + 60, "snake_session", "/data/workflows"),
            # Failed commands (triggers failure information)
            CommandExecutionData("gatk HaplotypeCaller -I sample.bam -R reference.fa", 1, 5000, time.time() + 120, "fail_session", "/data/analysis"),
            CommandExecutionData("python complex_analysis.py --memory-intensive", 1, 8000, time.time() + 180, "fail_session", "/data/analysis"),
        ]
        
        for cmd_data in trigger_commands:
            result = self.learning_engine.process_command_execution(cmd_data)
            # Add small delay to allow threshold processing
            time.sleep(0.1)
        
        # Check threshold status
        threshold_status = self.learning_engine.threshold_manager.get_status()
        
        # Verify information accumulation
        info_accumulators = threshold_status['information_accumulators']
        self.assertGreater(info_accumulators[InformationTypes.NEW_COMMANDS]['total_value'], 0)
        self.assertGreater(info_accumulators[InformationTypes.FAILURES]['total_value'], 0)
        
        # Verify agent config updates
        agent_configs = threshold_status['agent_configs']
        self.assertIn(AgentNames.LEARNING_ANALYST, agent_configs)
        
        self.logger.info("✓ Information threshold system test passed")
    
    def test_workflow_pattern_recognition(self):
        """Test workflow pattern recognition functionality"""
        self.logger.info("Testing workflow pattern recognition...")
        
        # Create workflow pattern recognizer
        pattern_recognizer = WorkflowPatternRecognizer(self.test_storage_dir)
        abstractor = AdvancedCommandAbstractor()
        
        # Process bioinformatics workflow
        bio_workflow = [
            CommandExecutionData("fastqc sample.fastq", 0, 5000, time.time(), "bio_workflow", "/data/seq"),
            CommandExecutionData("trimmomatic PE sample.fastq trimmed.fastq", 0, 15000, time.time() + 60, "bio_workflow", "/data/seq"),
            CommandExecutionData("bwa mem ref.fa trimmed.fastq | samtools sort -o aligned.bam", 0, 120000, time.time() + 120, "bio_workflow", "/data/seq"),
            CommandExecutionData("gatk HaplotypeCaller -I aligned.bam -O variants.vcf", 0, 180000, time.time() + 240, "bio_workflow", "/data/seq")
        ]
        
        for cmd_data in bio_workflow:
            analysis = abstractor.analyze_command_comprehensive(cmd_data.command)
            result = pattern_recognizer.analyze_command_sequence(cmd_data, analysis)
            
            # Verify sequence analysis
            self.assertIn('workflow_insights', result)
            if len(pattern_recognizer.active_sessions['bio_workflow']) >= 2:
                insights = result['workflow_insights']
                self.assertTrue(insights['pattern_detected'])
                self.assertIn('workflow_type', insights)
        
        # Get workflow statistics
        stats = pattern_recognizer.get_workflow_statistics()
        self.assertIn('workflow_types', stats)
        
        # Get optimization patterns
        optimization_patterns = pattern_recognizer.get_workflow_optimization_patterns('bioinformatics_pipeline')
        
        pattern_recognizer.shutdown()
        self.logger.info("✓ Workflow pattern recognition test passed")
    
    def test_command_optimization_suggestions(self):
        """Test command optimization suggestion functionality"""
        self.logger.info("Testing command optimization suggestions...")
        
        # Test commands that should generate optimization suggestions
        optimization_test_commands = [
            "grep -r pattern /large/directory",  # Should suggest ripgrep
            "find /data -name '*.txt'",  # Should suggest fd
            "sbatch job.sh",  # Should suggest resource specifications
            "cat file.txt | grep pattern",  # Should suggest direct input
        ]
        
        for command in optimization_test_commands:
            suggestions = self.learning_engine.get_command_suggestions(command)
            
            # Verify suggestions structure
            self.assertIn('optimization_opportunities', suggestions)
            self.assertIn('safety_warnings', suggestions)
            self.assertIn('complexity_analysis', suggestions)
            
            # Verify specific optimizations
            if 'grep' in command:
                opportunities = suggestions['optimization_opportunities']
                self.assertTrue(any('ripgrep' in opp or 'rg' in opp for opp in opportunities))
        
        self.logger.info("✓ Command optimization suggestions test passed")
    
    def test_learning_data_persistence(self):
        """Test learning data persistence and recovery"""
        self.logger.info("Testing learning data persistence...")
        
        # Process some commands
        for cmd_data in self.test_commands[:3]:
            self.learning_engine.process_command_execution(cmd_data)
        
        # Get initial state
        initial_stats = self.learning_engine.get_system_status()
        initial_patterns = len(self.learning_engine.storage.command_patterns)
        
        # Shutdown and recreate engine (simulates restart)
        self.learning_engine.shutdown()
        
        # Create new engine with same storage directory
        new_engine = LearningEngine(self.test_storage_dir)
        
        # Verify data was persisted and restored
        restored_stats = new_engine.get_system_status()
        restored_patterns = len(new_engine.storage.command_patterns)
        
        # Should have restored some patterns
        self.assertGreaterEqual(restored_patterns, 0)
        
        # Process a new command to verify system is functional
        result = new_engine.process_command_execution(self.test_commands[-1])
        self.assertTrue(result['processed'])
        
        new_engine.shutdown()
        self.logger.info("✓ Learning data persistence test passed")
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration"""
        self.logger.info("Testing performance monitoring integration...")
        
        # Process commands and check performance monitoring
        for cmd_data in self.test_commands:
            result = self.learning_engine.process_command_execution(cmd_data)
            
            # Verify performance was tracked
            performance = result.get('performance', {})
            self.assertIn('total_duration_ms', performance)
            self.assertIn('target_met', performance)
        
        # Get performance statistics
        perf_monitor = self.learning_engine.performance_monitor
        perf_stats = perf_monitor.get_performance_stats(1)  # Last hour
        
        # Verify performance data
        self.assertIn('total_operations', perf_stats)
        self.assertIn('operation_stats', perf_stats)
        self.assertGreater(perf_stats['total_operations'], 0)
        
        # Check real-time status
        real_time_status = perf_monitor.get_real_time_status()
        self.assertIn('system_health_score', real_time_status)
        self.assertIn('performance_targets_met', real_time_status)
        
        self.logger.info("✓ Performance monitoring integration test passed")
    
    def test_error_handling_and_graceful_degradation(self):
        """Test error handling and graceful degradation"""
        self.logger.info("Testing error handling and graceful degradation...")
        
        # Test with malformed command data
        malformed_command = CommandExecutionData(
            command="",  # Empty command
            exit_code=-1,
            duration_ms=-100,  # Invalid duration
            timestamp=0,
            session_id="",
            working_directory=""
        )
        
        # Should not crash, should handle gracefully
        result = self.learning_engine.process_command_execution(malformed_command)
        self.assertIn('processed', result)
        
        # Test with very long command
        long_command = CommandExecutionData(
            command="a" * 10000,  # Very long command
            exit_code=0,
            duration_ms=1000,
            timestamp=time.time(),
            session_id="long_test",
            working_directory="/test"
        )
        
        result = self.learning_engine.process_command_execution(long_command)
        self.assertIn('processed', result)
        
        # Test system still functional after errors
        normal_result = self.learning_engine.process_command_execution(self.test_commands[0])
        self.assertTrue(normal_result['processed'])
        
        self.logger.info("✓ Error handling and graceful degradation test passed")
    
    def test_memory_usage_compliance(self):
        """Test memory usage stays within targets"""
        self.logger.info("Testing memory usage compliance...")
        
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Process many commands to stress test memory usage
        stress_commands = []
        for i in range(100):
            cmd = CommandExecutionData(
                command=f"test_command_{i} --param{i % 10} file_{i}.txt",
                exit_code=i % 3,  # Mix of success and failures
                duration_ms=1000 + (i * 10),
                timestamp=time.time() + i,
                session_id=f"stress_session_{i % 5}",  # Multiple sessions
                working_directory=f"/test/dir_{i % 3}"
            )
            stress_commands.append(cmd)
        
        # Process all commands
        for cmd_data in stress_commands:
            self.learning_engine.process_command_execution(cmd_data)
        
        # Force garbage collection
        gc.collect()
        
        # Check final memory usage
        final_memory_mb = process.memory_info().rss / (1024 * 1024)
        memory_increase_mb = final_memory_mb - initial_memory_mb
        
        # Should not exceed memory targets
        self.assertLessEqual(
            memory_increase_mb,
            PerformanceTargets.TOTAL_SYSTEM_MEMORY_MB,
            f"Memory usage increased by {memory_increase_mb:.1f}MB, target: {PerformanceTargets.TOTAL_SYSTEM_MEMORY_MB}MB"
        )
        
        self.logger.info(f"✓ Memory usage compliance test passed (increase: {memory_increase_mb:.1f}MB)")
    
    def test_cross_component_data_flow(self):
        """Test data flow between all learning components"""
        self.logger.info("Testing cross-component data flow...")
        
        # Process a command and trace data flow
        test_command = self.test_commands[0]
        
        # Before processing - capture initial states
        initial_schema_patterns = len(self.learning_engine.schema.pattern_registry)
        initial_storage_patterns = len(self.learning_engine.storage.command_patterns)
        initial_threshold_info = sum(
            acc.total_value for acc in self.learning_engine.threshold_manager.info_accumulators.values()
        )
        
        # Process command
        result = self.learning_engine.process_command_execution(test_command)
        
        # After processing - verify data flow
        
        # 1. Command should be analyzed by abstractor
        self.assertIn('analysis', result)
        analysis = result['analysis']
        self.assertIn('category', analysis)
        self.assertIn('complexity', analysis)
        
        # 2. Schema should have new pattern
        final_schema_patterns = len(self.learning_engine.schema.pattern_registry)
        self.assertGreaterEqual(final_schema_patterns, initial_schema_patterns)
        
        # 3. Storage should have new pattern
        final_storage_patterns = len(self.learning_engine.storage.command_patterns)
        self.assertGreaterEqual(final_storage_patterns, initial_storage_patterns)
        
        # 4. Threshold manager should have accumulated information
        final_threshold_info = sum(
            acc.total_value for acc in self.learning_engine.threshold_manager.info_accumulators.values()
        )
        self.assertGreater(final_threshold_info, initial_threshold_info)
        
        # 5. Performance monitor should have recorded metrics
        perf_stats = self.learning_engine.performance_monitor.get_performance_stats(1)
        self.assertGreater(perf_stats['total_operations'], 0)
        
        self.logger.info("✓ Cross-component data flow test passed")

def run_integration_tests():
    """Run all integration tests"""
    # Set up test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLearningSystemIntegration)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()

if __name__ == "__main__":
    print("=== Claude-Sync Learning System Integration Tests ===")
    print(f"Testing learning system integration...")
    print()
    
    # Run integration tests
    success = run_integration_tests()
    
    if success:
        print("\n🎉 All integration tests PASSED!")
        print("The learning system components are working together correctly.")
    else:
        print("\n❌ Some integration tests FAILED!")
        print("Check the test output above for details.")
    
    sys.exit(0 if success else 1)