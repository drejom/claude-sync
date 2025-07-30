#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cryptography>=41.0.0"
# ]
# ///
"""
Command Abstractor - Advanced pattern recognition and command classification

This extends the basic SecureAbstractor with advanced command pattern recognition,
semantic classification, and learning-oriented command analysis. Key features:

- Semantic command classification beyond basic pattern matching
- Command complexity analysis with learning significance
- Context-aware pattern recognition for workflow detection
- Integration with adaptive schema for pattern evolution
- Performance optimized for real-time hook usage

Based on REFACTOR_PLAN.md sections 48-139 (Command Pattern Analysis)
"""

import re
import json
import time
import hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import logging

# Import base abstraction
from learning.abstraction import SecureAbstractor

@dataclass
class CommandAnalysis:
    """Comprehensive command analysis result"""
    command_category: str
    subcategory: str
    complexity_score: float
    semantic_intent: str
    tool_chain: List[str]
    resource_indicators: Dict[str, Any]
    workflow_context: Dict[str, Any]
    safety_concerns: List[str]
    optimization_opportunities: List[str]
    learning_significance: float

@dataclass
class WorkflowPattern:
    """Detected workflow pattern"""
    workflow_type: str  # 'bioinformatics', 'data_analysis', 'ml_training', etc.
    pipeline_stage: str  # 'preprocessing', 'analysis', 'postprocessing'
    tools_involved: List[str]
    data_flow_direction: str  # 'input', 'processing', 'output'
    resource_requirements: Dict[str, Any]

class AdvancedCommandAbstractor(SecureAbstractor):
    """
    Advanced command pattern recognition and abstraction system.
    
    This class extends the basic SecureAbstractor with sophisticated pattern
    recognition capabilities for learning system optimization.
    """
    
    def __init__(self):
        super().__init__()
        
        # Advanced pattern recognition databases
        self.tool_signatures = self._build_tool_signatures()
        self.workflow_patterns = self._build_workflow_patterns()
        self.resource_indicators = self._build_resource_indicators()
        self.safety_patterns = self._build_safety_patterns()
        
        # Learning significance factors
        self.complexity_weights = {
            'pipe_complexity': 0.3,
            'flag_complexity': 0.2,
            'tool_sophistication': 0.3,
            'data_size_indicators': 0.2
        }
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_command_comprehensive(self, command: str, context: Optional[Dict[str, Any]] = None) -> CommandAnalysis:
        """
        Perform comprehensive command analysis for learning system.
        
        This method provides detailed analysis beyond basic abstraction,
        focusing on learning significance and pattern recognition.
        """
        try:
            # Basic classification
            category, subcategory = self._classify_command_semantic(command)
            
            # Complexity analysis
            complexity_score = self._calculate_comprehensive_complexity(command)
            
            # Semantic intent detection
            semantic_intent = self._detect_semantic_intent(command, context)
            
            # Tool chain analysis
            tool_chain = self._extract_tool_chain(command)
            
            # Resource requirement analysis
            resource_indicators = self._analyze_resource_requirements(command)
            
            # Workflow context detection
            workflow_context = self._detect_workflow_context(command, context)
            
            # Safety analysis
            safety_concerns = self._analyze_safety_concerns(command)
            
            # Optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(command)
            
            # Learning significance calculation
            learning_significance = self._calculate_learning_significance(
                command, complexity_score, category, tool_chain
            )
            
            return CommandAnalysis(
                command_category=category,
                subcategory=subcategory,
                complexity_score=complexity_score,
                semantic_intent=semantic_intent,
                tool_chain=tool_chain,
                resource_indicators=resource_indicators,
                workflow_context=workflow_context,
                safety_concerns=safety_concerns,
                optimization_opportunities=optimization_opportunities,
                learning_significance=learning_significance
            )
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive command analysis: {e}")
            return self._create_fallback_analysis(command)
    
    def abstract_command_enhanced(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced command abstraction with learning-oriented features.
        
        This extends the base abstract_command method with additional
        analysis for the learning system.
        """
        # Get basic abstraction
        basic_abstraction = self.abstract_command(command)
        
        # Add comprehensive analysis
        analysis = self.analyze_command_comprehensive(command, context)
        
        # Combine into enhanced abstraction
        enhanced = {
            **basic_abstraction,
            'semantic_analysis': {
                'category': analysis.command_category,
                'subcategory': analysis.subcategory,
                'semantic_intent': analysis.semantic_intent,
                'complexity': analysis.complexity_score,
                'learning_significance': analysis.learning_significance
            },
            'tool_analysis': {
                'tool_chain': analysis.tool_chain,
                'tool_sophistication': self._calculate_tool_sophistication(analysis.tool_chain)
            },
            'resource_analysis': analysis.resource_indicators,
            'workflow_analysis': analysis.workflow_context,
            'optimization_analysis': {
                'opportunities': analysis.optimization_opportunities,
                'safety_concerns': analysis.safety_concerns
            }
        }
        
        return enhanced
    
    def _classify_command_semantic(self, command: str) -> Tuple[str, str]:
        """Semantic command classification with detailed subcategories"""
        command_lower = command.lower().strip()
        
        # HPC and Job Scheduling
        if any(pattern in command_lower for pattern in ['sbatch', 'squeue', 'scancel', 'sinfo']):
            if 'sbatch' in command_lower:
                return 'hpc_job_submission', 'slurm_batch'
            elif 'squeue' in command_lower:
                return 'hpc_monitoring', 'slurm_queue'
            else:
                return 'hpc_management', 'slurm_control'
        
        elif any(pattern in command_lower for pattern in ['qsub', 'qstat', 'qdel']):
            return 'hpc_job_submission', 'pbs_torque'
        
        elif any(pattern in command_lower for pattern in ['bsub', 'bjobs', 'bkill']):
            return 'hpc_job_submission', 'lsf'
        
        # Container and Virtualization
        elif any(pattern in command_lower for pattern in ['singularity', 'apptainer']):
            if 'exec' in command_lower:
                return 'container_execution', 'singularity_exec'
            elif 'build' in command_lower:
                return 'container_build', 'singularity_build'
            else:
                return 'container_management', 'singularity'
        
        elif command_lower.startswith('docker'):
            if 'run' in command_lower:
                return 'container_execution', 'docker_run'
            elif 'build' in command_lower:
                return 'container_build', 'docker_build'
            else:
                return 'container_management', 'docker'
        
        # Data Analysis and Scientific Computing
        elif command_lower.startswith('r ') or 'rscript' in command_lower or any(r_pattern in command_lower for r_pattern in ['r cmd', 'r --vanilla', '--args', 'library(', 'install.packages']):
            return 'data_analysis', self._identify_r_subcategory(command_lower)
        
        elif command_lower.startswith('python'):
            if any(lib in command_lower for lib in ['pandas', 'numpy', 'scipy']):
                return 'data_analysis', 'python_data_science' 
            elif any(lib in command_lower for lib in ['torch', 'tensorflow', 'keras']):
                return 'machine_learning', 'python_ml'
            else:
                return 'programming', 'python_general'
        
        elif command_lower.startswith('jupyter'):
            return 'interactive_analysis', 'jupyter'
        
        # Workflow Engines
        elif any(pattern in command_lower for pattern in ['nextflow', 'nf']):
            return 'workflow_execution', 'nextflow'
        
        elif 'snakemake' in command_lower:
            return 'workflow_execution', 'snakemake'
        
        elif any(pattern in command_lower for pattern in ['cwl-runner', 'cwltool']):
            return 'workflow_execution', 'cwl'
        
        # Bioinformatics Tools
        elif any(tool in command_lower for tool in ['blast', 'bwa', 'samtools', 'gatk', 'bcftools', 'trimmomatic', 'cutadapt', 'fastqc', 'multiqc', 'bowtie2', 'hisat2', 'star', 'salmon', 'kallisto', 'featurecounts', 'deseq2', 'edger', 'bedtools', 'vcftools', 'picard', 'freebayes', 'varscan', 'annovar', 'vep', 'snpeff']):
            return 'bioinformatics', self._identify_bioinformatics_subcategory(command_lower)
        
        # Data Transfer and Management
        elif any(pattern in command_lower for pattern in ['rsync', 'scp', 'wget', 'curl']):
            return 'data_transfer', 'network_transfer'
        
        elif any(pattern in command_lower for pattern in ['tar', 'gzip', 'zip', 'unzip']):
            return 'data_management', 'compression'
        
        # System Administration
        elif any(pattern in command_lower for pattern in ['ssh', 'sudo', 'systemctl']):
            return 'system_admin', 'remote_management'
        
        # File Operations
        elif any(pattern in command_lower for pattern in ['find', 'grep', 'awk', 'sed']):
            return 'file_processing', 'text_processing'
        
        elif any(pattern in command_lower for pattern in ['ls', 'cp', 'mv', 'rm']):
            return 'file_operations', 'basic_file_ops'
        
        # Default classification
        else:
            return 'general_computing', 'unknown'
    
    def _calculate_comprehensive_complexity(self, command: str) -> float:
        """Calculate comprehensive complexity score for learning significance"""
        complexity_factors = {}
        
        # Pipe complexity
        pipe_count = command.count('|')
        complexity_factors['pipe_complexity'] = min(pipe_count / 3.0, 2.0)  # Normalize to max 2.0
        
        # Flag complexity
        flags = [part for part in command.split() if part.startswith('-')]
        flag_complexity = len(flags) / 10.0  # Normalize
        complexity_factors['flag_complexity'] = min(flag_complexity, 2.0)
        
        # Tool sophistication
        tool_sophistication = self._calculate_tool_sophistication(self._extract_tool_chain(command))
        complexity_factors['tool_sophistication'] = tool_sophistication
        
        # Data size indicators
        data_size_complexity = self._analyze_data_size_complexity(command)
        complexity_factors['data_size_indicators'] = data_size_complexity
        
        # Weighted complexity score
        total_complexity = sum(
            complexity_factors[factor] * self.complexity_weights[factor]
            for factor in complexity_factors
        )
        
        return min(total_complexity, 5.0)  # Cap at 5.0
    
    def _detect_semantic_intent(self, command: str, context: Optional[Dict[str, Any]]) -> str:
        """Detect the semantic intent of the command"""
        command_lower = command.lower()
        
        # Data processing intents
        if any(pattern in command_lower for pattern in ['sort', 'uniq', 'cut', 'awk']):
            return 'data_transformation'
        
        # Analysis intents
        elif any(pattern in command_lower for pattern in ['analyze', 'process', 'calculate']):
            return 'data_analysis'
        
        # Workflow intents
        elif any(pattern in command_lower for pattern in ['run', 'execute', 'submit']):
            return 'workflow_execution'
        
        # Resource management intents
        elif any(pattern in command_lower for pattern in ['monitor', 'status', 'info']):
            return 'resource_monitoring'
        
        # Data movement intents
        elif any(pattern in command_lower for pattern in ['copy', 'move', 'sync', 'transfer']):
            return 'data_movement'
        
        # Environment setup intents
        elif any(pattern in command_lower for pattern in ['install', 'setup', 'configure']):
            return 'environment_setup'
        
        else:
            return 'general_operation'
    
    def _extract_tool_chain(self, command: str) -> List[str]:
        """Extract the chain of tools used in the command"""
        tools = []
        
        # Split by pipes to find tool chain
        pipe_segments = command.split('|')
        
        for segment in pipe_segments:
            # Extract the first word as the tool
            segment_words = segment.strip().split()
            if segment_words:
                tool = segment_words[0]
                # Clean up common prefixes
                if tool.startswith('./'):
                    tool = tool[2:]
                tools.append(tool)
        
        return tools
    
    def _analyze_resource_requirements(self, command: str) -> Dict[str, Any]:
        """Analyze resource requirements indicated by the command"""
        requirements = {
            'compute_intensive': False,
            'memory_intensive': False,
            'io_intensive': False,
            'gpu_required': False,
            'parallel_processing': False,
            'estimated_duration': 'unknown'
        }
        
        command_lower = command.lower()
        
        # Compute intensive indicators
        if any(pattern in command_lower for pattern in ['blast', 'bwa', 'gatk', 'training', 'compile']):
            requirements['compute_intensive'] = True
        
        # Memory intensive indicators
        if any(pattern in command_lower for pattern in ['--mem', '--memory', 'java -xmx', 'sort -S']):
            requirements['memory_intensive'] = True
        
        # I/O intensive indicators
        if any(pattern in command_lower for pattern in ['find', 'rsync', 'tar', 'gzip']):
            requirements['io_intensive'] = True
        
        # GPU requirements
        if any(pattern in command_lower for pattern in ['cuda', 'gpu', '--gres=gpu', 'nvidia']):
            requirements['gpu_required'] = True
        
        # Parallel processing indicators
        if any(pattern in command_lower for pattern in ['-j', '--parallel', '--threads', '--cores']):
            requirements['parallel_processing'] = True
        
        # Duration estimation based on tool types
        if any(pattern in command_lower for pattern in ['blast', 'bwa mem', 'gatk']):
            requirements['estimated_duration'] = 'hours'
        elif any(pattern in command_lower for pattern in ['python', 'r ', 'analysis']):
            requirements['estimated_duration'] = 'minutes'
        elif any(pattern in command_lower for pattern in ['ls', 'cp', 'mv']):
            requirements['estimated_duration'] = 'seconds'
        
        return requirements
    
    def _detect_workflow_context(self, command: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect workflow context and pipeline stage"""
        workflow_info = {
            'workflow_type': 'unknown',
            'pipeline_stage': 'unknown',
            'data_type': 'unknown',
            'workflow_engine': None
        }
        
        command_lower = command.lower()
        
        # Workflow engine detection
        if any(engine in command_lower for engine in ['nextflow', 'snakemake', 'cwl']):
            if 'nextflow' in command_lower:
                workflow_info['workflow_engine'] = 'nextflow'
            elif 'snakemake' in command_lower:
                workflow_info['workflow_engine'] = 'snakemake'
            elif 'cwl' in command_lower:
                workflow_info['workflow_engine'] = 'cwl'
        
        # Bioinformatics workflow detection
        if any(tool in command_lower for tool in ['fastq', 'bam', 'vcf', 'fasta']):
            workflow_info['workflow_type'] = 'bioinformatics'
            workflow_info['data_type'] = 'genomics'
            
            # Pipeline stage detection
            if any(stage in command_lower for stage in ['trim', 'clean', 'preprocess']):
                workflow_info['pipeline_stage'] = 'preprocessing'
            elif any(stage in command_lower for stage in ['align', 'map', 'blast']):
                workflow_info['pipeline_stage'] = 'alignment'
            elif any(stage in command_lower for stage in ['variant', 'call', 'gatk']):
                workflow_info['pipeline_stage'] = 'variant_calling'
            elif any(stage in command_lower for stage in ['annotate', 'predict']):
                workflow_info['pipeline_stage'] = 'annotation'
        
        # Data analysis workflow detection
        elif any(tool in command_lower for tool in ['python', 'r ', 'jupyter', 'pandas']):
            workflow_info['workflow_type'] = 'data_analysis'
            
            if any(stage in command_lower for stage in ['clean', 'preprocess', 'munge']):
                workflow_info['pipeline_stage'] = 'data_preparation'
            elif any(stage in command_lower for stage in ['analyze', 'model', 'fit']):
                workflow_info['pipeline_stage'] = 'analysis'
            elif any(stage in command_lower for stage in ['plot', 'visualize', 'report']):
                workflow_info['pipeline_stage'] = 'visualization'
        
        # Machine learning workflow detection
        elif any(tool in command_lower for tool in ['torch', 'tensorflow', 'keras', 'sklearn']):
            workflow_info['workflow_type'] = 'machine_learning'
            workflow_info['data_type'] = 'ml_data'
            
            if any(stage in command_lower for stage in ['train', 'fit']):
                workflow_info['pipeline_stage'] = 'training'
            elif any(stage in command_lower for stage in ['predict', 'inference']):
                workflow_info['pipeline_stage'] = 'prediction'
            elif any(stage in command_lower for stage in ['evaluate', 'test']):
                workflow_info['pipeline_stage'] = 'evaluation'
        
        return workflow_info
    
    def _analyze_safety_concerns(self, command: str) -> List[str]:
        """Analyze potential safety concerns with the command"""
        concerns = []
        command_lower = command.lower()
        
        # Destructive operations
        if any(pattern in command_lower for pattern in ['rm -rf', 'rm -r', 'rmdir']):
            concerns.append('destructive_file_operation')
        
        if any(pattern in command_lower for pattern in ['dd if=', 'mkfs', 'fdisk']):
            concerns.append('disk_operation_risk')
        
        # Network security
        if any(pattern in command_lower for pattern in ['wget http://', 'curl http://']):
            concerns.append('unencrypted_download')
        
        # Privilege escalation
        if any(pattern in command_lower for pattern in ['sudo', 'su -', 'chmod 777']):
            concerns.append('privilege_escalation')
        
        # Resource exhaustion risks
        if any(pattern in command_lower for pattern in ['fork bomb', ':(){ :|:& };:']):
            concerns.append('resource_exhaustion_risk')
        
        # Large data operations without limits
        if any(pattern in command_lower for pattern in ['find /', 'grep -r /']):
            concerns.append('potentially_expensive_operation')
        
        return concerns
    
    def _identify_optimization_opportunities(self, command: str) -> List[str]:
        """Identify optimization opportunities for the command"""
        opportunities = []
        command_lower = command.lower()
        
        # Tool replacements
        if 'grep' in command_lower and 'rg' not in command_lower:
            opportunities.append('replace_grep_with_ripgrep')
        
        if 'find' in command_lower and 'fd' not in command_lower:
            opportunities.append('replace_find_with_fd')
        
        if 'cat' in command_lower and '|' in command_lower:
            opportunities.append('consider_direct_input_redirection')
        
        # Parallel processing opportunities
        if any(tool in command_lower for tool in ['gzip', 'bzip2', 'xz']) and 'pigz' not in command_lower:
            opportunities.append('use_parallel_compression')
        
        # SLURM optimization opportunities
        if 'sbatch' in command_lower:
            if '--mem=' not in command_lower:
                opportunities.append('specify_memory_requirement')
            if '--time=' not in command_lower:
                opportunities.append('specify_time_limit')
            if '--cpus-per-task=' not in command_lower and any(parallel in command_lower for parallel in ['-j', '--threads']):
                opportunities.append('specify_cpu_requirement')
        
        # Container optimization
        if any(container in command_lower for container in ['docker', 'singularity']):
            if '--bind' not in command_lower and '--mount' not in command_lower:
                opportunities.append('consider_explicit_bind_mounts')
        
        return opportunities
    
    def _calculate_learning_significance(self, command: str, complexity: float, category: str, tools: List[str]) -> float:
        """Calculate how significant this command is for learning purposes"""
        significance = 1.0
        
        # Complexity contributes to significance
        significance += complexity * 0.3
        
        # Rare or sophisticated tools are more significant
        tool_sophistication = self._calculate_tool_sophistication(tools)
        significance += tool_sophistication * 0.4
        
        # Certain categories are inherently more significant for learning
        category_weights = {
            'hpc_job_submission': 2.0,
            'container_execution': 1.8,
            'workflow_execution': 2.2,
            'bioinformatics': 1.9,
            'machine_learning': 2.1,
            'data_analysis': 1.5,
            'system_admin': 1.3,
            'general_computing': 0.8
        }
        
        category_weight = category_weights.get(category, 1.0)
        significance *= category_weight
        
        # Cap significance at reasonable level
        return min(significance, 10.0)
    
    def _calculate_tool_sophistication(self, tools: List[str]) -> float:
        """Calculate sophistication score for tools"""
        if not tools:
            return 0.0
        
        sophistication_scores = {
            # High sophistication tools
            'nextflow': 3.0, 'snakemake': 3.0, 'gatk': 2.8, 'blast': 2.5,
            'singularity': 2.3, 'docker': 2.0, 'bwa': 2.2, 'samtools': 2.1,
            
            # Medium sophistication tools  
            'python': 1.8, 'r': 1.8, 'jupyter': 1.7, 'awk': 1.5,
            'rsync': 1.4, 'tar': 1.2, 'gzip': 1.1,
            
            # Basic tools
            'grep': 1.0, 'find': 1.0, 'ls': 0.8, 'cp': 0.8, 'mv': 0.8,
            'cat': 0.7, 'echo': 0.5
        }
        
        tool_scores = [sophistication_scores.get(tool, 1.0) for tool in tools]
        return sum(tool_scores) / len(tool_scores)
    
    def _analyze_data_size_complexity(self, command: str) -> float:
        """Analyze complexity based on data size indicators"""
        complexity = 0.0
        command_lower = command.lower()
        
        # File extension indicators of large data
        large_data_patterns = ['*.fastq', '*.bam', '*.vcf', '*.h5', '*.hdf5', '*.parquet']
        if any(pattern in command_lower for pattern in large_data_patterns):
            complexity += 1.0
        
        # Size specifications
        if any(size in command_lower for size in ['gb', 'tb', 'gigabyte', 'terabyte']):
            complexity += 1.5
        
        # Batch processing indicators
        if any(batch in command_lower for batch in ['*.', 'find', 'xargs', 'parallel']):
            complexity += 0.8
        
        return min(complexity, 2.0)
    
    def _identify_bioinformatics_subcategory(self, command_lower: str) -> str:
        """Identify specific bioinformatics tool subcategory"""
        # Sequence alignment tools
        if any(tool in command_lower for tool in ['blast', 'blastn', 'blastp', 'blastx', 'tblastn']):
            return 'sequence_alignment'
        
        # Read mapping and alignment
        elif any(tool in command_lower for tool in ['bwa', 'bowtie', 'bowtie2', 'hisat', 'hisat2', 'star', 'minimap2']):
            return 'read_mapping'
        
        # Quality control and preprocessing
        elif any(tool in command_lower for tool in ['fastqc', 'multiqc', 'trimmomatic', 'cutadapt', 'fastp', 'trim_galore']):
            return 'quality_control'
        
        # Quantification and expression analysis
        elif any(tool in command_lower for tool in ['salmon', 'kallisto', 'rsem', 'featurecounts', 'htseq', 'stringtie']):
            return 'expression_quantification'
        
        # Differential expression analysis
        elif any(tool in command_lower for tool in ['deseq2', 'edger', 'limma', 'ballgown']):
            return 'differential_expression'
        
        # Sequence processing and manipulation
        elif any(tool in command_lower for tool in ['samtools', 'bcftools', 'vcftools', 'bedtools', 'seqtk', 'picard']):
            return 'sequence_processing'
        
        # Variant calling and analysis
        elif any(tool in command_lower for tool in ['gatk', 'freebayes', 'varscan', 'mutect2', 'haplotypecaller']):
            return 'variant_calling'
        
        # Annotation tools
        elif any(tool in command_lower for tool in ['annovar', 'vep', 'snpeff', 'funcotator']):
            return 'annotation'
        
        # Assembly tools
        elif any(tool in command_lower for tool in ['spades', 'velvet', 'canu', 'flye', 'megahit']):
            return 'genome_assembly'
        
        # Phylogenetic analysis
        elif any(tool in command_lower for tool in ['muscle', 'mafft', 'raxml', 'iqtree', 'beast']):
            return 'phylogenetic_analysis'
        
        else:
            return 'general_bioinformatics'
    
    def _identify_r_subcategory(self, command_lower: str) -> str:
        """Identify specific R computing subcategory"""
        # Bioinformatics R packages
        if any(pkg in command_lower for pkg in ['bioconductor', 'deseq2', 'edger', 'limma', 'genomicranges', 'biomart', 'tcga', 'seurats']):
            return 'r_bioinformatics'
        
        # Statistical modeling packages
        elif any(pkg in command_lower for pkg in ['lme4', 'nlme', 'survival', 'coxph', 'glm', 'lm(', 'aov(', 'anova']):
            return 'r_statistical_modeling'
        
        # Machine learning packages
        elif any(pkg in command_lower for pkg in ['randomforest', 'caret', 'e1071', 'nnet', 'kernlab', 'gbm', 'xgboost']):
            return 'r_machine_learning'
        
        # Data manipulation and visualization
        elif any(pkg in command_lower for pkg in ['dplyr', 'ggplot2', 'tidyr', 'readr', 'stringr', 'lubridate', 'tidyverse', 'shiny']):
            return 'r_data_science'
        
        # Genomics and sequencing analysis
        elif any(pkg in command_lower for pkg in ['seurat', 'monocle', 'scanpy', 'cellranger', 'scran', 'scater']):
            return 'r_single_cell_analysis'
        
        # Time series analysis
        elif any(pkg in command_lower for pkg in ['forecast', 'ts', 'zoo', 'xts', 'quantmod']):
            return 'r_time_series'
        
        # Spatial analysis
        elif any(pkg in command_lower for pkg in ['sp', 'sf', 'raster', 'maptools', 'rgdal', 'leaflet']):
            return 'r_spatial_analysis'
        
        # Package management and installation
        elif any(pattern in command_lower for pattern in ['install.packages', 'devtools', 'remotes', 'pak', 'renv']):
            return 'r_package_management'
        
        # Report generation
        elif any(pattern in command_lower for pattern in ['rmarkdown', 'knitr', 'bookdown', 'blogdown', 'xaringan']):
            return 'r_reporting'
        
        else:
            return 'r_general_computing'
    
    def _create_fallback_analysis(self, command: str) -> CommandAnalysis:
        """Create fallback analysis when comprehensive analysis fails"""
        return CommandAnalysis(
            command_category='general_computing',
            subcategory='unknown',
            complexity_score=1.0,
            semantic_intent='general_operation',
            tool_chain=[command.split()[0] if command.split() else 'unknown'],
            resource_indicators={},
            workflow_context={},
            safety_concerns=[],
            optimization_opportunities=[],
            learning_significance=1.0
        )
    
    def _build_tool_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Build database of tool signatures for pattern recognition"""
        return {
            'bioinformatics': {
                'blast': {'complexity': 2.5, 'resource_intensive': True, 'typical_duration': 'hours'},
                'bwa': {'complexity': 2.2, 'resource_intensive': True, 'typical_duration': 'hours'}, 
                'gatk': {'complexity': 2.8, 'resource_intensive': True, 'typical_duration': 'hours'},
                'samtools': {'complexity': 2.1, 'resource_intensive': False, 'typical_duration': 'minutes'}
            },
            'workflow_engines': {
                'nextflow': {'complexity': 3.0, 'resource_intensive': True, 'typical_duration': 'hours'},
                'snakemake': {'complexity': 3.0, 'resource_intensive': True, 'typical_duration': 'hours'}
            },
            'containers': {
                'singularity': {'complexity': 2.3, 'resource_intensive': False, 'typical_duration': 'variable'},
                'docker': {'complexity': 2.0, 'resource_intensive': False, 'typical_duration': 'variable'}
            }
        }
    
    def _build_workflow_patterns(self) -> Dict[str, List[str]]:
        """Build database of common workflow patterns"""
        return {
            'genomics_pipeline': ['fastqc', 'trim', 'bwa', 'samtools', 'gatk', 'vcftools'],
            'rnaseq_pipeline': ['fastqc', 'trim', 'hisat2', 'samtools', 'stringtie', 'deseq2'],
            'ml_pipeline': ['preprocess', 'train', 'validate', 'predict', 'evaluate']
        }
    
    def _build_resource_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Build database of resource requirement indicators"""
        return {
            'memory_intensive': {
                'patterns': ['sort -S', 'java -Xmx', '--mem=', '--memory='],
                'tools': ['gatk', 'picard', 'blast', 'bwa mem']
            },
            'cpu_intensive': {
                'patterns': ['-j', '--threads', '--cores', '--parallel'],
                'tools': ['blast', 'bwa', 'bowtie2', 'gatk']
            },
            'gpu_required': {
                'patterns': ['--gres=gpu', 'cuda', 'nvidia-smi'],
                'tools': ['tensorflow', 'pytorch', 'blast+']
            }
        }
    
    def _build_safety_patterns(self) -> Dict[str, List[str]]:
        """Build database of potentially dangerous command patterns"""
        return {
            'destructive': ['rm -rf', 'rm -r', 'rmdir', 'dd if=', 'mkfs'],
            'privilege_escalation': ['sudo', 'su -', 'chmod 777', 'chown'],
            'network_risks': ['wget http://', 'curl http://', 'nc -l'],
            'resource_exhaustion': ['fork bomb', ':(){ :|:& };:', 'find /']
        }

if __name__ == "__main__":
    # Example usage and testing
    abstractor = AdvancedCommandAbstractor()
    
    # Test commands
    test_commands = [
        "sbatch --mem=32G --time=4:00:00 --cpus-per-task=8 run_blast.sh",
        "singularity exec container.sif python analysis.py | grep significant",
        "nextflow run pipeline.nf --input data/*.fastq --genome hg38",
        "bwa mem reference.fa reads1.fastq reads2.fastq | samtools sort -o output.bam",
        "grep -r pattern /large/directory | awk '{print $1}' | sort | uniq"
    ]
    
    for command in test_commands:
        print(f"\nAnalyzing: {command}")
        analysis = abstractor.analyze_command_comprehensive(command)
        print(f"Category: {analysis.command_category}/{analysis.subcategory}")
        print(f"Complexity: {analysis.complexity_score:.2f}")
        print(f"Learning Significance: {analysis.learning_significance:.2f}")
        print(f"Tools: {analysis.tool_chain}")
        print(f"Optimizations: {analysis.optimization_opportunities}")
        if analysis.safety_concerns:
            print(f"Safety Concerns: {analysis.safety_concerns}")
        
        # Test enhanced abstraction
        enhanced = abstractor.abstract_command_enhanced(command)
        print(f"Enhanced abstraction keys: {list(enhanced.keys())}")