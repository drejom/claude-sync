#!/usr/bin/env python3
"""
Documentation Intelligence Core
Central system for AI-powered documentation review and learning.
"""

import re
import ast
import json
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import hashlib

# Import learning infrastructure
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'learning'))
    from encryption import get_secure_storage
    from abstraction import get_abstractor
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    get_secure_storage = lambda: None
    get_abstractor = lambda: None

@dataclass
class DocumentContext:
    """Context information for documentation analysis"""
    doc_type: str  # 'function', 'vignette', 'readme', 'blogpost', 'internal', 'external'
    file_path: str
    language: str  # 'r', 'python', 'markdown', 'rst'
    audience: str  # 'developer', 'user', 'scientist', 'general'
    package_context: Optional[str] = None
    function_type: Optional[str] = None  # 'exported', 'internal', 'method'

@dataclass
class DocumentationIssue:
    """Represents a documentation issue or suggestion"""
    severity: str  # 'critical', 'major', 'minor', 'suggestion'
    category: str  # 'style', 'completeness', 'examples', 'clarity', 'accuracy'
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    confidence: float = 0.8

class DocumentationIntelligence:
    """Central documentation analysis and learning system"""
    
    def __init__(self):
        self.storage = get_secure_storage() if LEARNING_AVAILABLE else None
        self.abstractor = get_abstractor() if LEARNING_AVAILABLE else None
        
        # Document type patterns
        self.doc_patterns = {
            'function': {
                'required_sections': ['@param', '@return', '@export', '@examples'],
                'r_sections': ['@param', '@return', '@export', '@examples', '@description'],
                'python_sections': ['Args:', 'Returns:', 'Examples:', 'Note:']
            },
            'vignette': {
                'required_sections': ['Introduction', 'Installation', 'Usage', 'Examples'],
                'flow_requirements': ['setup', 'demonstration', 'conclusion']
            },
            'readme': {
                'required_sections': ['Installation', 'Usage', 'Examples'],
                'critical_first_impression': True
            }
        }
        
        # Style patterns learned from good documentation
        self.style_patterns = self._load_style_patterns()
    
    def analyze_document(self, content: str, context: DocumentContext) -> List[DocumentationIssue]:
        """Comprehensive document analysis"""
        issues = []
        
        # Run all analysis modules
        issues.extend(self._analyze_completeness(content, context))
        issues.extend(self._analyze_style(content, context))
        issues.extend(self._analyze_examples(content, context))
        issues.extend(self._analyze_audience_appropriateness(content, context))
        
        # Learn from this analysis
        if self.storage:
            self._learn_from_analysis(content, context, issues)
        
        return sorted(issues, key=lambda x: self._issue_priority(x), reverse=True)
    
    def _analyze_completeness(self, content: str, context: DocumentContext) -> List[DocumentationIssue]:
        """Check for missing required documentation elements"""
        issues = []
        
        if context.doc_type == 'function':
            issues.extend(self._check_function_completeness(content, context))
        elif context.doc_type == 'readme':
            issues.extend(self._check_readme_completeness(content, context))
        elif context.doc_type == 'vignette':
            issues.extend(self._check_vignette_completeness(content, context))
        
        return issues
    
    def _check_function_completeness(self, content: str, context: DocumentContext) -> List[DocumentationIssue]:
        """Check function documentation completeness"""
        issues = []
        
        # Check for R function documentation
        if context.language == 'r':
            required_tags = self.doc_patterns['function']['r_sections']
            
            # Check for missing @param tags
            params = self._extract_function_parameters(content, 'r')
            documented_params = re.findall(r'@param\s+(\w+)', content)
            
            missing_params = set(params) - set(documented_params)
            if missing_params:
                issues.append(DocumentationIssue(
                    severity='major',
                    category='completeness',
                    message=f"Missing @param documentation for: {', '.join(missing_params)}",
                    suggestion=f"Add: {chr(10).join([f'@param {p} Description of {p}' for p in missing_params])}"
                ))
            
            # Check for @return documentation
            if '@return' not in content and context.function_type == 'exported':
                issues.append(DocumentationIssue(
                    severity='major',
                    category='completeness',
                    message="Missing @return documentation for exported function",
                    suggestion="Add: @return Description of return value"
                ))
            
            # Check for @examples
            if '@examples' not in content and context.function_type == 'exported':
                issues.append(DocumentationIssue(
                    severity='minor',
                    category='completeness',
                    message="Missing @examples section for exported function",
                    suggestion="Add: @examples\\n# Example usage\\nfunction_name(param1, param2)"
                ))
        
        # Check for Python function documentation
        elif context.language == 'python':
            if '"""' not in content and "'''" not in content:
                issues.append(DocumentationIssue(
                    severity='critical',
                    category='completeness',
                    message="Missing docstring for Python function",
                    suggestion='Add docstring with """Description, Args:, Returns:, Examples:"""'
                ))
        
        return issues
    
    def _check_readme_completeness(self, content: str, context: DocumentContext) -> List[DocumentationIssue]:
        """Check README completeness"""
        issues = []
        
        required_sections = ['installation', 'usage', 'example']
        content_lower = content.lower()
        
        missing_sections = []
        for section in required_sections:
            if section not in content_lower:
                missing_sections.append(section)
        
        if missing_sections:
            issues.append(DocumentationIssue(
                severity='major',
                category='completeness',
                message=f"README missing sections: {', '.join(missing_sections)}",
                suggestion=f"Add sections for: {', '.join(missing_sections)}"
            ))
        
        # Check for one-liner installation
        if 'install' in content_lower and not any(cmd in content for cmd in ['install.packages', 'devtools::install', 'pip install', 'conda install']):
            issues.append(DocumentationIssue(
                severity='minor',
                category='completeness',
                message="Installation section lacks clear command",
                suggestion="Add specific installation command (e.g., install.packages('package'))"
            ))
        
        return issues
    
    def _analyze_style(self, content: str, context: DocumentContext) -> List[DocumentationIssue]:
        """Analyze documentation style and consistency"""
        issues = []
        
        # Check line length
        lines = content.split('\n')
        long_lines = [(i+1, line) for i, line in enumerate(lines) if len(line) > 80]
        
        if long_lines and len(long_lines) > len(lines) * 0.1:  # >10% of lines
            issues.append(DocumentationIssue(
                severity='minor',
                category='style',
                message=f"{len(long_lines)} lines exceed 80 characters",
                suggestion="Consider breaking long lines for readability"
            ))
        
        # Check for consistent parameter style
        if context.language == 'r':
            param_styles = re.findall(r'@param\s+(\w+)\s+(.+)', content)
            if param_styles:
                # Check if descriptions start with capital letters
                inconsistent_caps = [p for p, desc in param_styles if desc and not desc[0].isupper()]
                if inconsistent_caps and len(inconsistent_caps) > len(param_styles) * 0.3:
                    issues.append(DocumentationIssue(
                        severity='minor',
                        category='style',
                        message="Inconsistent capitalization in @param descriptions",
                        suggestion="Start all parameter descriptions with capital letters"
                    ))
        
        # Check for passive voice (scientific writing preference)
        passive_indicators = ['is done', 'are used', 'will be', 'can be']
        passive_count = sum(content.lower().count(indicator) for indicator in passive_indicators)
        
        if passive_count > 5 and context.audience == 'user':
            issues.append(DocumentationIssue(
                severity='suggestion',
                category='style',
                message="Consider reducing passive voice for user documentation",
                suggestion="Use active voice: 'Calculate X' instead of 'X is calculated'"
            ))
        
        return issues
    
    def _analyze_examples(self, content: str, context: DocumentContext) -> List[DocumentationIssue]:
        """Analyze code examples for quality and correctness"""
        issues = []
        
        # Extract code blocks
        code_blocks = []
        
        # R code blocks in roxygen comments
        if context.language == 'r':
            r_examples = re.findall(r'@examples\s*\n(.*?)(?=@\w+|$)', content, re.DOTALL)
            code_blocks.extend(r_examples)
        
        # Markdown code blocks
        md_blocks = re.findall(r'```(?:r|R|python)?\n(.*?)\n```', content, re.DOTALL)
        code_blocks.extend(md_blocks)
        
        for i, code_block in enumerate(code_blocks):
            if not code_block.strip():
                continue
            
            # Check for realistic examples
            if len(code_block.strip()) < 20:
                issues.append(DocumentationIssue(
                    severity='minor',
                    category='examples',
                    message=f"Example {i+1} is very brief",
                    suggestion="Consider adding more comprehensive example"
                ))
            
            # Check for hardcoded paths
            if '/Users/' in code_block or 'C:\\' in code_block:
                issues.append(DocumentationIssue(
                    severity='minor',
                    category='examples',
                    message=f"Example {i+1} contains hardcoded path",
                    suggestion="Use relative paths or system-independent examples"
                ))
            
            # Check for output demonstration
            if context.doc_type == 'vignette' and '#>' not in code_block and 'print(' not in code_block:
                issues.append(DocumentationIssue(
                    severity='suggestion',
                    category='examples',
                    message=f"Example {i+1} doesn't show output",
                    suggestion="Consider showing expected output with #> comments"
                ))
        
        return issues
    
    def _analyze_audience_appropriateness(self, content: str, context: DocumentContext) -> List[DocumentationIssue]:
        """Check if documentation matches intended audience"""
        issues = []
        
        # Technical complexity assessment
        technical_terms = ['algorithm', 'implementation', 'optimization', 'complexity', 'efficiency']
        jargon_count = sum(content.lower().count(term) for term in technical_terms)
        
        if context.audience == 'user' and jargon_count > 3:
            issues.append(DocumentationIssue(
                severity='minor',
                category='clarity',
                message="High technical complexity for user documentation",
                suggestion="Consider simplifying language or adding explanations for technical terms"
            ))
        
        # Check for assumption of prior knowledge
        if context.audience == 'general':
            assumed_knowledge = ['bioinformatics', 'genomics', 'RNA-seq', 'SLURM', 'HPC']
            assumptions = [term for term in assumed_knowledge if term in content and f'{term} is' not in content]
            
            if assumptions:
                issues.append(DocumentationIssue(
                    severity='suggestion',
                    category='clarity',
                    message=f"May assume knowledge of: {', '.join(assumptions)}",
                    suggestion="Consider brief explanations for domain-specific terms"
                ))
        
        return issues
    
    def _extract_function_parameters(self, content: str, language: str) -> List[str]:
        """Extract function parameter names"""
        if language == 'r':
            # Extract from function definition
            func_match = re.search(r'(\w+)\s*<-\s*function\s*\((.*?)\)', content, re.DOTALL)
            if func_match:
                params_str = func_match.group(2)
                # Simple parameter extraction (doesn't handle complex defaults)
                params = re.findall(r'(\w+)(?:\s*=.*?)?(?:,|$)', params_str)
                return [p.strip() for p in params if p.strip()]
        
        elif language == 'python':
            try:
                # Parse Python AST to extract parameters
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        return [arg.arg for arg in node.args.args]
            except:
                pass
        
        return []
    
    def _learn_from_analysis(self, content: str, context: DocumentContext, issues: List[DocumentationIssue]):
        """Learn patterns from documentation analysis"""
        if not self.storage:
            return
        
        # Store analysis patterns
        analysis_data = {
            'doc_type': context.doc_type,
            'language': context.language,
            'audience': context.audience,
            'issue_count': len(issues),
            'issue_categories': Counter([issue.category for issue in issues]),
            'severity_distribution': Counter([issue.severity for issue in issues]),
            'content_length': len(content),
            'timestamp': __import__('time').time()
        }
        
        # Load existing learning data
        learning_data = self.storage.load_learning_data('docs_intelligence', {
            'analysis_patterns': [],
            'style_preferences': defaultdict(Counter),
            'completeness_patterns': defaultdict(list),
            'audience_insights': defaultdict(dict)
        })
        
        # Add new analysis
        learning_data['analysis_patterns'].append(analysis_data)
        
        # Learn style preferences
        for issue in issues:
            if issue.category == 'style':
                key = f"{context.doc_type}_{context.language}"
                learning_data['style_preferences'][key][issue.message] += 1
        
        # Keep only recent analyses
        learning_data['analysis_patterns'] = learning_data['analysis_patterns'][-100:]
        
        # Store updated learning data
        self.storage.store_learning_data('docs_intelligence', dict(learning_data))
    
    def _load_style_patterns(self) -> Dict:
        """Load learned style patterns"""
        if not self.storage:
            return {}
        
        learning_data = self.storage.load_learning_data('docs_intelligence', {})
        return learning_data.get('style_preferences', {})
    
    def _issue_priority(self, issue: DocumentationIssue) -> int:
        """Calculate issue priority for sorting"""
        severity_weights = {'critical': 100, 'major': 75, 'minor': 50, 'suggestion': 25}
        category_weights = {'completeness': 10, 'examples': 8, 'style': 6, 'clarity': 7, 'accuracy': 9}
        
        return (severity_weights.get(issue.severity, 0) + 
                category_weights.get(issue.category, 0) +
                int(issue.confidence * 10))

def get_docs_intelligence():
    """Get configured documentation intelligence instance"""
    return DocumentationIntelligence()

if __name__ == '__main__':
    # Test documentation analysis
    intel = get_docs_intelligence()
    
    # Test R function documentation
    r_function = '''
    #' Calculate mean values
    #' @param x numeric vector
    #' @export
    calculate_mean <- function(x, na.rm = FALSE) {
        mean(x, na.rm = na.rm)
    }
    '''
    
    context = DocumentContext(
        doc_type='function',
        file_path='test.R',
        language='r',
        audience='user',
        function_type='exported'
    )
    
    issues = intel.analyze_document(r_function, context)
    
    print("Documentation Analysis Results:")
    for issue in issues:
        print(f"{issue.severity.upper()}: {issue.message}")
        if issue.suggestion:
            print(f"  Suggestion: {issue.suggestion}")
        print()