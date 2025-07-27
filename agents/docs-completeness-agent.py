#!/usr/bin/env python3
"""
Documentation Completeness Agent
Detects missing documentation elements and ensures comprehensive coverage.
"""

import json
import sys
import re
import ast
from pathlib import Path
from collections import defaultdict

# Import documentation intelligence
try:
    from docs_intelligence import get_docs_intelligence, DocumentContext
    INTEL_AVAILABLE = True
except ImportError:
    INTEL_AVAILABLE = False

def main():
    hook_input = json.loads(sys.stdin.read())
    
    if hook_input.get('tool_name') not in ['Edit', 'Write', 'MultiEdit']:
        sys.exit(0)
    
    file_path = hook_input.get('tool_input', {}).get('file_path', '')
    
    # Only analyze documentation files
    if not is_documentation_file(file_path):
        sys.exit(0)
    
    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        sys.exit(0)
    
    # Check documentation completeness
    if INTEL_AVAILABLE:
        completeness_result = check_documentation_completeness(content, file_path)
        if completeness_result:
            result = {
                'block': False,
                'message': completeness_result
            }
            print(json.dumps(result))
    
    sys.exit(0)

def is_documentation_file(file_path):
    """Check if file contains documentation"""
    if not file_path:
        return False
    
    path = Path(file_path)
    return (path.suffix in ['.md', '.rst', '.Rmd', '.R', '.py'] or 
            path.name.lower().startswith('readme') or
            'vignette' in path.name.lower())

def check_documentation_completeness(content, file_path):
    """Check documentation completeness and suggest missing elements"""
    intel = get_docs_intelligence()
    
    # Determine document context
    context = determine_document_context(content, file_path)
    
    # Run completeness analysis
    issues = intel.analyze_document(content, context)
    
    # Filter for completeness issues
    completeness_issues = [issue for issue in issues if issue.category == 'completeness']
    
    if not completeness_issues:
        # Check if documentation is actually complete
        completeness_score = calculate_completeness_score(content, context)
        if completeness_score >= 0.9:
            return format_completeness_success(completeness_score, context)
        return None
    
    return format_completeness_feedback(completeness_issues, content, context)

def determine_document_context(content, file_path):
    """Determine documentation context"""
    path = Path(file_path)
    
    # Document type detection
    if path.name.lower().startswith('readme'):
        doc_type = 'readme'
    elif path.suffix == '.Rmd' or 'vignette' in path.name.lower():
        doc_type = 'vignette'
    elif ('#\'' in content and 'function' in content) or ('def ' in content and '"""' in content):
        doc_type = 'function'
    else:
        doc_type = 'general'
    
    # Language detection
    if path.suffix == '.R' or path.suffix == '.Rmd':
        language = 'r'
    elif path.suffix == '.py':
        language = 'python'
    else:
        language = 'markdown'
    
    # Audience detection
    if any(term in content.lower() for term in ['internal', '@keywords internal', '@noRd']):
        audience = 'developer'
    elif '@export' in content or 'user' in content.lower():
        audience = 'user'
    else:
        audience = 'general'
    
    # Function type for R
    function_type = None
    if language == 'r' and doc_type == 'function':
        if '@export' in content:
            function_type = 'exported'
        elif '@noRd' in content or '@keywords internal' in content:
            function_type = 'internal'
    
    return DocumentContext(
        doc_type=doc_type,
        file_path=file_path,
        language=language,
        audience=audience,
        function_type=function_type
    )

def calculate_completeness_score(content, context):
    """Calculate documentation completeness score (0-1)"""
    score = 0.0
    max_score = 0.0
    
    if context.doc_type == 'function':
        if context.language == 'r':
            # R function completeness
            checks = {
                'has_description': bool(re.search(r'#\'\s*[A-Z]', content)),
                'has_params': bool(re.search(r'@param', content)),
                'has_return': bool(re.search(r'@return', content)),
                'has_examples': bool(re.search(r'@examples', content)),
                'has_export': bool('@export' in content)
            }
            
            # Adjust weights based on function type
            if context.function_type == 'exported':
                weights = {'has_description': 2, 'has_params': 2, 'has_return': 2, 'has_examples': 1, 'has_export': 1}
            else:
                weights = {'has_description': 2, 'has_params': 2, 'has_return': 1, 'has_examples': 0, 'has_export': 0}
            
            for check, passed in checks.items():
                weight = weights.get(check, 1)
                max_score += weight
                if passed:
                    score += weight
        
        elif context.language == 'python':
            # Python function completeness
            has_docstring = '"""' in content or "'''" in content
            has_args = 'Args:' in content or 'Parameters:' in content
            has_returns = 'Returns:' in content or 'Return:' in content
            
            max_score = 3
            if has_docstring: score += 1
            if has_args: score += 1
            if has_returns: score += 1
    
    elif context.doc_type == 'readme':
        # README completeness
        content_lower = content.lower()
        checks = {
            'has_title': bool(re.search(r'^#\s+', content, re.MULTILINE)),
            'has_installation': 'install' in content_lower,
            'has_usage': 'usage' in content_lower or 'use' in content_lower,
            'has_examples': 'example' in content_lower,
            'has_description': len(content) > 200
        }
        
        max_score = len(checks)
        score = sum(checks.values())
    
    elif context.doc_type == 'vignette':
        # Vignette completeness
        content_lower = content.lower()
        checks = {
            'has_intro': 'introduction' in content_lower or 'overview' in content_lower,
            'has_setup': 'library(' in content or 'install' in content_lower,
            'has_examples': content.count('```') >= 2,
            'has_conclusion': 'conclusion' in content_lower or 'summary' in content_lower,
            'reasonable_length': len(content) > 1000
        }
        
        max_score = len(checks)
        score = sum(checks.values())
    
    return score / max_score if max_score > 0 else 0.0

def format_completeness_feedback(issues, content, context):
    """Format completeness feedback"""
    parts = ["Documentation Completeness Review"]
    parts.append("")
    
    # Calculate current completeness score
    completeness_score = calculate_completeness_score(content, context)
    parts.append(f"Current completeness: {completeness_score:.0%}")
    parts.append("")
    
    # Group issues by severity
    critical_issues = [i for i in issues if i.severity == 'critical']
    major_issues = [i for i in issues if i.severity == 'major']
    minor_issues = [i for i in issues if i.severity in ['minor', 'suggestion']]
    
    if critical_issues:
        parts.append("Missing Critical Elements:")
        for issue in critical_issues:
            parts.append(f"  • {issue.message}")
            if issue.suggestion:
                parts.append(f"    Add: {issue.suggestion}")
        parts.append("")
    
    if major_issues:
        parts.append("Missing Important Elements:")
        for issue in major_issues:
            parts.append(f"  • {issue.message}")
            if issue.suggestion:
                parts.append(f"    Add: {issue.suggestion}")
        parts.append("")
    
    if minor_issues:
        parts.append("Additional Improvements:")
        for issue in minor_issues[:3]:  # Top 3
            parts.append(f"  • {issue.message}")
        parts.append("")
    
    # Add context-specific completeness guide
    parts.extend(get_completeness_guide(context))
    
    return "\n".join(parts)

def format_completeness_success(score, context):
    """Format message for complete documentation"""
    return f"Documentation completeness: {score:.0%} - Well documented {context.doc_type}!"

def get_completeness_guide(context):
    """Get context-specific completeness guide"""
    guide = []
    
    if context.doc_type == 'function':
        if context.language == 'r':
            guide.append("R Function Documentation Checklist:")
            guide.append("  • Clear description of what function does")
            guide.append("  • @param for each parameter with type and description")
            guide.append("  • @return describing return value and type")
            if context.function_type == 'exported':
                guide.append("  • @examples with working code")
                guide.append("  • @export tag for user-facing functions")
        
        elif context.language == 'python':
            guide.append("Python Function Documentation Checklist:")
            guide.append("  • Docstring with clear description")
            guide.append("  • Args: section with parameter descriptions")
            guide.append("  • Returns: section with return value description")
            guide.append("  • Examples: section with usage examples")
    
    elif context.doc_type == 'readme':
        guide.append("README Completeness Checklist:")
        guide.append("  • Clear project title and description")
        guide.append("  • Installation instructions")
        guide.append("  • Basic usage examples")
        guide.append("  • Link to detailed documentation")
    
    elif context.doc_type == 'vignette':
        guide.append("Vignette Completeness Checklist:")
        guide.append("  • Introduction explaining the problem")
        guide.append("  • Setup and installation steps")
        guide.append("  • Step-by-step examples with output")
        guide.append("  • Summary or conclusion")
    
    return guide

if __name__ == '__main__':
    main()