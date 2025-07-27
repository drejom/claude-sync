#!/usr/bin/env python3
"""
Documentation Style Agent
Analyzes and learns documentation style patterns for consistency and readability.
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter

# Import documentation intelligence
try:
    from docs_intelligence import get_docs_intelligence, DocumentContext, DocumentationIssue
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
    
    # Analyze documentation style
    if INTEL_AVAILABLE:
        review_result = analyze_documentation_style(content, file_path)
        if review_result:
            result = {
                'block': False,
                'message': review_result
            }
            print(json.dumps(result))
    
    sys.exit(0)

def is_documentation_file(file_path):
    """Check if file contains documentation"""
    if not file_path:
        return False
    
    path = Path(file_path)
    
    # Direct documentation files
    if path.suffix in ['.md', '.rst', '.Rmd']:
        return True
    
    # R files with roxygen documentation
    if path.suffix == '.R':
        return True
    
    # Python files with docstrings
    if path.suffix == '.py':
        return True
    
    # README files
    if path.name.lower().startswith('readme'):
        return True
    
    return False

def analyze_documentation_style(content, file_path):
    """Analyze documentation style and provide feedback"""
    intel = get_docs_intelligence()
    
    # Determine document context
    context = determine_document_context(content, file_path)
    
    # Run style analysis
    issues = intel.analyze_document(content, context)
    
    # Filter for style-related issues
    style_issues = [issue for issue in issues if issue.category in ['style', 'clarity']]
    
    if not style_issues:
        return None
    
    return format_style_feedback(style_issues, context)

def determine_document_context(content, file_path):
    """Determine documentation context from file and content"""
    path = Path(file_path)
    
    # Determine document type
    if path.name.lower().startswith('readme'):
        doc_type = 'readme'
    elif path.suffix == '.Rmd':
        doc_type = 'vignette'
    elif 'function' in content or 'def ' in content:
        doc_type = 'function'
    elif path.suffix == '.md':
        doc_type = 'guide'
    else:
        doc_type = 'general'
    
    # Determine language
    if path.suffix == '.R' or path.suffix == '.Rmd':
        language = 'r'
    elif path.suffix == '.py':
        language = 'python'
    else:
        language = 'markdown'
    
    # Determine audience (heuristic-based)
    content_lower = content.lower()
    if any(term in content_lower for term in ['internal', 'developer', 'implementation']):
        audience = 'developer'
    elif any(term in content_lower for term in ['user', 'tutorial', 'guide']):
        audience = 'user'
    elif any(term in content_lower for term in ['scientist', 'research', 'analysis']):
        audience = 'scientist'
    else:
        audience = 'general'
    
    # Determine function type for R
    function_type = None
    if language == 'r' and doc_type == 'function':
        if '@export' in content:
            function_type = 'exported'
        elif '#\'' in content:
            function_type = 'internal'
    
    return DocumentContext(
        doc_type=doc_type,
        file_path=file_path,
        language=language,
        audience=audience,
        function_type=function_type
    )

def format_style_feedback(issues, context):
    """Format style feedback for display"""
    if not issues:
        return None
    
    # Separate by severity
    critical_issues = [i for i in issues if i.severity == 'critical']
    major_issues = [i for i in issues if i.severity == 'major']
    minor_issues = [i for i in issues if i.severity in ['minor', 'suggestion']]
    
    parts = ["Documentation Style Review"]
    parts.append("")
    
    # Show critical issues first
    if critical_issues:
        parts.append("Critical Style Issues:")
        for issue in critical_issues:
            parts.append(f"  • {issue.message}")
            if issue.suggestion:
                parts.append(f"    Fix: {issue.suggestion}")
        parts.append("")
    
    # Major issues
    if major_issues:
        parts.append("Style Improvements Needed:")
        for issue in major_issues:
            parts.append(f"  • {issue.message}")
            if issue.suggestion:
                parts.append(f"    Suggestion: {issue.suggestion}")
        parts.append("")
    
    # Minor suggestions (limit to most important)
    if minor_issues:
        parts.append("Style Suggestions:")
        for issue in minor_issues[:3]:  # Show top 3
            parts.append(f"  • {issue.message}")
        
        if len(minor_issues) > 3:
            parts.append(f"  ... and {len(minor_issues) - 3} more suggestions")
        parts.append("")
    
    # Add context-specific advice
    parts.extend(get_context_specific_advice(context))
    
    return "\n".join(parts)

def get_context_specific_advice(context):
    """Get context-specific style advice"""
    advice = []
    
    if context.doc_type == 'function' and context.language == 'r':
        advice.append("R Function Documentation:")
        advice.append("  • Use consistent parameter description style")
        advice.append("  • Start descriptions with verbs for actions")
        advice.append("  • Include units for numeric parameters")
    
    elif context.doc_type == 'readme':
        advice.append("README Best Practices:")
        advice.append("  • Lead with clear value proposition")
        advice.append("  • Keep installation steps simple and testable")
        advice.append("  • Use working examples with expected output")
    
    elif context.doc_type == 'vignette':
        advice.append("Vignette Style:")
        advice.append("  • Maintain tutorial narrative flow")
        advice.append("  • Show progressive complexity")
        advice.append("  • Include session info and reproducible setup")
    
    if context.audience == 'scientist':
        advice.append("Scientific Audience:")
        advice.append("  • Reference methods and citations where appropriate")
        advice.append("  • Use precise terminology")
        advice.append("  • Include performance and accuracy notes")
    
    return advice

if __name__ == '__main__':
    main()