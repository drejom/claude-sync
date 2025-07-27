#!/usr/bin/env python3
"""
Documentation Audience Optimizer Agent
Ensures documentation matches intended audience and adjusts complexity appropriately.
"""

import json
import sys
import re
from pathlib import Path
from collections import Counter, defaultdict

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
    
    # Optimize for audience
    if INTEL_AVAILABLE:
        audience_result = optimize_for_audience(content, file_path)
        if audience_result:
            result = {
                'block': False,
                'message': audience_result
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

def optimize_for_audience(content, file_path):
    """Analyze and optimize documentation for target audience"""
    # Determine current audience and context
    context = determine_document_context(content, file_path)
    
    # Analyze audience appropriateness
    audience_analysis = analyze_audience_match(content, context)
    
    if not audience_analysis['issues']:
        return None
    
    return format_audience_feedback(audience_analysis, context)

def determine_document_context(content, file_path):
    """Determine document context and intended audience"""
    path = Path(file_path)
    
    # Document type
    if path.name.lower().startswith('readme'):
        doc_type = 'readme'
        default_audience = 'user'
    elif path.suffix == '.Rmd' or 'vignette' in path.name.lower():
        doc_type = 'vignette'
        default_audience = 'user'
    elif 'internal' in path.name.lower() or '@keywords internal' in content or '@noRd' in content:
        doc_type = 'function'
        default_audience = 'developer'
    elif '@export' in content or 'def ' in content:
        doc_type = 'function'
        default_audience = 'user'
    else:
        doc_type = 'general'
        default_audience = 'general'
    
    # Language
    if path.suffix == '.R' or path.suffix == '.Rmd':
        language = 'r'
    elif path.suffix == '.py':
        language = 'python'
    else:
        language = 'markdown'
    
    # Detect actual audience from content
    detected_audience = detect_audience_from_content(content)
    audience = detected_audience if detected_audience else default_audience
    
    return DocumentContext(
        doc_type=doc_type,
        file_path=file_path,
        language=language,
        audience=audience
    )

def detect_audience_from_content(content):
    """Detect intended audience from content cues"""
    content_lower = content.lower()
    
    # Developer audience indicators
    developer_terms = ['implementation', 'internal', 'algorithm', 'optimization', 'complexity', 'refactor']
    developer_score = sum(content_lower.count(term) for term in developer_terms)
    
    # User audience indicators  
    user_terms = ['tutorial', 'guide', 'how to', 'getting started', 'example', 'usage']
    user_score = sum(content_lower.count(term) for term in user_terms)
    
    # Scientist audience indicators
    scientist_terms = ['research', 'analysis', 'method', 'dataset', 'experiment', 'publication']
    scientist_score = sum(content_lower.count(term) for term in scientist_terms)
    
    # Determine primary audience
    scores = {'developer': developer_score, 'user': user_score, 'scientist': scientist_score}
    max_audience = max(scores, key=scores.get)
    
    if scores[max_audience] > 2:  # Threshold for confidence
        return max_audience
    
    return None

def analyze_audience_match(content, context):
    """Analyze how well documentation matches intended audience"""
    issues = []
    suggestions = []
    
    # Analyze complexity level
    complexity_analysis = analyze_complexity(content)
    
    # Analyze jargon usage
    jargon_analysis = analyze_jargon(content, context)
    
    # Analyze assumption level
    assumption_analysis = analyze_assumptions(content, context)
    
    # Check audience-specific requirements
    if context.audience == 'user':
        issues.extend(check_user_audience_issues(content, complexity_analysis, jargon_analysis))
    elif context.audience == 'developer':
        issues.extend(check_developer_audience_issues(content, complexity_analysis))
    elif context.audience == 'scientist':
        issues.extend(check_scientist_audience_issues(content, jargon_analysis))
    
    # General audience mismatch issues
    issues.extend(check_general_audience_issues(content, context, complexity_analysis))
    
    return {
        'issues': issues,
        'complexity': complexity_analysis,
        'jargon': jargon_analysis,
        'assumptions': assumption_analysis
    }

def analyze_complexity(content):
    """Analyze technical complexity of content"""
    # Count technical indicators
    technical_terms = [
        'algorithm', 'implementation', 'optimization', 'complexity', 'performance',
        'efficiency', 'scalability', 'architecture', 'framework', 'infrastructure'
    ]
    
    jargon_count = sum(content.lower().count(term) for term in technical_terms)
    
    # Analyze sentence complexity
    sentences = re.split(r'[.!?]+', content)
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    
    # Analyze code-to-text ratio
    code_blocks = len(re.findall(r'```|@examples|    \w+', content))
    total_lines = len(content.split('\n'))
    code_ratio = code_blocks / max(total_lines, 1)
    
    complexity_score = min(100, (jargon_count * 5) + (avg_sentence_length / 2) + (code_ratio * 20))
    
    return {
        'score': complexity_score,
        'jargon_count': jargon_count,
        'avg_sentence_length': avg_sentence_length,
        'code_ratio': code_ratio,
        'level': 'high' if complexity_score > 40 else 'medium' if complexity_score > 20 else 'low'
    }

def analyze_jargon(content, context):
    """Analyze domain-specific jargon usage"""
    # Scientific/bioinformatics terms
    bio_terms = [
        'genomics', 'transcriptomics', 'proteomics', 'bioinformatics', 'sequencing',
        'fastq', 'bam', 'vcf', 'gtf', 'fasta', 'rna-seq', 'chip-seq', 'single-cell'
    ]
    
    # Technical computing terms
    tech_terms = [
        'hpc', 'cluster', 'slurm', 'singularity', 'docker', 'container',
        'parallel', 'distributed', 'scalable', 'pipeline', 'workflow'
    ]
    
    # R/statistics terms
    stats_terms = [
        'dataframe', 'tibble', 'dplyr', 'ggplot', 'bioconductor', 'cran',
        'p-value', 'significance', 'correlation', 'regression', 'model'
    ]
    
    content_lower = content.lower()
    
    jargon_counts = {
        'bio': sum(content_lower.count(term) for term in bio_terms),
        'tech': sum(content_lower.count(term) for term in tech_terms),
        'stats': sum(content_lower.count(term) for term in stats_terms)
    }
    
    total_jargon = sum(jargon_counts.values())
    
    return {
        'counts': jargon_counts,
        'total': total_jargon,
        'density': total_jargon / max(len(content.split()), 1) * 100,
        'primary_domain': max(jargon_counts, key=jargon_counts.get) if total_jargon > 0 else None
    }

def analyze_assumptions(content, context):
    """Analyze assumed prior knowledge"""
    assumptions = []
    
    content_lower = content.lower()
    
    # Check for unexplained technical terms
    technical_first_mentions = []
    for term in ['bioconductor', 'slurm', 'singularity', 'genomics', 'rna-seq']:
        if term in content_lower:
            # Check if term is explained when first mentioned
            term_pos = content_lower.find(term)
            surrounding = content_lower[max(0, term_pos-100):term_pos+100]
            if not any(explain in surrounding for explain in ['is a', 'refers to', 'means', 'which is']):
                technical_first_mentions.append(term)
    
    if technical_first_mentions and context.audience in ['user', 'general']:
        assumptions.append({
            'type': 'unexplained_terms',
            'terms': technical_first_mentions,
            'severity': 'minor'
        })
    
    # Check for assumed software knowledge
    software_mentions = re.findall(r'\b(R|Python|bash|git|docker|singularity)\b', content)
    if len(set(software_mentions)) > 2 and context.audience == 'general':
        assumptions.append({
            'type': 'software_knowledge',
            'software': list(set(software_mentions)),
            'severity': 'minor'
        })
    
    return assumptions

def check_user_audience_issues(content, complexity_analysis, jargon_analysis):
    """Check for user audience specific issues"""
    issues = []
    
    # Too complex for users
    if complexity_analysis['level'] == 'high':
        issues.append({
            'severity': 'major',
            'category': 'complexity',
            'message': 'Content complexity too high for user audience',
            'suggestion': 'Simplify technical language and add more explanatory text'
        })
    
    # Too much unexplained jargon
    if jargon_analysis['density'] > 5:  # >5% jargon
        issues.append({
            'severity': 'minor',
            'category': 'jargon',
            'message': f"High jargon density ({jargon_analysis['density']:.1f}%) for user docs",
            'suggestion': 'Add brief explanations for technical terms'
        })
    
    # Missing practical examples
    if 'example' not in content.lower() and len(content) > 200:
        issues.append({
            'severity': 'minor',
            'category': 'examples',
            'message': 'User documentation lacks practical examples',
            'suggestion': 'Add real-world usage examples'
        })
    
    return issues

def check_developer_audience_issues(content, complexity_analysis):
    """Check for developer audience specific issues"""
    issues = []
    
    # Too simple for developers
    if complexity_analysis['level'] == 'low' and len(content) > 500:
        issues.append({
            'severity': 'suggestion',
            'category': 'depth',
            'message': 'Content may be too basic for developer audience',
            'suggestion': 'Consider adding implementation details or technical context'
        })
    
    # Missing technical details
    if 'implementation' not in content.lower() and 'algorithm' not in content.lower():
        issues.append({
            'severity': 'minor',
            'category': 'technical_depth',
            'message': 'Developer docs may lack sufficient technical detail',
            'suggestion': 'Add implementation notes or algorithm description'
        })
    
    return issues

def check_scientist_audience_issues(content, jargon_analysis):
    """Check for scientist audience specific issues"""
    issues = []
    
    # Missing domain context
    if jargon_analysis['total'] == 0 and len(content) > 300:
        issues.append({
            'severity': 'minor',
            'category': 'domain_context',
            'message': 'Scientific documentation lacks domain-specific context',
            'suggestion': 'Add relevant scientific context or use cases'
        })
    
    # Missing citations or references
    if len(content) > 500 and not re.search(r'\bcitation|\bref|\bdoi|\bpubmed', content.lower()):
        issues.append({
            'severity': 'suggestion',
            'category': 'references',
            'message': 'Scientific docs may benefit from citations or references',
            'suggestion': 'Consider adding relevant literature references'
        })
    
    return issues

def check_general_audience_issues(content, context, complexity_analysis):
    """Check for general audience mismatch issues"""
    issues = []
    
    # Check for audience confusion indicators
    mixed_complexity = (
        ('tutorial' in content.lower() and complexity_analysis['level'] == 'high') or
        ('internal' in content.lower() and context.audience == 'user')
    )
    
    if mixed_complexity:
        issues.append({
            'severity': 'major',
            'category': 'audience_mismatch',
            'message': 'Mixed signals about intended audience',
            'suggestion': 'Clarify target audience and adjust complexity accordingly'
        })
    
    return issues

def format_audience_feedback(audience_analysis, context):
    """Format audience optimization feedback"""
    parts = [f"Documentation Audience Review (Target: {context.audience})"]
    parts.append("")
    
    # Show complexity analysis
    complexity = audience_analysis['complexity']
    parts.append(f"Content complexity: {complexity['level']} (score: {complexity['score']:.0f})")
    
    # Show jargon analysis
    jargon = audience_analysis['jargon']
    if jargon['total'] > 0:
        parts.append(f"Domain jargon: {jargon['density']:.1f}% ({jargon['primary_domain']} focus)")
    
    parts.append("")
    
    # Group issues by severity
    issues = audience_analysis['issues']
    critical_issues = [i for i in issues if i['severity'] == 'critical']
    major_issues = [i for i in issues if i['severity'] == 'major']
    minor_issues = [i for i in issues if i['severity'] in ['minor', 'suggestion']]
    
    if critical_issues:
        parts.append("Critical Audience Mismatches:")
        for issue in critical_issues:
            parts.append(f"  • {issue['message']}")
            if issue.get('suggestion'):
                parts.append(f"    Fix: {issue['suggestion']}")
        parts.append("")
    
    if major_issues:
        parts.append("Audience Optimization Needed:")
        for issue in major_issues:
            parts.append(f"  • {issue['message']}")
            if issue.get('suggestion'):
                parts.append(f"    Suggestion: {issue['suggestion']}")
        parts.append("")
    
    if minor_issues:
        parts.append("Audience Enhancement Suggestions:")
        for issue in minor_issues[:3]:  # Top 3
            parts.append(f"  • {issue['message']}")
        parts.append("")
    
    # Add audience-specific recommendations
    parts.extend(get_audience_recommendations(context, complexity))
    
    return "\n".join(parts)

def get_audience_recommendations(context, complexity):
    """Get audience-specific writing recommendations"""
    recommendations = []
    
    if context.audience == 'user':
        recommendations.append("User Documentation Best Practices:")
        recommendations.append("  • Start with clear problem statement")
        recommendations.append("  • Use step-by-step instructions")
        recommendations.append("  • Show expected outcomes")
        recommendations.append("  • Explain technical terms briefly")
    
    elif context.audience == 'developer':
        recommendations.append("Developer Documentation Best Practices:")
        recommendations.append("  • Include implementation details")
        recommendations.append("  • Show code architecture")
        recommendations.append("  • Document edge cases and limitations")
        recommendations.append("  • Reference related technical concepts")
    
    elif context.audience == 'scientist':
        recommendations.append("Scientific Documentation Best Practices:")
        recommendations.append("  • Provide methodological context")
        recommendations.append("  • Reference relevant literature")
        recommendations.append("  • Include performance or accuracy notes")
        recommendations.append("  • Show validation or benchmarking")
    
    return recommendations

if __name__ == '__main__':
    main()