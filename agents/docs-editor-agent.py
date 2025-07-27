#!/usr/bin/env python3
"""
Documentation Editor Agent
Ruthless editor that maintains tight, direct documentation standards.
Learns user's editing style and detects when docs need serious cleanup.
"""

import json
import sys
import re
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

# Import documentation intelligence
try:
    from docs_intelligence import get_docs_intelligence, DocumentContext
    sys.path.insert(0, str(Path(__file__).parent.parent / 'learning'))
    from encryption import get_secure_storage
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

def main():
    hook_input = json.loads(sys.stdin.read())
    
    if hook_input.get('tool_name') != 'Bash':
        sys.exit(0)
    
    command = hook_input.get('tool_input', {}).get('command', '')
    
    # Trigger on git commit commands
    if is_git_commit_command(command):
        editor_result = editorial_review_before_commit()
        if editor_result:
            result = {
                'block': False,
                'message': editor_result
            }
            print(json.dumps(result))
    
    sys.exit(0)

def is_git_commit_command(command):
    """Check if this is a git commit command"""
    return (command.strip().startswith('git commit') or 
            'git commit' in command)

def editorial_review_before_commit():
    """Editorial review of documentation changes before commit"""
    try:
        # Get documentation files that changed
        doc_changes = get_documentation_changes()
        if not doc_changes:
            return None
        
        # Analyze changes for editorial issues
        editorial_analysis = analyze_documentation_changes(doc_changes)
        
        if editorial_analysis and editorial_analysis['needs_editing']:
            return format_editorial_feedback(editorial_analysis)
        
    except Exception:
        # Don't break git operations
        pass
    
    return None

def get_documentation_changes():
    """Get documentation files that have changed"""
    try:
        # Get staged files
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return []
        
        changed_files = result.stdout.strip().split('\n')
        
        # Filter for documentation files
        doc_files = []
        for file_path in changed_files:
            if file_path and is_documentation_file(file_path):
                doc_files.append(file_path)
        
        return doc_files
        
    except Exception:
        return []

def is_documentation_file(file_path):
    """Check if file is documentation"""
    if not file_path:
        return False
    
    path = Path(file_path)
    return (path.suffix in ['.md', '.rst', '.Rmd'] or 
            path.name.lower().startswith('readme') or
            'doc' in path.name.lower())

def analyze_documentation_changes(doc_files):
    """Analyze documentation changes for editorial issues"""
    if not LEARNING_AVAILABLE:
        return None
    
    storage = get_secure_storage()
    editorial_issues = []
    total_bloat_score = 0
    
    for file_path in doc_files:
        try:
            # Get current and previous versions
            current_content = read_file_content(file_path)
            if not current_content:
                continue
            
            previous_content = get_previous_file_version(file_path)
            
            # Analyze the change
            change_analysis = analyze_single_file_change(
                file_path, previous_content, current_content, storage
            )
            
            if change_analysis:
                editorial_issues.append(change_analysis)
                total_bloat_score += change_analysis.get('bloat_score', 0)
        
        except Exception:
            continue
    
    if not editorial_issues:
        return None
    
    # Determine if editorial intervention is needed
    needs_editing = (
        total_bloat_score > 50 or  # High bloat score
        any(issue.get('severity') == 'critical' for issue in editorial_issues) or
        any(issue.get('author_type') in ['inexperienced', 'ai_generated'] for issue in editorial_issues)
    )
    
    return {
        'needs_editing': needs_editing,
        'issues': editorial_issues,
        'total_bloat_score': total_bloat_score,
        'files_analyzed': len(doc_files)
    }

def analyze_single_file_change(file_path, previous_content, current_content, storage):
    """Analyze a single file's documentation changes"""
    if not previous_content:
        # New file - check if it meets standards
        return analyze_new_documentation(file_path, current_content, storage)
    
    # Analyze the diff
    change_analysis = {
        'file_path': file_path,
        'change_type': 'modification',
        'bloat_score': 0,
        'issues': [],
        'author_type': detect_author_type(current_content, previous_content),
        'severity': 'minor'
    }
    
    # Calculate length change
    prev_length = len(previous_content)
    curr_length = len(current_content)
    length_ratio = curr_length / max(prev_length, 1)
    
    # Detect sprawl
    if length_ratio > 1.5:  # 50% increase
        change_analysis['bloat_score'] += 30
        change_analysis['issues'].append("Significant length increase - check for sprawl")
    
    # Detect style degradation
    style_issues = detect_style_degradation(previous_content, current_content)
    change_analysis['issues'].extend(style_issues)
    change_analysis['bloat_score'] += len(style_issues) * 10
    
    # Detect structural problems
    structure_issues = detect_structure_problems(current_content)
    change_analysis['issues'].extend(structure_issues)
    change_analysis['bloat_score'] += len(structure_issues) * 15
    
    # Learn from this analysis
    learn_editorial_patterns(change_analysis, storage)
    
    # Set severity based on bloat score
    if change_analysis['bloat_score'] > 40:
        change_analysis['severity'] = 'critical'
    elif change_analysis['bloat_score'] > 20:
        change_analysis['severity'] = 'major'
    
    return change_analysis if change_analysis['issues'] else None

def analyze_new_documentation(file_path, content, storage):
    """Analyze new documentation file"""
    analysis = {
        'file_path': file_path,
        'change_type': 'new_file',
        'bloat_score': 0,
        'issues': [],
        'author_type': detect_author_type(content, ""),
        'severity': 'minor'
    }
    
    # Check if new file follows tight documentation principles
    length_issues = check_document_length(content, file_path)
    analysis['issues'].extend(length_issues)
    analysis['bloat_score'] += len(length_issues) * 20
    
    # Check structure
    structure_issues = detect_structure_problems(content)
    analysis['issues'].extend(structure_issues)
    analysis['bloat_score'] += len(structure_issues) * 15
    
    # Check for learned bad patterns
    bad_patterns = detect_learned_bad_patterns(content, storage)
    analysis['issues'].extend(bad_patterns)
    analysis['bloat_score'] += len(bad_patterns) * 25
    
    if analysis['bloat_score'] > 30:
        analysis['severity'] = 'major'
    
    return analysis if analysis['issues'] else None

def detect_author_type(current_content, previous_content):
    """Detect likely author type based on content patterns"""
    
    # AI-generated indicators
    ai_indicators = [
        r'here.*step.*guide',  # "Here's a step-by-step guide"
        r'comprehensive.*overview',  # AI loves "comprehensive"
        r'leverage.*power',     # "leverage the power of"
        r'seamlessly.*integrate',  # AI buzzwords
        r'robust.*solution',    # More AI favorites
        r'cutting.*edge',
        r'revolutionize.*workflow'
    ]
    
    ai_score = sum(1 for pattern in ai_indicators 
                   if re.search(pattern, current_content, re.IGNORECASE))
    
    # Inexperienced author indicators
    inexperienced_indicators = [
        r'obviously',
        r'clearly',
        r'simply',
        r'just.*need.*to',
        r'easy.*steps',
        r'don\'t.*worry',
        r'as.*you.*can.*see'
    ]
    
    inexperienced_score = sum(1 for pattern in inexperienced_indicators 
                             if re.search(pattern, current_content, re.IGNORECASE))
    
    # Drunk/careless indicators
    drunk_indicators = [
        r'\b(teh|hte|adn|nad|taht|thsi)\b',  # Typos
        r'\.{3,}',  # Excessive ellipses
        r'!{2,}',   # Multiple exclamation marks
        r'\?{2,}',  # Multiple question marks
        r'\b\w+\s+\1\b'  # Repeated words
    ]
    
    drunk_score = sum(1 for pattern in drunk_indicators 
                     if re.search(pattern, current_content, re.IGNORECASE))
    
    # Verbose/academic indicators (opposite of tight style)
    verbose_indicators = [
        r'furthermore',
        r'moreover',
        r'in.*addition.*to',
        r'it.*should.*be.*noted',
        r'worth.*mentioning',
        r'comprehensive.*analysis',
        r'detailed.*examination'
    ]
    
    verbose_score = sum(1 for pattern in verbose_indicators 
                       if re.search(pattern, current_content, re.IGNORECASE))
    
    # Determine author type
    if ai_score >= 3:
        return 'ai_generated'
    elif inexperienced_score >= 2:
        return 'inexperienced'
    elif drunk_score >= 2:
        return 'careless'
    elif verbose_score >= 3:
        return 'academic_verbose'
    else:
        return 'unknown'

def detect_style_degradation(previous_content, current_content):
    """Detect degradation in documentation style"""
    issues = []
    
    # Check for introduction of fluff words
    fluff_words = ['obviously', 'clearly', 'simply', 'just', 'seamlessly', 'robust', 'comprehensive']
    prev_fluff = sum(previous_content.lower().count(word) for word in fluff_words)
    curr_fluff = sum(current_content.lower().count(word) for word in fluff_words)
    
    if curr_fluff > prev_fluff + 2:
        issues.append("Introduction of fluff words - maintain direct style")
    
    # Check for emoji explosion
    prev_emoji = len(re.findall(r'[🚀🎯⚡🔧🧠🌍💡✅❌⚠️]', previous_content))
    curr_emoji = len(re.findall(r'[🚀🎯⚡🔧🧠🌍💡✅❌⚠️]', current_content))
    
    if curr_emoji > prev_emoji + 5:
        issues.append("Emoji overuse - maintain clean visual style")
    
    # Check for passive voice increase
    passive_patterns = [r'is.*done', r'are.*used', r'will.*be', r'can.*be.*found']
    prev_passive = sum(len(re.findall(pattern, previous_content, re.IGNORECASE)) for pattern in passive_patterns)
    curr_passive = sum(len(re.findall(pattern, current_content, re.IGNORECASE)) for pattern in passive_patterns)
    
    if curr_passive > prev_passive + 3:
        issues.append("Increased passive voice - use active, direct language")
    
    return issues

def detect_structure_problems(content):
    """Detect structural problems in documentation"""
    issues = []
    
    # Check for excessive nesting
    header_levels = re.findall(r'^(#{1,6})', content, re.MULTILINE)
    if header_levels and max(len(h) for h in header_levels) > 4:
        issues.append("Excessive header nesting - flatten structure")
    
    # Check for wall of text
    paragraphs = content.split('\n\n')
    long_paragraphs = [p for p in paragraphs if len(p.strip()) > 500]
    if len(long_paragraphs) > 2:
        issues.append("Wall of text detected - break into smaller sections")
    
    # Check for list explosion
    list_items = len(re.findall(r'^\s*[-*+]\s', content, re.MULTILINE))
    bullet_lists = len(re.findall(r'^\s*[-*+]\s', content, re.MULTILINE))
    if list_items > 20:
        issues.append("List overload - consolidate or use tables")
    
    # Check for redundant sections
    headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
    header_words = []
    for header in headers:
        header_words.extend(header.lower().split())
    
    word_counts = Counter(header_words)
    repeated_words = [word for word, count in word_counts.items() if count > 3 and len(word) > 4]
    if repeated_words:
        issues.append(f"Repetitive section naming: {', '.join(repeated_words[:3])}")
    
    return issues

def check_document_length(content, file_path):
    """Check if document length is appropriate"""
    issues = []
    length = len(content)
    
    path = Path(file_path)
    
    # README should be concise
    if path.name.lower().startswith('readme') and length > 3000:
        issues.append("README too long - keep under 3000 chars for readability")
    
    # Any doc file getting too verbose
    if length > 8000:
        issues.append("Document getting verbose - consider splitting or linking to detailed docs")
    
    # Check line count
    lines = len(content.split('\n'))
    if lines > 200:
        issues.append("Too many lines - aim for scannable documentation")
    
    return issues

def detect_learned_bad_patterns(content, storage):
    """Detect patterns previously identified as problematic"""
    issues = []
    
    if not storage:
        return issues
    
    # Load learned bad patterns
    editorial_data = storage.load_learning_data('editorial_patterns', {
        'bad_phrases': [],
        'bloat_patterns': [],
        'style_violations': []
    })
    
    # Check against learned bad patterns
    for bad_phrase in editorial_data.get('bad_phrases', []):
        if bad_phrase.lower() in content.lower():
            issues.append(f"Previously flagged phrase: '{bad_phrase}'")
    
    for bloat_pattern in editorial_data.get('bloat_patterns', []):
        if re.search(bloat_pattern, content, re.IGNORECASE):
            issues.append("Matches previously identified bloat pattern")
    
    return issues

def learn_editorial_patterns(change_analysis, storage):
    """Learn from editorial analysis for future improvement"""
    if not storage:
        return
    
    # Load existing editorial learning data
    editorial_data = storage.load_learning_data('editorial_patterns', {
        'author_type_patterns': defaultdict(list),
        'bloat_indicators': defaultdict(int),
        'style_preferences': [],
        'cleanup_successes': []
    })
    
    # Learn author type patterns
    author_type = change_analysis.get('author_type', 'unknown')
    if author_type != 'unknown':
        editorial_data['author_type_patterns'][author_type].extend(change_analysis.get('issues', []))
    
    # Learn bloat indicators
    if change_analysis.get('bloat_score', 0) > 20:
        for issue in change_analysis.get('issues', []):
            editorial_data['bloat_indicators'][issue] += 1
    
    # Store updated learning data
    storage.store_learning_data('editorial_patterns', dict(editorial_data))

def get_previous_file_version(file_path):
    """Get previous version of file from git"""
    try:
        result = subprocess.run(
            ['git', 'show', f'HEAD:{file_path}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return result.stdout
        
    except Exception:
        pass
    
    return ""

def read_file_content(file_path):
    """Read current file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""

def format_editorial_feedback(editorial_analysis):
    """Format editorial feedback"""
    issues = editorial_analysis['issues']
    bloat_score = editorial_analysis['total_bloat_score']
    
    parts = ["Documentation Editorial Review"]
    parts.append("")
    
    # Overall assessment
    if bloat_score > 50:
        parts.append("🚨 CRITICAL: Documentation sprawl detected")
        parts.append("   Docs are losing focus and directness")
    elif bloat_score > 30:
        parts.append("⚠️  WARNING: Documentation getting verbose")
        parts.append("   Tighten up before it gets out of hand")
    else:
        parts.append("📝 REVIEW: Minor editorial issues detected")
    
    parts.append(f"   Bloat score: {bloat_score}/100")
    parts.append("")
    
    # Group issues by severity
    critical_issues = [i for i in issues if i.get('severity') == 'critical']
    major_issues = [i for i in issues if i.get('severity') == 'major']
    minor_issues = [i for i in issues if i.get('severity') == 'minor']
    
    if critical_issues:
        parts.append("CRITICAL EDITORIAL ISSUES:")
        for issue in critical_issues:
            parts.append(f"  • {Path(issue['file_path']).name}: Author type '{issue['author_type']}'")
            for problem in issue['issues'][:3]:  # Top 3 issues
                parts.append(f"    - {problem}")
        parts.append("")
    
    if major_issues:
        parts.append("MAJOR CLEANUP NEEDED:")
        for issue in major_issues:
            parts.append(f"  • {Path(issue['file_path']).name}")
            for problem in issue['issues'][:2]:  # Top 2 issues
                parts.append(f"    - {problem}")
        parts.append("")
    
    # Editorial principles reminder
    parts.append("EDITORIAL PRINCIPLES:")
    parts.append("  • Direct, no fluff - every word must earn its place")
    parts.append("  • Scannable structure - headers, bullets, white space")
    parts.append("  • Link to details rather than explaining everything")
    parts.append("  • Active voice, concrete examples")
    parts.append("  • Assume intelligent readers, respect their time")
    
    return "\n".join(parts)

if __name__ == '__main__':
    main()