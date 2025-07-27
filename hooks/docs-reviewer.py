#!/usr/bin/env python3
"""
Documentation Review Hook
Analyzes code changes and proposes documentation updates before repo pushes.
"""

import json
import sys
import subprocess
import re
import os
from pathlib import Path
from collections import defaultdict

def main():
    hook_input = json.loads(sys.stdin.read())
    
    if hook_input.get('tool_name') != 'Bash':
        sys.exit(0)
    
    command = hook_input.get('tool_input', {}).get('command', '')
    
    # Trigger on git push commands
    if is_git_push_command(command):
        review_result = review_documentation_before_push()
        if review_result:
            result = {
                'block': False,
                'message': review_result
            }
            print(json.dumps(result))
    
    sys.exit(0)

def is_git_push_command(command):
    """Check if this is a git push command"""
    return (command.strip().startswith('git push') or 
            'git push' in command)

def review_documentation_before_push():
    """Review and propose documentation updates"""
    try:
        # Get changes to be pushed
        changes = get_changes_to_push()
        if not changes:
            return None
        
        # Analyze changes for documentation impact
        doc_suggestions = analyze_changes_for_docs(changes)
        
        if doc_suggestions:
            return format_documentation_suggestions(doc_suggestions)
        
    except Exception:
        # Don't break git operations if review fails
        pass
    
    return None

def get_changes_to_push():
    """Get list of changes about to be pushed"""
    try:
        # Get the remote tracking branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            # No upstream branch
            return []
        
        remote_branch = result.stdout.strip()
        
        # Get diff between local and remote
        diff_result = subprocess.run(
            ['git', 'diff', '--name-status', remote_branch],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if diff_result.returncode != 0:
            return []
        
        changes = []
        for line in diff_result.stdout.strip().split('\n'):
            if line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    status, filename = parts
                    changes.append({
                        'status': status,
                        'file': filename,
                        'type': classify_file_type(filename)
                    })
        
        return changes
        
    except Exception:
        return []

def classify_file_type(filename):
    """Classify file type for documentation impact assessment"""
    path = Path(filename)
    
    if path.suffix == '.py':
        return 'python_code'
    elif path.suffix == '.sh':
        return 'shell_script'
    elif path.suffix == '.md':
        return 'documentation'
    elif path.suffix == '.json':
        return 'configuration'
    elif path.name in ['install.sh', 'update.sh']:
        return 'installer_script'
    elif 'hook' in filename.lower():
        return 'hook_script'
    elif 'learning' in filename.lower():
        return 'learning_module'
    else:
        return 'other'

def analyze_changes_for_docs(changes):
    """Analyze changes and suggest documentation updates"""
    suggestions = []
    
    # Group changes by type
    changes_by_type = defaultdict(list)
    for change in changes:
        changes_by_type[change['type']].append(change)
    
    # Check for new hooks
    new_hooks = [c for c in changes if c['status'] == 'A' and c['type'] == 'hook_script']
    if new_hooks:
        hook_names = [Path(c['file']).stem for c in new_hooks]
        suggestions.append({
            'type': 'new_features',
            'priority': 'high',
            'suggestion': f"📚 **New hooks added**: {', '.join(hook_names)}",
            'details': [
                "Consider updating README.md with:",
                f"  • Description of new hook capabilities: {', '.join(hook_names)}",
                "  • Updated installation/configuration instructions",
                "  • Examples of new functionality"
            ]
        })
    
    # Check for new learning modules
    new_learning = [c for c in changes if c['status'] == 'A' and c['type'] == 'learning_module']
    if new_learning:
        module_names = [Path(c['file']).stem for c in new_learning]
        suggestions.append({
            'type': 'architecture_changes',
            'priority': 'high', 
            'suggestion': f"🧠 **New learning modules**: {', '.join(module_names)}",
            'details': [
                "Consider updating documentation with:",
                f"  • Architecture overview including: {', '.join(module_names)}",
                "  • Security model explanation",
                "  • Cross-host learning capabilities"
            ]
        })
    
    # Check for installer changes
    installer_changes = [c for c in changes if c['type'] == 'installer_script']
    if installer_changes:
        suggestions.append({
            'type': 'installation_changes',
            'priority': 'medium',
            'suggestion': "⚙️ **Installation scripts modified**",
            'details': [
                "Consider updating:",
                "  • Installation instructions in README.md",
                "  • Prerequisites or dependencies",
                "  • Troubleshooting section"
            ]
        })
    
    # Check for configuration changes
    config_changes = [c for c in changes if c['type'] == 'configuration']
    if config_changes:
        suggestions.append({
            'type': 'configuration_changes',
            'priority': 'medium',
            'suggestion': "⚙️ **Configuration files modified**",
            'details': [
                "Consider updating:",
                "  • Configuration examples in README.md",
                "  • Template settings documentation",
                "  • User configuration guide"
            ]
        })
    
    # Check if README was updated proportionally
    readme_updated = any(c['file'].lower() == 'readme.md' for c in changes)
    significant_changes = len([c for c in changes if c['type'] in ['hook_script', 'learning_module', 'installer_script']]) > 0
    
    if significant_changes and not readme_updated:
        suggestions.append({
            'type': 'missing_docs',
            'priority': 'high',
            'suggestion': "❗ **README.md not updated despite significant changes**",
            'details': [
                "Consider updating README.md to reflect:",
                "  • New functionality or capabilities",
                "  • Changed installation process",
                "  • Updated examples or usage patterns"
            ]
        })
    
    # Analyze commit messages for documentation hints
    recent_commit_analysis = analyze_recent_commits()
    if recent_commit_analysis:
        suggestions.extend(recent_commit_analysis)
    
    return suggestions

def analyze_recent_commits():
    """Analyze recent commit messages for documentation clues"""
    try:
        # Get recent commit messages
        result = subprocess.run(
            ['git', 'log', '--oneline', '-5', '--no-merges'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return []
        
        suggestions = []
        commits = result.stdout.strip().split('\n')
        
        for commit in commits:
            if not commit:
                continue
            
            commit_msg = commit.split(' ', 1)[1] if ' ' in commit else commit
            
            # Look for feature-indicating keywords
            if any(keyword in commit_msg.lower() for keyword in ['new', 'add', 'implement', 'create']):
                suggestions.append({
                    'type': 'commit_analysis',
                    'priority': 'medium',
                    'suggestion': f"💭 **Recent commit suggests new features**: \"{commit_msg[:50]}...\"",
                    'details': [
                        "Consider documenting:",
                        "  • What problem this feature solves",
                        "  • How users can benefit from it",
                        "  • Usage examples or configuration"
                    ]
                })
                break  # Only show one commit analysis
        
        return suggestions
        
    except Exception:
        return []

def format_documentation_suggestions(suggestions):
    """Format documentation suggestions for display"""
    if not suggestions:
        return None
    
    # Sort by priority
    priority_order = {'high': 1, 'medium': 2, 'low': 3}
    suggestions.sort(key=lambda x: priority_order.get(x['priority'], 3))
    
    parts = ["📋 **Documentation Review Before Push**"]
    parts.append("")
    
    high_priority = [s for s in suggestions if s['priority'] == 'high']
    medium_priority = [s for s in suggestions if s['priority'] == 'medium']
    
    if high_priority:
        parts.append("🔴 **High Priority Updates Needed:**")
        for suggestion in high_priority:
            parts.append(f"  {suggestion['suggestion']}")
            for detail in suggestion['details']:
                parts.append(f"    {detail}")
            parts.append("")
    
    if medium_priority:
        parts.append("🟡 **Consider Updating:**")
        for suggestion in medium_priority:
            parts.append(f"  {suggestion['suggestion']}")
            for detail in suggestion['details'][:2]:  # Limit details for medium priority
                parts.append(f"    {detail}")
            parts.append("")
    
    # Add quick action suggestions
    parts.append("⚡ **Quick Actions:**")
    parts.append("  • Update README.md with new features")
    parts.append("  • Add examples for new hooks")
    parts.append("  • Update installation instructions if needed")
    parts.append("  • Consider adding troubleshooting notes")
    parts.append("")
    parts.append("💡 *This review helps keep documentation current with code changes*")
    
    return "\n".join(parts)

def get_file_analysis():
    """Analyze specific files for documentation needs"""
    analysis = {}
    
    # Check for hooks directory
    hooks_dir = Path('hooks')
    if hooks_dir.exists():
        hook_files = list(hooks_dir.glob('*.py'))
        analysis['hook_count'] = len(hook_files)
        analysis['hook_names'] = [f.stem for f in hook_files]
    
    # Check for learning directory
    learning_dir = Path('learning')
    if learning_dir.exists():
        learning_files = list(learning_dir.glob('*.py'))
        analysis['learning_modules'] = len(learning_files)
        analysis['learning_names'] = [f.stem for f in learning_files]
    
    # Check README length vs code complexity
    readme_path = Path('README.md')
    if readme_path.exists():
        readme_lines = len(readme_path.read_text().split('\n'))
        analysis['readme_lines'] = readme_lines
        
        # Rough heuristic: complex projects should have substantial docs
        total_code_files = len(list(Path('.').rglob('*.py'))) + len(list(Path('.').rglob('*.sh')))
        analysis['docs_to_code_ratio'] = readme_lines / max(total_code_files, 1)
    
    return analysis

if __name__ == '__main__':
    main()