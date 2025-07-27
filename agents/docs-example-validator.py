#!/usr/bin/env python3
"""
Documentation Example Validator Agent
Tests code examples in documentation for correctness and quality.
"""

import json
import sys
import re
import subprocess
import tempfile
import os
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
    
    # Only analyze documentation files with examples
    if not is_documentation_with_examples(file_path):
        sys.exit(0)
    
    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        sys.exit(0)
    
    # Validate examples
    if INTEL_AVAILABLE:
        validation_result = validate_documentation_examples(content, file_path)
        if validation_result:
            result = {
                'block': False,
                'message': validation_result
            }
            print(json.dumps(result))
    
    sys.exit(0)

def is_documentation_with_examples(file_path):
    """Check if file likely contains code examples"""
    if not file_path:
        return False
    
    path = Path(file_path)
    
    # Files that commonly contain examples
    if (path.suffix in ['.md', '.rst', '.Rmd', '.R', '.py'] or 
        path.name.lower().startswith('readme') or
        'vignette' in path.name.lower() or
        'example' in path.name.lower()):
        return True
    
    return False

def validate_documentation_examples(content, file_path):
    """Validate code examples in documentation"""
    # Extract code examples
    examples = extract_code_examples(content, file_path)
    
    if not examples:
        return None
    
    # Validate each example
    validation_results = []
    for i, example in enumerate(examples):
        result = validate_single_example(example, i + 1)
        if result:
            validation_results.append(result)
    
    if validation_results:
        return format_validation_feedback(validation_results, len(examples))
    
    return None

def extract_code_examples(content, file_path):
    """Extract code examples from documentation"""
    examples = []
    path = Path(file_path)
    
    # R examples in roxygen comments
    if path.suffix == '.R':
        # Extract @examples sections
        examples_sections = re.findall(r'@examples\s*\n(.*?)(?=@\w+|#\'\s*$|$)', content, re.DOTALL)
        for section in examples_sections:
            # Clean up roxygen comment markers
            cleaned = re.sub(r'#\'\s?', '', section).strip()
            if cleaned:
                examples.append({
                    'code': cleaned,
                    'language': 'r',
                    'type': 'roxygen_example',
                    'line_start': content.find(section)
                })
    
    # Markdown code blocks
    if path.suffix in ['.md', '.Rmd', '.rst']:
        # Extract fenced code blocks
        code_blocks = re.finditer(r'```(?:r|R|python|py)?\s*\n(.*?)\n```', content, re.DOTALL)
        for match in code_blocks:
            code_content = match.group(1).strip()
            if code_content:
                # Determine language from fence or content
                fence_line = content[max(0, match.start()-20):match.start()].split('\n')[-1]
                if 'python' in fence_line.lower() or 'py' in fence_line:
                    language = 'python'
                else:
                    language = 'r'  # Default to R for scientific packages
                
                examples.append({
                    'code': code_content,
                    'language': language,
                    'type': 'markdown_block',
                    'line_start': content[:match.start()].count('\n') + 1
                })
    
    # Python docstring examples
    if path.suffix == '.py':
        # Extract examples from docstrings
        docstring_examples = re.findall(r'Examples?:\s*\n(.*?)(?=\n\s*[A-Z][a-z]+:|"""|\'\'\')' , content, re.DOTALL)
        for example in docstring_examples:
            cleaned = example.strip()
            if cleaned:
                examples.append({
                    'code': cleaned,
                    'language': 'python',
                    'type': 'docstring_example',
                    'line_start': content.find(example)
                })
    
    return examples

def validate_single_example(example, example_num):
    """Validate a single code example"""
    issues = []
    code = example['code']
    language = example['language']
    
    # Basic syntax checks
    syntax_issues = check_syntax(code, language)
    issues.extend(syntax_issues)
    
    # Quality checks
    quality_issues = check_example_quality(code, language)
    issues.extend(quality_issues)
    
    # Try to run example (if safe)
    if is_safe_to_run(code, language):
        runtime_issues = check_runtime(code, language)
        issues.extend(runtime_issues)
    
    if issues:
        return {
            'example_num': example_num,
            'code_preview': code[:100] + '...' if len(code) > 100 else code,
            'issues': issues,
            'type': example['type']
        }
    
    return None

def check_syntax(code, language):
    """Check code syntax"""
    issues = []
    
    if language == 'r':
        # Check for common R syntax issues
        if re.search(r'\blibrary\s*\(\s*[^)]*[^\'"][^)]*\)', code):
            issues.append({
                'severity': 'minor',
                'message': 'Library call without quotes',
                'suggestion': 'Use library("package") with quotes'
            })
        
        # Check for assignment operators
        if '=' in code and '<-' not in code:
            equals_assignments = len(re.findall(r'\w+\s*=\s*[^=]', code))
            if equals_assignments > 0:
                issues.append({
                    'severity': 'suggestion',
                    'message': 'Consider using <- for assignment in R',
                    'suggestion': 'R convention prefers <- over ='
                })
    
    elif language == 'python':
        # Try to parse Python syntax
        try:
            compile(code, '<example>', 'exec')
        except SyntaxError as e:
            issues.append({
                'severity': 'critical',
                'message': f'Python syntax error: {e.msg}',
                'suggestion': f'Fix syntax error at line {e.lineno}'
            })
        except:
            pass
    
    return issues

def check_example_quality(code, language):
    """Check example quality and best practices"""
    issues = []
    
    # Check for hardcoded paths
    hardcoded_paths = re.findall(r'["\'](?:/Users/|C:\\|/home/[^/]+)', code)
    if hardcoded_paths:
        issues.append({
            'severity': 'major',
            'message': 'Example contains hardcoded system paths',
            'suggestion': 'Use relative paths or system.file() for examples'
        })
    
    # Check for very short examples
    if len(code.strip()) < 10:
        issues.append({
            'severity': 'minor',
            'message': 'Example is very brief',
            'suggestion': 'Consider a more comprehensive example'
        })
    
    # Check for missing output demonstration
    if language == 'r' and len(code) > 50:
        if not re.search(r'#>\s|print\(|cat\(', code):
            issues.append({
                'severity': 'suggestion',
                'message': 'Example doesn\'t show expected output',
                'suggestion': 'Add #> comments showing expected results'
            })
    
    # Check for data dependencies
    if language == 'r':
        data_calls = re.findall(r'data\s*\(\s*([^)]+)\s*\)', code)
        if data_calls:
            issues.append({
                'severity': 'minor',
                'message': f'Example depends on data: {", ".join(data_calls)}',
                'suggestion': 'Ensure data dependencies are available or use built-in datasets'
            })
    
    # Check for undefined variables
    if language == 'r':
        # Simple check for variables that appear to be used but not defined
        lines = code.split('\n')
        defined_vars = set()
        used_vars = set()
        
        for line in lines:
            # Variables being assigned
            assignments = re.findall(r'(\w+)\s*<-', line)
            defined_vars.update(assignments)
            
            # Variables being used (simplified)
            uses = re.findall(r'\b([a-zA-Z_]\w*)\s*[^\s<-]', line)
            used_vars.update(uses)
        
        # Filter out functions and common R objects
        r_builtins = {'c', 'list', 'data.frame', 'mean', 'sum', 'length', 'print', 'cat', 'head', 'tail'}
        undefined = used_vars - defined_vars - r_builtins
        
        if undefined and len(undefined) > 2:  # Only flag if many undefined
            issues.append({
                'severity': 'suggestion',
                'message': f'Potentially undefined variables: {", ".join(list(undefined)[:3])}',
                'suggestion': 'Ensure all variables are defined in the example'
            })
    
    return issues

def is_safe_to_run(code, language):
    """Check if code is safe to execute"""
    # Don't run code with dangerous operations
    dangerous_patterns = [
        r'\bsystem\s*\(',      # System calls
        r'\bunlink\s*\(',      # File deletion
        r'\bfile\.remove\s*\(',  # File removal
        r'\brm\s+',            # Shell rm command
        r'\binstall\.',        # Package installation
        r'\bdownload\.',       # Downloads
        r'\burl\s*\(',         # URL access
        r'\bhttr::|curl::'     # HTTP requests
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return False
    
    # Don't run very long code
    if len(code) > 500:
        return False
    
    return True

def check_runtime(code, language):
    """Check if code runs without errors"""
    issues = []
    
    try:
        if language == 'r':
            # Try to run R code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    ['Rscript', '--vanilla', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr.strip()[:200]  # Truncate long errors
                    issues.append({
                        'severity': 'major',
                        'message': f'Example fails to run: {error_msg}',
                        'suggestion': 'Fix runtime error or add necessary context'
                    })
                
            finally:
                os.unlink(temp_file)
        
        elif language == 'python':
            # Try to run Python code
            try:
                exec(code, {'__name__': '__main__'})
            except Exception as e:
                issues.append({
                    'severity': 'major',
                    'message': f'Example fails to run: {str(e)[:200]}',
                    'suggestion': 'Fix runtime error or add necessary imports/setup'
                })
    
    except subprocess.TimeoutExpired:
        issues.append({
            'severity': 'minor',
            'message': 'Example takes too long to run',
            'suggestion': 'Consider simpler or faster example'
        })
    except Exception:
        # Don't fail the hook if we can't test runtime
        pass
    
    return issues

def format_validation_feedback(validation_results, total_examples):
    """Format example validation feedback"""
    issues_count = sum(len(result['issues']) for result in validation_results)
    
    parts = [f"Documentation Examples Review ({len(validation_results)}/{total_examples} examples have issues)"]
    parts.append("")
    
    for result in validation_results:
        parts.append(f"Example {result['example_num']} ({result['type']}):")
        parts.append(f"  Code: {result['code_preview']}")
        
        # Group issues by severity
        critical = [i for i in result['issues'] if i['severity'] == 'critical']
        major = [i for i in result['issues'] if i['severity'] == 'major']
        minor = [i for i in result['issues'] if i['severity'] in ['minor', 'suggestion']]
        
        for issue in critical + major:
            severity_mark = "❌" if issue['severity'] == 'critical' else "⚠️"
            parts.append(f"  {severity_mark} {issue['message']}")
            if issue.get('suggestion'):
                parts.append(f"     Fix: {issue['suggestion']}")
        
        for issue in minor[:2]:  # Limit minor issues
            parts.append(f"  💡 {issue['message']}")
        
        parts.append("")
    
    # Add best practices
    parts.append("Example Best Practices:")
    parts.append("  • Use realistic but simple data")
    parts.append("  • Show expected output with comments")
    parts.append("  • Avoid system-specific paths")
    parts.append("  • Test examples before committing")
    
    return "\n".join(parts)

if __name__ == '__main__':
    main()