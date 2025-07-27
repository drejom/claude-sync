#!/usr/bin/env python3
"""
Documentation Agent Orchestrator
Coordinates multiple documentation review agents for comprehensive analysis.
"""

import json
import sys
import subprocess
from pathlib import Path
import threading
import time

def main():
    hook_input = json.loads(sys.stdin.read())
    
    if hook_input.get('tool_name') not in ['Edit', 'Write', 'MultiEdit']:
        sys.exit(0)
    
    file_path = hook_input.get('tool_input', {}).get('file_path', '')
    
    # Only analyze documentation files
    if not is_documentation_file(file_path):
        sys.exit(0)
    
    # Run documentation agent orchestration
    orchestration_result = orchestrate_docs_review(hook_input, file_path)
    
    if orchestration_result:
        result = {
            'block': False,
            'message': orchestration_result
        }
        print(json.dumps(result))
    
    sys.exit(0)

def is_documentation_file(file_path):
    """Check if file needs documentation review"""
    if not file_path:
        return False
    
    path = Path(file_path)
    
    # Documentation files
    doc_extensions = ['.md', '.rst', '.Rmd']
    if path.suffix in doc_extensions:
        return True
    
    # Code files with documentation
    if path.suffix in ['.R', '.py']:
        return True
    
    # Special documentation files
    if (path.name.lower().startswith('readme') or 
        'vignette' in path.name.lower() or
        'documentation' in path.name.lower()):
        return True
    
    return False

def orchestrate_docs_review(hook_input, file_path):
    """Orchestrate multiple documentation review agents"""
    
    # Available agents
    agents = {
        'style': 'docs-style-agent.py',
        'completeness': 'docs-completeness-agent.py', 
        'examples': 'docs-example-validator.py',
        'audience': 'docs-audience-optimizer.py'
    }
    
    # Determine which agents to run based on file type and size
    active_agents = select_agents_for_file(file_path)
    
    if not active_agents:
        return None
    
    # Run agents concurrently
    agent_results = run_agents_concurrently(hook_input, active_agents, agents)
    
    # Combine and prioritize results
    if agent_results:
        return format_orchestrated_feedback(agent_results, file_path)
    
    return None

def select_agents_for_file(file_path):
    """Select appropriate agents based on file characteristics"""
    path = Path(file_path)
    active_agents = []
    
    # Always run style and completeness for docs
    active_agents.extend(['style', 'completeness'])
    
    # Run example validator if file likely contains examples
    if (path.suffix in ['.md', '.Rmd', '.R'] or
        'example' in path.name.lower() or
        'vignette' in path.name.lower()):
        active_agents.append('examples')
    
    # Run audience optimizer for user-facing docs
    if (path.name.lower().startswith('readme') or
        path.suffix in ['.md', '.Rmd'] or
        'guide' in path.name.lower() or
        'tutorial' in path.name.lower()):
        active_agents.append('audience')
    
    # For R functions, run all agents
    if path.suffix == '.R':
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            if '@export' in content or '#\'' in content:
                active_agents = ['style', 'completeness', 'examples', 'audience']
        except:
            pass
    
    return list(set(active_agents))  # Remove duplicates

def run_agents_concurrently(hook_input, active_agents, agents):
    """Run selected agents concurrently for faster feedback"""
    agent_results = {}
    threads = []
    
    def run_agent(agent_name, agent_script):
        try:
            # Run agent script
            agents_dir = Path(__file__).parent.parent / 'agents'
            agent_path = agents_dir / agent_script
            
            if not agent_path.exists():
                return
            
            result = subprocess.run(
                ['python3', str(agent_path)],
                input=json.dumps(hook_input),
                text=True,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    agent_output = json.loads(result.stdout)
                    if agent_output.get('message'):
                        agent_results[agent_name] = agent_output['message']
                except json.JSONDecodeError:
                    pass
                    
        except Exception:
            # Don't let one agent failure break others
            pass
    
    # Start agent threads
    for agent_name in active_agents:
        if agent_name in agents:
            thread = threading.Thread(
                target=run_agent,
                args=(agent_name, agents[agent_name])
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
    
    # Wait for all agents to complete (with timeout)
    start_time = time.time()
    for thread in threads:
        remaining_time = max(0, 30 - (time.time() - start_time))
        thread.join(timeout=remaining_time)
    
    return agent_results

def format_orchestrated_feedback(agent_results, file_path):
    """Format combined feedback from all agents"""
    if not agent_results:
        return None
    
    path = Path(file_path)
    
    parts = [f"Comprehensive Documentation Review: {path.name}"]
    parts.append("")
    
    # Agent priority order
    agent_priority = ['completeness', 'style', 'examples', 'audience']
    
    # Show results in priority order
    for agent_name in agent_priority:
        if agent_name in agent_results:
            result = agent_results[agent_name]
            
            # Add section header
            agent_titles = {
                'completeness': '📋 Completeness Analysis',
                'style': '✏️ Style Review', 
                'examples': '🧪 Example Validation',
                'audience': '👥 Audience Optimization'
            }
            
            parts.append(agent_titles.get(agent_name, f'{agent_name.title()} Analysis'))
            parts.append("-" * 40)
            
            # Add agent results (remove duplicate headers)
            result_lines = result.split('\n')
            # Skip first line if it's a title/header
            if result_lines and ('Review' in result_lines[0] or 'Analysis' in result_lines[0]):
                result_lines = result_lines[1:]
            
            parts.extend(result_lines)
            parts.append("")
    
    # Add summary recommendations
    parts.extend(generate_summary_recommendations(agent_results, path))
    
    return "\n".join(parts)

def generate_summary_recommendations(agent_results, path):
    """Generate high-level recommendations based on all agent feedback"""
    recommendations = []
    
    # Count issues across agents
    total_issues = sum(result.count('•') for result in agent_results.values())
    critical_issues = sum(result.lower().count('critical') for result in agent_results.values())
    
    recommendations.append("📝 Summary & Next Steps")
    recommendations.append("-" * 40)
    
    if total_issues == 0:
        recommendations.append("✅ Documentation meets quality standards")
        recommendations.append("   Well-structured and comprehensive")
    elif critical_issues > 0:
        recommendations.append("🔴 Address critical issues before publishing")
        recommendations.append("   Focus on completeness and correctness first")
    elif total_issues > 10:
        recommendations.append("🟡 Multiple improvements needed")
        recommendations.append("   Consider gradual refinement over time")
    else:
        recommendations.append("🟢 Minor improvements will enhance quality")
        recommendations.append("   Documentation is on the right track")
    
    # File-specific recommendations
    if path.name.lower().startswith('readme'):
        recommendations.append("")
        recommendations.append("README-specific priorities:")
        recommendations.append("  1. Clear installation instructions")
        recommendations.append("  2. Working examples") 
        recommendations.append("  3. Quick start guide")
    
    elif path.suffix == '.Rmd':
        recommendations.append("")
        recommendations.append("Vignette priorities:")
        recommendations.append("  1. Logical flow from simple to complex")
        recommendations.append("  2. Reproducible examples with output")
        recommendations.append("  3. Clear section structure")
    
    elif path.suffix == '.R':
        recommendations.append("")
        recommendations.append("R documentation priorities:")
        recommendations.append("  1. Complete @param and @return tags")
        recommendations.append("  2. Working @examples section")
        recommendations.append("  3. Clear function description")
    
    # Learning opportunity
    recommendations.append("")
    recommendations.append("💡 This analysis helps improve documentation quality")
    recommendations.append("   Patterns learned will improve future suggestions")
    
    return recommendations

if __name__ == '__main__':
    main()