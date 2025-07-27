#!/bin/bash
# One-liner installation for private repos
# Copy and paste this entire block on your remote host

mkdir -p ~/.claude/hooks && \
cd ~/.claude && \
git clone git@github.com:drejom/claude-hooks.git temp-install && \
cp temp-install/hooks/*.py hooks/ && \
cp temp-install/agents/*.py hooks/ && \
cp temp-install/learning/*.py hooks/ && \
chmod +x hooks/*.py && \
[ ! -f settings.local.json ] && cp temp-install/templates/settings.local.json . && \
cp temp-install/update.sh update-claude-hooks && \
chmod +x update-claude-hooks && \
rm -rf temp-install && \
echo "✅ Claude hooks installed! Files:" && \
ls -1 hooks/*.py | wc -l && echo "hooks installed" && \
echo "📋 To activate in a project:" && \
echo "  mkdir -p .claude && cp ~/.claude/settings.local.json .claude/"