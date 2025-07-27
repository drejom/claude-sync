# Foolproof Manual Installation for Private Repos

Copy and paste these commands one by one on your remote host:

## Step 1: Setup directories
```bash
mkdir -p ~/.claude/hooks
cd ~/.claude
```

## Step 2: Clone the repository  
```bash
git clone git@github.com:drejom/claude-hooks.git temp-install
```

## Step 3: Copy files to correct locations
```bash
# Copy all hook files
cp temp-install/hooks/*.py hooks/
cp temp-install/agents/*.py hooks/
cp temp-install/learning/*.py hooks/

# Make them executable
chmod +x hooks/*.py

# Copy settings template (only if it doesn't exist)
[ ! -f settings.local.json ] && cp temp-install/templates/settings.local.json .

# Copy update script
cp temp-install/update.sh update-claude-hooks
chmod +x update-claude-hooks
```

## Step 4: Add to PATH (optional)
```bash
echo 'export PATH="$HOME/.claude:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Step 5: Cleanup
```bash
rm -rf temp-install
```

## Step 6: Verify installation
```bash
ls -la ~/.claude/hooks/
echo "✅ Installation complete!"
```

## Activate in a project
```bash
cd /path/to/your/project
mkdir -p .claude
cp ~/.claude/settings.local.json .claude/
```

That's it! The hooks are now active for any project where you copy the settings file.