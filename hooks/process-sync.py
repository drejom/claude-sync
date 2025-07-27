#!/usr/bin/env python3
"""
Process Sync Script
Processes incoming mesh learning data from other hosts.
"""

import sys
from pathlib import Path

# Try to import mesh sync
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'learning'))
    from mesh_sync import get_mesh_sync
    
    # Process incoming sync data
    mesh_sync = get_mesh_sync()
    mesh_sync.process_incoming_sync('/tmp/claude_sync_incoming.enc')
    
except ImportError:
    # Mesh sync not available
    pass
except Exception:
    # Clean up on any error
    sync_file = Path('/tmp/claude_sync_incoming.enc')
    sync_file.unlink(missing_ok=True)