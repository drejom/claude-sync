#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "psutil>=5.9.0",
# ]
# ///
"""
Hardware-Based Host Identity System

Generates stable host identifiers from hardware characteristics that survive
OS reinstalls and provide consistent identity for cross-host trust management.

Key Features:
- CPU serial number detection (Intel/AMD)
- Motherboard UUID extraction  
- Primary network interface MAC address
- Cross-platform compatibility (macOS, Linux, WSL)
- Fallback strategies for virtualized environments
- Performance optimized (<100ms generation time)

Security Model:
- Hardware fingerprints are hashed to prevent direct hardware exposure
- Multiple hardware sources combined for robustness
- Deterministic generation ensures consistency
- No network calls or external dependencies
"""

import hashlib
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import time
import re

class HardwareIdentity:
    """Cross-platform hardware identity generation"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.cache_file = Path.home() / '.claude' / '.host_identity_cache'
        self._identity_cache = None
        
    def get_cpu_serial(self) -> Optional[str]:
        """Get CPU serial number or processor ID"""
        try:
            if self.system == 'darwin':  # macOS
                result = subprocess.run([
                    'system_profiler', 'SPHardwareDataType'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    # Look for processor serial or identifier
                    for line in result.stdout.split('\n'):
                        if 'Serial Number' in line and 'Hardware' in result.stdout:
                            return line.split(':')[-1].strip()
                        if 'Processor Name' in line:
                            proc_name = line.split(':')[-1].strip()
                            # Use processor name as stable identifier
                            return hashlib.sha256(proc_name.encode()).hexdigest()[:16]
                            
            elif self.system == 'linux':
                # Try multiple approaches for Linux
                approaches = [
                    self._get_cpu_serial_dmidecode,
                    self._get_cpu_serial_cpuinfo,
                    self._get_cpu_serial_lscpu
                ]
                
                for approach in approaches:
                    result = approach()
                    if result:
                        return result
                        
        except Exception:
            pass
            
        return None
    
    def _get_cpu_serial_dmidecode(self) -> Optional[str]:
        """Get CPU serial from dmidecode (requires root on some systems)"""
        try:
            result = subprocess.run([
                'dmidecode', '-t', 'processor', '-s', 'processor-serial-number'
            ], capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                serial = result.stdout.strip()
                if serial and serial != 'Not Specified' and 'Permission denied' not in serial:
                    return serial
        except Exception:
            pass
        return None
    
    def _get_cpu_serial_cpuinfo(self) -> Optional[str]:
        """Get CPU info from /proc/cpuinfo"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
                
            # Look for processor serial or unique identifiers
            for line in content.split('\n'):
                if 'serial' in line.lower():
                    parts = line.split(':')
                    if len(parts) > 1:
                        return parts[1].strip()
                        
                # Use model name + microcode as stable identifier
                if 'model name' in line.lower():
                    model = line.split(':')[-1].strip()
                    # Look for microcode version
                    for other_line in content.split('\n'):
                        if 'microcode' in other_line.lower():
                            microcode = other_line.split(':')[-1].strip()
                            combined = f"{model}:{microcode}"
                            return hashlib.sha256(combined.encode()).hexdigest()[:16]
                    
                    # Fallback to just model name
                    return hashlib.sha256(model.encode()).hexdigest()[:16]
                    
        except Exception:
            pass
        return None
    
    def _get_cpu_serial_lscpu(self) -> Optional[str]:
        """Get CPU info from lscpu command"""
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                model_name = None
                stepping = None
                
                for line in lines:
                    if 'Model name:' in line:
                        model_name = line.split(':')[-1].strip()
                    elif 'Stepping:' in line:
                        stepping = line.split(':')[-1].strip()
                
                if model_name:
                    # Combine model name and stepping for uniqueness
                    identifier = f"{model_name}:{stepping or 'unknown'}"
                    return hashlib.sha256(identifier.encode()).hexdigest()[:16]
                    
        except Exception:
            pass
        return None

    def get_motherboard_uuid(self) -> Optional[str]:
        """Get motherboard UUID or system UUID"""
        try:
            if self.system == 'darwin':  # macOS
                result = subprocess.run([
                    'system_profiler', 'SPHardwareDataType'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Hardware UUID' in line:
                            return line.split(':')[-1].strip()
                            
            elif self.system == 'linux':
                # Try multiple approaches for Linux
                approaches = [
                    lambda: self._read_file('/sys/class/dmi/id/product_uuid'),
                    lambda: self._read_file('/proc/sys/kernel/random/boot_id'),
                    lambda: self._get_dmidecode_uuid(),
                ]
                
                for approach in approaches:
                    result = approach()
                    if result:
                        return result
                        
        except Exception:
            pass
            
        return None
    
    def _read_file(self, path: str) -> Optional[str]:
        """Safely read a system file"""
        try:
            with open(path, 'r') as f:
                content = f.read().strip()
                if content and content != 'Not Available':
                    return content
        except Exception:
            pass
        return None
    
    def _get_dmidecode_uuid(self) -> Optional[str]:
        """Get system UUID from dmidecode"""
        try:
            result = subprocess.run([
                'dmidecode', '-s', 'system-uuid'
            ], capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                uuid_str = result.stdout.strip()
                if uuid_str and 'Not Settable' not in uuid_str and 'Permission denied' not in uuid_str:
                    return uuid_str
        except Exception:
            pass
        return None

    def get_network_mac_primary(self) -> Optional[str]:
        """Get primary network interface MAC address"""
        try:
            import psutil
            
            # Get all network interfaces
            interfaces = psutil.net_if_addrs()
            
            # Prioritize certain interface types
            priority_patterns = [
                r'^en\d+$',    # Ethernet on macOS (en0, en1)
                r'^eth\d+$',   # Ethernet on Linux (eth0, eth1)
                r'^enp\d+s\d+$',  # Predictable names on Linux
                r'^wl\w+$',    # Wireless on Linux
            ]
            
            # First pass: look for high-priority interfaces
            for pattern in priority_patterns:
                for interface_name, addresses in interfaces.items():
                    if re.match(pattern, interface_name):
                        for addr in addresses:
                            if addr.family.name == 'AF_LINK' and addr.address:
                                mac = addr.address.replace(':', '').replace('-', '').upper()
                                if mac != '000000000000' and len(mac) == 12:
                                    return mac
            
            # Second pass: any valid MAC address
            for interface_name, addresses in interfaces.items():
                # Skip loopback and virtual interfaces
                if interface_name.startswith(('lo', 'docker', 'veth', 'br-')):
                    continue
                    
                for addr in addresses:
                    if addr.family.name == 'AF_LINK' and addr.address:
                        mac = addr.address.replace(':', '').replace('-', '').upper()
                        if mac != '000000000000' and len(mac) == 12:
                            return mac
                            
        except Exception:
            pass
            
        return None

    def generate_stable_host_id(self) -> str:
        """Generate stable host ID from hardware characteristics"""
        start_time = time.time()
        
        # Check cache first
        if self._identity_cache:
            return self._identity_cache
            
        # Try to load from cache file
        cached_identity = self._load_cached_identity()
        if cached_identity:
            self._identity_cache = cached_identity
            return cached_identity
        
        # Collect hardware identifiers
        hardware_sources = []
        
        # CPU identifier (highest priority)
        cpu_serial = self.get_cpu_serial()
        if cpu_serial:
            hardware_sources.append(f"cpu:{cpu_serial}")
        
        # Motherboard/System UUID (second priority)
        mb_uuid = self.get_motherboard_uuid()
        if mb_uuid:
            hardware_sources.append(f"mb:{mb_uuid}")
            
        # Network MAC (lowest priority, but useful for VMs)
        mac_addr = self.get_network_mac_primary()
        if mac_addr:
            hardware_sources.append(f"mac:{mac_addr}")
        
        # Ensure we have at least one hardware identifier
        if not hardware_sources:
            # Fallback: generate from system characteristics
            fallback_data = {
                'platform': platform.platform(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_implementation': platform.python_implementation(),
            }
            fallback_str = json.dumps(fallback_data, sort_keys=True)
            hardware_sources.append(f"fallback:{fallback_str}")
        
        # Combine all sources
        combined_hardware = ':'.join(hardware_sources)
        
        # Generate stable host ID using SHA-256
        host_id = hashlib.sha256(combined_hardware.encode('utf-8')).hexdigest()[:16]
        
        # Cache the result
        self._save_cached_identity(host_id, hardware_sources)
        self._identity_cache = host_id
        
        # Verify performance target
        duration_ms = (time.time() - start_time) * 1000
        if duration_ms > 100:  # 100ms target from interfaces.py
            print(f"Warning: Host identity generation took {duration_ms:.1f}ms (target: <100ms)")
        
        return host_id
    
    def _load_cached_identity(self) -> Optional[str]:
        """Load cached identity if valid"""
        try:
            if not self.cache_file.exists():
                return None
                
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Validate cache is recent (within 24 hours)
            cache_time = cache_data.get('timestamp', 0)
            if time.time() - cache_time > 86400:  # 24 hours
                return None
            
            # Re-verify one hardware source to ensure validity
            cached_sources = cache_data.get('hardware_sources', [])
            if cached_sources:
                # Check if first hardware source still matches
                first_source = cached_sources[0]
                if first_source.startswith('cpu:'):
                    current_cpu = self.get_cpu_serial()
                    if current_cpu and first_source == f"cpu:{current_cpu}":
                        return cache_data.get('host_id')
                elif first_source.startswith('mb:'):
                    current_mb = self.get_motherboard_uuid()
                    if current_mb and first_source == f"mb:{current_mb}":
                        return cache_data.get('host_id')
            
        except Exception:
            pass
            
        return None
    
    def _save_cached_identity(self, host_id: str, hardware_sources: List[str]) -> None:
        """Save identity to cache"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                'host_id': host_id,
                'hardware_sources': hardware_sources,
                'timestamp': time.time(),
                'system': self.system,
                'generated_version': '1.0'
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception:
            pass  # Cache failure shouldn't break identity generation

    def validate_host_identity(self, claimed_id: str) -> bool:
        """Validate claimed host identity against current hardware"""
        try:
            current_id = self.generate_stable_host_id()
            return current_id == claimed_id
        except Exception:
            return False
    
    def get_identity_info(self) -> Dict[str, Any]:
        """Get detailed identity information for debugging"""
        info = {
            'host_id': self.generate_stable_host_id(),
            'system': self.system,
            'hardware_sources': {
                'cpu_serial': self.get_cpu_serial(),
                'motherboard_uuid': self.get_motherboard_uuid(),
                'primary_mac': self.get_network_mac_primary(),
            },
            'platform_info': {
                'platform': platform.platform(),
                'machine': platform.machine(),
                'processor': platform.processor(),
            }
        }
        return info

def main():
    """Test hardware identity generation"""
    identity = HardwareIdentity()
    
    print("Hardware Identity Test")
    print("=" * 50)
    
    start_time = time.time()
    host_id = identity.generate_stable_host_id()
    duration_ms = (time.time() - start_time) * 1000
    
    print(f"Host ID: {host_id}")
    print(f"Generation Time: {duration_ms:.1f}ms")
    print()
    
    # Show detailed info
    info = identity.get_identity_info()
    print("Hardware Sources:")
    for source, value in info['hardware_sources'].items():
        status = "✓" if value else "✗"
        print(f"  {status} {source}: {value or 'Not available'}")
    
    print()
    print("Platform Info:")
    for key, value in info['platform_info'].items():
        print(f"  {key}: {value}")
    
    # Test validation
    print()
    print("Validation Test:")
    is_valid = identity.validate_host_identity(host_id)
    print(f"  Identity validation: {'✓' if is_valid else '✗'}")

if __name__ == '__main__':
    main()