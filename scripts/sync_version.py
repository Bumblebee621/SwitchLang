"""
Synchronizes the version from core/version.py to other files like setup.iss.
"""

import os
import re
import sys

# Add project root to path to import core.version
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

try:
    from core.version import __version__
except ImportError:
    print("Error: Could not import core.version. Run this script from the project root or scripts directory.")
    sys.exit(1)

def sync_inno_setup(version):
    iss_path = os.path.join(project_dir, 'setup.iss')
    if not os.path.exists(iss_path):
        print(f"Warning: {iss_path} not found.")
        return

    with open(iss_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace #define MyAppVersion "..."
    new_content = re.sub(
        r'(#define MyAppVersion\s+")[^"]+(")',
        rf'\g<1>{version}\g<2>',
        content
    )

    if content != new_content:
        with open(iss_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Synced version {version} to setup.iss")
    else:
        print("setup.iss version is already up to date.")

def main():
    print(f"Syncing version {__version__}...")
    sync_inno_setup(__version__)
    print("Done.")

if __name__ == "__main__":
    main()
