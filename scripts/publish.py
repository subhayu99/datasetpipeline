#!/usr/bin/env python3
"""
Script to help publish datasetpipeline to PyPI.
"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, check=True):
    """Run a command and print it."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def main():
    # Ensure we're in the right directory
    if not Path("pyproject.toml").exists():
        print("Error: pyproject.toml not found. Run this from the project root.")
        sys.exit(1)
    
    print("🚀 Publishing datasetpipeline to PyPI")
    print("=" * 50)
    
    # Clean previous builds
    print("\n1. Cleaning previous builds...")
    run_cmd("rm -rf dist/ build/ *.egg-info/")
    
    # Install build dependencies
    print("\n2. Installing build dependencies...")
    run_cmd("pip install --upgrade build twine")
    
    # Build the package
    print("\n3. Building package...")
    run_cmd("python -m build")
    
    # Check the package
    print("\n4. Checking package...")
    run_cmd("twine check dist/*")
    
    # Show what will be uploaded
    print("\n5. Package contents:")
    run_cmd("ls -la dist/", check=False)
    
    # Ask for confirmation
    response = input("\n6. Upload to PyPI? (y/N): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Upload to PyPI
    print("\n7. Uploading to PyPI...")
    print("Note: You'll need to enter your PyPI credentials or API token")
    run_cmd("twine upload dist/*")
    
    print("\n✅ Successfully published to PyPI!")
    print("\nUsers can now install with:")
    print("  pip install datasetpipeline")
    print("  uv pip install datasetpipeline")
    print("  uv tool install datasetpipeline")

def test_upload():
    """Upload to TestPyPI first for testing."""
    print("🧪 Testing upload to TestPyPI")
    print("=" * 40)
    
    # Clean and build
    run_cmd("rm -rf dist/ build/ *.egg-info/")
    run_cmd("pip install --upgrade build twine")
    run_cmd("python -m build")
    run_cmd("twine check dist/*")
    
    # Upload to TestPyPI
    print("\nUploading to TestPyPI...")
    run_cmd("twine upload --repository testpypi dist/*")
    
    print("\n✅ Test upload complete!")
    print("\nTest installation with:")
    print("  pip install --index-url https://test.pypi.org/simple/ datasetpipeline")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_upload()
    else:
        main()