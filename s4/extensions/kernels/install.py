#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def main():
    """Install structured kernels CUDA extensions."""
    kernel_dir = Path(__file__).parent
    setup_py = kernel_dir / "setup.py"

    if not setup_py.exists():
        print(f"Error: {setup_py} not found", file=sys.stderr)
        sys.exit(1)

    # Change to the kernel directory and run setup.py
    original_dir = os.getcwd()
    try:
        os.chdir(kernel_dir)
        result = subprocess.run([
            sys.executable, "setup.py", "install"
        ], check=True)
        print("Kernels installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing kernels: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()