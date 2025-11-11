#!/usr/bin/env python3

import sys
import os

# Initialize environment variables and paths before any other imports
# This replicates the behavior of start_portable_nvidia.bat
is_frozen = getattr(sys, 'frozen', False)

if is_frozen:
    # Running as executable - use PyInstaller temp directory
    base_dir = sys._MEIPASS
else:
    # Running as script - use parent directory of code/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set environment variables exactly as in start_portable_nvidia.bat
os.environ['appdata'] = os.path.join(base_dir, 'system', 'tmp')
os.environ['userprofile'] = os.path.join(base_dir, 'system', 'tmp')
os.environ['temp'] = os.path.join(base_dir, 'system', 'tmp')

# Add system paths to PATH (prepend in the same order as batch file)
system_paths = [
    os.path.join(base_dir, 'system', 'git', 'cmd'),
    os.path.join(base_dir, 'system', 'python'),
    os.path.join(base_dir, 'system', 'python', 'scripts'),
    os.path.join(base_dir, 'system', 'ffmpeg'),
    os.path.join(base_dir, 'system', 'CUDA'),
    os.path.join(base_dir, 'system', 'CUDA', 'lib'),
    os.path.join(base_dir, 'system', 'CUDA', 'bin'),
]

# Prepend to PATH (batch file adds to existing PATH)
current_path = os.environ.get('PATH', '')
new_path = os.pathsep.join(system_paths) + os.pathsep + current_path
os.environ['PATH'] = new_path

# Set CUDA environment variables
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['CUDA_PATH'] = os.path.join(base_dir, 'system', 'CUDA', 'bin')

# Change working directory to code/ folder (as batch file does: cd /d code)
code_dir = os.path.join(base_dir, 'code')
if os.path.exists(code_dir):
    os.chdir(code_dir)

# Store base_dir for use in other modules
sys._base_dir = base_dir

from modules import core

if __name__ == '__main__':
    core.run()
