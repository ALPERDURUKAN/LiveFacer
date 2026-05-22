#!/usr/bin/env python3

import sys
import os
import importlib

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
]

# Check if running with CUDA provider
is_cuda = False
# Check args for execution provider
for i, arg in enumerate(sys.argv):
    if arg == '--execution-provider' and i + 1 < len(sys.argv):
        if sys.argv[i + 1] == 'cuda':
            is_cuda = True
            break

if is_cuda:
    system_paths.extend([
        os.path.join(base_dir, 'system', 'CUDA'),
        os.path.join(base_dir, 'system', 'CUDA', 'lib'),
        os.path.join(base_dir, 'system', 'CUDA', 'bin'),
    ])

    # Set CUDA environment variables
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    os.environ['CUDA_PATH'] = os.path.join(base_dir, 'system', 'CUDA', 'bin')

# Prepend to PATH (batch file adds to existing PATH)
current_path = os.environ.get('PATH', '')
new_path = os.pathsep.join(system_paths) + os.pathsep + current_path
os.environ['PATH'] = new_path

# Change working directory to code/ folder (as batch file does: cd /d code)
code_dir = os.path.join(base_dir, 'code')
if os.path.exists(code_dir):
    os.chdir(code_dir)

# Store base_dir for use in other modules
sys._base_dir = base_dir


def enforce_runtime_integrity_or_exit() -> None:
    try:
        from modules.integrity import is_integrity_enforced, verify_runtime_integrity
    except Exception as exception:
        print(f"[SECURITY] Integrity module unavailable: {exception}")
        return

    if not is_integrity_enforced():
        return

    is_valid, message = verify_runtime_integrity()
    if not is_valid:
        print(f"[SECURITY] Integrity check failed: {message}")
        sys.exit(13)
    print(f"[SECURITY] {message}")


def patch_protobuf_message_factory() -> None:
    try:
        from google.protobuf import message_factory
    except Exception:
        return

    def get_prototype(self, descriptor):
        return message_factory.GetMessageClass(descriptor)

    for module_name in (
        "google.protobuf.message_factory",
        "google.protobuf.pyext._message",
        "google._upb._message",
    ):
        try:
            module = importlib.import_module(module_name)
            factory_class = getattr(module, "MessageFactory", None)
            if factory_class is None or hasattr(factory_class, "GetPrototype"):
                continue
            setattr(factory_class, "GetPrototype", get_prototype)
        except Exception:
            continue


patch_protobuf_message_factory()
enforce_runtime_integrity_or_exit()

try:
    import modules.tkinter_fix  # noqa: F401
except Exception:
    pass

from modules import core

if __name__ == '__main__':
    from modules import platform_info
    platform_info.print_banner()
    core.run()
