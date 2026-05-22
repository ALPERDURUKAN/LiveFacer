import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime

import modules.globals
import modules.metadata
import modules.ui as ui
from modules import license_manager
from modules import model_bootstrap
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    pass

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='filter the NSFW image or video', dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--map-faces', help='map source target faces', dest='map_faces', action='store_true', default=False)
    program.add_argument('--mouth-mask', help='mask the mouth region', dest='mouth_mask', action='store_true', default=False)
    program.add_argument('--eyes-mask', help='mask the eyes region', dest='eyes_mask', action='store_true', default=False)
    program.add_argument('--eyebrows-mask', help='mask the eyebrows region', dest='eyebrows_mask', action='store_true', default=False)
    program.add_argument('--poisson-blend', help='use poisson blending for swapped face', dest='poisson_blend', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('-l', '--lang', help='UI language', dest='lang', default='en')
    program.add_argument('--live-mirror', help='The live camera display as you see it in the front-facing camera frame', dest='live_mirror', action='store_true', default=False)
    program.add_argument('--live-resizable', help='The live camera frame is resizable', dest='live_resizable', action='store_true', default=False)
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('--license-key', help='license key override', dest='license_key', default=None)
    program.add_argument('--license-enforced', help='enable license enforcement', dest='license_enforced', action='store_true', default=False)
    program.add_argument('--license-offline-grace-days', help='offline grace period in days', dest='license_offline_grace_days', type=int, default=None)
    program.add_argument('--model-bootstrap', help='download protected model set after license validation', dest='model_bootstrap', action='store_true', default=False)
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.eyes_mask = args.eyes_mask
    modules.globals.eyebrows_mask = args.eyebrows_mask
    modules.globals.poisson_blend = args.poisson_blend
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads
    modules.globals.lang = args.lang
    modules.globals.license_key = args.license_key or os.getenv('LIVEFACER_LICENSE_KEY')
    modules.globals.license_enforced = (
        args.license_enforced or parse_env_bool('LIVEFACER_LICENSE_ENFORCE', False)
    )
    configured_grace = args.license_offline_grace_days
    if configured_grace is None:
        env_grace = os.getenv('LIVEFACER_LICENSE_GRACE_DAYS')
        if env_grace:
            try:
                configured_grace = int(env_grace)
            except ValueError:
                configured_grace = None
    modules.globals.license_offline_grace_days = max(
        1, configured_grace if configured_grace is not None else 7
    )
    modules.globals.model_bootstrap_enabled = (
        args.model_bootstrap or parse_env_bool('LIVEFACER_MODEL_BOOTSTRAP', False)
    )

    #for ENHANCER tumbler:
    if 'face_enhancer' in args.frame_processor:
        modules.globals.fp_ui['face_enhancer'] = True
    else:
        modules.globals.fp_ui['face_enhancer'] = False

    # translate deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    available_providers = onnxruntime.get_available_providers()
    encoded_available_providers = encode_execution_providers(available_providers)
    
    decoded = [provider for provider, encoded_execution_provider in zip(available_providers, encoded_available_providers)
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]
            
    # Fallback/Force mapping for providers that might be installed but not detected by default
    provider_map = {
        'cpu': 'CPUExecutionProvider',
        'cuda': 'CUDAExecutionProvider',
        'dml': 'DmlExecutionProvider',
        'openvino': 'OpenVINOExecutionProvider',
        'coreml': 'CoreMLExecutionProvider',
        'rocm': 'ROCMExecutionProvider',
        'tensorrt': 'TensorrtExecutionProvider'
    }
    
    for ep in execution_providers:
        if ep in provider_map and provider_map[ep] not in decoded:
            decoded.append(provider_map[ep])
            
    return decoded


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return ['cpu', 'cuda', 'dml', 'openvino', 'coreml', 'rocm', 'tensorrt']


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 8

def limit_resources() -> None:
    # limit memory usage
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        pass


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')
    if not modules.globals.headless:
        ui.update_status(message)


def ensure_runtime_license() -> bool:
    validation = license_manager.validate_license(modules.globals.license_key)
    modules.globals.license_status_message = validation.get('message', '')
    if validation.get('ok', False):
        update_status(validation.get('message', 'License validated.'), 'DLC.LICENSE')
        return True

    update_status(validation.get('message', 'License validation failed.'), 'DLC.LICENSE')
    if modules.globals.headless:
        return False

    if not validation.get('requires_activation', True):
        return False

    if hasattr(ui, 'prompt_license_activation'):
        entered_key = ui.prompt_license_activation(validation.get('message', 'Activation required.'))
    else:
        entered_key = None
    if not entered_key:
        return False

    activated = license_manager.activate_license(entered_key)
    modules.globals.license_status_message = activated.get('message', '')
    if activated.get('ok', False):
        modules.globals.license_key = entered_key
        update_status(activated.get('message', 'License activated.'), 'DLC.LICENSE')
        return True

    update_status(activated.get('message', 'Activation failed.'), 'DLC.LICENSE')
    return False

def start() -> None:
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    update_status('Processing...')
    # process image to image
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return
        try:
            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        except Exception as e:
            print("Error copying file:", str(e))
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
            release_resources()
        if is_image(modules.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
    if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
        return

    if not modules.globals.map_faces:
        update_status('Creating temp resources...')
        create_temp(modules.globals.target_path)
        update_status('Extracting frames...')
        extract_frames(modules.globals.target_path)

    temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Progressing...', frame_processor.NAME)
        frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
        release_resources()
    # handles fps
    if modules.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(modules.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(modules.globals.target_path, fps)
    else:
        update_status('Creating video with 30.0 fps...')
        create_video(modules.globals.target_path)
    # handle audio
    if modules.globals.keep_audio:
        if modules.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)
    # clean and validate
    clean_temp(modules.globals.target_path)
    if is_video(modules.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit: quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return

    if modules.globals.license_enforced or license_manager.load_license_config().license_enforced:
        if not ensure_runtime_license():
            return
        if modules.globals.model_bootstrap_enabled or parse_env_bool('LIVEFACER_MODEL_BOOTSTRAP', False):
            bootstrap = model_bootstrap.ensure_models_available(modules.globals.license_key)
            if not bootstrap.get('ok', False):
                update_status(bootstrap.get('message', 'Model bootstrap failed.'), 'DLC.MODEL-BOOTSTRAP')
                return
            update_status(bootstrap.get('message', 'Model bootstrap complete.'), 'DLC.MODEL-BOOTSTRAP')

    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if modules.globals.headless:
        start()
    else:
        lang = getattr(modules.globals, "lang", "en")
        window = ui.init(start, destroy, lang)
        window.mainloop()
