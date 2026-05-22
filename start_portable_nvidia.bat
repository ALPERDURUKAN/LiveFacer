@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM LiveFacer portable NVIDIA launcher
REM - Uses only /system for Python, CUDA, ffmpeg, git and temp/cache
REM - Installs missing dependencies into portable Python (system/python)
REM
REM Usage:
REM   start_portable_nvidia.bat
REM   start_portable_nvidia.bat --deps
REM   start_portable_nvidia.bat --check
REM   set MAX_MEMORY_GB=8 && start_portable_nvidia.bat
REM   start_portable_nvidia.bat [additional run.py args]

set "BASE_DIR=%~dp0"
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"

set "SYSTEM_DIR=%BASE_DIR%\system"
set "CODE_DIR=%BASE_DIR%\code"
set "PYTHON_EXE=%SYSTEM_DIR%\python\python.exe"
set "REQ_FILE=%CODE_DIR%\requirements.txt"
set "PIP_CACHE_LOCAL=%SYSTEM_DIR%\tmp\pip-cache"
set "WHEELHOUSE=%SYSTEM_DIR%\wheels"
set "MODE=run"

if /I "%~1"=="--deps" (
    set "MODE=deps"
    shift
) else if /I "%~1"=="--check" (
    set "MODE=check"
    shift
)

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Portable Python not found: "%PYTHON_EXE%"
    exit /b 1
)
if not exist "%CODE_DIR%\run.py" (
    echo [ERROR] Could not find "%CODE_DIR%\run.py"
    exit /b 1
)
if not exist "%REQ_FILE%" (
    echo [ERROR] Could not find "%REQ_FILE%"
    exit /b 1
)
if not exist "%SYSTEM_DIR%\tmp" mkdir "%SYSTEM_DIR%\tmp"
if not exist "%PIP_CACHE_LOCAL%" mkdir "%PIP_CACHE_LOCAL%"

REM Keep runtime data portable (no user profile pollution)
set "appdata=%SYSTEM_DIR%\tmp"
set "userprofile=%SYSTEM_DIR%\tmp"
set "TEMP=%SYSTEM_DIR%\tmp"
set "TMP=%SYSTEM_DIR%\tmp"
set "PIP_CACHE_DIR=%PIP_CACHE_LOCAL%"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"

REM NVIDIA/CUDA performance-oriented env
set "CUDA_MODULE_LOADING=LAZY"
set "OMP_NUM_THREADS=1"
set "OMP_WAIT_POLICY=PASSIVE"
set "TF_CPP_MIN_LOG_LEVEL=2"

set "PATH=%SYSTEM_DIR%\git\cmd;%SYSTEM_DIR%\python;%SYSTEM_DIR%\python\Scripts;%SYSTEM_DIR%\ffmpeg;%SYSTEM_DIR%\CUDA;%SYSTEM_DIR%\CUDA\lib;%SYSTEM_DIR%\CUDA\bin;%PATH%"

if exist "%SYSTEM_DIR%\python\Lib\site-packages-gpu\" (
    if defined PYTHONPATH (
        set "PYTHONPATH=%SYSTEM_DIR%\python\Lib\site-packages-gpu;%PYTHONPATH%"
    ) else (
        set "PYTHONPATH=%SYSTEM_DIR%\python\Lib\site-packages-gpu"
    )
)

for %%D in (
    "%SYSTEM_DIR%\python\Lib\site-packages\torch\lib"
    "%SYSTEM_DIR%\python\Lib\site-packages\nvidia\cublas\bin"
    "%SYSTEM_DIR%\python\Lib\site-packages\nvidia\cuda_runtime\bin"
    "%SYSTEM_DIR%\python\Lib\site-packages\nvidia\cudnn\bin"
    "%SYSTEM_DIR%\python\Lib\site-packages\nvidia\cufft\bin"
    "%SYSTEM_DIR%\python\Lib\site-packages\nvidia\curand\bin"
    "%SYSTEM_DIR%\python\Lib\site-packages\nvidia\cusolver\bin"
    "%SYSTEM_DIR%\python\Lib\site-packages\nvidia\cusparse\bin"
    "%SYSTEM_DIR%\python\Lib\site-packages\nvidia\nvjitlink\bin"
    "%SYSTEM_DIR%\python\Lib\site-packages\nvidia\nvtx\bin"
) do (
    if exist "%%~fD" set "PATH=%%~fD;%PATH%"
)

if /I "%MODE%"=="check" (
    echo [INFO] Portable environment check
    echo [INFO] BASE_DIR = %BASE_DIR%
    echo [INFO] PYTHON   = %PYTHON_EXE%
    "%PYTHON_EXE%" --version
    "%PYTHON_EXE%" -c "import onnxruntime as ort; print('onnxruntime:', ort.__version__); print('onnxruntime path:', ort.__file__); print('onnxruntime providers:', ort.get_available_providers())"
    call :warn_if_cuda_missing
    exit /b %ERRORLEVEL%
)

call :ensure_dependencies
if errorlevel 1 exit /b %errorlevel%

if /I "%MODE%"=="deps" (
    echo [OK] Dependencies are ready in "%SYSTEM_DIR%\python"
    exit /b 0
)

set "RUN_ARGS=--execution-provider cuda --execution-threads 1"
if defined MAX_MEMORY_GB set "RUN_ARGS=%RUN_ARGS% --max-memory %MAX_MEMORY_GB%"

pushd "%CODE_DIR%"
echo [INFO] Launching LiveFacer with CUDA...
call :warn_if_cuda_missing
"%PYTHON_EXE%" run.py %RUN_ARGS% %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%

:ensure_dependencies
"%PYTHON_EXE%" -c "import onnxruntime, insightface, cv2, customtkinter" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Dependencies already available.
    goto :eof
)

echo [INFO] Missing dependencies detected. Installing into portable Python...
"%PYTHON_EXE%" -m ensurepip --upgrade >nul 2>&1

if exist "%WHEELHOUSE%\" (
    echo [INFO] Using local wheelhouse: "%WHEELHOUSE%"
    "%PYTHON_EXE%" -m pip install --no-index --find-links "%WHEELHOUSE%" --prefer-binary -r "%REQ_FILE%"
) else (
    echo [INFO] Local wheelhouse not found. Installing from internet.
    "%PYTHON_EXE%" -m pip install --prefer-binary -r "%REQ_FILE%"
)

if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    exit /b 1
)

echo [OK] Dependencies installed successfully.
goto :eof

:warn_if_cuda_missing
"%PYTHON_EXE%" -c "import onnxruntime as ort,sys; sys.exit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)" >nul 2>&1
if errorlevel 1 (
    echo [WARN] CUDAExecutionProvider not detected by onnxruntime.
    echo [WARN] If you installed dependencies manually, run:
    echo [WARN]   "%PYTHON_EXE%" -m pip install --upgrade onnxruntime-gpu
) else (
    echo [OK] CUDAExecutionProvider detected.
)
goto :eof
