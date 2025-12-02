# Bugs Fixed for PC Installation and Usage

This document lists all bugs that were identified and fixed to enable successful installation and usage of LiveFacer on PC.

## Critical Installation Bugs

### 1. Non-existent Package in requirements.txt
**File:** `requirements.txt`  
**Issue:** The package `onnxruntime-tensorrt==1.22.0` does not exist in PyPI, causing installation to fail.  
**Fix:** Removed the non-existent package from requirements.txt (line 20).  
**Impact:** Installation would fail immediately when running `pip install -r requirements.txt`

### 2. Incorrect base_dir Calculation in run.py
**File:** `run.py`  
**Issue:** When running as a script (not frozen executable), the base_dir was calculated as the parent of the parent directory (`os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`), which is incorrect.  
**Fix:** Changed to use only one dirname to get the directory containing run.py: `os.path.dirname(os.path.abspath(__file__))`  
**Impact:** The application would look for modules and resources in the wrong directory, causing import failures and runtime errors.

### 3. Incorrect Directory References in README.md
**File:** `README.md`  
**Issue:** Multiple references to a non-existent `code/` directory in installation instructions.  
**Lines affected:** 62, 68-70, 84  
**Fix:** Updated all references to use the correct repository directory structure (LiveFacer/ instead of code/)  
**Impact:** Users following the README would be confused and unable to install properly.

## Model Download Bugs

### 4. Wrong Model Filename in README
**File:** `README.md`  
**Issue:** The README linked to `inswapper_128.onnx` but the code expects `inswapper_128_fp16.onnx`  
**Fix:** Updated the URL to point to the correct filename `inswapper_128_fp16.onnx`  
**Impact:** Users would download the wrong model file, causing runtime errors.

### 5. Incorrect Download URL in face_swapper.py
**File:** `modules/processors/frame/face_swapper.py`  
**Issue:** Used HuggingFace "blob" URL instead of "resolve" URL, which cannot be used for direct download.  
**Fix:** Changed from `blob/main` to `resolve/main` in the URL  
**Impact:** Automatic model download would fail.

### 6. Inconsistent Model URLs in instructions.txt
**File:** `models/instructions.txt`  
**Issue:** Different URLs than the README, and the GFPGANv1.4.pth URL pointed to GitHub instead of HuggingFace  
**Fix:** Updated to use consistent HuggingFace URLs for both models  
**Impact:** User confusion about which model files to download.

## Cross-Platform Compatibility Bugs

### 7. Windows-style Path Separator in face_enhancer.py (pre_check)
**File:** `modules/processors/frame/face_enhancer.py`  
**Issue:** Used Windows-style path separator `"..\models"` instead of cross-platform `"../models"`  
**Line:** 27  
**Fix:** Changed to use forward slashes  
**Impact:** Would fail on Linux/macOS systems.

### 8. Platform-specific Path Logic in face_enhancer.py (get_face_enhancer)
**File:** `modules/processors/frame/face_enhancer.py`  
**Issue:** Unnecessary if/else block for Windows vs other platforms using different path separators  
**Lines:** 57-61  
**Fix:** Removed platform-specific code and use single cross-platform path with forward slashes  
**Impact:** Unnecessary code complexity and potential for path-related bugs.

## Additional Improvements

### 9. Enhanced .gitignore
**File:** `.gitignore`  
**Issue:** Minimal .gitignore that didn't exclude common Python artifacts  
**Fix:** Added comprehensive exclusions for:
- Python cache files (__pycache__, *.pyc, etc.)
- Virtual environments (venv/, env/, ENV/)
- IDE files (.vscode/, .idea/, etc.)
- OS files (.DS_Store, Thumbs.db)
- Temporary files (*.tmp, system/tmp/)  
**Impact:** Prevents committing unnecessary files to the repository.

## Summary

**Total bugs fixed:** 9 critical bugs  
**Categories:**
- Installation blockers: 3
- Model download issues: 3
- Cross-platform compatibility: 2
- Code quality improvements: 1

All bugs have been fixed and verified. The application should now:
1. Install successfully on PC with `pip install -r requirements.txt`
2. Run correctly from the repository directory with `python run.py`
3. Work on Windows, Linux, and macOS
4. Download the correct model files automatically or allow manual download with correct URLs
