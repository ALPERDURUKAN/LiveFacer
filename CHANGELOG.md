# Changelog

This changelog includes only **feature updates and enhancements** since `1.0b`.

## [1.5b] - 2026-02-15

### 🚀 New Features
- 🎨 **Poisson Blend option added** (`poisson_blend`) for more natural face merge flow using `seamlessClone`.
- 🧩 **New modular face masking system** (`face_masking.py`) with:
  - 👄 mouth mask
  - 👀 eyes mask
  - 🪄 eyebrows mask
  - 📦 optional mask box visualization toggles
- ⚡ **GPU processing helper module added** (`gpu_processing.py`) with:
  - Gaussian blur
  - resize
  - color conversion
  - addWeighted
  - automatic CUDA acceleration with CPU fallback
- 📷 **Video capture abstraction added** (`video_capture.py`) with backend fallback support for live mode.
- 🛡️ **Tkinter compatibility patch added** (`tkinter_fix.py`) and auto-loaded on startup.

### ✨ Enhancements
- 🧵 **Live preview pipeline improved** with separate capture and processing threads for smoother and more stable real-time flow.
- 🧠 **Face swap post-processing upgraded** via modular region blending and improved processing path consistency.
- 🧰 **CLI expanded** with new flags:
  - `--eyes-mask`
  - `--eyebrows-mask`
  - `--poisson-blend`
- 🌍 **UI localization coverage improved** for switches, popups, status text, and file picker labels.
- 📦 **Portable startup flow strengthened** by auto-loading runtime compatibility patches in `run.py`.

