<h1 align="center">🎭 LiveFacer</h1>

<p align="center">
  <strong>The #1 Real-Time Face Swap App for Streamers, Creators & Filmmakers</strong><br/>
  GPU-Accelerated • 100% Offline • No Setup • Virtual Webcam Built-In
</p>

<p align="center">
  <a href="https://durq.gumroad.com/l/livefacer"><img src="https://img.shields.io/badge/🚀%20Download%20Now-Gumroad-FF90E8?style=for-the-badge" alt="Download on Gumroad"/></a>
  <img src="https://img.shields.io/badge/Version-1.5b-blue?style=for-the-badge" alt="Version 1.5b"/>
  <img src="https://img.shields.io/badge/Platform-Windows%2010%2F11-0078D4?style=for-the-badge&logo=windows" alt="Windows"/>
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?style=for-the-badge&logo=nvidia" alt="NVIDIA CUDA"/>
</p>

<p align="center">
  <img src="media/demo.gif" alt="LiveFacer Demo" width="700"/>
</p>

---

## ⚡ What Is LiveFacer?

**LiveFacer** is a portable, GPU-accelerated face-swapping application that lets you swap faces in **real time** on live webcam feeds and video files — with a single image and a single click.

> No Python. No Git. No installation. Just download, run, and create.

**Built for:**
- 🎥 Streamers on Twitch, YouTube & TikTok
- 🎬 Filmmakers & video editors
- 🎨 VFX artists & content creators
- 🧩 AI researchers & developers
- 💼 Agencies & freelancers

---

## 🚀 Download LiveFacer (No Setup Required)

> ⬇️ **[Download the pre-built .exe on Gumroad](https://durq.gumroad.com/l/livefacer)** — No manual installation. No dependencies. Just run.

[![Download LiveFacer](media/download.png)](https://durq.gumroad.com/l/livefacer)

- ✅ Portable Windows executable (.exe)
- ✅ Includes all models pre-bundled (GFPGAN, inswapper)
- ✅ GPU acceleration out of the box (NVIDIA CUDA)
- ✅ Virtual webcam built-in — works with OBS, Discord, Zoom, Twitch
- ✅ Download once. Own your version forever.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| ⚡ **One-Click Launch** | Fully portable .exe — no Python, no Git, no config needed |
| 🎭 **Real-Time Face Swap** | Up to 30 FPS with GPU acceleration |
| 👥 **Multi-Face Mapping** | Swap multiple faces simultaneously in the same scene |
| 📸 **Built-In Virtual Webcam** | Works directly in Discord, Zoom, Twitch, Teams — no OBS setup needed |
| 🎥 **OBS Integration** | Native OBS Studio support for broadcast-ready streaming |
| 🔐 **100% Offline & Private** | No cloud. No uploads. Your data never leaves your machine |
| 🪄 **GFPGAN AI Enhancement** | Built-in AI face upscaling for ultra-clean, artifact-free results |
| 🖥️ **Resizable Preview Window** | Dynamically resize the live window for better performance |

<p align="center">
  <img src="media/avgpcperformancedemo.gif" alt="Performance Demo" width="700"/>
</p>

---

## 🔥 What's New in v1.5b

### 🆕 New Features
- 🎨 **Poisson Blend** — More natural, seamless face merges using `seamlessClone` for lifelike skin blending
- 🧩 **Modular Face Masking** — Independent mouth, eyes & eyebrow masks with optional visualization toggles
- ⚡ **Enhanced GPU Pipeline** — Gaussian blur, resize, color conversion & `addWeighted` via CUDA (auto-falls back to CPU)
- 📷 **Video Capture Fallback** — Multi-backend support for more reliable live mode across systems
- 🛡 **Tkinter Compatibility Patch** — Auto-loads at startup for a smoother portable experience on every machine

### ✨ Improvements
- 🧵 **Smoother Live Preview** — Separate capture and processing threads for stable, stutter-free real-time output
- 🧠 **Better Post-Processing** — Modular region blending for cleaner, sharper swaps
- 📦 **Stronger Portable Startup** — Compatibility patches load automatically on any machine

---

## 📌 How It Works — 3 Steps

```
1. Download  →  Run the .exe (no installation)
2. Select    →  Choose your source face image & target (webcam / video)
3. Swap      →  Go live instantly on any platform
```

**Webcam Mode:**

<p align="center">
  <img src="media/demo.gif" alt="Webcam Mode Demo" width="600"/>
</p>

**Face Mapping (multi-face):**

<p align="center">
  <img src="media/face_mapping_result.gif" alt="Face Mapping Demo" width="600"/>
</p>

---

## 🔥 Subscribe for Updates. Own Your Software Forever.

This project is distributed via **[Gumroad](https://durq.gumroad.com/l/livefacer)**.

> **You are NOT renting access.** Every version you download is yours to keep permanently — even after cancelling.

Your subscription unlocks:
- ✨ Access to every new version released during your subscription
- ⚙️ Performance upgrades and new AI model improvements
- 🧩 New features added regularly (like v1.5b's Poisson Blend & modular masking)
- 💬 Priority support from the development team
- 🔒 Cancel anytime — your downloaded version stays with you permanently

**[Subscribe / Download Now →](https://durq.gumroad.com/l/livefacer)**

---

## 🖥️ System Requirements

| | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 (64-bit) | Windows 11 (64-bit) |
| **GPU** | Any (CPU mode) | NVIDIA RTX / AMD RX 8GB+ VRAM |
| **CPU** | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| **RAM** | 8 GB | 16 GB |

> ⚠️ CPU-only mode is supported but significantly slower. A modern NVIDIA GPU is strongly recommended for real-time use.

---

## 🔧 Manual Installation (Advanced Users)

> ⚠️ **The pre-built .exe is strongly recommended for most users.** Manual installation requires technical skills. Do NOT open platform/installation issues on GitHub before discussing on the Discord server.

### Basic Installation (CPU)

**1. Setup Your Platform**
- Python 3.10
- pip, git
- [ffmpeg](https://www.youtube.com/watch?v=OlNWCpFdVMA)
- [Visual Studio 2022 Runtimes (Windows)](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

**2. Clone Repository**
```bash
git clone https://github.com/ALPERDURUKAN/LiveFacer.git
```

**3. Download Models**
- [GFPGANv1.4](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth)
- [inswapper_128_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx)

Place both files in the `models/` folder.

**4. Install Dependencies**
```bash
pip install -r requirements.txt
```

**5. Run**
```bash
python run.py
```

<details>
<summary>🐛 GPU Acceleration Setup (NVIDIA / AMD / OpenVINO)</summary>

For NVIDIA CUDA:
```bash
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.16.3
```
Then run with:
```bash
python run.py --execution-provider cuda
```
</details>

---

## 💻 Command Line Arguments

```
-s, --source        Source face image
-t, --target        Target image or video
-o, --output        Output file or directory
--frame-processor   face_swapper, face_enhancer, ...
--keep-fps          Keep original FPS
--keep-audio        Keep original audio
--many-faces        Process every face in frame
--map-faces         Map source to target faces
--live-mirror       Mirror the live camera display
--live-resizable    Allow resizing the live window
--execution-provider cpu, cuda, ...
--max-memory        Max RAM in GB
-v, --version       Show version
```

> Passing `-s` / `--source` runs the program in **CLI mode**.

---

## 🛡️ Ethical Use

LiveFacer is built for **creativity, education, and entertainment**.

| ✅ Allowed | ❌ Not Allowed |
|---|---|
| Creative & artistic projects | Deceptive or non-consensual content |
| Clearly labelled AI-generated content | Impersonation or fraud |
| Educational and research use | Illegal or harmful content |
| Consented use of real faces | Any content involving minors |

A built-in filter prevents processing of inappropriate media (nudity, graphic content, etc.).

---

## 📊 Roadmap

- [x] Multi-face support
- [x] UI/UX enhancements
- [x] GPU acceleration (NVIDIA, AMD, OpenVINO)
- [x] Built-in virtual webcam
- [x] Modular face masking (v1.5b)
- [x] Poisson Blend natural merging (v1.5b)
- [ ] Web app / browser-based version
- [ ] Faster model loading
- [ ] Further real-time speed improvements

---

## 🙏 Credits

- [ffmpeg](https://ffmpeg.org/) — video processing
- [deepinsight / insightface](https://github.com/deepinsight/insightface) — face analysis models *(non-commercial research use only)*
- [havok2-htwo](https://github.com/havok2-htwo) — webcam code
- [GosuDRM](https://github.com/GosuDRM) — open roop base
- [pereiraroland26](https://github.com/pereiraroland26) — multi-face support
- [vic4key](https://github.com/vic4key), [KRSHH](https://github.com/KRSHH) — contributions
- All contributors to [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) and its dependencies

---

<p align="center">
  <strong>🌟 Start Creating. No Setup. No Cloud. No Waiting.</strong><br/><br/>
  <a href="https://durq.gumroad.com/l/livefacer">
    <img src="https://img.shields.io/badge/⬇%20Download%20LiveFacer%20Now-FF90E8?style=for-the-badge&logo=gumroad&logoColor=black" alt="Download LiveFacer"/>
  </a>
</p>
