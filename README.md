# LiveFacer: Real-Time AI Face Swap for Webcam, OBS & Video

**The #1 Real-Time Face Swap App for Creators.** LiveFacer is a GPU-accelerated, portable **Windows 10/11 face swap application** for real-time webcam previews, OBS virtual camera workflows, photos and videos. Choose a face, refine the result, and create without manually installing Python or AI dependencies in the prebuilt edition.

**30 FPS, Near-Native Performance · 100% Offline & Private · One-click Portable .exe**

[**Get the Portable Windows Edition on Gumroad →**](https://durq.gumroad.com/l/livefacer?utm_source=github&utm_medium=readme&utm_campaign=v18) · [**Explore LiveFacer 1.8 features →**](https://livefacer.com/?utm_source=github&utm_medium=readme&utm_campaign=v18)

![Real LiveFacer desktop interface with face selection, webcam preview and masks](https://livefacer.com/images/image.png)

> **Free source or ready-to-run?** The source-based workflow is available in this repository. The **paid self-extracting edition** bundles Python, the CUDA runtime, FFmpeg and included AI models. Double-click to extract, then start with a launcher that matches your hardware. Subscriptions provide new builds and priority support; your last downloaded version remains usable after cancellation.

## What's new in LiveFacer 1.8

*The Gumroad product currently lists version 1.8 as an alpha-testing release. The source repository and packaged builds may differ.*

| 1.8 improvement | What it does |
| --- | --- |
| **In-memory video rendering** | Streams frames straight through the AI pipeline and into the encoder without writing thousands of temporary image files |
| **GPU video export** | Automatic NVIDIA **NVENC**, AMD **AMF** or Intel **QSV**, with high-quality software encoding fallback |
| **3 face enhancers** | Bundled offline **GFPGAN**; optional GPEN-512 and GPEN-256 downloaded on first use |
| **Enhancer Intensity** | 0–100% slider blends the original detail and the enhanced result |
| **Feathered face blending** | Reduces visible hard edges, halos and seams in face paste-back |
| **Improved Poisson Blend** | Geometry-aligned seamless blending for steadier skin-tone matching in motion |
| **Detection sizes** | Choose **160, 320 or 640** for FPS/accuracy balance |
| **12-point Mouth Mask** | Adjustable lip contour, 0–100 size slider and live adjustment preview |
| **Random AI Face** | One-click non-existent realistic face for creative testing and privacy |
| **Ease of use** | Hover tooltips, multi-backend/MJPG webcam detection, Unicode file paths and self-extracting .exe |

## What LiveFacer can do

- **Real-time webcam face swap:** Live preview, FPS overlay, custom resolution and mirror mode.
- **OBS and virtual camera:** Output through a compatible driver, such as **OBS Virtual Camera**, for streaming and supported video-call applications.
- **Image and video face swap:** Keep audio and FPS, control output quality, and render with GPU encoding when available.
- **Multiple faces:** Many Faces mode and Map Faces, including distinct source faces for different people.
- **More natural expressions:** Mouth Mask, Eyes Mask and Eyebrows Mask preserve original lip movement and gaze.
- **Creative controls:** Color correction (Fix Blueish Cam), sharpness, similarity/opacity, face enhancers and face blending.
- **Hardware options:** One-click launchers for NVIDIA CUDA, AMD DirectML, Intel OpenVINO and CPU.

## Download the ready-to-run Windows edition

### [Get LiveFacer on Gumroad →](https://durq.gumroad.com/l/livefacer?utm_source=github&utm_medium=readme&utm_campaign=v18&utm_content=portable)

1. Download the approximately **4 GB** self-extracting executable.
2. Double-click to extract. No WinRAR or 7-Zip required. Allow approximately **10 GB** of free disk space.
3. Select your NVIDIA, AMD, Intel or CPU launcher.
4. Choose a face and open a webcam, image or video workflow.

The paid build is portable and can run from a folder or external drive. A virtual camera driver is required for third-party app virtual-camera use.

**Pricing:** $10 monthly · $25 quarterly · $40 every 6 months · $60 annually. Every plan contains the complete edition. You keep the version you downloaded after cancellation. Check the [Gumroad page](https://durq.gumroad.com/l/livefacer) for current terms, billing and refund information.

## System requirements

| | |
| --- | --- |
| OS | Windows **10/11, 64-bit** |
| Download size | Approximately 4 GB |
| Storage after extraction | Approximately 10 GB free |
| Recommended live hardware | NVIDIA RTX GPU |
| Alternative execution | AMD DirectML, Intel OpenVINO and CPU-only; performance may be lower |
| Internet | Required for initial optional GPEN downloads and the Random AI Face button |

**30 FPS, Near-Native Performance.** Actual FPS depends on your GPU, resolution, face detection size and chosen settings.

## Build from source

This is the developer-oriented path. It requires a compatible Python/GPU environment and additional dependencies or model files.

\`\`\`bash
git clone https://github.com/ALPERDURUKAN/LiveFacer.git
cd LiveFacer/code
pip install -r requirements.txt
python run.py --execution-provider cuda
\`\`\`

Follow [code/README.md](code/README.md) and [model instructions](code/models/instructions.txt) for environment and model setup. The example above is for compatible NVIDIA environments, not a universal installer.

## Frequently asked questions

**Can it run without an NVIDIA GPU?** Yes. AMD, Intel and CPU-only launchers are included. NVIDIA RTX is recommended for real-time use.

**Can it run offline?** Core processing is local, and bundled GFPGAN works offline. Optional GPEN downloads and Random AI Face generation need internet.

**Can I use it with OBS, Discord, Zoom or streaming software?** Use a compatible virtual camera driver such as OBS Virtual Camera; support depends on the destination application's camera options.

**Is LiveFacer related to Deep-Live-Cam?** Yes. LiveFacer builds on [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam), with a focus on the portable Windows workflow and added creative controls.

**Where can I ask for help?** See [GitHub Issues](https://github.com/ALPERDURUKAN/LiveFacer/issues) for technical reports and the product page for purchase-related support.

## Responsible use and licenses

LiveFacer is for lawful, consensual creative work. Obtain permission before using a real person's likeness, disclose AI-edited content when shared, and never use it for deceptive or non-consensual impersonation.

The project builds on open-source software and third-party AI models, which can have separate terms and license restrictions. Review applicable model, source-code and commercial usage rights before redistribution or commercial use.

---

[**LiveFacer.com**](https://livefacer.com/) · [**Get LiveFacer**](https://durq.gumroad.com/l/livefacer?utm_source=github&utm_medium=readme&utm_campaign=v18) · [**Webcam face swap guide**](https://livefacer.com/real-time-webcam-face-swap/) · [**Video face swap guide**](https://livefacer.com/ai-video-face-swap/) · [**Deep-Live-Cam alternative**](https://livefacer.com/deep-live-cam-alternative/)
