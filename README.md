# 🎭 LiveFacer

![LiveFacer Header](https://via.placeholder.com/1000x300/1e1e1e/ffffff?text=LiveFacer+-+Real-Time+Face+Swapping)

**LiveFacer** is an advanced, high-performance, real-time face swapping application. Designed for seamless live video streaming and interactive use, it leverages state-of-the-art AI models to map and swap faces with incredible accuracy, high framerates, and minimal latency.

Whether you're a content creator, a developer, or just want to have fun on your next video call, LiveFacer delivers ultra-smooth performance tailored for modern GPUs.

---

## 🚀 Get Started Immediately (1-Click Version)

We believe in open-source, and all the code is freely available here for anyone to compile. However, setting up AI environments, installing CUDA toolkits, configuring Python paths, and compiling from source can be a massive headache that takes hours.

**Want to skip the setup and jump straight into the action?**

We've prepared a **Pre-compiled, 1-Click Portable Version** that works out of the box! No Python installation, no environment headaches, no compiling errors. Just extract and double-click. 

By purchasing the pre-compiled version, you also directly support the continuous development of this open-source project. ❤️

👉 **[Get the 1-Click Portable Version on Gumroad](https://durq.gumroad.com/l/livefacer)**

---

## ✨ Key Features

- **Real-Time Performance:** Engineered for live camera feeds and high-FPS rendering.
- **Hardware Acceleration:** Native support for NVIDIA GPUs (CUDA), AMD (DirectML), and CPU fallbacks.
- **Advanced Masking:** Dynamic generation for Face, Eyes, Eyebrows, and Mouth masks using Convex Hull rendering to handle extreme facial expressions (like sticking your tongue out) without glitching.
- **Live UI Control:** Adjust swapping parameters, opacity, and mask boundaries on the fly.
- **Multi-Face Mapping:** Swap multiple faces simultaneously or target specific faces in a crowded frame.
- **Color Correction:** Built-in Poisson blending and color fixers to eliminate "blueish" or mismatched skin tones.

---

## 🛠️ Build it Yourself (Open Source)

If you are a developer and prefer to build the project from scratch, LiveFacer is 100% open-source.

### Prerequisites
- Python 3.10+
- NVIDIA CUDA Toolkit & cuDNN (if using NVIDIA GPU)
- Visual Studio Build Tools (for C++ dependencies)
- FFmpeg

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ALPERDURUKAN/livefacer.git
   cd livefacer/code
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python run.py --execution-provider cuda
   ```

*(Note: Depending on your hardware, you may need to configure specific ONNX Runtime providers manually.)*

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/ALPERDURUKAN/livefacer/issues). 

## 📜 License

This project is open-source and available under the standard MIT License. The core technology relies on several open-source AI models; please ensure you respect their respective licenses and ethical usage guidelines.

> **Disclaimer:** LiveFacer is created for ethical, educational, and entertainment purposes only. Do not use this software to deceive others, create non-consensual content, or violate any laws.
