# Fruit Ninja Aim Bot 🍉⚔️

## 🎥 Demo Video

[🎥 Click here to watch the Demo Video on GitHub](https://github.com/rutikbhadane/fruit-ninja-aimbot/raw/main/20260425_16_36_02_209.mp4)

An AI-powered computer vision agent capable of playing Fruit Ninja with inhuman precision. Utilizing a highly optimized YOLOv8 object detection model combined with OpenVINO runtime for low-latency execution, and a physics-based predictive targeting system, this bot observes the screen, calculates trajectories, and executes precise mouse-drag slices to cut fast-moving fruits while avoiding bombs.

## ✨ Features

- **Real-Time Object Detection**: Uses a custom-trained YOLOv8 model (`best.pt` / `best.onnx`) to identify fruits and bombs on the screen in real-time.
- **Physics-Based Predictive Slicing**: Bypasses the limitation of static detection by tracking fruits across consecutive frames. It calculates their velocity and predicts their future position at the time of the mouse slice to ensure accurate hits on fast-moving targets.
- **Hardware Optimized Inference**: By exporting the tracking model to ONNX format and utilizing the OpenVINO runtime, the AI runs effortlessly on local Intel hardware, hitting 60+ FPS inference speeds with virtually zero lag.
- **Native Input Simulation**: Mimics genuine human play using low-level OS mouse hooks, executing swift click-and-drag movements rather than static clicks, triggering the game's actual slicing mechanics efficiently.
- **Bomb Avoidance System**: Employs spatial awareness to abort any predictive slice paths that intersect with an active bomb's trajectory.

## 🗂️ Project Structure

- `src/aim_bot.py`, `src/aim_bot_v2.py`, `claude_aim_bot.py`: Various iterations of the core gameplay loop (which includes fast screen capturing, YOLOv8 inference, trajectory physics, and mouse execution).
- `src/fruit_ninja/helpers/`: Small utilities to assist in dataset sorting, frame capturing, and annotation formatting.
- `debug_mov.py`: Script to debug mouse movement events locally without firing game inputs.
- `best.pt` / `best.onnx`: The weights of our custom-trained fruit and bomb object detection vision models.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rutikbhadane/fruit-ninja-aimbot.git
   cd fruit-ninja-aimbot
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have correct driver implementations for your PyTorch hardware whether CUDA or OpenVINO base version)*

## 🚀 Usage

1. **Start Fruit Ninja**: Ensure the game runs in windowed mode roughly taking up a recognizable portion of the screen.
2. **Launch the Bot**:
   Run the main predictive aim bot script. Ensure your Python terminal is run with sufficient OS privileges if mouse commands are being blocked (e.g., Run as Administrator on Windows).
   ```bash
   python src/aim_bot_v2.py
   ```
3. The bot will automatically analyze the application window, begin inference, and start slashing fruits autonomously!

## 🧗 Key Challenges Conquered

During the development of this project, several intricate challenges were solved:
1. **The Game Triggered No Slices from Static Clicks**: The game engine ignores standard "click" inputs. Finding a solution required building an exact simulation of holding the mouse down and interpolating drag movements over time.
2. **Target Tracking Latency**: Fast fruits consistently outpaced the screen capture + interference pipeline. This was completely solved using **velocity calculation and position projection**, where the physics-based system calculates where the fruit *will* be by the time the blade strikes.
3. **Hardware Framerate Drop**: Migrating local YOLOv8 inference to the OpenVINO framework was essential in maintaining real-time playback rates. It cut input delay by magnitudes.

---
*Disclaimer: This project was built for educational purposes involving Computer Vision, Reinforcement Logic, and Process Automation.*
