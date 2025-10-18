# Progress Report — Rubik’s Cube AI Solver

## Overview

This repository is aimed at developing an **AI-driven Rubik’s Cube solving system** that combines reinforcement learning, cube state representation, and (eventually) computer vision.  

The current focus is on the **learning component** — specifically, training a Deep Q-Network (DQN) agent to learn how to solve a **2×2 Rubik’s Cube** from scrambled states simulated in code.  
Future goals include:
- Extending to a 3×3 solver.
- Integrating a vision system to parse cube states from camera input (`opencv.py`).
- Potentially connecting to a physical robot arm to perform moves.

---

## Current Progress

### ✅ 1. Rubiks2x2 Environment (in `rub.py`)

- Built a **custom OpenAI Gym–style environment** (`Rubiks2x2Env`) using the `magiccube` library as the cube simulator.  
- Includes:
  - Full cube representation via **facelet strings** (e.g., `"YYYYRRRRGGGGOOOOBBBBWWWW"`).
  - **One-hot encoding** of facelets for neural network input (shape `24 × 6` = 144-dim vector).
  - **Scramble generation** with logic to avoid immediate inverse moves.
  - **Step function** that applies moves, checks for solved state, and returns rewards.
  - **Step penalty** and **early termination** to encourage efficient solving.

This forms a solid foundation for reinforcement learning.

---

### ✅ 2. Deep Q-Network Agent

- Implemented a **DQN agent** (`DQNAgent`) with:
  - A 3-layer fully connected **Q-network** (`QNet`) and a **target network**.
  - **Soft updates** (controlled by `tau`) for target network stability.
  - **Double DQN** target computation for more stable training.
  - **Gradient clipping** to prevent exploding gradients.
  - **Epsilon-greedy** exploration during training.

- The agent learns via transitions stored in a **Replay Buffer**, which:
  - Uses a named tuple `Transition` for structured samples.
  - Randomly samples batches for training.
  - Supports up to 300k experiences for replay.

---

### ✅ 3. Training Loop

- Wrote a robust `train()` function that:
  - Supports **curriculum learning** — gradually increasing scramble length per epoch.
  - Implements **linear epsilon decay** across episodes.
  - Periodically evaluates agent performance on validation scrambles.
  - Saves model checkpoints after each epoch.
  - Reports intermediate success rates and average solve steps.

- Training parameters include:
  - `gamma = 0.98` (discount factor)
  - `lr = 1e-3` (learning rate)
  - `batch_size` tunable (default 64–256)
  - `max_steps = 60`
  - Adjustable `episodes_per_epoch` and `scramble_schedule`.

This structure supports scalable training and fine-tuning.

---

### ✅ 4. Evaluation and Inference

- Added `evaluate()` to test the agent’s success rate over multiple episodes.
- Implemented `solve_with_agent()`:
  - Loads a trained model.
  - Runs **greedy inference** (no exploration).
  - Returns a **sequence of moves** predicted to solve a given 2×2 cube state.
  - Optionally avoids immediate inverses at inference for smoother sequences.

---

### ⚙️ 5. `opencv.py` Placeholder

- Exists as a placeholder file — intended for **future vision module**.
- Likely planned to handle:
  - Capturing cube face images via webcam.
  - Detecting sticker colors and mapping them to cube notation.
  - Producing a valid facelet string input for the DQN solver.

At the moment, it’s empty, meaning the system still relies on simulated cube states rather than real camera input.

---

## Current State Summary

| Component | Status | Description |
|------------|---------|-------------|
| 2×2 cube environment | ✅ Working | Fully functional simulator built around `magiccube`. |
| RL agent (DQN) | ✅ Working | Functional and trainable; implements stable DQN variant. |
| Training framework | ✅ Working | Supports multi-epoch training, logging, checkpointing. |
| Evaluation + inference | ✅ Working | Allows testing trained agent’s solving ability. |
| Vision module (`opencv.py`) | ⏳ Not yet implemented | Placeholder for cube color detection from images. |
| 3×3 extension | ❌ Not started | No 3×3 cube logic yet. |
| Physical robot integration | ❌ Not started | Future phase. |

---

## Next Steps / Roadmap

1. **Add vision pipeline (`opencv.py`):**
   - Capture each cube face using OpenCV.
   - Segment and identify sticker colors using k-means or HSV thresholding.
   - Build a facelet string (`"YRGOBW"` mapping).

2. **Refine reward shaping:**
   - Consider adding intermediate rewards based on “number of correct faces” or “Hamming distance from solved”.
   - Helps the agent learn more efficiently.

3. **Track and visualize metrics:**
   - Add TensorBoard or matplotlib logging for loss curves and success rates.

4. **Extend to 3×3 cubes:**
   - Redefine facelet representation for 54 stickers.
   - Update environment, moves, and reward logic accordingly.

5. **Experimentation and tuning:**
   - Test different architectures (e.g., convolutional Q-nets).
   - Try prioritized replay or dueling DQN for stability.

6. **Robotic integration (long-term):**
   - Once the DQN agent consistently solves cubes in simulation, connect to a robotic arm (e.g., via ROS or Arduino interface).
   - Convert move sequences to real actuator commands.

---

## Challenges & Learnings

- **Sparse rewards** make cube solving hard — the agent only gets a reward upon full solve.  
- Curriculum learning (increasing scramble length gradually) is essential for convergence.  
- Avoiding immediate inverse moves helps exploration efficiency.  
- Efficient one-hot encoding and state representation are crucial for speed.  
- Future integration with real cubes will require robust color detection and calibration.

---

## Current Deliverables

| File | Description |
|------|--------------|
| `rub.py` | Main DQN training and inference logic for 2×2 Rubik’s Cube. |
| `opencv.py` | Placeholder for future vision / color detection module. |
| `requirements.txt` | Python dependencies (likely includes `magiccube`, `torch`, `opencv-python`). |
| `README.md` | Basic project overview; will need expansion. |
| `dqn_2x2.pt` | Model checkpoint saved after training (produced by `train()`). |

---
