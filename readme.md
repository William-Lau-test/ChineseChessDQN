# Chinese Chess DQN

This repository contains an implementation of a Deep Q-Network (DQN) agent to play Chinese Chess (Xiangqi) against a human opponent. The agent is trained using reinforcement learning to make strategic moves, with a graphical interface provided via Pygame for visualization.

## Overview

- **Language**: Python
- **Libraries**: NumPy, PyTorch, Pygame, Matplotlib
- **Purpose**: Develop an AI to play Chinese Chess, with adjustable difficulty levels (easy, medium, hard).
- **Features**:
  - Trainable DQN agent with experience replay.
  - Pygame-based GUI for rendering the chessboard.
  - Logging system to track training progress and errors.
  - Visualization of training curves (reward, loss, invalid actions).

## Prerequisites

- Python 3.8 or higher
- Required packages:
  - `numpy`
  - `torch`
  - `pygame`
  - `matplotlib`
- Install dependencies using:
  ```bash
  pip install numpy torch pygame matplotlib