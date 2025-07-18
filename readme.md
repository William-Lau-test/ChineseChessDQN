# DQN for Chinese Chess

This project implements a Deep Q-Network (DQN) agent to play a complete version of Chinese Chess using PyTorch. The environment includes all standard pieces and rules (capture, knight leg block, elephant eye block, check, checkmate). It features a Pygame interface for visualization and mouse-based move selection, with options to choose the computer's side (red or black) and difficulty level (easy, medium, hard).

## Project Structure

- `main.py`: Main training and testing loop, handles command-line arguments and reward visualization.
- `dqn.py`: DQN agent class, manages Q-network, target network, and updates.
- `environment.py`: Complete Chinese Chess environment with Pygame rendering and mouse interaction.
- `models.py`: Defines the enhanced Q-network architecture.
- `replay_buffer.py`: Implements the experience replay buffer.
- `utils.py`: Utility functions, including epsilon-greedy action selection.
- `README.md`: This file.

## Dependencies

- Python 3.8+
- PyTorch (>=1.9)
- NumPy (>=1.21.0)
- Matplotlib (for reward visualization)
- Pygame (for chessboard rendering and mouse interaction)

Install dependencies:
```bash
pip install torch numpy matplotlib pygame