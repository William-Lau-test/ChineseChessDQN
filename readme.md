# Chinese Chess DQN Agent

This project implements a robust Deep Q-Network (DQN) agent to play Chinese Chess (Xiangqi), trained through reinforcement learning and equipped with safety-aware action selection, dynamic difficulty adaptation, and Pygame visualization. The codebase emphasizes training stability, action legality, and structured environment design.

> **Current active branch**: `feature/modifyTrain`  
> Status: ✅ Mixed-precision training, ✅ Legal move masking, ✅ Improved environment interface

---

## 🧠 Key Features

- **Safe DQN Training Pipeline**: With epsilon-greedy strategy adapted by difficulty, constrained action selection, and gradient clipping.
- **Custom Environment**:
  - 1530-dimensional state vector combining board layers and control maps.
  - Full legality check for moves, including "king face-to-face" and self-check.
  - Reward shaping includes capturing value and check status.
- **Advanced Architecture**:
  - Dual Q-Networks (online + target) with `LayerNorm` and mixed precision (AMP).
  - Action space encoded as linear indices over all possible (from, to) positions.
- **Logging & Debugging**:
  - Logging via Python `logging` module with saved `.log` files.
  - Dimension mismatch checks and error recovery mechanisms.
- **Interactive GUI**:
  - Built with Pygame, supports piece selection, move highlights, and in-game status panels.

---

## 🔧 Installation

```bash
git clone https://github.com/your-username/ChineseChessDQN.git
cd ChineseChessDQN
pip install -r requirements.txt
```

If no `requirements.txt`, manually install:

```bash
pip install torch pygame numpy matplotlib
```

---

## 🚀 How to Train

Run the following in terminal:

```bash
python main.py
```

This launches training using the current settings:
- Batch size: 64
- Optimizer: AdamW
- Mixed Precision: Enabled (if CUDA available)
- Target update frequency: 5000 steps

All hyperparameters are defined in `dqn.py` and configurable via agent initialization.

---

## 🧪 Environment (Custom)

Defined in [`environment.py`](./environment.py), the `ChineseChessEnv` class includes:
- `get_state()`: Returns a 1530-dim vector with multiple spatial channels
- `get_legal_moves()`: Strict legal move generator considering check and blocking rules
- `step(action)`: Applies move and returns `(next_state, reward, done, info)`

Board state includes 1 layer of raw board positions, 14 binary piece layers, and 2 control layers (friendly/enemy control zones).

---

## 🏗️ DQN Agent Design

Located in [`dqn.py`](./dqn.py), the `DQNAgent` includes:

- `select_action(state, legal_moves)`: Epsilon-greedy with legal move filtering.
- `store_transition()`: Records (s, a, r, s', done) with validation.
- `update()`: Mixed-precision backward pass, loss clipping, target updates.
- `save(path) / load(path)`: Checkpointing support.

---

## 🧩 Action Representation

Moves are encoded as `(i, j, ni, nj)` tuples and mapped to a linear index:

```
index = (i * 9 + j) * 90 + (ni * 9 + nj)
```

This enables consistent indexing across state and Q-value vectors.

---

## 🎮 GUI Instructions

- On launch, the GUI will render the chessboard.
- Click to select a piece and see legal moves.
- The agent plays automatically as the opposite side.
- Captured pieces, game state (check/checkmate), and timers are displayed.

---

## 📈 Visualization (Optional)

To visualize training progress (e.g., average reward, loss):

```bash
python visualize.py
```

Ensure logs or saved metrics are available for plotting.

---

## 📁 Project Structure

```
├── main.py               # Entry point (train loop)
├── dqn.py                # DQN agent
├── environment.py        # Custom chess environment
├── visualize.py          # Optional training curve visualization
├── training.log          # Log file (auto-generated)
├── models/               # (Optional) Saved checkpoints
└── README.md
```

---

## 📌 Notes

- The environment assumes red plays at bottom, black at top.
- The current player switches after each valid move.
- Auto-resets if the board becomes invalid (e.g., king missing).

---

## 📜 License

MIT License. Feel free to fork, modify, and contribute!

---

## 🤝 Acknowledgements

- Inspired by classic DQN and AlphaZero-style agents.
- Thanks to the Chinese Chess open-source rule databases and visualization tools.
