# Chinese Chess DQN Agent

This project implements a Deep Q-Network (DQN) agent to play Chinese Chess (Xiangqi), using reinforcement learning in a custom environment with strict rule enforcement. It supports mixed-precision training, structured debugging, and detailed unit testing.

> **Active Branch**: `feature/modifyTrain`

---

## 🧠 Key Features

- **Safe DQN Training Pipeline**: Includes epsilon-greedy with difficulty levels, constrained legal actions, and stable loss handling.
- **Custom Game Environment**:
  - Full Chinese Chess rules with move legality, check detection, and self-check prevention.
  - 1530-dimensional state representation combining board layers and control zones.
- **Agent Architecture**:
  - DQN with dual networks (online/target), `LayerNorm`, AMP support.
  - Legal move masking using move-index encoding.
- **Debug & Recovery**:
  - Logging system with strict dimension checks and error rollbacks.
  - Auto-handling of invalid board states or illegal moves.
- **Unit Testing Support**:
  - Organized under `/test` for model, environment, and dimension tests.

---

## 📁 Project Structure

```
├── dqn.py                # DQN agent implementation
├── environment.py        # Chinese Chess environment
├── train.py              # Main training loop
├── evaluate.py           # Model evaluation script
├── battle.py             # Agent vs agent/human interface
├── cmd.txt               # Example commands
├── main.py               # (Entry wrapper, customizable)
├── test/                 # Pytest-compatible unit tests
│   ├── fix_dimension_test.py
│   ├── test_dqn.py
│   ├── test_environment.py
│   ├── test_dimensions.py
│   ├── test_train.py
│   └── test_verify_state.py
├── utils/                # Helper modules
│   ├── memory.py         # Experience replay memory
│   ├── scheduler.py      # Learning rate or epsilon scheduler
│   └── __init__.py
└── README.md
```

---

## 🔧 Installation

```bash
pip install torch pygame numpy
```

---

## 🚀 Training

```bash
python train.py
```

Training parameters can be configured in `train.py` or through modifying the `DQNAgent` class in `dqn.py`.

---

## 🤖 Evaluation & Battle

```bash
python evaluate.py       # Runs evaluation
python battle.py         # Launches GUI game (human vs AI)
```

---

## 🧪 Testing

Unit tests are located in the `test/` directory. To run all tests:

```bash
pytest test/
```

---

## 📌 Notes

- The environment enforces all Chinese Chess rules including king facing rule and 80-move draw rule.
- Action space is encoded into a linear index for (i, j, ni, nj) mapping.
- AMP is enabled if CUDA is available.

---

## 📜 License

MIT License

---

## 🤝 Acknowledgements

Thanks to open-source contributions in reinforcement learning and Chinese Chess rule references.
