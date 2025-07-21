import numpy as np
import logging

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        logging.info(f"ReplayBuffer initialized with capacity={capacity}")

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        idx = self.position
        self.position = (self.position + 1) % self.capacity
        logging.info(f"Pushed transition to buffer, position={self.position}, size={len(self.buffer)}")
        return idx

    def sample(self, indices):
        """Sample a batch of transitions using specified indices."""
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        logging.info(f"Sampled batch of size={len(indices)} using prioritized indices")
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)