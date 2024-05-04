from torch.utils.data import Dataset


class MemoryBuffer(Dataset):
    def __init__(self, batch_size=256):
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def push(self, state, action, prob, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def __getitem__(self, index):
        return (
            self.states[index],
            self.actions[index],
            self.probs[index],
            self.rewards[index],
            self.next_states[index],
            self.dones[index],
        )

    def __len__(self):
        return len(self.batch_size)
