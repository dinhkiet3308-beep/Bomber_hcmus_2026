import torch
import numpy as np


class RolloutBuffer:
    """Collects trajectory data and computes GAE advantages for PPO.
    
    Args:
        gamma: Discount factor for future rewards.
        lam: GAE lambda for bias-variance tradeoff in advantage estimation.
        reward_scale: Divisor applied to raw rewards before GAE computation.
                      Use 100.0 when rewards are in [-100, +100] to normalize
                      to [-1, +1] range matching a tanh value head.
    """

    def __init__(self, gamma=0.99, lam=0.95, reward_scale=100.0):
        self.gamma = gamma
        self.lam = lam
        self.reward_scale = reward_scale
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()

    def push(self, state, action, log_prob, reward, value, done, mask):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(mask)

    def compute_gae(self, next_value):
        """Computes Generalized Advantage Estimation and returns clean tensors.
        
        Rewards are scaled by 1/reward_scale before GAE computation so that
        the resulting returns live in a range compatible with the value head.
        
        Args:
            next_value: V(s_terminal) bootstrap value for the final state.
            
        Returns:
            Tuple of (states, actions, log_probs, returns, advantages, action_masks)
            as PyTorch tensors.
        """
        # Scale rewards to match value head range
        rewards = np.array(self.rewards, dtype=np.float32) / self.reward_scale
        values = np.array(self.values + [next_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        
        trajectory_len = len(rewards)
        advantages = np.zeros(trajectory_len, dtype=np.float32)
        gae = 0.0
        
        # Backward pass from the end of the episode
        for t in reversed(range(trajectory_len)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            
        returns = advantages + values[:-1]
        
        # Normalize advantages for stable gradient updates
        if trajectory_len > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.log_probs)),
            torch.FloatTensor(returns),
            torch.FloatTensor(advantages),
            torch.FloatTensor(np.array(self.action_masks))
        )