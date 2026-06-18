import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class PPOUpdater:
    """Proximal Policy Optimization with clipped surrogate objective.
    
    Supports:
    - Action masking (heuristic shield for illegal/suicidal moves)
    - Entropy bonus to prevent premature convergence
    - Gradient clipping for training stability
    """

    def __init__(self, ppo_net, optimizer, clip_eps=0.2, c_val=0.5, c_ent=0.01,
                 max_grad_norm=0.5):
        self.policy_net = ppo_net
        self.optimizer = optimizer
        self.clip_eps = clip_eps      # Policy clipping range (epsilon)
        self.c_val = c_val            # Value loss coefficient
        self.c_ent = c_ent            # Entropy bonus coefficient
        self.max_grad_norm = max_grad_norm

    def train_step(self, states, actions, old_log_probs, returns, advantages, action_masks):
        """
        Executes one PPO optimization step on a mini-batch.
        All inputs are PyTorch Tensors.
        
        Returns:
            (actor_loss, critic_loss, total_loss, entropy) — all floats for logging
        """
        self.policy_net.train()
        
        # 1. Forward pass through the dual heads
        policy_logits, value_preds = self.policy_net(states)
        
        # 2. Apply the heuristic shield: set illegal actions to -inf probability
        masked_logits = policy_logits.clone()
        masked_logits[action_masks == 0] = -1e8
        
        # 3. Compute probabilities via categorical log-softmax
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = F.softmax(masked_logits, dim=-1)
        
        # Gather the log-probabilities of the actions that were actually taken
        gathered_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 4. Probability ratio: r_t(theta) = pi_new(a|s) / pi_old(a|s)
        ratios = torch.exp(gathered_log_probs - old_log_probs)
        
        # 5. Clipped actor (policy) loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 6. Critic (value) loss — squeeze value_preds to match returns shape
        critic_loss = F.mse_loss(value_preds.squeeze(-1), returns)
        
        # 7. Entropy bonus — prevents premature convergence
        # Only computed across valid (masked) actions
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # 8. Total joint objective
        total_loss = actor_loss + (self.c_val * critic_loss) - (self.c_ent * entropy)
        
        # 9. Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), total_loss.item(), entropy.item()