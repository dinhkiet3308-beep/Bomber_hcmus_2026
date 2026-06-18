import torch
from torch.distributions import Categorical
import numpy as np


class RLAgent:
    """Neural-network agent for PPO training and inference.
    
    Wraps a BomberlandCNN model with action masking (heuristic shield)
    to prevent illegal and suicidal moves. Supports both training mode
    (stochastic sampling) and evaluation mode (greedy argmax).
    
    Args:
        model: BomberlandCNN network (policy + value heads).
        agent_id: Player index in the game (0-3).
        device: torch device for inference.
        train_mode: If True, sample actions stochastically; if False, use argmax.
    """

    def __init__(self, model, agent_id, device, train_mode=True):
        self.model = model
        self.agent_id = agent_id
        self.device = device
        self.train_mode = train_mode
        self.step_count = 0

    def get_action(self, obs, danger_times, base_agent):
        """Queries the network and applies action masking.
        
        Args:
            obs: Game observation dict with 'map', 'players', 'bombs'.
            danger_times: Precomputed danger map from base_agent._build_danger_map.
            base_agent: TacticalRuleAgent instance (must have matching agent_id)
                        used for _tensorize_state and action validation.
        
        Returns:
            (action, log_prob, value, state_tensor, mask)
        """
        self.step_count += 1
        grid = obs["map"]
        players = obs["players"]
        
        # Safety: if agent is dead, return no-op with zero value
        if players[self.agent_id][2] == 0:
            dummy_state = np.zeros((9, 13, 13), dtype=np.float32)
            dummy_mask = np.zeros(6, dtype=np.float32)
            dummy_mask[0] = 1.0  # only STOP is valid
            return 0, 0.0, 0.0, dummy_state, dummy_mask
        
        my_x, my_y = int(players[self.agent_id][0]), int(players[self.agent_id][1])
        my_pos = (my_x, my_y)
        
        # Build the 9-channel state tensor using base_agent's tensorizer
        state_tensor = base_agent._tensorize_state(obs, danger_times)
        state_input = torch.FloatTensor(state_tensor).unsqueeze(0).to(self.device)
        
        # Compute blocked positions (bombs minus our own position)
        bomb_positions = {(int(b[0]), int(b[1])) for b in obs["bombs"]}
        blocked = set(bomb_positions)
        blocked.discard(my_pos)
        
        # Compute enemy positions for blocking consideration
        enemy_set = set()
        for i, p in enumerate(players):
            if i != self.agent_id and p[2] == 1:
                enemy_set.add((int(p[0]), int(p[1])))
        
        # Get physically valid movement actions (with enemy blocking)
        valid_actions = base_agent._valid_actions(grid, my_pos, blocked, enemy_set)
        
        # Filter to actions that don't walk into explosions at step 1
        safe_actions = [
            a for a in valid_actions
            if not base_agent._is_tile_dangerous_at(danger_times, base_agent._next_pos(my_pos, a), 1)
        ]
        if not safe_actions:
            safe_actions = valid_actions if valid_actions else [0]
        
        # Add bomb placement (action=5) to safe actions when valid:
        # - Agent has bombs remaining
        # - Not already standing on a bomb
        # - Can escape after placing (checked via danger model)
        bombs_left = int(players[self.agent_id][3])
        standing_on_bomb = my_pos in bomb_positions
        if bombs_left > 0 and not standing_on_bomb:
            # Quick escape check: at least 2 open neighbors to run to
            open_exits = base_agent._open_neighbors(grid, my_pos, blocked)
            if open_exits >= 2:
                safe_actions.append(5)
            
        # Build binary action mask (6 actions)
        mask = np.zeros(6, dtype=np.float32)
        for a in safe_actions:
            mask[a] = 1.0
        
        # Forward pass (no gradients during action selection)
        with torch.no_grad():
            policy_logits, value = self.model(state_input)
            
        # FIXED: Squeeze out the batch dimension [1, 6] -> [6]
        policy_logits = policy_logits.squeeze(0).cpu()
        value = value.cpu().item()
        
        # Apply the action mask directly onto the flat 1D vector
        masked_logits = policy_logits.clone()
        masked_logits[mask == 0.0] = -1e8
        
        dist = Categorical(logits=masked_logits)
        
        if self.train_mode:
            action = dist.sample().item()
        else:
            action = torch.argmax(masked_logits, dim=-1).item()
            
        log_prob = dist.log_prob(torch.tensor(action)).item()
        
        return action, log_prob, value, state_tensor, mask