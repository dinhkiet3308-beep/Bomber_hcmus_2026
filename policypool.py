import os
import random
import torch
from glob import glob


class PolicyPoolManager:
    """Manages a pool of historical policy checkpoints for self-play training.
    
    The pool stores snapshots of the learning agent at various training stages.
    Opponents are sampled from this pool to create an evolutionary arms race.
    
    The seed checkpoint (iteration 0) is always protected from eviction to
    ensure training always has an "easy" curriculum opponent available.
    
    Args:
        pool_dir: Directory to store checkpoint files.
        max_pool_size: Maximum number of checkpoints to keep (including seed).
    """

    def __init__(self, pool_dir="pi_pool", max_pool_size=20):
        self.pool_dir = pool_dir
        self.max_pool_size = max_pool_size
        os.makedirs(self.pool_dir, exist_ok=True)

    def save_checkpoint(self, model_state_dict, current_iteration):
        """Saves a snapshot of the current policy to the opponent pool.
        
        Evicts the second-oldest checkpoint (never the seed) when the pool
        exceeds max_pool_size.
        """
        checkpoint_path = os.path.join(self.pool_dir, f"checkpoint_iter_{current_iteration}.pth")
        torch.save(model_state_dict, checkpoint_path)
        print(f"  [Pool] Added opponent: {checkpoint_path}")
        
        # Maintain rolling cap — protect the seed checkpoint
        checkpoints = sorted(glob(os.path.join(self.pool_dir, "*.pth")))
        while len(checkpoints) > self.max_pool_size:
            # Never evict the first (seed) checkpoint
            if len(checkpoints) > 1:
                evict_path = checkpoints[1]  # second-oldest
                os.remove(evict_path)
                checkpoints.pop(1)
                print(f"  [Pool] Evicted old opponent: {os.path.basename(evict_path)}")
            else:
                break

    def sample_opponent_weights(self, device="cpu"):
        """Returns a random historical checkpoint path, or None if pool is empty."""
        checkpoints = glob(os.path.join(self.pool_dir, "*.pth"))
        if not checkpoints:
            return None
        return random.choice(checkpoints)

    def sample_n_opponents(self, n, device="cpu"):
        """Sample n opponent checkpoint paths (with replacement if pool < n).
        
        Returns:
            List of n checkpoint file paths.
        """
        checkpoints = glob(os.path.join(self.pool_dir, "*.pth"))
        if not checkpoints:
            return [None] * n
        return [random.choice(checkpoints) for _ in range(n)]

    def pool_size(self):
        """Returns the current number of checkpoints in the pool."""
        return len(glob(os.path.join(self.pool_dir, "*.pth")))

    def seed_pool(self, model_state_dict):
        """Initialize the pool with the starting BC-pretrained weights."""
        seed_path = os.path.join(self.pool_dir, "checkpoint_iter_0.pth")
        if not os.path.exists(seed_path):
            torch.save(model_state_dict, seed_path)
            print(f"  [Pool] Seeded with initial weights: {seed_path}")