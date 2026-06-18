"""
bomberland_env.py — Self-Play Environment Wrapper

Wraps engine.BomberEnv (4-player Bomberland) for RL self-play training.
Supports 1 learning agent (agent_id=0) vs 3 opponents (agent_ids 1,2,3)
drawn from the policy pool.

Usage:
    runner = BomberlandLocalMatchRunner()
    obs = runner.reset()
    while not done:
        next_obs, done = runner.step(actions)  # actions: list of 4 ints
"""

import sys
import os
import numpy as np

# Add project root so we can import engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import BomberEnv


class BomberlandLocalMatchRunner:
    """Thin wrapper around BomberEnv for self-play training loops."""

    def __init__(self, max_steps=500):
        self.max_steps = max_steps
        self._episode_count = 0
        self.env = None

    def reset(self):
        """Reset the environment with a new random seed.
        
        Returns:
            obs: dict with keys 'map', 'players', 'bombs'
        """
        seed = self._episode_count * 7919 + 42  # deterministic but varied
        self._episode_count += 1
        self.env = BomberEnv(
            width=13, height=13,
            max_steps=self.max_steps,
            seed=seed,
        )
        obs = self.env.reset(seed=seed)
        return obs

    def step(self, actions):
        """Advance the game by one tick.
        
        Args:
            actions: list/tuple of 4 ints, one per player.
                     0=STOP, 1=UP(LEFT), 2=DOWN(RIGHT), 3=LEFT(UP), 4=RIGHT(DOWN), 5=BOMB
        
        Returns:
            (next_obs, done): next observation dict and whether the game ended.
        """
        obs, terminated, truncated = self.env.step(actions)
        done = terminated or truncated
        return obs, done

    def get_stats(self):
        """Return per-player stats from the engine for logging."""
        if self.env is None:
            return []
        return [
            {
                "alive": p.alive,
                "kills": p.stats["kills"],
                "boxes": p.stats["boxes"],
                "items": p.stats["items"],
                "bombs_placed": p.stats["bombs"],
            }
            for p in self.env.players
        ]

    @property
    def current_step(self):
        """Current game tick."""
        if self.env is None:
            return 0
        return self.env.current_step
