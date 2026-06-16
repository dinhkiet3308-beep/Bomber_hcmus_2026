import numpy as np
import os
import glob
import re


class DataHarvester:
    def __init__(self, save_dir="expert_data"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.states = []
        self.actions = []
        self._already_saved = False  # Guard against duplicate save_match calls

        # Auto-resume: scan existing files to avoid overwriting previous data
        self.match_count = self._find_next_match_id()

    def _find_next_match_id(self):
        """Scan the save directory for existing match_*.npz files and return
        the next available ID so we never overwrite previous match data."""
        existing = glob.glob(os.path.join(self.save_dir, "match_*.npz"))
        if not existing:
            return 0
        max_id = -1
        pattern = re.compile(r"match_(\d+)\.npz$")
        for fp in existing:
            m = pattern.search(os.path.basename(fp))
            if m:
                max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    def record_step(self, state_tensor, action_taken):
        """Called inside your agent's act() method right before returning the action."""
        self.states.append(state_tensor)
        self.actions.append(action_taken)

    def save_match(self, match_won: bool, survival_ratio: float = 1.0,
                   kills: int = 0, boxes_destroyed: int = 0):
        """Called at the end of a match to save the arrays to disk.
        
        Args:
            match_won: Whether this agent won the match.
            survival_ratio: steps_survived / max_steps (0.0 to 1.0).
            kills: Number of enemies killed during the match.
            boxes_destroyed: Number of boxes destroyed during the match.
        """
        # Guard: only save once per match
        if self._already_saved:
            return
        self._already_saved = True

        if len(self.states) == 0:
            return

        # Convert to numpy arrays
        states_np = np.array(self.states, dtype=np.float32)
        actions_np = np.array(self.actions, dtype=np.int64)

        # Enriched outcome signal: combine win/loss with survival quality
        # Binary win flag still stored for simple filtering during training
        win_flag = 1.0 if match_won else -1.0
        # Composite outcome: win bonus + survival ratio + kill/box credit
        # Range roughly [-1.0, +2.5] — captures nuance beyond binary win/loss
        composite_outcome = (
            win_flag * 0.5
            + survival_ratio * 0.3
            + min(kills * 0.1, 0.3)       # cap kill bonus
            + min(boxes_destroyed * 0.01, 0.2)  # cap box bonus
        )
        outcomes_np = np.full((len(self.actions), 1), composite_outcome, dtype=np.float32)

        # Save to disk as compressed numpy files
        filename = os.path.join(self.save_dir, f"match_{self.match_count}.npz")
        np.savez_compressed(
            filename,
            states=states_np,
            actions=actions_np,
            outcomes=outcomes_np,
            # Store metadata for potential filtering during training
            match_won=np.array([match_won]),
            survival_ratio=np.array([survival_ratio]),
            kills=np.array([kills]),
            boxes_destroyed=np.array([boxes_destroyed]),
        )

        num_steps = len(actions_np)
        self.match_count += 1
        print(f"Saved {filename} with {num_steps} steps. "
              f"Won={match_won}, Survival={survival_ratio:.2f}, "
              f"Kills={kills}, Boxes={boxes_destroyed}")

        # Reset for the next match
        self.states = []
        self.actions = []
        self._already_saved = False  # Reset guard for next match