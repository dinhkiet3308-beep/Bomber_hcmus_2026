import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from model import BomberlandCNN


# ==============================================================================
# 1. Custom PyTorch Dataset to load your harvested .npz files
# ==============================================================================

class ExpertDataset(Dataset):
    def __init__(self, data_dir):
        self.states = []
        self.actions = []
        self.outcomes = []
        self.match_won_flags = []  # Per-frame win flag for sample weighting

        file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        print(f"Found {len(file_paths)} match files. Loading data into RAM...")

        for fp in file_paths:
            data = np.load(fp)
            n_frames = len(data['actions'])
            self.states.append(data['states'])
            self.actions.append(data['actions'])
            self.outcomes.append(data['outcomes'])

            # Extract per-match win flag and broadcast to all frames in that match
            if 'match_won' in data:
                won = bool(data['match_won'][0])
            else:
                # Backward compat: infer from outcome sign
                won = float(data['outcomes'][0, 0]) > 0
            self.match_won_flags.append(np.full(n_frames, won, dtype=bool))

        self.states = np.concatenate(self.states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.outcomes = np.concatenate(self.outcomes, axis=0)
        self.match_won_flags = np.concatenate(self.match_won_flags, axis=0)

        print(f"Loaded {len(self.actions)} total frames "
              f"({self.match_won_flags.sum()} from wins, "
              f"{(~self.match_won_flags).sum()} from losses).")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.states[idx]),
                torch.LongTensor([self.actions[idx]])[0],
                torch.FloatTensor(self.outcomes[idx]),
                torch.FloatTensor([float(self.match_won_flags[idx])]))


# ==============================================================================
# 2. Compute action class weights to counteract imbalanced action distributions
# ==============================================================================

def compute_class_weights(actions, num_actions=6, smoothing=0.1):
    """Compute inverse-frequency class weights with label smoothing.
    
    Without this, the model would bias toward predicting the most common
    action (usually 'stay' or a directional move), ignoring rare but
    critical actions like 'place bomb'.
    """
    counts = np.bincount(actions, minlength=num_actions).astype(np.float64)
    # Add smoothing to prevent division by zero for unseen actions
    counts = counts + smoothing
    freq = counts / counts.sum()
    # Inverse frequency, normalized so weights sum to num_actions
    weights = (1.0 / freq)
    weights = weights / weights.sum() * num_actions
    return torch.FloatTensor(weights)


# ==============================================================================
# 3. Build a weighted sampler that oversamples winning match frames
# ==============================================================================

def build_win_weighted_sampler(dataset, win_weight=2.0):
    """Create a WeightedRandomSampler that samples winning-match frames
    more frequently than losing-match frames.
    
    This addresses the fact that an imperfect expert agent's behavior
    during losses may contain bad decisions that shouldn't be learned.
    """
    sample_weights = np.where(dataset.match_won_flags, win_weight, 1.0)
    sample_weights = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                  replacement=True)


# ==============================================================================
# 4. The Training Execution
# ==============================================================================

def train_behavioral_cloning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Load data ---
    full_dataset = ExpertDataset("expert_data")

    # --- Train / Validation split (80/20) ---
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Split: {n_train} training frames, {n_val} validation frames.")

    # --- Weighted sampler for training (oversamples winning matches) ---
    # We need to extract the indices from the train_dataset Subset
    train_indices = train_dataset.indices
    train_won_flags = full_dataset.match_won_flags[train_indices]
    train_sample_weights = np.where(train_won_flags, 2.0, 1.0)
    train_sampler = WeightedRandomSampler(
        torch.DoubleTensor(train_sample_weights),
        num_samples=len(train_sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=512, sampler=train_sampler,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False,
                             num_workers=2, pin_memory=True)

    # --- Model ---
    model = BomberlandCNN(input_channels=9, num_actions=6).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters.")

    # --- Optimizer with lower LR + Cosine schedule ---
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

    # --- Class weights for action imbalance ---
    all_actions = full_dataset.actions[train_indices]
    class_weights = compute_class_weights(all_actions, num_actions=6).to(device)
    print(f"Action class weights: {class_weights.tolist()}")

    # --- Loss functions ---
    criterion_policy = nn.CrossEntropyLoss(weight=class_weights)
    criterion_value = nn.MSELoss()

    # --- Training config ---
    epochs = 20
    value_loss_weight = 0.5  # Down-weight noisy value head signal
    max_grad_norm = 1.0       # Gradient clipping threshold

    # --- LR Scheduler ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # --- Best model tracking ---
    best_val_loss = float('inf')
    best_epoch = -1

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_states, batch_actions, batch_outcomes, _ in train_loader:
            batch_states = batch_states.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            batch_outcomes = batch_outcomes.to(device, non_blocking=True)

            optimizer.zero_grad()

            policy_logits, value_preds = model(batch_states)

            loss_p = criterion_policy(policy_logits, batch_actions)
            loss_v = criterion_value(value_preds, batch_outcomes)
            loss = loss_p + value_loss_weight * loss_v

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_policy_loss += loss_p.item() * batch_states.size(0)
            train_value_loss += loss_v.item() * batch_states.size(0)
            _, predicted = policy_logits.max(1)
            train_correct += predicted.eq(batch_actions).sum().item()
            train_total += batch_states.size(0)

        # --- Validation phase ---
        model.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_states, batch_actions, batch_outcomes, _ in val_loader:
                batch_states = batch_states.to(device, non_blocking=True)
                batch_actions = batch_actions.to(device, non_blocking=True)
                batch_outcomes = batch_outcomes.to(device, non_blocking=True)

                policy_logits, value_preds = model(batch_states)

                loss_p = criterion_policy(policy_logits, batch_actions)
                loss_v = criterion_value(value_preds, batch_outcomes)

                val_policy_loss += loss_p.item() * batch_states.size(0)
                val_value_loss += loss_v.item() * batch_states.size(0)
                _, predicted = policy_logits.max(1)
                val_correct += predicted.eq(batch_actions).sum().item()
                val_total += batch_states.size(0)

        # --- Metrics ---
        avg_train_p = train_policy_loss / max(train_total, 1)
        avg_train_v = train_value_loss / max(train_total, 1)
        avg_train_combined = avg_train_p + value_loss_weight * avg_train_v
        train_acc = 100.0 * train_correct / max(train_total, 1)

        avg_val_p = val_policy_loss / max(val_total, 1)
        avg_val_v = val_value_loss / max(val_total, 1)
        avg_val_combined = avg_val_p + value_loss_weight * avg_val_v
        val_acc = 100.0 * val_correct / max(val_total, 1)

        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.6f}")
        print(f"  Train — Policy: {avg_train_p:.4f}  Value: {avg_train_v:.4f}  "
              f"Combined: {avg_train_combined:.4f}  Acc: {train_acc:.1f}%")
        print(f"  Val   — Policy: {avg_val_p:.4f}  Value: {avg_val_v:.4f}  "
              f"Combined: {avg_val_combined:.4f}  Acc: {val_acc:.1f}%")

        # --- Save best checkpoint ---
        if avg_val_combined < best_val_loss:
            best_val_loss = avg_val_combined
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "quan_bc_best.pth")
            print(f"  ★ New best model saved! (val_loss={best_val_loss:.4f})")

        # --- Periodic checkpoint ---
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"quan_bc_epoch_{epoch+1}.pth")
            print(f"  Checkpoint saved: quan_bc_epoch_{epoch+1}.pth")

        scheduler.step()

    # --- Final save ---
    torch.save(model.state_dict(), "quan_bc_weights.pth")
    print(f"\nTraining complete.")
    print(f"Best model was at epoch {best_epoch} with val_loss={best_val_loss:.4f}")
    print(f"Files saved: quan_bc_weights.pth (final), quan_bc_best.pth (best)")


if __name__ == "__main__":
    train_behavioral_cloning()