"""
train_rl.py — PPO Self-Play Training Loop for Bomberland (v2 — FIXED)

Major improvements over v1:
  1. Multi-episode rollouts: Accumulates 8 episodes before each PPO update
     (v1 used only 1 episode per update → too few samples for stable gradients)
  2. Mixed opponents: 50% heuristic agents + 50% pool agents
     (v1 used only pool clones → echo chamber of passive play)
  3. Rebalanced reward scale (200.0) to match new reward range [-50, +200]
  4. Better hyperparameters: higher entropy, more epochs, larger batch

Usage:
    python train_rl.py
"""

import os
import sys
import time
import torch
import numpy as np

# Ensure imports work from the agent/ directory
sys.path.insert(0, os.path.dirname(__file__))

from model import BomberlandCNN
from buffer import RolloutBuffer
from PPO import PPOUpdater
from rl_agent import RLAgent
from policypool import PolicyPoolManager
from agent_quan_14_6 import TacticalRuleAgent
from bomberland_env import BomberlandLocalMatchRunner


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
TOTAL_ITERATIONS   = 350       # Each iteration = EPISODES_PER_ITER games
EPISODES_PER_ITER  = 8         # Accumulate this many episodes before PPO update
UPDATES_PER_ITER   = 6         # PPO epochs per rollout batch
BATCH_SIZE         = 128       # Mini-batch size (bigger = more stable)
LEARNING_RATE      = 1e-5      # Slightly higher LR for faster learning
POOL_SAVE_INTERVAL = 10        # Save to pool every N iterations
BEST_SAVE_INTERVAL = 10        # Evaluate & save best model every N iterations
GAMMA              = 0.99
GAE_LAMBDA         = 0.95
REWARD_SCALE       = 200.0     # Match new reward range [-50, +200]
CLIP_EPS           = 0.1      # Slightly wider clip for more policy change
ENTROPY_COEFF      = 0.01      # DOUBLED from 0.01 — prevents collapse
VALUE_COEFF        = 0.5
MAX_GRAD_NORM      = 0.5
POOL_DIR           = "pi_pool"
HEURISTIC_MIX      = 0.3       # 50% of opponents are heuristic agents
WEIGHTS_PATH       = "quan_rl_best.pth"  # Start from best RL or BC weights


class HeuristicOpponent:
    """Wraps TacticalRuleAgent to match the interface expected by the training loop."""
    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.rule_agent = TacticalRuleAgent(agent_id=agent_id)
        self.is_heuristic = True
    
    def get_action(self, obs, danger_times, base_agent):
        """Returns (action, 0.0, 0.0, dummy_state, dummy_mask) to match RLAgent interface."""
        action = self.rule_agent._decide_action(obs)
        return action, 0.0, 0.0, None, None


def execute_self_play_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  Bomberland PPO Self-Play Training v2 (FIXED)          ║")
    print(f"║  Device: {str(device):<47s} ║")
    print(f"║  Mode: 1 learner vs 3 mixed opponents (4-player)      ║")
    print(f"║  Episodes per update: {EPISODES_PER_ITER:<34d} ║")
    print(f"║  Heuristic mix: {HEURISTIC_MIX:<38.0%} ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    
    # ─────────────────────────────────────────────────────────────────
    # 1. Initialize networks
    # ─────────────────────────────────────────────────────────────────
    learning_net = BomberlandCNN(input_channels=9, num_actions=6).to(device)
    
    if os.path.exists(WEIGHTS_PATH):
        learning_net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print(f"  ✓ Loaded foundation weights from {WEIGHTS_PATH}")
    elif os.path.exists("quan_bc_best.pth"):
        learning_net.load_state_dict(torch.load("quan_bc_best.pth", map_location=device))
        print(f"  ✓ Loaded BC foundation weights from quan_bc_best.pth")
    else:
        print(f"  ⚠ Warning: No pretrained weights found — training from scratch!")
    
    # Opponent networks (only needed for pool-based opponents)
    opponent_nets = []
    for i in range(3):
        opp_net = BomberlandCNN(input_channels=9, num_actions=6).to(device)
        opp_net.eval()
        opponent_nets.append(opp_net)
    
    # Optimizer & PPO updater
    optimizer = torch.optim.Adam(learning_net.parameters(), lr=LEARNING_RATE)
    updater = PPOUpdater(
        learning_net, optimizer,
        clip_eps=CLIP_EPS,
        c_val=VALUE_COEFF,
        c_ent=ENTROPY_COEFF,
        max_grad_norm=MAX_GRAD_NORM,
    )
    
    # ─────────────────────────────────────────────────────────────────
    # 2. Initialize policy pool
    # ─────────────────────────────────────────────────────────────────
    pool = PolicyPoolManager(pool_dir=POOL_DIR, max_pool_size=20)
    pool.seed_pool(learning_net.state_dict())
    
    # Base rule agents for each player ID (needed for perspective-correct
    # state tensorization — channels 3/4 are relative to agent_id)
    base_rules = [TacticalRuleAgent(agent_id=i) for i in range(4)]
    
    # ─────────────────────────────────────────────────────────────────
    # 3. Training metrics
    # ─────────────────────────────────────────────────────────────────
    metrics = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_reward": 0.0,
        "total_steps": 0,
        "total_kills": 0,
        "best_win_rate": 0.0,
        "window_wins": [],  # rolling window for win rate
    }
    WINDOW_SIZE = 50
    
    # Environment
    runner = BomberlandLocalMatchRunner(max_steps=500)
    
    total_games = TOTAL_ITERATIONS * EPISODES_PER_ITER
    print(f"\n  Starting training: {TOTAL_ITERATIONS} iterations × {EPISODES_PER_ITER} episodes = {total_games} total games\n")
    
    # ─────────────────────────────────────────────────────────────────
    # 4. Training loop
    # ─────────────────────────────────────────────────────────────────
    game_count = 0
    
    for iteration in range(1, TOTAL_ITERATIONS + 1):
        iter_start = time.time()
        
        # Shared buffer across all episodes in this iteration
        buffer = RolloutBuffer(gamma=GAMMA, lam=GAE_LAMBDA, reward_scale=REWARD_SCALE)
        
        iter_wins = 0
        iter_losses = 0
        iter_draws = 0
        iter_reward = 0.0
        iter_kills = 0
        iter_steps = 0
        
        for episode in range(EPISODES_PER_ITER):
            game_count += 1
            
            # ── Decide opponent composition for this episode ──
            # Each of the 3 opponent slots independently picks heuristic or pool
            agents_opp = []
            opp_labels = []
            
            for opp_idx in range(3):
                opp_id = opp_idx + 1
                use_heuristic = (np.random.random() < HEURISTIC_MIX)
                
                if use_heuristic:
                    agents_opp.append(HeuristicOpponent(agent_id=opp_id))
                    opp_labels.append("H")
                else:
                    # Load a random pool checkpoint
                    opp_paths = pool.sample_n_opponents(1, device=device)
                    opp_path = opp_paths[0]
                    if opp_path is not None:
                        opponent_nets[opp_idx].load_state_dict(
                            torch.load(opp_path, map_location=device)
                        )
                        opp_labels.append("P")
                    else:
                        opponent_nets[opp_idx].load_state_dict(learning_net.state_dict())
                        opp_labels.append("S")
                    agents_opp.append(
                        RLAgent(opponent_nets[opp_idx], agent_id=opp_id, device=device, train_mode=True)
                    )
            
            # ── Create learner agent ──
            agent_learner = RLAgent(learning_net, agent_id=0, device=device, train_mode=True)
            
            # ── Run one episode ──
            obs = runner.reset()
            episode_reward = 0.0
            episode_buffer = RolloutBuffer(gamma=GAMMA, lam=GAE_LAMBDA, reward_scale=REWARD_SCALE)
            
            done = False
            while not done:
                danger_times = base_rules[0]._build_danger_map(
                    obs["map"], obs["bombs"], obs["players"]
                )
                
                # Learner picks action
                act_me, log_p, val, state_tensor, mask = agent_learner.get_action(
                    obs, danger_times, base_rules[0]
                )
                
                # Opponents pick actions
                actions = [act_me]
                for i, agent_opp in enumerate(agents_opp):
                    opp_id = i + 1
                    if obs["players"][opp_id][2] == 1:  # alive
                        if hasattr(agent_opp, 'is_heuristic') and agent_opp.is_heuristic:
                            act_opp, _, _, _, _ = agent_opp.get_action(obs, danger_times, base_rules[opp_id])
                        else:
                            act_opp, _, _, _, _ = agent_opp.get_action(
                                obs, danger_times, base_rules[opp_id]
                            )
                        actions.append(act_opp)
                    else:
                        actions.append(0)  # dead → no-op
                
                # Step the environment
                next_obs, env_done = runner.step(actions)
                
                # Compute reward for the learner
                step_reward = base_rules[0].calculate_step_reward(obs, next_obs, act_me)
                episode_reward += step_reward
                
                # Check if the LEARNER specifically died this step
                learner_died = (obs["players"][0][2] == 1 and next_obs["players"][0][2] == 0)
                buffer_done = env_done or learner_died
                
                # Store transition
                episode_buffer.push(state_tensor, act_me, log_p, step_reward, val, buffer_done, mask)
                
                obs = next_obs
                
                if learner_died:
                    break
                
                done = env_done
            
            # ── Compute bootstrap value ──
            danger_times_final = base_rules[0]._build_danger_map(
                obs["map"], obs["bombs"], obs["players"]
            )
            _, _, final_val, _, _ = agent_learner.get_action(
                obs, danger_times_final, base_rules[0]
            )
            
            # ── Compute GAE for this episode and merge into main buffer ──
            ep_states, ep_actions, ep_log_probs, ep_returns, ep_advantages, ep_masks = \
                episode_buffer.compute_gae(final_val)
            
            # Merge into the iteration buffer
            for idx in range(ep_states.size(0)):
                buffer.states.append(ep_states[idx].numpy())
                buffer.actions.append(ep_actions[idx].item())
                buffer.log_probs.append(ep_log_probs[idx].item())
                buffer.rewards.append(ep_returns[idx].item())    # store returns, not raw rewards
                buffer.values.append(ep_advantages[idx].item())  # store advantages
                buffer.dones.append(0)  # placeholder
                buffer.action_masks.append(ep_masks[idx].numpy())
            
            # ── Track outcome ──
            stats = runner.get_stats()
            learner_alive = stats[0]["alive"]
            enemies_alive = sum(1 for s in stats[1:] if s["alive"])
            learner_kills = stats[0]["kills"]
            
            if learner_alive and enemies_alive == 0:
                iter_wins += 1
                metrics["window_wins"].append(1)
            elif not learner_alive:
                iter_losses += 1
                metrics["window_wins"].append(0)
            else:
                iter_draws += 1
                metrics["window_wins"].append(0)
            
            iter_reward += episode_reward
            iter_kills += learner_kills
            iter_steps += runner.current_step
        
        # ── PPO update on the accumulated multi-episode buffer ──
        b_states = torch.FloatTensor(np.array(buffer.states)).to(device)
        b_actions = torch.LongTensor(np.array(buffer.actions)).to(device)
        
        # FIXED: Correctly map the variables straight from the buffer arrays
        b_old_log_probs = torch.FloatTensor(np.array(buffer.log_probs)).to(device) 
        b_returns = torch.FloatTensor(np.array(buffer.rewards)).to(device)       
        b_advantages = torch.FloatTensor(np.array(buffer.values)).to(device)     
        b_masks = torch.FloatTensor(np.array(buffer.action_masks)).to(device)
        
        # Re-normalize advantages across all episodes
        if b_advantages.numel() > 1:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        dataset_size = b_states.size(0)
        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        epoch_entropy = 0.0
        n_updates = 0
        
        for _ in range(UPDATES_PER_ITER):
            permutation = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, BATCH_SIZE):
                batch_indices = permutation[start_idx:start_idx + BATCH_SIZE]
                
                a_loss, c_loss, _, entropy = updater.train_step(
                    states=b_states[batch_indices],
                    actions=b_actions[batch_indices],
                    old_log_probs=b_old_log_probs[batch_indices],
                    returns=b_returns[batch_indices],
                    advantages=b_advantages[batch_indices],
                    action_masks=b_masks[batch_indices],
                )
                epoch_actor_loss += a_loss
                epoch_critic_loss += c_loss
                epoch_entropy += entropy
                n_updates += 1
        
        if n_updates > 0:
            epoch_actor_loss /= n_updates
            epoch_critic_loss /= n_updates
            epoch_entropy /= n_updates
        
        # ── Update metrics ──
        metrics["wins"] += iter_wins
        metrics["losses"] += iter_losses
        metrics["draws"] += iter_draws
        metrics["total_reward"] += iter_reward
        metrics["total_steps"] += iter_steps
        metrics["total_kills"] += iter_kills
        
        # Keep rolling window
        if len(metrics["window_wins"]) > WINDOW_SIZE:
            metrics["window_wins"] = metrics["window_wins"][-WINDOW_SIZE:]
        
        win_rate = sum(metrics["window_wins"]) / max(len(metrics["window_wins"]), 1)
        iter_time = time.time() - iter_start
        
        # ── Logging ──
        opp_str = "/".join(opp_labels)
        print(
            f"  [{iteration:3d}/{TOTAL_ITERATIONS}] "
            f"W/L/D={iter_wins}/{iter_losses}/{iter_draws} | "
            f"R={iter_reward/EPISODES_PER_ITER:+7.1f} | "
            f"Steps={iter_steps//EPISODES_PER_ITER:3d} | "
            f"K={iter_kills} | "
            f"WR={win_rate:.1%} | "
            f"π={epoch_actor_loss:.4f} | "
            f"V={epoch_critic_loss:.4f} | "
            f"H={epoch_entropy:.3f} | "
            f"N={dataset_size} | "
            f"{iter_time:.1f}s"
        )
        
        # ── Save checkpoint to pool ──
        if iteration % POOL_SAVE_INTERVAL == 0:
            pool.save_checkpoint(learning_net.state_dict(), iteration)
        
        # ── Save best model based on win rate ──
        if iteration % BEST_SAVE_INTERVAL == 0 and win_rate > metrics["best_win_rate"]:
            metrics["best_win_rate"] = win_rate
            best_path = "quan_rl_best.pth"
            torch.save(learning_net.state_dict(), best_path)
            print(f"  ★ New best model saved! WR={win_rate:.1%} → {best_path}")
        
        # ── Entropy collapse warning ──
        if epoch_entropy < 0.1 and iteration > 10:
            print(f"  ⚠ Warning: Entropy very low ({epoch_entropy:.4f}) — "
                  f"policy may be collapsing. Consider increasing c_ent.")
    
    # ─────────────────────────────────────────────────────────────────
    # 5. Final save
    # ─────────────────────────────────────────────────────────────────
    final_path = "quan_rl_final.pth"
    torch.save(learning_net.state_dict(), final_path)
    
    total_games = metrics["wins"] + metrics["losses"] + metrics["draws"]
    wld_str = f"{metrics['wins']}/{metrics['losses']}/{metrics['draws']}"
    final_wr = f"{sum(metrics['window_wins'])/max(len(metrics['window_wins']),1):.1%}"
    best_wr = f"{metrics['best_win_rate']:.1%}"
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║  Training Complete!                                    ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║  Total games:    {total_games:<40d} ║")
    print(f"║  Wins/Loss/Draw: {wld_str:<40s} ║")
    print(f"║  Final win rate: {final_wr:<40s} ║")
    print(f"║  Best win rate:  {best_wr:<40s} ║")
    print(f"║  Total kills:    {metrics['total_kills']:<40d} ║")
    print(f"║  Saved: {final_path:<49s} ║")
    print(f"╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    execute_self_play_training()