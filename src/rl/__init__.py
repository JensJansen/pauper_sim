"""RL / self-play training harness: the attention perception stack (arch),
per-card featurization + vocab (features), per-deck actor-critic (deck), the
fixed/pointer action-table bridge (action_bridge), the shared deck pool
(pool), the historical-opponent league (league), the PPO training loop
(train), and the win/loss reward (rewards).

Sits on top of the `game` engine and the `drl_env` action layer (both imported
absolutely, since they live at src/ root). The entry-point script
run_league.py stays at src/ root and imports from here."""
