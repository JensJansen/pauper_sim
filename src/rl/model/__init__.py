"""Network/observation shape: per-card featurization + vocab (features), the
attention perception stack (arch), the per-deck actor-critic (deck), and the
per-deck pregame mulligan model (mulligan). No training-loop or
opponent-management logic lives here -- see rl.training / rl.league."""
