# env_pong.py
# Minimal, deterministic Pong-like environment with vector state.
# Coordinate system: x,y in [-1, 1]. Agent controls RIGHT paddle.

from dataclasses import dataclass
import numpy as np

@dataclass
class PongConfig:
    dt: float = 0.02               # time step (s)
    r_paddle_speed: float = 1.7    # RIGHT paddle max speed
    l_paddle_speed: float = 1.1    # LEFT paddle (heuristic) speed — easier opponent
    paddle_height: float = 0.30    # paddle half-extent in y (slightly larger hitbox)
    ball_speed: float = 1.2        # base ball speed (units/s)
    speedup_on_hit: float = 1.03   # ball speed multiplier per hit
    wall_y: float = 1.0            # top/bottom boundaries at |y| = 1
    goal_x: float = 1.0            # scoring at |x| >= 1
    action_accel: float = 6.0      # accel to change paddle velocity
    friction: float = 3.5          # velocity damping for paddle
    rng_seed: int = 42
    reward_step_penalty: float = -0.005
    reward_hit: float = 1.0
    reward_score: float = 5.0
    reward_miss: float = -5.0
    obs_noise: float = 0.0         # set small >0 for noisy observations
    start_bias_right: bool = True  # serve towards the agent initially

class PongEnv:
    """Right-paddle agent vs. simple heuristic left paddle."""
    def __init__(self, cfg: PongConfig = PongConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.rng_seed)
        self.reset()

    def reset(self):
        c = self.cfg
        self.ball_pos = np.array([0.0, 0.0], dtype=np.float32)
        angle = self.rng.uniform(-0.6, 0.6)
        dir_x = 1.0 if c.start_bias_right else self.rng.choice([-1.0, 1.0])
        self.ball_vel = np.array([dir_x*np.cos(angle), np.sin(angle)], dtype=np.float32)
        self.ball_vel = self.ball_vel / (np.linalg.norm(self.ball_vel) + 1e-8) * c.ball_speed
        self.r_paddle_y = 0.0
        self.r_paddle_v = 0.0
        self.l_paddle_y = 0.0
        self.score_r = 0
        self.score_l = 0
        return self._get_state()

    def _get_state(self):
        c = self.cfg
        obs = np.array([
            self.ball_pos[0],
            self.ball_pos[1],
            self.ball_vel[0],
            self.ball_vel[1],
            self.r_paddle_y,
            self.r_paddle_v,
        ], dtype=np.float32)
        if c.obs_noise > 0:
            obs += self.rng.normal(0, c.obs_noise, size=obs.shape)
        return obs

    def _left_paddle_policy(self):
        # Simple heuristic: track ball y with capped speed (slower than agent)
        c = self.cfg
        err = (self.ball_pos[1] - self.l_paddle_y)
        self.l_paddle_y += np.clip(err, -c.l_paddle_speed*c.dt, c.l_paddle_speed*c.dt)
        self.l_paddle_y = float(np.clip(self.l_paddle_y, -c.wall_y + c.paddle_height, c.wall_y - c.paddle_height))

    def step(self, action: int):
        """
        action ∈ {-1,0,+1}: up / stay / down for RIGHT paddle.
        Returns: obs, reward, done, info
        """
        c = self.cfg
        # --- Right paddle dynamics ---
        accel = float(action) * c.action_accel
        self.r_paddle_v += accel * c.dt
        # friction/damping
        self.r_paddle_v *= np.exp(-c.friction * c.dt)
        self.r_paddle_v = float(np.clip(self.r_paddle_v, -c.r_paddle_speed, c.r_paddle_speed))
        self.r_paddle_y += self.r_paddle_v * c.dt
        self.r_paddle_y = float(np.clip(self.r_paddle_y, -c.wall_y + c.paddle_height, c.wall_y - c.paddle_height))

        # --- Left paddle (heuristic) ---
        self._left_paddle_policy()

        # --- Ball physics ---
        self.ball_pos += self.ball_vel * c.dt

        # wall bounce (top/bottom)
        if self.ball_pos[1] >= c.wall_y:
            self.ball_pos[1] = 2*c.wall_y - self.ball_pos[1]
            self.ball_vel[1] *= -1
        elif self.ball_pos[1] <= -c.wall_y:
            self.ball_pos[1] = -2*c.wall_y - self.ball_pos[1]
            self.ball_vel[1] *= -1

        reward = c.reward_step_penalty
        done = False
        info = {}

        # paddle x-positions
        x_r = c.goal_x - 0.02
        x_l = -c.goal_x + 0.02

        # Right paddle collision
        if self.ball_pos[0] >= x_r and self.ball_vel[0] > 0:
            if abs(self.ball_pos[1] - self.r_paddle_y) <= c.paddle_height:
                # reflect X, add a small vertical tweak based on hit offset
                offset = (self.ball_pos[1] - self.r_paddle_y) / c.paddle_height
                self.ball_vel[0] *= -1
                self.ball_vel[1] += 0.35 * offset
                # renormalize speed, speed up slightly
                v = self.ball_vel
                self.ball_vel = v / (np.linalg.norm(v) + 1e-8) * (np.linalg.norm(v) * c.speedup_on_hit)
                reward += c.reward_hit
            else:
                reward += c.reward_miss
                done = True
                self.score_l += 1

        # Left paddle collision
        if self.ball_pos[0] <= x_l and self.ball_vel[0] < 0:
            if abs(self.ball_pos[1] - self.l_paddle_y) <= c.paddle_height:
                self.ball_vel[0] *= -1
                offset = (self.ball_pos[1] - self.l_paddle_y) / c.paddle_height
                self.ball_vel[1] += 0.30 * offset
                v = self.ball_vel
                self.ball_vel = v / (np.linalg.norm(v) + 1e-8) * (np.linalg.norm(v) * c.speedup_on_hit)
            else:
                # Agent scores if left misses
                reward += c.reward_score
                done = True
                self.score_r += 1

        # Out-of-bounds goal (failsafe)
        if self.ball_pos[0] > c.goal_x or self.ball_pos[0] < -c.goal_x:
            done = True

        # Dense shaping: small reward for aligning paddle with ball (within hitbox)
        if abs(self.ball_pos[1] - self.r_paddle_y) <= c.paddle_height:
            reward += 0.02

        return self._get_state(), float(reward), bool(done), info
