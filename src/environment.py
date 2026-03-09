import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import yaml

from src.car import Car
from src.utils import load_track


class CarRacingEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):

        super().__init__()

        # Load configuration
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        self.num_rays = config["env_params"]["num_rays"]
        self.max_steps = config["env_params"]["max_steps_per_episode"]

        self.render_mode = render_mode

        # Observation space
        # rays + velocity
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_rays + 1,),
            dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Discrete(5)

        # Load track
        self.walls = load_track("tracks/track_1.txt")

        # Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()

        # Car
        self.car = Car(400, 300)

        self.step_count = 0


    # ---------------------------------------------------
    # Reset
    # ---------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.car = Car(400, 300)
        self.step_count = 0

        obs = self._get_observation()

        return obs, {}


    # ---------------------------------------------------
    # Step
    # ---------------------------------------------------

    def step(self, action):

        reward = 0
        terminated = False

        # Apply action
        if action == 1:
            self.car.accelerate()

        elif action == 2:
            self.car.brake()

        elif action == 3:
            self.car.turn(-1)

        elif action == 4:
            self.car.turn(1)

        # Store previous position
        old_x = self.car.x
        old_y = self.car.y

        # Update car
        self.car.update()

        # Collision check
        if self.car.check_collision(self.walls):

            reward = -10
            terminated = True

            self.car.x = old_x
            self.car.y = old_y

        else:
            reward = 0.1

        # Time penalty
        reward -= 0.01

        self.step_count += 1

        truncated = self.step_count >= self.max_steps

        obs = self._get_observation()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}


    # ---------------------------------------------------
    # Observation
    # ---------------------------------------------------

    def _get_observation(self):

        readings = self.car.cast_rays(self.walls)

        distances = []

        for dist, _ in readings:
            distances.append(dist / self.car.max_ray_distance)

        velocity = self.car.velocity / self.car.max_speed

        obs = np.array(distances + [velocity], dtype=np.float32)

        return obs


    # ---------------------------------------------------
    # Render
    # ---------------------------------------------------

    def render(self):

        self.screen.fill((30, 30, 30))

        # draw track walls
        for x1, y1, x2, y2 in self.walls:
            pygame.draw.line(self.screen, (255, 255, 255), (x1, y1), (x2, y2), 3)

        # draw car
        self.car.draw(self.screen)

        # draw rays
        readings = self.car.cast_rays(self.walls)
        self.car.draw_rays(self.screen, readings)

        pygame.display.flip()

        if self.render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(self.screen), (1, 0, 2)
            )

        self.clock.tick(self.metadata["render_fps"])

    # ---------------------------------------------------
    # Close
    # ---------------------------------------------------

    def close(self):

        pygame.quit()