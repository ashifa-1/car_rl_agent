import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.environment import CarRacingEnv


# --------------------------------------------------
# Callback for logging rewards
# --------------------------------------------------

class RewardLogger(BaseCallback):

    def __init__(self, check_freq=5000):
        super().__init__()
        self.check_freq = check_freq
        self.timesteps = []
        self.mean_rewards = []

    def _on_step(self):

        # PPO stores episode info here
        if len(self.model.ep_info_buffer) > 0:

            if self.n_calls % self.check_freq == 0:

                rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]

                mean_reward = np.mean(rewards)

                self.timesteps.append(self.num_timesteps)
                self.mean_rewards.append(float(mean_reward))

        return True


# --------------------------------------------------
# Training function
# --------------------------------------------------

def main():

    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Ensure folders exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Create environment
    env = CarRacingEnv()

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["ppo_params"]["learning_rate"],
        n_steps=config["ppo_params"]["n_steps"],
        batch_size=config["ppo_params"]["batch_size"],
        gamma=config["ppo_params"]["gamma"],
        verbose=1
    )

    # Logger callback
    logger = RewardLogger()

    print("Starting training...")

    # Train agent
    model.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=logger
    )

    print("Training finished. Saving artifacts...")

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model.save("models/ppo_car_agent")

    # --------------------------------------------------
    # Save training log
    # --------------------------------------------------

    log_data = {
        "timesteps": logger.timesteps,
        "mean_rewards": logger.mean_rewards
    }

    with open("results/training_log.json", "w") as f:
        json.dump(log_data, f)

    # --------------------------------------------------
    # Generate reward curve
    # --------------------------------------------------

    plt.figure()

    plt.plot(logger.timesteps, logger.mean_rewards)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Training Reward Curve")

    plt.savefig("results/reward_curve.png")

    print("Model saved to: models/ppo_car_agent.zip")
    print("Training log saved to: results/training_log.json")
    print("Reward plot saved to: results/reward_curve.png")


# --------------------------------------------------

if __name__ == "__main__":
    main()