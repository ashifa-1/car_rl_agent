import numpy as np
from stable_baselines3 import PPO
from src.environment import CarRacingEnv


def main():

    env = CarRacingEnv()

    model = PPO.load("models/ppo_car_agent")

    rewards = []

    num_episodes = 20

    for episode in range(num_episodes):

        obs, _ = env.reset()

        done = False
        total_reward = 0

        while not done:

            action, _ = model.predict(obs)

            obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            total_reward += reward

        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print(f"Mean Reward: {mean_reward}")
    print(f"Std Reward: {std_reward}")


if __name__ == "__main__":
    main()