from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
from src.environment import CarRacingEnv


def main():

    env = CarRacingEnv(render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder="results",
        name_prefix="agent_demonstration",
        episode_trigger=lambda x: True
    )

    model = PPO.load("models/ppo_car_agent")

    obs, _ = env.reset()

    done = False

    while not done:

        action, _ = model.predict(obs)

        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    main()