import os
import gym
import minerl
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

os.environ['MALMO_MINECRAFT_OUTPUT_LOG'] = 'true'
os.environ['MALMO_MINECRAFT_INITIAL_MEMORY'] = '2G'
os.environ['MALMO_MINECRAFT_MAX_MEMORY'] = '4G'
os.environ['MINERL_HEADLESS'] = '1'

minerl.env.ENV_KWARGS = {
    'timeout': 30000,
    'retry_on_timeout': True,
    'retry_count': 3,
}

set_random_seed(42)
os.makedirs("models_dqn", exist_ok=True)
os.makedirs("logs_dqn", exist_ok=True)
class MinecraftCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        in_channels = observation_space.shape[0]
        super(MinecraftCNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        # compute the output size after conv layers
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x))
class PovWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)

    def observation(self, observation):
        pov = observation["pov"]
        return np.transpose(pov, (2, 0, 1))  # HWC → CHW
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = [
            {"forward": 1, "jump": 1, "attack": 1, "camera": [0, 10]},  # Look up + move
            {"forward": 1, "jump": 0, "attack": 1, "camera": [0, 0]},   # Straight ahead
            {"forward": 0, "jump": 0, "attack": 1, "camera": [0, -5]},  # Look down
        ]
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, act):
        return self.actions[act]

class ErrorHandlingWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        for _ in range(3):
            try:
                return self.env.reset(**kwargs)
            except Exception:
                pass
        raise RuntimeError("Env failed to reset after retries.")

    def step(self, action):
        try:
            return self.env.step(action)
        except Exception:
            return self.reset(), 0, True, {}
def make_env():
    def _init():
        env = gym.make("MineRLTreechop-v0")
        env = ActionWrapper(env)
        env = PovWrapper(env)
        env = ErrorHandlingWrapper(env)
        return env
    return _init
class LossTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.losses = []

    def _on_step(self) -> bool:
        # Only OffPolicyAlgorithms like DQN have `logger.name_to_value`
        if hasattr(self.model, "logger") and "train/loss" in self.model.logger.name_to_value:
            loss = self.model.logger.name_to_value["train/loss"]
            self.losses.append(loss)
        return True

def train_dqn_agent():
    env = DummyVecEnv([make_env()])
    train_env = VecFrameStack(env, n_stack=2)
    
    policy_kwargs = dict(
        features_extractor_class=MinecraftCNN,
        features_extractor_kwargs=dict(features_dim=512)
    )

    model = DQN(
        "CnnPolicy", 
        train_env, 
        learning_rate=5e-4, 
        buffer_size=10000,
        learning_starts=500,
        batch_size=32,
        tau=1.0,
        gamma=0.98,
        train_freq=1,
        target_update_interval=100,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs_dqn/"
    )
    
    loss_callback = LossTrackingCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models_dqn/',
        name_prefix='dqn_treechop'
    )
    

    model.learn(total_timesteps=50_000, progress_bar=True,callback=[checkpoint_callback, loss_callback])
    model.save("models_dqn/final_dqn_treechop")
    np.save("train_losses.npy", loss_callback.losses)
    env.close()
    
def evaluate_dqn_agent(model_path="models_dqn/final_dqn_treechop", num_episodes=5, save_path="eval_metrics_dqn_200000.npz", log_interval=10):
    env = make_env()()
    model = DQN.load(model_path)
    total_rewards = []
    lengths = []

    print(f"Evaluating over {num_episodes} episodes...")
    for ep in tqdm(range(num_episodes), desc="Evaluation Progress"):
        obs = env.reset()
        done = False
        episode_reward = 0
        length = 0
        while not done:
            # Ensure the observation is contiguous (fixes negative stride issue)
            obs_contiguous = np.ascontiguousarray(obs)
            action, _ = model.predict(obs_contiguous, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            length += 1
        total_rewards.append(episode_reward)
        lengths.append(length)

        if (ep + 1) % log_interval == 0:
            print(f"Episode {ep+1} | Reward: {episode_reward:.2f} | Length: {length}")

    np.savez(save_path, rewards=total_rewards, lengths=lengths)
    print(f"\nSaved evaluation results to {save_path}")
    print(f"Average Reward: {np.mean(total_rewards):.2f} | Average Length: {np.mean(lengths):.2f}")
    env.close()

def moving_average(data, *, window_size = 50):
    data = np.array(data)
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

def plot_eval_metrics(rewards_file="eval_metrics_dqn_200000.npz", losses_file="train_losses.npy"):
    data = np.load(rewards_file)
    rewards = data["rewards"]
    lengths = data["lengths"]
    losses = np.load(losses_file)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(rewards, label="Evaluation Rewards")
    axs[0].plot(moving_average(rewards), label="Moving Average")
    axs[0].axhline(np.mean(rewards), color="r", linestyle="--", label=f"Mean: {np.mean(rewards):.2f}")
    axs[0].set_title("Evaluation Rewards")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(lengths, label="Episode Lengths", color="orange")
    axs[1].set_title("Episode Lengths")
    axs[1].set_ylabel("Steps")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(losses, label="Training Loss", color="green", alpha=0.7)
    axs[2].set_title("Training Losses")
    axs[2].set_xlabel("Training Steps")
    axs[2].set_ylabel("Loss")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()



# === Load Training Rewards from TensorBoard ===
def plot_training_rewards(tb_log_dir="logs_dqn/DQN_1/"):  # change folder as needed
    event_acc = EventAccumulator(tb_log_dir)
    event_acc.Reload()
    rewards = event_acc.Scalars("train/loss")

    steps = [e.step for e in rewards]
    values = [e.value for e in rewards]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, label="Train Loss (Mean)")
    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.title("DQN Training Loss (Mean Episode Loss)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    train_dqn_agent()
    evaluate_dqn_agent()
    plot_eval_metrics()
    plot_training_rewards()
    