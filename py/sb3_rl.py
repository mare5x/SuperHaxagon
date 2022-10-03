import pathlib
import sys
import queue
import threading
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.logger import TensorBoardOutputFormat

from plot import moving_average


GymObs = Union[Tuple, Dict, np.ndarray, int]


env_queue = queue.Queue()  # Input observation
action_queue = queue.Queue()  # Output action

INPUT_SIZE = 6*2 + 1 + 3 + 6 + 1 + 1
OUT_SIZE = 3

KYS_FLAG = "kys"
GAME_OVER_FLAG = "game_over"
ENV_STATE_FLAG = "env_state"


def latest_checkpoint(path):
    path = pathlib.Path(path)
    checkpoints = list(path.glob("*_steps.zip"))
    if len(checkpoints) > 0:
        n, p = max([(int(p.stem.rsplit('_')[2]), p) for p in checkpoints])
        replay_buffers = list(path.glob(f"*_replay_buffer_{n}_steps.pkl"))
        if len(replay_buffers) > 0:
            return p, replay_buffers[0]
        return p, None
    return None


def get_env_queue():
    flag, value = env_queue.get()
    if flag == KYS_FLAG:
        return sys.exit()  # Exit the learner thread only
    return flag, value


class SupaEnv(gym.Env):
    def __init__(self):
        super(SupaEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(INPUT_SIZE,))
        self.action_space = gym.spaces.Discrete(OUT_SIZE)

        self.last_state = None 
        self.real_episode_scores = []
        self.hparams = {
            "reward_slot_center": True,  # Receive reward for being close to the center of a slot
            "reward_slot_center_amount": 0.1,
            "reward_far_wall": True,  # Receive reward for being on a slot where the wall is far away
            "reward_far_wall_amount": 0.5,
            "reward_interval": 60,  # Receive interval_reward reward after this many successful steps
            "interval_reward": 0,
            "default_reward": 0.1,  # Receive reward each step
            "loss_reward": -1,      # Reward when game over
        }
        for attr, value in self.hparams.items():
            setattr(self, attr, value)
        self.frame_number = 0

    def reset(self) -> GymObs:
        """
        Called at the beginning of an episode.
        :return: the first observation of the episode
        """
        flag, state = get_env_queue()
        self.frame_number = 1
        return state

    def step(self, action: Union[int, np.ndarray]) -> Tuple[GymObs, float, bool, Dict]:
        """
        Step into the environment.
        :return: A tuple containing the new observation, the reward signal, 
        whether the episode is over and additional informations.
        """
        action_queue.put(action)
        flag, state = get_env_queue()
        if flag == GAME_OVER_FLAG:
            self.real_episode_scores.append(state)
            state = self.last_state
            reward = self.loss_reward
            done = True
        elif flag == ENV_STATE_FLAG:
            self.last_state = state
            self.frame_number += 1

            reward = self.get_reward(state)
            done = False
        else:
            raise ValueError(f"Unknown flag {flag}")
        info = {}
        return state, reward, done, info

    def get_reward(self, state):
        reward = self.default_reward
        # Reward shaping
        if self.frame_number % self.reward_interval == 0:
            reward += self.interval_reward
        state_struct = self.state_to_struct(state)
        if self.reward_slot_center:
            center_offset = self.get_cur_center_offset(state_struct)
            reward += self.reward_slot_center_amount * pow(1 - abs(center_offset), 4)
        if self.reward_far_wall:
            wall_dist, wall_width = self.get_cur_wall_dist(state_struct)
            reward += self.reward_far_wall_amount * wall_dist
        return reward

    @staticmethod
    def state_to_struct(state: list) -> dict:
        # This function must be defined based on the C++ GameState_DQN struct.
        walls = np.array(state[:12]).reshape(6, 2)
        n_slots = state[13:16].index(1) + 4
        cur_slot = next(i for i, x in enumerate(state[16:16+6]) if x > 0)
        return {
            "walls": walls,
            "n_slots": n_slots,
            "cur_slot": cur_slot,
            "player_pos": state[22] * n_slots,
            "world_rotation": state[23],
        }

    @staticmethod
    def get_cur_wall_dist(state_struct: dict) -> tuple:
        dist, width = state_struct["walls"][state_struct["cur_slot"]]
        return dist, width

    @staticmethod
    def get_cur_center_offset(state_struct: dict) -> float:
        # -1: left edge, 0: center, 1: right edge
        pos = state_struct["player_pos"] * state_struct["n_slots"] 
        return (pos % 1.0) * 2.0 - 1.0


class TensorboardCallback(BaseCallback):
    def _on_training_start(self):
        # Save reference to tensorboard formatter object
        output_formats = self.logger.output_formats
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

        # TODO https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-hyperparameters
        # self.tb_formatter.writer.add_text("params", str(self.model._hparams), self.steps_taken)

    def _on_step(self) -> bool:
        is_episode_done = self.locals["dones"][0]
        if is_episode_done:
            scores = self.training_env.get_attr("real_episode_scores")[0]
            self.tb_formatter.writer.add_scalar("Score/Score vs episode", scores[-1], len(scores))
            self.tb_formatter.writer.add_scalar("Score/Time vs episode", scores[-1] / 60, len(scores))
            self.logger.record("Score/Score vs steps", scores[-1])
        return True


class SavePlotCallback(BaseCallback):
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = pathlib.Path(save_path)

    def _on_training_start(self):
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        def plot(ax, data):
            ax.set_title("Q-learning score history")
            x = np.array(data) / 60  # Seconds
            ax.set_ylabel('Time [s]')
            ax.set_xlabel('Attempt number')
            ax.plot(x)
            ax.plot(moving_average(x, k=10))

        score_history = self.training_env.get_attr("real_episode_scores")[0]
        path = self.save_path / f"{self.num_timesteps}.png"
        fig, ax = plt.subplots()
        plot(ax, score_history)
        fig.savefig(path)
        plt.close(fig)
        return True


class CheckpointWithEnvCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        # Save env variables by adding them as model variables so they get saved
        real_episode_scores = self.training_env.get_attr("real_episode_scores")[0]
        hparams = self.training_env.get_attr("hparams")[0]
        self.model._env_real_episode_scores = real_episode_scores
        self.model._env_hparams = hparams
        return super()._on_step()


class SupaSB3:
    def __init__(self, experiment_name="sb3_0"):
        self.experiment_name = experiment_name

        sb3_params = dict(
            train_freq=16,
            gradient_steps=8,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.07,
            target_update_interval=600,
            learning_starts=1000,
            buffer_size=10000,
            batch_size=128,
            learning_rate=4e-3,
            policy_kwargs=dict(net_arch=[256, 256])
        )

        # The model works with indices [0, 3), but the server expects [-1,0,1].
        self.actions_tr = [-1, 0, 1]  # Map action index to action
        self.actions_tr_inv = { v: i for i, v in enumerate(self.actions_tr) }

        checkpoint_path = f"./checkpoints/{self.experiment_name}"
        checkpoint_callback = CheckpointWithEnvCallback(
            save_freq=50_000,  # steps
            save_path=checkpoint_path,
            name_prefix=self.experiment_name,
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=2
        )
        save_plot_callback_ = SavePlotCallback(save_path=checkpoint_path)
        save_plot_callback = EveryNTimesteps(n_steps=checkpoint_callback.save_freq, callback=save_plot_callback_)

        self.callbacks = [
            TensorboardCallback(),
            checkpoint_callback,
            save_plot_callback
        ]

        self.env = SupaEnv()
        if not self.load_checkpoint(checkpoint_path):
            self.model = DQN("MlpPolicy",
                self.env,
                **sb3_params,
                verbose=2,
                tensorboard_log=f"runs/{self.experiment_name}",
                seed=42)
        self.learn_thread = None 

    def on_episode_end(self, score=None):
        if self.learn_thread and self.learn_thread.is_alive():
            env_queue.put((GAME_OVER_FLAG, score))

    def get_action(self, state):
        action = None
        if self.learn_thread and self.learn_thread.is_alive():
            env_queue.put((ENV_STATE_FLAG, state))
            try:
                action = action_queue.get(timeout=15)
            except queue.Empty:
                # This should happen only when the total timesteps is reached (learning has stopped)
                print("action_queue timeout")
        if action is None:
            action, _ = self.model.predict(state, deterministic=True)
        return self.actions_tr[action]

    def set_is_learning(self, is_learning):
        def learn_worker():
            self.model.learn(
                total_timesteps=100000,
                log_interval=10,  # Log every n episodes 
                reset_num_timesteps=False, 
                callback=self.callbacks)
            print("Learning complete!")

        if is_learning:
            if self.learn_thread is None or not self.learn_thread.is_alive():
                self.learn_thread = threading.Thread(target=learn_worker)
                self.learn_thread.start()
        else:
            if self.learn_thread:
                if self.learn_thread.is_alive():
                    env_queue.put((KYS_FLAG, None))
                self.learn_thread.join()
            self.learn_thread = None

    def load_checkpoint(self, checkpoint_path):
        # Resume latest checkpoint
        path = latest_checkpoint(checkpoint_path)
        if path is None:
            return False

        model_path, replay_buffer_path = path
        self.model = DQN.load(model_path, env=self.env, print_system_info=True)
        if replay_buffer_path:
            self.model.load_replay_buffer(replay_buffer_path)

        # Restore manually saved env variables
        self.env.real_episode_scores = self.model._env_real_episode_scores
        self.env.hparams = self.model._env_hparams

        return True 
    