import pathlib
import pickle
import sys
import queue
import threading
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.logger import TensorBoardOutputFormat

from plot import moving_average


GymObs = Union[Tuple, Dict, np.ndarray, int]


env_queue = queue.Queue()  # Input observation
action_queue = queue.Queue()  # Output action

N_EXTRA_FEATURES = 1 + 2
INPUT_SIZE = 6*2 + 1 + 3 + 6 + 1 + 1 + N_EXTRA_FEATURES
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
    # TODO use rl_zoo3 frame skipping wrapper
    def __init__(self, hparams: dict = None):
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
        if hparams:
            self.hparams.update(hparams)
        for attr, value in self.hparams.items():
            setattr(self, attr, value)
        self.frame_number = 0

    def reset(self) -> GymObs:
        """
        Called at the beginning of an episode.
        :return: the first observation of the episode
        """
        flag, state = get_env_queue()
        state_struct = self.state_to_struct(state)
        state_struct = self.add_state_features(state_struct)
        state = state_struct["_packed"]
        self.last_state = state
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
            state_struct = self.state_to_struct(state)
            state_struct = self.add_state_features(state_struct)
            state = state_struct["_packed"]
            self.last_state = state
            self.frame_number += 1
            reward = self.get_reward(state_struct)
            done = False
        else:
            raise ValueError(f"Unknown flag {flag}")
        info = {}
        return state, reward, done, info

    def get_reward(self, state_struct: dict) -> float:
        reward = self.default_reward
        # Reward shaping
        if self.frame_number % self.reward_interval == 0:
            reward += self.interval_reward
        if self.reward_slot_center:
            reward += self.reward_slot_center_amount * pow(1 - abs(state_struct["center_offset"]), 4)
        if self.reward_far_wall:
            wall_dist, wall_width = self.get_cur_wall_dist(state_struct)
            reward += self.reward_far_wall_amount * wall_dist
        return reward

    @staticmethod
    def add_state_features(state_struct: dict) -> dict:
        # Add extra features to the input
        center_offset = SupaEnv.get_cur_center_offset(state_struct)
        state_struct["center_offset"] = center_offset
        state_struct["_packed"].append(center_offset)
        phi = state_struct["player_pos"] / state_struct["n_slots"] * 2 * np.pi
        state_struct["pos_x"] = np.cos(phi)
        state_struct["pos_y"] = np.sin(phi)
        state_struct["_packed"].append(state_struct["pos_x"])
        state_struct["_packed"].append(state_struct["pos_y"])
        return state_struct

    @staticmethod
    def state_to_struct(state: list) -> dict:
        # This function must be defined based on the C++ GameState_DQN struct.
        walls = np.array(state[:12]).reshape(6, 2)
        wall_speed = state[12]
        n_slots = state[13:16].index(1) + 4
        cur_slot = next(i for i, x in enumerate(state[16:16+6]) if x > 0)
        return {
            "_packed": state,
            "walls": walls,
            "wall_speed": wall_speed,
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


# Adapted from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/wrappers.py
class HistoryWrapper(gym.Wrapper):
    """
    Stack past observations and actions to give an history to the agent.
    :param env:
    :param horizon:Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 2):
        horizon = env.hparams.get("horizon", horizon)
        
        # Overwrite the observation space
        input_size = INPUT_SIZE + (INPUT_SIZE + 1) * horizon
        env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(input_size,))
        
        super().__init__(env)

        self.horizon = horizon
        self.obs_history = np.zeros(INPUT_SIZE * (horizon + 1))
        self.action_history = np.zeros(horizon)

    def _create_obs_from_history(self):
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self):
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs = self.env.reset()
        obs = np.array(obs)
        self.obs_history[..., -obs.shape[-1] :] = obs
        return self._create_obs_from_history()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.array(obs)
        action = np.array(action)
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1] :] = obs

        self.action_history = np.roll(self.action_history, shift=-1, axis=-1)
        self.action_history[..., -1:] = action
        return self._create_obs_from_history(), reward, done, info


class TensorboardCallback(BaseCallback):
    def _on_training_start(self):
        # Save reference to tensorboard formatter object
        output_formats = self.logger.output_formats
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

        # TODO https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-hyperparameters
        env_params = self.training_env.get_attr("hparams")[0]
        model_params = self.model._hparams
        self.tb_formatter.writer.add_text("params/model", str(model_params), self.num_timesteps)
        self.tb_formatter.writer.add_text("params/env", str(env_params), self.num_timesteps)

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
    def __init__(self, experiment_name="sb3_9"):
        self.experiment_name = experiment_name

        sb3_params = dict(
            train_freq=16,
            gradient_steps=32,
            gamma=0.96,
            exploration_fraction=0.05,
            exploration_final_eps=0.005,
            target_update_interval=100,
            learning_starts=100,
            buffer_size=100_000,
            batch_size=128,
            learning_rate=6.3e-4,
            policy_kwargs=dict(net_arch=[256, 256])
        )
        # PPO
        # sb3_params = dict(
        #     n_steps = 1024,
        #     batch_size = 64,
        #     gae_lambda = 0.98,
        #     gamma = 0.95,
        #     n_epochs = 4,
        #     ent_coef = 0.01,
        # )

        self.total_timesteps = 500_000
        self.horizon = 1  # Stack this many states into one

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

        env = SupaEnv()
        self.env = HistoryWrapper(env, horizon=self.horizon)
        if not self.load_checkpoint(checkpoint_path):
            self.model = DQN("MlpPolicy",
                self.env,
                **sb3_params,
                verbose=2,
                tensorboard_log=f"runs/{self.experiment_name}",
                seed=42)
            # self.model = PPO(
            #     "MlpPolicy",
            #     self.env,
            #     **sb3_params,
            #     verbose=2,
            #     tensorboard_log=f"runs/{self.experiment_name}",
            #     seed=42
            # )
            self.model._hparams = sb3_params  # Store params for easy logging and saving
        self.learn_thread = None 

        self.last_action = None

    def on_episode_end(self, score=None):
        if self.learn_thread and self.learn_thread.is_alive():
            env_queue.put((GAME_OVER_FLAG, score))
        self.last_action = None

    def get_action(self, state):
        action = None
        env_queue.put((ENV_STATE_FLAG, state))
        if self.learn_thread and self.learn_thread.is_alive():
            try:
                action = action_queue.get(timeout=15)
            except queue.Empty:
                # This should happen only when the total timesteps is reached (learning has stopped)
                print("action_queue timeout")        
        if action is None:  # When not learning 
            if self.last_action is None:
                state = self.env.reset()
            else:
                state, *_ = self.env.step(self.last_action)
                action_queue.get()  # Empty out the queue (== self.last_action)
            action, _ = self.model.predict(state, deterministic=True)
        
        self.last_action = action
        action = self.actions_tr[action]
        return action

    def set_is_learning(self, is_learning):
        def learn_worker():
            self.model.learn(
                total_timesteps=self.total_timesteps - self.model.num_timesteps, 
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
        # self.model = PPO.load(model_path, env=self.env, print_system_info=True)
        if replay_buffer_path:
            self.model.load_replay_buffer(replay_buffer_path)

        # Restore manually saved env variables
        self.env.real_episode_scores = self.model._env_real_episode_scores
        self.env.hparams = self.model._env_hparams

        return True 



import optuna
import optuna.visualization as opt_vis


class OptunaTrialCallback(BaseCallback):
    def __init__(self, trial: optuna.Trial, eval_freq: int, verbose: int = 0):
        super().__init__(verbose)

        self.eval_freq = eval_freq
        self.trial = trial
        self.last_n_mean = 50
        
        self.is_pruned = False

    def _on_step(self) -> bool:
        # if self.num_timesteps > 40_000:
        #     # Prune if the run is absolute garbage (worse than random)
        #     scores = self.training_env.get_attr("real_episode_scores")[0]
        #     score = np.median(scores[-min(len(scores), self.last_n_mean):])
        #     if score < 3 * 60:
        #         self.is_pruned = True
        #         return False
        if self.num_timesteps % self.eval_freq == 0:
            scores = self.training_env.get_attr("real_episode_scores")[0]
            score = np.median(scores[-min(len(scores), self.last_n_mean):])
            self.trial.report(score, self.num_timesteps)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.97, 0.99])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
    batch_size = 64
    # buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5)])
    buffer_size = 100_000

    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0, 0.1)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [300, 1000, 5000])

    # train_freq = trial.suggest_categorical("train_freq", [1, 8, 16])
    # subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2])
    # gradient_steps = max(train_freq // subsample_steps, 1)
    train_freq = 8
    gradient_steps = -1

    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": 100,
        "policy_kwargs": dict(net_arch=net_arch),
    }
    return hyperparams


def sample_env_params(trial: optuna.Trial) -> Dict[str, Any]:
    reward_slot_center_amount = trial.suggest_float("reward_slot_center_amount", 0, 1)
    reward_far_wall_amount = trial.suggest_float("reward_far_wall_amount", 0, 1)
    # default_reward = trial.suggest_float("default_reward", 0, 1)
    default_reward = 0.1
    horizon = trial.suggest_categorical("horizon", [0, 1, 2])
    # horizon = 2

    hyperparams = {
        "reward_slot_center": True,  # Receive reward for being close to the center of a slot
        "reward_slot_center_amount": reward_slot_center_amount,
        "reward_far_wall": True,  # Receive reward for being on a slot where the wall is far away
        "reward_far_wall_amount": reward_far_wall_amount,
        "reward_interval": 60,  # Receive interval_reward reward after this many successful steps
        "interval_reward": 0,
        "default_reward": default_reward,  # Receive reward each step
        "loss_reward": -1,      # Reward when game over
        "horizon": horizon,  # Stack this many previous states together
    }
    return hyperparams


class SupaSB3Optuna:
    def __init__(self, experiment_name="sb3_optuna_13"):
        self.experiment_name = experiment_name

        self.n_trials = 50
        self.n_startup_trials = 3  # Pruning is disabled until the given number of trials finish in the same study.
        self.eval_freq = 2000  # Report metrics to pruner every this many steps
        self.total_timesteps = 100_000
        self.n_warmup_steps = int(self.total_timesteps * 0.3)  # Do not prune before % of the max budget is used

        # The model works with indices [0, 3), but the server expects [-1,0,1].
        self.actions_tr = [-1, 0, 1]  # Map action index to action
        self.actions_tr_inv = { v: i for i, v in enumerate(self.actions_tr) }

        # A seperate thread is used for listening to C++ requests and for optimization.
        # Another option would be to make the gym.Env handle server requests instead.
        self.learn_thread = threading.Thread(target=self.optimize_hyperparameters)
        self.learn_thread.start()

    def on_episode_end(self, score=None):
        env_queue.put((GAME_OVER_FLAG, score))

    def get_action(self, state):
        env_queue.put((ENV_STATE_FLAG, state))
        try:
            action = action_queue.get(timeout=10)
        except queue.Empty:
            while not env_queue.empty():
                env_queue.get_nowait()
            env_queue.put((ENV_STATE_FLAG, state))
            action = action_queue.get()
        action = self.actions_tr[action]
        return action

    def set_is_learning(self, is_learning):
        return

    def optimize_hyperparameters(self):
        print("Optimizing hyperparameters...")

        def objective(trial: optuna.Trial):            
            print(f"Running trial {trial._trial_id} ...")

            env_params = sample_env_params(trial)
            dqn_params = sample_dqn_params(trial)
            print(env_params)
            print(dqn_params)

            env = SupaEnv(hparams=env_params)
            env = HistoryWrapper(env)

            model = DQN(
                "MlpPolicy",
                env,
                **dqn_params,
                verbose=2,
                # tensorboard_log=None,
                tensorboard_log=f"runs/{self.experiment_name}/{trial._trial_id}",
                seed=None)
            model._hparams = dqn_params  # Store params for easy logging and saving

            trial_cb = OptunaTrialCallback(trial, eval_freq=self.eval_freq, verbose=2)
            callbacks = [
                TensorboardCallback(),
                trial_cb
            ]
            model.learn(
                total_timesteps=self.total_timesteps, 
                log_interval=10,  # Log every n episodes 
                reset_num_timesteps=False, 
                callback=callbacks)
            print("Learning complete!")

            model.save(f"runs/{self.experiment_name}/{trial._trial_id}/model")
            
            if trial_cb.is_pruned:
                print(f"Pruning trial {trial._trial_id}")
                raise optuna.exceptions.TrialPruned()

            scores = env.env.real_episode_scores
            score = np.mean(scores[-min(len(scores), trial_cb.last_n_mean):])
            return score

        sampler = optuna.samplers.TPESampler(n_startup_trials=self.n_startup_trials)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_warmup_steps)

        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        log_path = f"runs/{self.experiment_name}/report"
        print(f"Writing report to {log_path}")
        study.trials_dataframe().to_csv(f"{log_path}.csv")

        # Save python object to inspect/re-use it later
        with open(f"{log_path}.pkl", "wb+") as f:
            pickle.dump(study, f)

        # Plot optimization result
        try:
            fig1 = opt_vis.plot_optimization_history(study)
            fig2 = opt_vis.plot_param_importances(study)
            fig3 = opt_vis.plot_intermediate_values(study)
            fig4 = opt_vis.plot_parallel_coordinate(study)

            fig1.show()
            fig2.show()
            fig3.show()
            fig4.show()
        except (ValueError, ImportError, RuntimeError):
            pass

        print("Done optimizing hyperparameters!")
    