"""Script for training a drone to fly through gates using PPO + CNN.

The drone navigates through 6 sequentially ordered gates in an S-curve
with height variation, using only monocular RGB (grayscale) camera input.

Example
-------
In a terminal, run as:

    $ python learn_throughgate.py

"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack, VecEnv, VecEnvWrapper

from gym_pybullet_drones.envs.ThroughGateAviary import ThroughGateAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_N_ENVS = 4
FRAME_STACK = 4

OBS_TYPE = ObservationType.RGB
ACT_TYPE = ActionType.VEL

TOTAL_TIMESTEPS = int(2e7)


class SingleEnvVec(VecEnv):
    def __init__(self, env):
        self.env = env
        super().__init__(1, env.observation_space, env.action_space)

    def reset(self):
        o, i = self.env.reset()
        return np.array([o])

    def step_async(self, actions):
        self._action = np.array(actions[0]).reshape(1, -1)

    def step_wait(self):
        o, r, terminated, truncated, info = self.env.step(self._action)
        return (np.array([o]),
                np.array([r]),
                np.array([terminated or truncated]),
                [info])

    def close(self):
        self.env.close()

    def get_attr(self, name, indices=None):
        return [getattr(self.env, name)]

    def set_attr(self, name, value, indices=None):
        setattr(self.env, name, value)

    def env_method(self, method_name, *args, **kwargs):
        return [getattr(self.env, method_name)(*args, **kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]

    def seed(self, seed=None):
        self.env.reset(seed=seed)
        return [seed]

    def get_images(self):
        raise NotImplementedError

    def render(self, mode='human'):
        self.env.render()


def run(output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        n_envs=DEFAULT_N_ENVS):

    filename = os.path.join(output_folder, 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    os.makedirs(filename + '/', exist_ok=True)

    train_env = make_vec_env(ThroughGateAviary,
                             env_kwargs=dict(obs=OBS_TYPE, act=ACT_TYPE, gui=False),
                             n_envs=n_envs,
                             seed=0)
    train_env = VecFrameStack(train_env, n_stack=FRAME_STACK, channels_order='first')

    eval_env = SingleEnvVec(ThroughGateAviary(obs=OBS_TYPE, act=ACT_TYPE, gui=False))
    eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK, channels_order='first')

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    model = PPO('CnnPolicy',
                train_env,
                n_steps=256,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                learning_rate=3e-4,
                verbose=1)

    target_reward = 800.0
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename + '/',
                                 log_path=filename + '/',
                                 eval_freq=int(2000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=eval_callback,
                log_interval=10)

    model.save(filename + '/final_model.zip')
    print("[INFO] Saved to", filename)

    if os.path.isfile(filename + '/best_model.zip'):
        path = filename + '/best_model.zip'
    elif os.path.isfile(filename + '/final_model.zip'):
        path = filename + '/final_model.zip'
    else:
        print("[ERROR] No model found under", filename)
        return

    model = PPO.load(path)

    test_env = ThroughGateAviary(gui=gui, obs=OBS_TYPE, act=ACT_TYPE, record=record_video)
    test_env_nogui = ThroughGateAviary(obs=OBS_TYPE, act=ACT_TYPE, gui=False)

    wrapped_eval = VecFrameStack(SingleEnvVec(test_env_nogui), n_stack=FRAME_STACK, channels_order='first')
    mean_reward, std_reward = evaluate_policy(model, wrapped_eval, n_eval_episodes=5)
    print(f"\nMean reward {mean_reward:.1f} +- {std_reward:.1f}\n")

    obs, info = test_env.reset(seed=42, options={})
    obs_stack = np.zeros((FRAME_STACK, 360, 640), dtype=np.float32)
    obs_stack[:] = obs

    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        stacked = obs_stack[np.newaxis, :, :, :]
        action, _ = model.predict(stacked, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs_stack[:-1] = obs_stack[1:]
        obs_stack[-1] = obs

        gate_idx = info.get("next_gate", -1)
        print(f"Step {i}: gate={gate_idx} | reward={reward:.1f} | done={terminated}")
        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs, info = test_env.reset(seed=42, options={})
            obs_stack[:] = obs

    test_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a drone to fly through gates with monocular vision')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool,
                        help='Use PyBullet GUI (default: True)')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool,
                        help='Record a video (default: False)')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder for saving logs and models (default: "results")')
    parser.add_argument('--n_envs', default=DEFAULT_N_ENVS, type=int,
                        help='Number of parallel environments (default: 4)')
    ARGS = parser.parse_args()
    run(**vars(ARGS))
