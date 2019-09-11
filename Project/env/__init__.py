import gym
from env.atari_wrappers import make_atari, wrap_deepmind


class SaveToInfoEnv(gym.Wrapper):
    def __init__(self, env, prefix):
        gym.Wrapper.__init__(self, env)
        self.prefix = prefix

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        info[self.prefix + '_obs'] = obs
        info[self.prefix + '_reward'] = reward
        info[self.prefix + '_done'] = done
        return obs, reward, done, info


def make_env(game, monitor_dir=None):
    env = make_atari(game + "NoFrameskip-v4")
    env = SaveToInfoEnv(env, prefix='real')
    if monitor_dir:
        env = gym.wrappers.Monitor(env, directory=monitor_dir, force=True)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=False)
    return env
