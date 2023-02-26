import crafter
import numpy as np
from pathlib import Path
import pygame
import warnings

import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym
from dreamerv3.embodied.core.basics import convert


LOGDIR = Path('/home/jonanthan/logdir/crafter_small_1')


def main():
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    config = embodied.Config.load(LOGDIR / 'config.yaml')

    env = crafter.Env()
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, config)

    step = embodied.Counter()
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(LOGDIR / 'checkpoint.ckpt', keys=['agent'])

    state = None
    act = {'action': env.act_space['action'].sample(), 'reset': np.array(True)}
    while True:
        obs = env.step(act)
        obs = {k: convert(v) for k, v in obs.items()}
        act, state = agent.policy(obs, state, mode='eval')
        act = {'action': act['action'][0], 'reset': obs['is_last'][0]}


# def main():

#     warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

#     # See configs.yaml for all options.
#     config = embodied.Config.load(str(LOGDIR / 'config.yaml'))

#     env = crafter.Env()  # Replace this with your Gym env.
#     env = from_gym.FromGym(env)
#     env = dreamerv3.wrap_env(env, config)
#     # env = embodied.BatchEnv([env], parallel=False)

#     step = embodied.Counter()
#     agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
#     checkpoint = embodied.Checkpoint()
#     checkpoint.agent = agent
#     checkpoint.load(str(LOGDIR / 'checkpoint.ckpt'), keys=['agent'])

#     # pygame.init()
#     # size = (512, 512)
#     # screen = pygame.display.set_mode(size)
#     # clock = pygame.time.Clock()

#     step = 0
#     fps = 30

#     obs = env._env.reset()
#     obs = env._obs(obs, 0.0, is_first=True)
#     obs = {k: embodied.convert(v) for k, v in obs.items()}
#     state = None

#     while True:

#         # Rendering.
#         # image = env.render()
#         # surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
#         # screen.blit(surface, (0, 0))
#         # pygame.display.flip()
#         # clock.tick(fps)

#         act, state = agent.policy(obs, state, mode='eval')
#         acts = {k: v for k, v in act.items() if not k.startswith('log_')}
#         obs, reward, done, _ = env.step(acts)  # act['action'])

#         step += 1

if __name__ == '__main__':
  main()
