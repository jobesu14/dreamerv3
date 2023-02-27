import pygame
import warnings

import dreamerv3
from dreamerv3 import embodied


def _init_pygame():
  pygame.init()
  size = (512, 512)
  screen = pygame.display.set_mode(size)
  clock = pygame.time.Clock()
  return screen, clock


def _render_pygame(obs, env_no, envs, screen, clock, fps):
  # Could be useful to render observations in some cases...
  # image = obs['image']

  # Rendering environment.
  env = envs._envs[env_no]
  image_future = env.render()
  image = image_future()
  image = image.transpose((1, 0, 2))
  surface = pygame.surfarray.make_surface(image)
  surface = pygame.transform.scale(surface, screen.get_size())
  screen.blit(surface, (0, 0))
  pygame.display.flip()
  clock.tick(fps)


def main(argv=None):
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  config = embodied.Config(dreamerv3.agent.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(dreamerv3.agent.Agent.configs[name])
  config = embodied.Flags(config).parse(other)
  args = embodied.Config(
    **config.run, logdir=config.logdir,
    batch_steps=config.batch_size * config.batch_length)
  print(config)

  step = embodied.Counter()
  logger = dreamerv3.train.make_logger(parsed, config.logdir, step, config)

  # Rendering.
  fps = 30
  screen, clock = _init_pygame()
  on_step_cb = lambda obs, env_no: _render_pygame(obs, env_no, envs, screen, clock, fps)

  cleanup = []
  try:
    envs = dreamerv3.train.make_envs(config)  # mode='eval'
    cleanup.append(envs)
    agent = dreamerv3.agent.Agent(envs.obs_space, envs.act_space, step, config)
    embodied.run.eval_only(agent, envs, logger, args, on_step_cb)
  finally:
    for obj in cleanup:
      obj.close()


if __name__ == '__main__':
  main()
