import numpy as np
import pygame
import re
import warnings

import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym
from dreamerv3.embodied.core.basics import convert


def _init_pygame():
    pygame.init()
    size = (512, 512)
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    return screen, clock


def _render_obs_pygame(obs, env_no, envs, screen, clock, fps):
    # Could be useful to render observation in some case...
    # image = obs['image']
    
    # Rendering environment.
    env = envs._envs[env_no]
    image_future = env.render()
    image = image_future()
    image = image.transpose((1, 0, 2))
    # image = image.repeat(8, axis=0).repeat(8, axis=1)
    surface = pygame.surfarray.make_surface(image)
    surface = pygame.transform.scale(surface, screen.get_size())
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(fps)

def _eval_only(agent, envs, logger, args):

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print('Logdir', logdir)
    should_log = embodied.when.Clock(args.log_every)
    step = logger.step
    metrics = embodied.Metrics()
    print('Observation space:', envs.obs_space)
    print('Action space:', envs.act_space)

    timer = embodied.Timer()
    timer.wrap('agent', agent, ['policy'])
    timer.wrap('env', envs, ['step'])
    timer.wrap('logger', logger, ['write'])

    nonzeros = set()
    def per_episode(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        logger.add({'length': length, 'score': score}, prefix='episode')
        print(f'Episode has {length} steps and return {score:.1f}.')
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f'policy_{key}'] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f'sum_{key}'] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f'mean_{key}'] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f'max_{key}'] = ep[key].max(0).mean()
        metrics.add(stats, prefix='stats')

    driver = embodied.Driver(envs)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())

    screen, clock = _init_pygame()
    driver.on_step(lambda obs, env_no: _render_obs_pygame(obs, env_no, envs, screen, clock, 30))

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=['agent'])

    print('Start evaluation loop.')
    policy = lambda *args: agent.policy(*args, mode='eval')

    while step < args.steps:
        driver(policy, steps=100)
        if should_log(step):
            logger.add(metrics.result())
            logger.add(timer.stats(), prefix='timer')
            logger.write(fps=True)
    logger.write()


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

    cleanup = []
    try:
        envs = dreamerv3.train.make_envs(config)  # mode='eval'
        cleanup.append(envs)
        agent = dreamerv3.agent.Agent(envs.obs_space, envs.act_space, step, config)
        _eval_only(agent, envs, logger, args)
    finally:
        for obj in cleanup:
            obj.close()
