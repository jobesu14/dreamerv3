
import pygame
import crafter
import warnings
import dreamerv3
from dreamerv3 import embodied
from embodied.envs import from_gym

def main():

    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['medium'])
    config = config.update({
        'run.logdir': '~/logdir/run3',
        'run.script': 'run_eval',
        'run.train_ratio': 64,
        'run.log_every': 30,  # Seconds
        'batch_size': 16,
        'encoder.mlp_keys': '$^',
        'decoder.mlp_keys': '$^',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
        # 'jax.platform': 'cpu',
        'replay_online': True,
    })

    logdir = embodied.Path(config.run.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(logdir.name, config),
        # embodied.logger.MLFlowOutput(logdir.name),
    ])

    env = crafter.Env()  # Replace this with your Gym env.
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, config.wrapper)
    # env = embodied.BatchEnv([env], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay')
    # args = config.run.update(batch_steps=config.batch_size * config.batch_length)
    # embodied.run.train(agent, env, replay, logger, args)

    agent.dataset(replay.dataset) # load weights from pickle?
    # agent.policy_initial(config.batch_size)


    pygame.init()
    size = (512, 512)
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()

    step = 0
    fps = 30
    obs = env._env.reset()
    state = None
    while True:

        # Rendering.
        image = env.render()
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(fps)

        act, state = agent.policy(obs, state, mode='eval')
        obs, reward, done, _ = env.step(act['action'])

        step += 1

if __name__ == '__main__':
  main()
