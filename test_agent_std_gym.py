import pygame
import crafter

size = (512, 512)
env = crafter.Env(size=size)

pygame.init()
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
  
done = False
step = 0
fps = 30
env.reset()
while True:

    # Rendering.
    image = env.render()
    surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(fps)

    _, _, done, _ = env.step(env.action_space.sample())

    step += 1
