from ddpg import DDPG
from simulator.python.scenery import Scenery
from tkinter import Tk
import math
import numpy as np
import pygame

resolution = (1000, 300)

def process_state(state):
    (x, y, thetadot) = state
    theta = math.atan2(y, x)
    return (theta, thetadot)

def new_ddpg():
    return DDPG(
        num_inputs=2,
        num_outputs=1,
        noise=[0.01],
        actor_layers=[64, 32],
        critic_layers=[64, 32],
        memory_size=100000
    )


# ~  Simulator
Tk().withdraw()
pygame.init()
pygame.display.set_caption("Cart-pole simulator")
surface = pygame.display.set_mode(resolution)
clock = pygame.time.Clock()

if pygame.font.match_font("Monospace", True):
    font = pygame.font.SysFont("Monospace", 20, True)
elif pygame.font.match_font("Courier New", True):
    font = pygame.font.SysFont("Courier New", 20, True)
else:
    font = pygame.font.Font(None, 20)

scenery = Scenery(surface)


# ~  Algorithm
agent = new_ddpg()

if agent.load_weights("pendulum-model"):
    print("Weights loaded.")

episode_count = 0
episode_steps = 0
episode_reward = 0
episodes = 100
repetitions = 1
n = 0

state = process_state(scenery.reset())
rewards = np.zeros(episodes)

run = True
sum_dt = 0
fps = 0
sum_fps = 0
frame_count = 0
avg_fps = 0

# Body
while run:
    frame_count += 1
    dt = clock.get_time()
    sum_dt += dt
    if dt > 0:
        fps = 1000.0 / dt
    sum_fps += fps
    if sum_dt >= 100:
        avg_fps = sum_fps / frame_count
        sum_fps = 0
        frame_count = 0
        sum_dt = 0

    action = agent.action(state)

    new_state, reward, done = scenery.execute_action(action)
    scenery.tick(dt / 1000.0)

    agent.feed(action, reward, new_state)

    agent.train()

    # Update the target networks
    agent.update_target_networks()

    # Check if the episode is done
    if done:
        scenery.reset()
