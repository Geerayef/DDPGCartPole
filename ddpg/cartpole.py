import pygame
import numpy as np
import time
from tkinter import filedialog, Tk
from .ddpg import DDPG
from env.scenery import Scenery
from util.flags import TRACE, RECORD, SAVE_NEW_WEIGHTS, PREFILL_MEMORY


# ~  Constants
MAX_STEPS = 500
EPISODES = 2000
MEMORY_SIZE = 65536
WINDOW_DIM = (1000, 300)


# -------------------------------------------------------------------------------- #

# ~ Util Functions

def handle_recording():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                if scenery.is_recording():
                    scenery.stop_recording()
                else:
                    scenery.start_recording()
            elif event.key == pygame.K_p:
                if scenery.is_playing():
                    scenery.stop_playing()
                else:
                    scenery.start_playing(filedialog.askopenfilename())
            elif event.key == pygame.K_c:
                filename = filedialog.askopenfilename()
                if filename != "":
                    surface.fill((0, 0, 0))
                    msg = "Converting video ..."
                    (width, height) = font.size(msg)
                    text = font.render(msg, True, (255, 255, 255))
                    surface.blit(
                            text,
                            ((surface.get_width() - width) / 2,
                             (surface.get_height() - height) / 2)
                            )
                    pygame.display.update()
                    scenery.convert_recording(filename)


def init_simulator(clock, surface, font):
    clock = pygame.time.Clock()
    pygame.display.set_caption("Cart-pole simulator")
    surface = pygame.display.set_mode(WINDOW_DIM)

    Tk().withdraw()

    if pygame.font.match_font("Monospace", True):
        font = pygame.font.SysFont("Monospace", 20, True)
    elif pygame.font.match_font("Courier New", True):
        font = pygame.font.SysFont("Courier New", 20, True)
    else:
        font = pygame.font.Font(None, 20)

    return clock, surface, font


def fill_memory(memory_size, scenery, agent, clock):
    i = 0
    j = 0
    print("~~~~~ Filling memory")
    while (len(agent._memory) < memory_size):
        pygame.event.pump()
        dt = clock.get_time()
        action = agent.action(state, i)
        scenery._action += action[0]
        scenery.tick(dt / 1000.0, j)
        x, y, terminated = scenery.post_tick(j, action)
        agent.feed(action, y, x)

        j += 1
        if terminated:
            i += 1
            j = 0
            scenery.reset()
            x = scenery.get_current_state()

        clock.tick(50)

    scenery.reset()
    print(f"~~~~~ Memory filled: {len(agent._memory)}")

    return scenery, agent


# -------------------------------------------------------------------------------- #

# ~  Simulator

pygame.init()

if PREFILL_MEMORY:
    clock = None
    surface = None
    font = None
else:
    clock = pygame.time.Clock()
    pygame.display.set_caption("Cart-pole simulator")
    surface = pygame.display.set_mode(WINDOW_DIM)

    Tk().withdraw()

    if pygame.font.match_font("Monospace", True):
        font = pygame.font.SysFont("Monospace", 20, True)
    elif pygame.font.match_font("Courier New", True):
        font = pygame.font.SysFont("Courier New", 20, True)
    else:
        font = pygame.font.Font(None, 20)

sum_dt = 0
fps = 0
sum_fps = 0
frame_count = 0
avg_fps = 0


# -------------------------------------------------------------------------------- #

# ~  Algorithm

def new_ddpg():
    return DDPG(
        num_inputs=4,
        num_outputs=1,
        seed=1,
        tau=0.01,
        gamma=0.97,
        batch_size=256,
        memory_size=MEMORY_SIZE,
        noise_decay=EPISODES,
        my_ou=0,    # 0 - Gauss, 1 - My OU, 2 - tf-agents OU
        actor_layers=[128, 32],
        critic_layers=[128, 32],
        actor_lr=0.0002,
        critic_lr=0.0003
    )


run = True
episodes_count = 0
episode_steps = 0
repetitions = 1
n = 0

# Metrics
ep_reward = 0
rewards_episodes = np.zeros(EPISODES)
rewards_avg_step = rewards_episodes
pos_avg_episodes = rewards_episodes
ep_pos = 0
angle_avg_episodes = rewards_episodes
ep_angle = 0


# ~  Agent

agent = new_ddpg()

# ~  Fill memory before training

if PREFILL_MEMORY:
    clock, surface, font = init_simulator(clock, surface, font)
    scenery = Scenery(MAX_STEPS, surface)
    scenery.reset()
    state = scenery.get_current_state()

    scenery, agent = fill_memory(MEMORY_SIZE, scenery, agent, clock)
    print(f"~~~~~ Resetting environment for training with full memory")

# ~ Set up for training

scenery = Scenery(MAX_STEPS, surface)
scenery.reset()
state = scenery.get_current_state()
if TRACE:
    print(f"~~~~~ Initial state: {state}")

if agent.load_weights("cartpole-model"):
    print("~~~~~ Weights loaded")


# -------------------------------------------------------------------------------- #

# ~  Main Loop

while run:
    pygame.event.pump()

    dt = clock.get_time()
    frame_count += 1
    sum_dt += dt
    if dt > 0:
        fps = 1000.0 / dt
    sum_fps += fps
    if sum_dt >= 100:
        avg_fps = sum_fps / frame_count
        sum_fps = 0
        frame_count = 0
        sum_dt = 0

    # ~  Main algorithm
    action = agent.action(state, episodes_count)

    scenery._apply_action(action[0])
    scenery.tick(dt / 1000.0, episode_steps)
    state, step_reward, terminated = scenery.post_tick(episode_steps, action)

    agent.feed(action, step_reward, state)
    agent.train()

    ep_reward += step_reward
    ep_pos += state[0]
    ep_angle += state[2]
    episode_steps += 1

    scenery.draw()

    if terminated:
        rewards_episodes[episodes_count] += ep_reward
        rewards_avg_step[episodes_count] += ep_reward / episode_steps
        pos_avg_episodes[episodes_count] += ep_pos / episode_steps
        angle_avg_episodes[episodes_count] += ep_angle / episode_steps

        episodes_count += 1
        # agent.episode_counter = episodes_count
        episode_steps = 0
        ep_reward = 0
        ep_pos = 0
        ep_angle = 0

        scenery.reset()
        state = scenery.get_current_state()

        if episodes_count >= EPISODES:
            n += 1
            if n >= repetitions:
                break

            print(f"~~~~~ Repetition {n}")
            agent = new_ddpg()
            episodes_count = 0

    # text = font.render("Max reward: %.1f" % np.max(rewards_episodes), True, (255, 255, 255))
    # surface.blit(text, (775, 5))
    # text = font.render("Ep. reward: %.1f" % rewards_episodes[episodes_count - 1], True, (255, 255, 255))
    # surface.blit(text, (5, 25))
    text = font.render("Episode: %d" % episodes_count, True, (255, 255, 255))
    surface.blit(text, (5, 5))
    text_y = 5
    if pygame.time.get_ticks() % 1000 <= 500:
        msg = ""
        color = (255, 255, 255)
        if scenery.is_recording():
            msg = "RECORDING"
            color = (255, 64, 64)
        elif scenery.is_playing():
            msg = "PLAYING"
            color = (64, 255, 64)
        (width, height) = font.size(msg)
        text = font.render(msg, True, color)
        surface.blit(text, (surface.get_width() - width - 5, text_y))
        text_y += height

    pygame.display.update()

    if RECORD:
        handle_recording()

    if TRACE:
        print(f"~~~~~ Action to apply   : {action}")
        print(f"~~~~~ Episode reward    : {ep_reward}")
        print(f"~~~~~ Action post apply : {scenery._action}")
        print(f"~~~~~ State after tick  : {state}")

    clock.tick(50)
    time.sleep(0.02)

pygame.quit()

# print("~~~~~ DONE ~~~~~")
# print("~~~~~ Rewards per episode")
# print("~~~~~ Start of results:")
# for i in range(EPISODES):
#     print(rewards_episodes[i])
# print("~~~~~ End of results.")

np.save("reward_step.npy", rewards_avg_step)
np.save("pos_avg.npy", pos_avg_episodes)
np.save("angle_avg.npy", angle_avg_episodes)

if SAVE_NEW_WEIGHTS:
    agent.save_weights("cartpole-model")
