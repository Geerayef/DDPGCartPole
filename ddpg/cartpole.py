import pygame
import numpy as np
import time
from tkinter import filedialog, Tk
from .ddpg import DDPG
from env.scenery import Scenery
from util.flags import TRACE, RECORD, SAVE_NEW_WEIGHTS


def new_ddpg(episodes):
    return DDPG(
        num_inputs=4,
        num_outputs=1,
        noise=[1],
        tau=0.01,
        actor_layers=[64, 32],
        critic_layers=[64, 32],
        memory_size=8192,
        noise_decay=episodes,
        my_ou=True,
    )


# ~  Simulator

resolution = (1000, 300)
pygame.init()
pygame.display.set_caption("Cart-pole simulator")

surface = pygame.display.set_mode(resolution)
scenery = Scenery(surface)
clock = pygame.time.Clock()


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


# ~  Algorithm

run = True

episodes = 2500
episodes_count = 0
episode_steps = 0
repetitions = 1
n = 0

# Metrics
ep_reward = 0
rewards_episodes = np.zeros(episodes)
# rewards_avg_step = np.zeros(episodes)
# pos_avg_episodes   = np.zeros(episodes)
# ep_pos = 0
# angle_avg_episodes = np.zeros(episodes)
# ep_angle = 0

agent = new_ddpg(episodes)
if agent.load_weights("cartpole-model"):
    print("~~~~~ Weights loaded")

scenery.reset()
state = scenery.get_current_state()
if TRACE:
    print(f"~~~~~ Initial state: {state}")


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
    action = agent.action(state, agent.episode_counter)

    scenery._apply_action(action[0])
    scenery.tick(dt / 1000.0, episode_steps)

    state, step_reward, terminated = scenery.post_tick()

    agent.feed(action, step_reward, state)
    agent.train()
    agent.soft_update_target_networks()

    ep_reward += step_reward
    # ep_pos += state[0]
    # ep_angle += state[2]
    episode_steps += 1

    scenery.draw()

    if terminated:
        rewards_episodes[episodes_count] += ep_reward
        # rewards_avg_step[episodes_count] += ep_reward / episode_steps
        # pos_avg_episodes[episodes_count] += ep_pos / episode_steps
        # angle_avg_episodes[episodes_count] += ep_angle / episode_steps

        episodes_count += 1
        agent.episode_counter += 1
        episode_steps = 0
        ep_reward = 0
        # ep_pos = 0
        # ep_angle = 0

        scenery.reset()
        state = scenery.get_current_state()

        # Hard update target networks
        # agent.update_target_networks()

        if episodes_count >= episodes:
            n += 1
            if n >= repetitions:
                break

            print(f"~~~~~ Repetition {n}")
            agent = new_ddpg()
            episodes_count = 0

    text = font.render("Ep. reward: %.1f" % rewards_episodes[episodes_count - 1], True, (255, 255, 255))
    surface.blit(text, (5, 25))
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

print("~~~~~ Rewards per episode")
print("~~~~~ Start of results:")
for i in range(episodes):
    print(rewards_episodes[i])
print("~~~~~ End of results.")

if SAVE_NEW_WEIGHTS:
    agent.save_weights("cartpole-model")
