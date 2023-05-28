import gymnasium
import numpy as np
from .ddpg import DDPG
import tensorflow as tf
import math
import time


def process_state(state):
    print(f"Pre- State: {state}")
    if len(state) <= 2:
        (x, y, thetadot), placeholder = state
    else:
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
        memory_size=100000,
    )


env = gymnasium.make("Pendulum-v1", render_mode="rgb_array_list")
ddpg = new_ddpg()

if ddpg.load_weights("pendulum-model"):
    print("Weights loaded.")

episode_count = 0
episode_steps = 0
episode_reward = 0

state = process_state(env.reset())
print(f"Initial state: {state}")

episodes = 100
repetitions = 1
n = 0

rewards = np.zeros(episodes)

while True:
    env.render()

    action = ddpg.action(state)

    print(f"Action to be stepped {action}")

    state, reward, terminate, truncated, info = env.step(2.0 * action)
    state = process_state(state)
    print(f"Post- State: {state}")

    episode_steps += 1
    episode_reward += reward

    ddpg.feed(action, reward, state)
    ddpg.train()

    if terminate or truncated:
        print(
            "Episode {} finished in {} steps, average reward = {}".format(
                episode_count, episode_steps, episode_reward / episode_steps
            )
        )

        rewards[episode_count] += episode_reward / episode_steps

        episode_count += 1
        episode_steps = 0
        episode_reward = 0
        locked_direction = 0

        state = process_state(env.reset())
        ddpg.update_target_networks()

        if episode_count >= episodes:
            n += 1
            if n >= repetitions:
                break

            print("Repetition", n)
            ddpg = new_ddpg()
            episode_count = 0

    time.sleep(0.02)

for i in range(episodes):
    print(rewards[i] / n)

ddpg.save_weights("pendulum-model")
