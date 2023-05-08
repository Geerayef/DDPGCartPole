import gymnasium
import numpy as np
from ddpg import DDPG
import tensorflow as tf
import random, math, time

def process_state(state):
    print(f"The state looks like this :\n - {state}")
    (x, y, z), thetadot = state
    theta = math.atan2(x, y)
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

env = gymnasium.make('Pendulum-v1', render_mode="rgb_array")
ddpg = new_ddpg()

if ddpg.load_weights("pendulum-model"):
    print("Weights loaded.")

episode_count = 0
episode_steps = 0
episode_reward = 0

state = process_state(env.reset())

episodes = 100
repetitions = 1
n = 0

rewards = np.zeros(episodes)

while True:
    env.render()

    action = ddpg.action(state)

    state, reward, done, info = env.step(2.0*action)
    state = process_state(state)

    episode_steps += 1
    episode_reward += reward

    ddpg.feed(action, reward, state)
    ddpg.train()

    if done:
        print("Episode {} finished in {} steps, average reward = {}".format(
            episode_count, episode_steps, episode_reward / episode_steps))
        
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