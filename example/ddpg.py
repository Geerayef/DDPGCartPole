import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


class DDPG:
    def __init__(
            self, num_inputs, num_outputs, noise,
            actor_layers, critic_layers, memory_size,
            actor_lr=0.001, critic_lr=0.001
    ):
        assert num_inputs > 0
        assert num_outputs > 0
        assert len(noise) == num_outputs
        assert len(actor_layers) > 0
        assert len(critic_layers) > 0

        self._noise = noise

        # Construct the actor.
        self.actor = Sequential()
        self.actor.add(Dense(actor_layers[0], input_shape=(num_inputs,)))
        for i in range(1, len(actor_layers)):
            self.actor.add(Activation('relu'))
            self.actor.add(Dense(actor_layers[i]))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(num_outputs))
        self.actor.add(Activation('tanh'))

        # Construct the critic.
        self.critic = Sequential()
        self.critic.add(
            Dense(critic_layers[0], input_shape=(num_inputs + num_outputs,)))
        for i in range(1, len(critic_layers)):
            self.critic.add(Activation('relu'))
            self.critic.add(Dense(critic_layers[i]))
        self.critic.add(Activation('relu'))
        self.critic.add(Dense(1))
        self.critic.add(Activation('linear'))

        # Clone the actor and the critic as their target models.
        self.target_actor = clone_model(self.actor)
        self.target_critic = clone_model(self.critic)

        # Construct the optimizers.
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)

        # Initialize the replay memory.
        self._memory = []
        self._memory_size = memory_size
        self._memory_index = 0
        self._previous_state = []

    def load_weights(self, filename):
        try:
            self.actor.load_weights(filename + "-actor.h5")
            self.critic.load_weights(filename + "-critic.h5")
            self.target_actor.load_weights(filename + "-actor.h5")
            self.target_critic.load_weights(filename + "-critic.h5")
            return True
        except:
            return False

    def save_weights(self, filename):
        self.actor.save_weights(filename + "-actor.h5")
        self.critic.save_weights(filename + "-critic.h5")

    def action(self, state):
        # Let the actor return an action for the given state.
        action = self.actor(tf.convert_to_tensor(
            [state], dtype=tf.float32)).numpy()[0]

        # Add noise.
        for i in range(len(action)):
            action[i] += np.random.normal(0, self._noise[i])
            action[i] = np.clip(action[i], -1, 1)

        return action

    def feed(self, action, reward, new_state):
        if self._previous_state != []:
            if len(self._memory) < self._memory_size:
                self._memory.append(
                    (self._previous_state, action, reward, new_state))
            else:
                self._memory[self._memory_index] = (
                    self._previous_state, action, reward, new_state)
            self._memory_index = (self._memory_index + 1) % self._memory_size

        self._previous_state = new_state

    def train(self, batch_size=32, gamma=0.99):
        # Are there enough samples for a batch?
        if len(self._memory) < batch_size:
            return

        # Select a random batch.
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        for (state, action, reward, next_state) in random.sample(self._memory, batch_size):
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append([reward])
            next_state_batch.append(next_state)

        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(
            next_state_batch, dtype=tf.float32)

        # Train the actor.
        with tf.GradientTape() as tape:
            critic_values = self.critic(
                tf.concat(
                    [
                        state_batch,
                        self.actor(state_batch, training=True)
                    ],
                    axis=1
                ),
                training=True
            )
            target_critic_values = reward_batch + gamma * self.target_critic(
                tf.concat(
                    [
                        next_state_batch,
                        self.target_actor(next_state_batch, training=True)
                    ],
                    axis=1
                ),
                training=True
            )
            critic_loss = tf.math.reduce_mean(
                tf.math.square(target_critic_values - critic_values))
            critic_gradients = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_gradients, self.critic.trainable_variables))

    def update_target_networks(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
