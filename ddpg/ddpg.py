import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tf_agents.utils import common
from util.ornstein_uhlenbeck import OUNoise
from util.gaussian import GaussianNoise


class DDPG:
    def __init__(
            self, num_inputs, num_outputs, seed,
            tau, gamma, batch_size, memory_size,
            noise_decay, my_ou,
            actor_layers, critic_layers,
            actor_lr, critic_lr
    ):
        assert num_inputs > 0
        assert num_outputs > 0
        assert tau > 0
        assert gamma < 1
        assert gamma is not None
        assert batch_size > 0
        assert batch_size is not None
        assert len(actor_layers) > 0
        assert len(critic_layers) > 0

        self._tau = tau
        self._gamma = gamma
        self._batch_size = batch_size
        self._memory_size = memory_size
        self._episode_counter = 0

        tf.random.set_seed(seed)
        np.random.seed(seed)

        if my_ou == 0:
            self.gauss = GaussianNoise(num_outputs, (3 * noise_decay)/4)
        elif my_ou == 1:
            self.myOUNoise = OUNoise(action_space_size=1, decay_period=noise_decay)
        elif my_ou == 2:
            self.OU = common.ornstein_uhlenbeck_process(
                    initial_value=0.0,
                    damping=0.15,
                    stddev=0.2,
                    seed=np.random.normal(),
                    scope='ornstein_uhlenbeck_noise'
                    )

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
        self._memory_index = 0
        self._previous_state = []

    def action(self, state, ep_count):
        action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32)).numpy()[0] 

        if ep_count > self._episode_counter:
            self._episode_counter += 1
            self.gauss.reset()

        for i in range(len(action)):
            action[i] = np.clip(action[i], -1, 1)
            # action[i] += self.OU.__call__()
            # action[i] = self.myOUNoise.get_action(action[i], ep_count)
            action[i] += self.gauss.noise()
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

    def train(self):
        if len(self._memory) < 2 * self._batch_size:
            return

        # Select a random batch.
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        for (state, action, reward, next_state) in random.sample(self._memory, self._batch_size):
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append([reward])
            next_state_batch.append(next_state)

        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)

        # Train the critic.
        with tf.GradientTape() as tape:
            critic_values = self.critic(tf.concat([state_batch, action_batch], axis=1), training=True)
            target_critic_values = reward_batch + self._gamma * self.target_critic(
                tf.concat([next_state_batch, self.target_actor(next_state_batch)], axis=1)
            )
            critic_loss = tf.math.reduce_mean(tf.math.square(target_critic_values - critic_values))
            critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        if self._episode_counter % 2 == 0:
            # Train the actor.
            with tf.GradientTape() as tape:
                critic_values = self.critic(
                    tf.concat(
                        [state_batch, self.actor(state_batch, training=True)], axis=1), training=True
                        )
                actor_loss = -tf.math.reduce_mean(critic_values)
                actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            # Soft update actor
            for (a, b) in zip(self.target_actor.trainable_variables, self.actor.trainable_variables):
                a.assign(self._tau * b + (1 - self._tau) * a)

            # Soft update critic
            for (a, b) in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
                a.assign(self._tau * b + (1 - self._tau) * a)

    def update_target_networks(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def soft_update_target_networks(self):
        # Soft update actor
        for (a, b) in zip(self.target_actor.trainable_variables, self.actor.trainable_variables):
            a.assign(self._tau * b + (1 - self._tau) * a)

        # Soft update critic
        for (a, b) in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
            a.assign(self._tau * b + (1 - self._tau) * a)

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
