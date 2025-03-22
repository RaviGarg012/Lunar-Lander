import tensorflow as tf
import random
import numpy as np

MINIBATCH_SIZE = 64
E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy.
E_MIN = 0.1  # Minimum ε value for the ε-greedy policy.
random.seed(42)
def get_experiences(memory_buffer):
    """
    Returns a random sample of experience tuples drawn from the memory buffer.

    Returns:
        A tuple (states, actions, rewards, next_states, done_vals) where:

    """

    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
        dtype=tf.float32,
    )
    return (states, actions, rewards, next_states, done_vals)

def get_new_eps(epsilon):
    # update the value of epsilon
    return max(E_MIN, E_DECAY*epsilon)

