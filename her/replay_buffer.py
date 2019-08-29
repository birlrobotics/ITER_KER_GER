import threading

import numpy as np
from ipdb import set_trace
from baselines.her.mirror_learning_method import LOWER_MEMORY_BOUND,UPPER_EXTRACT_PROB_BOUND,INCREASE_RATE,START_EPOCH


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in episodes
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        # Buffer size measured by episode
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        # current_size is measured by episode, n_transitions_stored is measured by transition
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()
        # delay memory 
        self.dm_lower_memory_bound = LOWER_MEMORY_BOUND
        self.dm_increase_rate = INCREASE_RATE
        self.dm_start_epoch = START_EPOCH
        self.dm_upper_extract_prob_bound = UPPER_EXTRACT_PROB_BOUND * 100

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        # here we augmented the original transition with HER 
        transitions = self.sample_transitions(buffers, batch_size)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch, epoch_inx):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        # single episode here: batch_size = 1
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size,epoch_inx)

            # load inputs into buffers
            # buffer.keys = (['o', 'u', 'g', 'info_is_success', 'ag'])
            # self.buffer is the replay buffer with dict. Each key is the and restore the input episode one by one in order.
            for key in self.buffers.keys():
                # np.shape(self.buffers[key][idxs]) = (50, 25), where buffer.keys = (['o', 'u', 'g', 'info_is_success', 'ag'])
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None, epoch_inx=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            # set_trace()
            # randomly drop out the buffer samples
            if epoch_inx>self.dm_start_epoch:
                dm_lower_extract_prob_bound = epoch_inx * self.dm_increase_rate
                if np.random.randint(0, 100, 1) < min(dm_lower_extract_prob_bound,self.dm_upper_extract_prob_bound):
                    idx = np.random.randint(self.dm_lower_memory_bound, self.size, inc)
                else:
                    idx = np.random.randint(0, self.size, inc)
            else:
                # set_trace()
                idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx

        self.dm_lower_bound_memory = LOWER_BOUND_MEMORY
        self.dm_increase_rate = INCREASE_RATE
        self.dm_start_epoch = START_EPOCH