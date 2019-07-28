from collections import deque

import numpy as np
import pickle

from baselines.her.util import convert_episode_to_batch_major, store_args
from ipdb import set_trace

SYM_PLANE_Y = 0.48 * 2
BOOL_SYM = True
class RolloutWorker:

    @store_args
    def __init__(self, venv, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, monitor=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """

        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()

    def reset_all_rollouts(self):
        self.obs_dict = self.venv.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal']
        # set_trace()

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        dones = []
        info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        # ------------------symmetry-------------------------
        s_obs, s_achieved_goals, s_acts, s_goals = [], [], [], []
        s_info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        # ----------------end---------------------------
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations, do not return the reward, and get it from her_sampler.py
            obs_dict_new, _, done, info = self.venv.step(u)
            # set_trace()
            o_new = obs_dict_new['observation']
            ag_new = obs_dict_new['achieved_goal']
            success = np.array([i.get('is_success', 0.0) for i in info])

            # no need
            if any(done):
                # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
                # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
                # after a reset
                break
            # no need
            for i, info_dict in enumerate(info):
                for idx, key in enumerate(self.info_keys):
                    info_values[idx][t, i] = info[i][key]
            # no need
            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            dones.append(done)
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())

            # ----------------symmetry---------------------------
            s_obs.append(self.sym_plane_compute(o.copy()))
            s_achieved_goals.append(self.sym_plane_compute(ag.copy()))
            s_acts.append(self.sym_plane_compute(u.copy()))
            s_goals.append(self.sym_plane_compute(self.g.copy()))
            # ----------------end---------------------------

            o[...] = o_new
            ag[...] = ag_new

        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        # ----------------symmetry---------------------------
        s_obs.append(self.sym_plane_compute(o.copy()))
        s_achieved_goals.append(self.sym_plane_compute(ag.copy()))
        # ----------------end---------------------------


        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value


        # ----------------symmetry---------------------------
        s_episode = dict(o=s_obs,
                       u=s_acts,
                       g=s_goals,
                       ag=s_achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            s_episode['info_{}'.format(key)] = value
        # ----------------end---------------------------

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))

        # -------symmetry-------: original one no need to *2
        if BOOL_SYM:
            mul_factor = 2
        else:
            mul_factor = 1
        self.n_episodes += (mul_factor* self.rollout_batch_size)
        # ----------------end---------------------------


        # return dict: ['o', 'u', 'g', 'ag', 'info_is_success']
        episode_batch = convert_episode_to_batch_major(episode)


        # ----------------symmetry---------------------------
        if BOOL_SYM:
            s_episode_batch = convert_episode_to_batch_major(s_episode)
            return episode_batch, s_episode_batch
        # ----------------end---------------------------

        # set_trace()
        return episode_batch

    def sym_plane_compute(self,param):
        param_len = len(param[0])
        if param_len == 3:    #goal & achieved goal
            param[0][1] = SYM_PLANE_Y - param[0][1]

        elif param_len == 4:  #action
            param[0][1] = SYM_PLANE_Y - param[0][1]

        elif param_len == 10:     # observation without object
            param[0][1] = SYM_PLANE_Y - param[0][1]
            # vel do not need SYM_PLANE_Y
            param[0][4] = -param[0][4]
            
        elif param_len == 25:     # observation with object

            # sym_grip_pos
            param[0][1] = SYM_PLANE_Y - param[0][1]
            # sym_obj_pos
            param[0][4] = SYM_PLANE_Y - param[0][4]

            # sym_obj_rel_pos
            param[0][6] = param[0][3]-param[0][0]
            param[0][7] = param[0][4]-param[0][1]
            param[0][8] = param[0][5]-param[0][2]

            # sym_obj_rot_euler
            param[0][11] = -param[0][11]
            param[0][13] = -param[0][13]

            # get the original obj_velp first
            param[0][14] += param[0][20]
            param[0][15] += param[0][21]
            param[0][16] += param[0][22]
            # get the sym_obj_velp & sym_grip_velp
            param[0][15] = -param[0][15]
            param[0][21] = -param[0][21]
            # get the sym_obj_rel_velp
            param[0][14] -= param[0][20]
            param[0][15] -= param[0][21]
            param[0][16] -= param[0][22]

            # sym_obj_velr
            param[0][17] = -param[0][17]
            param[0][19] = -param[0][19]


        
        return param.copy()


    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

