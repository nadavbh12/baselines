import os
import numpy as np
import tempfile
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent(object):
    def __init__(self,
                 model,
                 double_q=False,    # TODO: change to true
                 memory_args=None,
                 ):
        """Create an RL agent

         Parameters
         -------
         model: torch.nn.Module
         the model that takes the following inputs:
             observation_in: torch.tensor
                 a batch of observations
             num_actions: int
                 number of actions
         and returns a tensor of shape (batch_size, num_actions) with values of every action.
         double_q: bool
            whether to use the double_q update
         memory_args: dict
            dictionary of arguments for the replay buffer
            buffer_size: int
                size of the replay buffer
            prioritized_replay: True
                if True prioritized replay buffer will be used.
            prioritized_replay_alpha: float
                alpha parameter for prioritized replay buffer
            prioritized_replay_beta0: float
                initial value of beta for prioritized replay buffer
            prioritized_replay_beta_iters: int
                number of iterations over which beta will be annealed from initial value
                to 1.0. If set to None equals to max_timesteps.
            prioritized_replay_eps: float
                epsilon to add to the TD errors when updating priorities.
        """
        memory_args = memory_args or {}
        memory_args.setdefault('buffer_size', 50000)
        memory_args.setdefault('prioritized_replay', False)
        if memory_args['prioritized_replay']:
            memory_args.setdefault('prioritized_replay_alpha', 0.6)
            memory_args.setdefault('prioritized_replay_beta0', 0.4)
            memory_args.setdefault('prioritized_replay_beta_iters', None)
            memory_args.setdefault('prioritized_replay_eps', 1e-6)

        # Create the replay buffer
        if memory_args['prioritized_replay']:
            self.replay_buffer = PrioritizedReplayBuffer(memory_args['buffer_size'],
                                                         alpha=memory_args['prioritized_replay_alpha'])
            if memory_args['prioritized_replay_beta_iters'] is None:
                prioritized_replay_beta_iters = memory_args['max_timesteps']
            self.beta_schedule = LinearSchedule(memory_args['prioritized_replay_beta_iters'],
                                           initial_p=memory_args['prioritized_replay_beta0'],
                                           final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(memory_args['buffer_size'])
            self.beta_schedule = None

        self.double_q = double_q

        # Initialize the parameters and copy them to the target network.
        self.net = model
        self.target_net = model
        self._update_target_net()

    def _update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    # @staticmethod
    # def load(path, num_cpu=16):
    #     """Load act function that was returned by learn function.
    #
    #     Parameters
    #     ----------
    #     path: str
    #         path to the act function pickle
    #     num_cpu: int
    #         number of cpus to use for executing the policy
    #
    #     Returns
    #     -------
    #     act: ActWrapper
    #         function that takes a batch of observations
    #         and returns actions.
    #     """
    #     # TODO: implement
    #     return Agent()

    # def save(self, path):
    #     # TODO: implement
    #     pass

    def act(self, observation, epsilon):
        """Chooses an action according to policy:

        Parameters
        ----------
        observation: torch.Tensor
            the observation from the environment
        epsilon: float
            probability for choosinga  random action

        Returns
        -------
        action: int
            The index of the selected action
        """
        if np.random.uniform() < epsilon:
            return np.random.randint(0, self.net.num_actions)
        else:
            # TODO return argmax!
            _, action = self.net(Variable(torch.from_numpy(observation).float(), volatile=False)).max()
            return action

    def _backward(self, obses_t, actions, rewards, obses_tp1, terminals, weights):
        """Takes a transition (s,a,r,s') and optimizes Bellman equation's error:

            td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
            loss = huber_loss[td_error]

        Parameters
        ----------
        obs_t: object
            a batch of observations
        action: np.array
            actions that were selected upon seeing obs_t.
            dtype must be int32 and shape must be (batch_size,)
        reward: np.array
            immediate reward attained after executing those actions
            dtype must be float32 and shape must be (batch_size,)
        obs_tp1: object
            observations that followed obs_t
        terminals: np.array
            1 if obs_t was the last observation in the episode and 0 otherwise
            obs_tp1 gets ignored, but must be of the valid shape.
            dtype must be float32 and shape must be (batch_size,)
        weight: np.array
            imporance weights for every element of the batch (gradient is multiplied
            by the importance weight) dtype must be float32 and shape must be (batch_size,)

        Returns
        -------
        td_error: np.array
            a list of differences between Q(s,a) and the target in Bellman's equation.
            dtype is float32 and shape is (batch_size,)
        """
        # TODO: handle GPU
        S = Variable(torch.Tensor(obses_t), requires_grad=False)
        a = Variable(torch.Tensor(actions), requires_grad=False)
        S2 = Variable(torch.Tensor(obses_tp1), requires_grad=False)
        t = Variable(torch.Tensor(terminals), requires_grad=False)

        # Q'(S_t+1, a)
        Qp = self.target_net(S2, volatile=True)
        if self.double_q:
            # TODO: don't forward terminal states
            _, argmax_q = self.net(S2, volatile=False)
            y = rewards + self.gamma*Qp[argmax_q]
        else:
            y = rewards + self.gamma*Qp.max()
        y.masked_scatter_(t, rewards)

        Q = self.net(S)[a]

        # clean previous gradients
        # huber loss
        loss = F.smooth_l1(y, Q)
        self.optimizer.zero_grad()
        torch.autograd.backward(loss)
        self.optimizer.step()


    def train(self,
              env,
              optimizer=optim.RMSprop,
              optimizer_args=None,
              max_timesteps=100000,
              exploration_fraction=0.1,
              exploration_final_eps=0.02,
              train_freq=1,
              batch_size=32,
              print_freq=1,
              checkpoint_freq=10000,
              learning_starts=1000,
              gamma=1.0,
              target_network_update_freq=500,
              num_cpu=16):
        """Train a deepq model.

        Parameters
        -------
        env : gym.Env
            environment to train on
        optimizer: torch.optim.Optimizer
            optimizer to use for the Q-learning objective.
        optimizer_args: dict
            parameters for the optimizer
        max_timesteps: int
            number of env steps to optimizer for
        exploration_fraction: float
            fraction of entire training period over which the exploration rate is annealed
        exploration_final_eps: float
            final value of random action probability
        train_freq: int
            update the model every `train_freq` steps.
        batch_size: int
            size of a batched sampled from replay buffer for training
        print_freq: int
            how often to print out training progress
            set to None to disable printing
        checkpoint_freq: int
            how often to save the model. This is so that the best version is restored
            at the end of the training. If you do not wish to restore the best version at
            the end of the training set this variable to None.
        learning_starts: int
            how many steps of the model to collect transitions for before learning starts
        gamma: float
            discount factor
        target_network_update_freq: int
            update the target network every `target_network_update_freq` steps.
        num_cpu: int
            number of cpus to use for training
        """

        optimizer_args = optimizer_args or {}
        optimizer_args.setdefault('lr', 5e-4)

        self.gamma = gamma

        self.optimizer = optimizer(self.net.parameters(), optimizer_args)

        # Create the schedule for exploration starting from 1.
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)

        episode_rewards = [0.0]
        saved_mean_reward = None
        obs = env.reset()
        with tempfile.TemporaryDirectory() as td:
            model_saved = False
            model_file = os.path.join(td, "model")
            for t in range(max_timesteps):
                # Take action and update exploration to the newest value
                action = self.act(np.array(obs)[None], epsilon=exploration.value(t))
                new_obs, reward, terminal, _ = env.step(action)
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(terminal))
                obs = new_obs

                episode_rewards[-1] += reward
                if terminal:
                    obs = env.reset()
                    episode_rewards.append(0.0)

                if t > learning_starts and t % train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(batch_size)
                    # TODO: handle prioritized

                    self._backward(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                    # td_errors = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                if t > learning_starts and t % target_network_update_freq == 0:
                    # Update target network periodically.
                    self._update_target_net()

                mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
                num_episodes = len(episode_rewards)
                if terminal and print_freq is not None and len(episode_rewards) % print_freq == 0:
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                    logger.dump_tabular()

                if (checkpoint_freq is not None and t > learning_starts and
                        num_episodes > 100 and t % checkpoint_freq == 0):
                    if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                        if print_freq is not None:
                            logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                       saved_mean_reward, mean_100ep_reward))
                        U.save_state(model_file)
                        model_saved = True
                        saved_mean_reward = mean_100ep_reward
            if model_saved:
                if print_freq is not None:
                    logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
                U.load_state(model_file)
