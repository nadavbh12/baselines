import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import logging.config
import gym
import logging

from baselines.common import set_global_seeds
from baselines.a2c.policies import mlp
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common.classic_control_wrappers import CartPoleNumpyWrapper, MountainCarNumpyWrapper
from DL_Logger.utils import AverageMeter


class A2CActor:
    def __init__(self, results, save_path, lr):
        """
        Parameters
        ----------
        results: DL_logger.ResultsLog
            class to log results
        save_path: string
            path where results are saved
        lr: float
            learning rate
        """
        # self.net = policy_network
        self.T = 0
        self.lr = lr
        self.results = results
        self.save_path = save_path

    def create_env_vec(self, env_id, seed, num_workers):

        # divide by 4 due to frameskip, then do a little extras so episodes end
        def make_env(rank):
            def _thunk():
                env = gym.make(env_id)
                env.seed(seed + rank)
                env = bench.Monitor(env, self.save_path and
                                    os.path.join(self.save_path, "{}.monitor.json".format(rank)))
                if env_id.startswith('CartPole'):
                    env = CartPoleNumpyWrapper(env)
                elif env_id.startswith('MountainCar'):
                    env = MountainCarNumpyWrapper(env)
                #     TODO: handle all other envs
                else:
                    env = wrap_deepmind(env)
                return env

            return _thunk

        set_global_seeds(seed)
        env = SubprocVecEnv([make_env(i) for i in range(num_workers)])
        return env

    def train(self, env_id, seed, num_workers, max_timesteps, gamma, ent_coef, value_coef, update_frequency, max_episode_len,
              max_grad_norm, optimizer=None):
        """Performs training of an A2C thread
        Parameters
        ----------
        env_id: string
            environment to train on, using Gym's id
        seed: int
            random seed
        num_workers: int
            number of workers
        max_timesteps: int
            total training steps
        gamma: float
            discount factor
        ent_coef: float
            controls the strength of the entropy regularization term
        value_coef: float
            controls the strength of the value loss term
        update_frequency: int
            number of steps in A2C
        max_episode_len: int
            maximum length of an episode
        max_grad_norm: float
            maximum gradient of the norm of the weights
        optimizer: torch.Optimizer
            the network's optimizer
        """

        env = self.create_env_vec(env_id, seed, num_workers)

        # TODO: move the network initialization elsewhere
        self.net = mlp([env.observation_space.shape[0]], env.action_space.n, [64])
        self.net.train()

        if optimizer is None:
            optimizer = optim.RMSprop(self.net.parameters(), self.lr)
        else:
            optimizer = optimizer(self.net.parameters(), self.lr)

        episode_len = 0
        episode_reward = 0
        epoch = 0
        state = env.reset()
        # TODO: handle stacking of frames in ale/rle
        state = torch.from_numpy(state)
        assert(state.shape[0] == num_workers)
        assert(state.shape[1:] == torch.Size(list(env.observation_space.shape)))

        avg_value_estimate = AverageMeter()
        avg_value_loss = AverageMeter()
        avg_policy_loss = AverageMeter()
        avg_entropy_loss = AverageMeter()

        while self.T < max_timesteps:

            # if self.T > 10000:
            #     env.render()
            rewards = []
            values = []
            entropies = []
            log_probs = []
            terminals = []

            for t in range(update_frequency):
                action_prob, value = self.net(Variable(state))
                avg_value_estimate.update(value.data.mean())
                action = action_prob.multinomial().data

                action_log_probs = torch.log(action_prob)
                entropy = -(action_log_probs * action_prob).sum(1)

                state, reward, terminal, info = env.step(action.numpy())
                # TODO: is the code below necessary?
                # for n, done in enumerate(dones):
                #     if done:
                #         self.obs[n] = self.obs[n] * 0
                state = torch.from_numpy(state)

                episode_len += 1

                # save rewards and values for later
                rewards.append(reward)
                terminals.append(terminal)
                values.append(value)
                entropies.append(entropy)
                log_probs.append(action_log_probs.gather(1, Variable(action)))

                self.T += 1

            # Convert lists to torch.Tensor/Variable
            rewards = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).transpose(0, 1)
            terminals = torch.from_numpy(np.asarray(terminals, dtype=np.uint8)).transpose(0, 1)
            values = torch.cat(values, 1)
            entropies = torch.cat(entropies, 0).view(values.size())
            log_probs = torch.cat(log_probs, 0).view(values.size())

            rewards = Variable(rewards, requires_grad=False)

            _, last_value = self.net(Variable(state))
            last_value.squeeze_()
            mask = Variable(torch.ones(terminals.size()) - terminals.float(), requires_grad=False)
            R = Variable(torch.zeros(rewards.size())) # VALIDATE: is this the correct place for Variable()?

            R[:, -1] = last_value * mask[:, -1]  # bootstrap from last state
            for i in reversed(range(update_frequency-1)):
                R[:, i] = (rewards[:, i] + gamma * R[:, i+1])*mask[:, i]

            advantage = R - values
            value_loss = advantage.pow(2)[:, :-1].mean()
            policy_loss = (-advantage*log_probs)[:, :-1].mean()
            entropy_loss = entropies[:, :-1].mean()

            optimizer.zero_grad()

            (policy_loss + value_coef*value_loss + ent_coef*entropy_loss).backward()
            avg_entropy_loss.update(entropy_loss.data[0])
            avg_value_loss.update(value_loss.data[0])
            avg_policy_loss.update(policy_loss.data[0])

            torch.nn.utils.clip_grad_norm(self.net.parameters(), max_grad_norm)
            optimizer.step()

            episode_len = 0
            if self.T % 1000 == 0:
                # save results
                json_results = bench.load_results(self.save_path)
                self.results.add(step=self.T, value=avg_value_estimate.avg(),
                                 avg_entropy_loss=avg_entropy_loss.avg(),
                                 avg_policy_loss=avg_policy_loss.avg(),
                                 avg_value_loss=avg_value_loss.avg(),
                                 time=time.time() - json_results['initial_reset_time'],
                                 mean_reward=np.mean(json_results['episode_rewards'][-10:])
                                 )
                avg_value_estimate.reset()
                avg_value_loss.reset()
                avg_policy_loss.reset()
                avg_entropy_loss.reset()
                # self.results.smooth('reward', window=10)
                # self.results.smooth('value', window=10)
                # self.results.smooth('avg_policy_loss', window=10)
                # self.results.smooth('avg_value_loss', window=10)
                # self.results.smooth('avg_entropy_loss', window=10)
                # self.results.plot(x='step', y='reward_smoothed',
                #                   title='Reward', ylabel='Reward')
                self.results.plot(x='time', y='mean_reward',
                                  title='mean_reward', ylabel='average reward')
                self.results.plot(x='step', y='value',
                                  title='value', ylabel='Avg value estimate')
                self.results.plot(x='step', y='avg_policy_loss',
                                  title='avg_policy_loss', ylabel='avg_policy_loss')
                self.results.plot(x='step', y='avg_value_loss',
                                  title='avg_value_loss', ylabel='avg_value_loss')
                self.results.plot(x='step', y='avg_entropy_loss',
                                  title='avg_entropy_loss', ylabel='avg_entropy_loss')
                self.results.save()

    def save(self):
        # TODO: implemented
        pass

    def load(self):
        # TODO: implemented
        pass