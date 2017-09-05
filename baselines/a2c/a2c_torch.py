import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.optim as optim
import logging.config

from baselines.a2c.policies import mlp
from DL_Logger.utils import AverageMeter


class A2CActor:
    def __init__(self, results, lr):
        """
        Parameters
        ----------
        results: DL_logger.ResultsLog
            class to logg results
        lr: float
            learning rate
        """
        # self.net = policy_network
        self.T = 0
        self.lr = lr
        self.results = results

    def train(self, env, seed, rank, max_timesteps, gamma, ent_coef, value_coef, update_frequency, max_episode_len,
              max_grad_norm, optimizer=None):
        """Performs training of an A2C thread
        Parameters
        ----------
        env: gym.Env
            environment to train on
        seed: int
            random seed
        rank: int
            the thread number
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

        torch.manual_seed(seed + rank)
        env.seed(seed + rank)
        # TODO: seed CUDA?

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
        state = torch.from_numpy(state)

        avg_value_estimate = AverageMeter()
        avg_value_loss = AverageMeter()
        avg_policy_loss = AverageMeter()
        avg_entropy_loss = AverageMeter()

        while self.T < max_timesteps:

            # if self.T > 10000:
            #     env.render()
#             TODO: synchronize parameters
            rewards = []
            values = []
            entropies = []
            log_probs = []
            terminal = False

            for t in range(update_frequency):
                action_prob, value = self.net(Variable(state))
                avg_value_estimate.update(value.data[0][0])
                action = action_prob.multinomial().data

                action_log_probs = torch.log(action_prob)
                entropy = -(action_log_probs * action_prob).sum(1)

                state, reward, terminal, info = env.step(action.numpy())
                state = torch.from_numpy(state)

                episode_len += 1
                episode_reward += reward

                # save rewards and values for later
                rewards.append(reward)
                values.append(value)
                entropies.append(entropy)
                log_probs.append(action_log_probs.gather(1, Variable(action)))

                self.T += 1
                if terminal or episode_len > max_episode_len:
                    epoch += 1
                    state = env.reset()
                    state = torch.from_numpy(state)

                    try:
                        self.results.add(epoch=epoch, step=self.T, reward=episode_reward, value=avg_value_estimate.avg(),
                                         avg_entropy_loss=avg_entropy_loss.avg(),
                                         avg_policy_loss=avg_policy_loss.avg(),
                                         avg_value_loss=avg_value_loss.avg())
                        avg_value_estimate.reset()
                        avg_value_loss.reset()
                        avg_policy_loss.reset()
                        avg_entropy_loss.reset()
                    #     handle end of episode before net update
                    except ZeroDivisionError:
                        pass

                    episode_reward = 0
                    episode_len = 0
                    if epoch % 100 == 0:
                        self.results.save()
                        self.results.smooth('reward', window=10)
                        self.results.smooth('value', window=10)
                        self.results.smooth('avg_policy_loss', window=10)
                        self.results.smooth('avg_value_loss', window=10)
                        self.results.smooth('avg_entropy_loss', window=10)
                        logging.info('Epoch {} reward = {}'.format(epoch, reward))
                        self.results.plot(x='step', y='reward_smoothed',
                                     title='Reward', ylabel='Reward')
                        self.results.plot(x='step', y='value_smoothed',
                                     title='value', ylabel='Avg value estimate')
                        self.results.plot(x='step', y='avg_policy_loss_smoothed',
                                     title='avg_policy_loss', ylabel='avg_policy_loss')
                        self.results.plot(x='step', y='avg_value_loss_smoothed',
                                     title='avg_value_loss', ylabel='avg_value_loss')
                        self.results.plot(x='step', y='avg_entropy_loss_smoothed',
                                     title='avg_entropy_loss', ylabel='avg_entropy_loss')
                        self.results.save()
                    break

            if terminal:
                R = torch.zeros(1, 1)
            else:
                # bootstrap for last state
                _, value = self.net(Variable(state))
                R = value.data

            # VERIFY: that the values array isn't larger than the rewards
            values.append(Variable(R))
            R = Variable(R)

            value_loss = 0
            policy_loss = 0
            entropy_loss = 0
            for i in reversed(range(len(rewards))):
                R = rewards[i] + gamma*R
                advantage = R - values[i]

                value_loss += 0.5*advantage.pow(2)
                policy_loss -= advantage*log_probs[i]
                entropy_loss += entropies[i]

            optimizer.zero_grad()

            (policy_loss + value_coef*value_loss + ent_coef*entropy_loss).backward()
            avg_entropy_loss.update(entropy_loss.data[0])
            avg_value_loss.update(value_loss.data[0][0])
            avg_policy_loss.update(policy_loss.data[0][0])

            torch.nn.utils.clip_grad_norm(self.net.parameters(), max_grad_norm)
            optimizer.step()

    def save(self):
        # TODO: implemented
        pass

    def load(self):
        # TODO: implemented
        pass