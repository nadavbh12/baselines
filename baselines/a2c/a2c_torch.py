import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.optim as optim

from baselines.a2c.policies import mlp


class A2CActor:
    def __init__(self, lr):
        # self.net = policy_network
        self.T = 0
        self.lr = lr

    def train(self, env, seed, rank, max_timesteps, gamma, beta, update_frequency, max_episode_len,
              max_grad_norm, optimizer=None):
        """Performs training of an A2C thread
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
        beta: float
            controls the strength of the entropy regularization term
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
        if env.unwrapped._spec._env_name == 'MountainCar':
            state = env.reset()
            state = torch.from_numpy(state).float()
        else:
            state, terminal = env.reset()
            state = torch.from_numpy(state)

        while self.T < max_timesteps:

            if self.T > 10000:
                env.render()
#             TODO: synchronize parameters
            rewards = []
            values = []
            entropies = []
            log_probs = []

            for t in range(update_frequency):
                action_prob, value = self.net(Variable(state))
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
                    if env.unwrapped._spec._env_name == 'MountainCar':
                        state = env.reset()
                        state = torch.from_numpy(state).float()
                    else:
                        state, terminal = env.reset()
                        state = torch.from_numpy(state)

                    print('Episode score = {}'.format(episode_reward))
                    episode_reward = 0
                    episode_len = 0
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
            for i in reversed(range(len(rewards))):
                R = rewards[i] + gamma*R
                advantage = R - values[i]
                value_loss = value_loss + 0.5*advantage.pow(2)
                # VERIFY: replace + with - ?
                policy_loss = policy_loss - advantage*log_probs[i] - beta*entropies[i]

            optimizer.zero_grad()

            (policy_loss + value_loss).backward()

            torch.nn.utils.clip_grad_norm(self.net.parameters(), max_grad_norm)
            optimizer.step()

    def save(self):
        # TODO: implemented
        pass

    def load(self):
        # TODO: implemented
        pass