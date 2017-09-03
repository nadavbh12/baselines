import gym
import torch

from baselines import deepq


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("CartPole-v0")
    num_actions = env.action_space.n
    state_shape = env.unwrapped.observation_space.shape[0]
    model = deepq.models_torch.mlp([state_shape], num_actions, [64])
    memory_args={'buffer_size': 50000}

    agent = deepq.DQNAgent(model, memory_args=memory_args)

    optimizer_args = {
        'lr':12-3

    }
    agent.train(
        env,
        optimizer=torch.optim.Adam,
        optimizer_args=optimizer_args,
        max_timesteps=100000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
    )
    print("Saving model to cartpole_model.pkl")
    agent.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
