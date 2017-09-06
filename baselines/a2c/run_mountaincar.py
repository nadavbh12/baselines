import os
from datetime import datetime

import numpy as np
import gym
import argparse
import torch
import torch.optim as optim
from DL_Logger.ResultsLog import ResultsLog, setup_logging


from baselines.a2c import policies, a2c_torch

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='A2C MountainCar training')
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                        help='results dir')
    parser.add_argument('--save', metavar='SAVE', default='',
                        help='saved folder')
    parser.add_argument('--env_id', default='CartPole-v0',
                        help='Gym env_id')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_workers', help='Number of processes to act samultaniously',
                        type=int, default=4)
    parser.add_argument('--policy', help='Policy architecture',
                        choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule',
                        choices=['constant', 'linear'], default='constant')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=40)
    args = parser.parse_args()

    # setup logging
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    agent = a2c_torch.A2CActor(results, save_path, lr=args.lr)
    agent.train(
        env_id=args.env_id,
        num_workers=args.num_workers,
        seed=args.seed,
        max_timesteps=1000000,
        gamma=0.99,
        ent_coef=0.01,
        value_coef=0.5,
        update_frequency=20,
        max_episode_len=600,
        max_grad_norm=50,
        optimizer=optim.RMSprop
    )
    print("Saving model to mountaincar_model.pkl")
    # agent.save("mountaincar_model.pkl")

if __name__ == '__main__':
    main()
