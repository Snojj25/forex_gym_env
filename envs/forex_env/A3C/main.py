import os
import gym
import torch
import torch.multiprocessing as mp
import envs
from model import ActorCritic
from train import train
#from test import test
import my_optim

# Gathering all the parameters (that we can modify to explore)
class Params():
    def __init__(self):
        self.lr = 0.01
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 8
        self.num_steps = 20
        self.max_episode_length = 10000
        self.load_model = True

# Main run
if __name__ == '__main__':
    # freeze_support()
    os.environ['OMP_NUM_THREADS'] = '1'
    params = Params()
    torch.manual_seed(params.seed)
    env = gym.make("ForexGym-v0")
    shared_model = ActorCritic(9, env.action_space)
    shared_model.share_memory()
    # optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=params.lr)
    # optimizer.share_memory()
    processes = []
    # p = mp.Process(target=test, args=(params.num_processes, params, shared_model))
    # p.start()
    # processes.append(p)
    for rank in range(2*params.num_processes):
        p = mp.Process(target=train, args=(rank, params, shared_model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()



