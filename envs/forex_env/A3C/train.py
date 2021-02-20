#Training
import time
import copy
import os
import numpy as np

from envs.forex_env import ForexEnv
from model import ActorCritic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def save_checkpoint(state, filename = "A3C_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state,filename)


def load_checkpoint(model, path ="A3C_checkpoint.pth.tar"):
    if os.path.exists(path):
        print("=> Loading checkpoint")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        pass

# MAIN TRAINING LOOP
def train(rank, params, shared_model):
    torch.manual_seed(params.seed + rank)
    env = ForexEnv(start_idx= rank * 10000)
    env.seed(params.seed + rank)
    state = env.reset()
    state = torch.from_numpy(np.array(state)).type(torch.FloatTensor)
    state.requires_grad = True
    model = ActorCritic(state.size()[1], env.action_space)
    done = True
    episode_length = 0
    optimizer = Adam(shared_model.parameters(), lr = params.lr)
    torch.autograd.set_detect_anomaly(True)
    if params.load_model:
        load_checkpoint(shared_model)
    all_rewards = []
    while True:
        start = time.time()
        if (episode_length % 100 == 0) and episode_length > 0:
            checkpoint = {
                "state_dict" : shared_model.state_dict(),
                "optimizer" : optimizer.state_dict()
            }
            save_checkpoint(checkpoint)
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done:
            hx  = torch.zeros((2, 1, 64), requires_grad = True)
            cx  = torch.zeros((2, 1, 64), requires_grad = True)
            # h_t2 = torch.rand((60, 64), requires_grad = True)
            # c_t2 = torch.rand((60, 64), requires_grad = True)
            # h_t3 = torch.rand((60, 64), requires_grad = True)
            # c_t3 = torch.rand((60, 64), requires_grad = True)
            
            # hidden_list = [(h_t, c_t), (h_t2, c_t2), (h_t3, c_t3)]

        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(params.num_steps):
            inputs = state.unsqueeze(0), (hx,cx)
            # print(hidden_list[0])
            # print("=================" + "\n" + "==============")
            value, action_values, hidden_list = model(inputs)
            prob = F.softmax(action_values.unsqueeze(0), dim = 1)
            log_prob = F.log_softmax(action_values.unsqueeze(0), dim = 1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial(num_samples = 1).data
            log_prob = log_prob.gather(1, action)
            values.append(value)
            log_probs.append(log_prob)
            state, reward, done = env.step(int(action.numpy()))
            done = (done or episode_length >= params.max_episode_length)
            if done:
                episode_length = 0
                state = env.reset()
            state = torch.from_numpy(np.array(state)).type(torch.FloatTensor)
            state.requires_grad = True
            rewards.append(reward)
            if done:
                break
        R = torch.zeros(1,1)
        if not done:
            value, _, _ = model(inputs)
            R = value.data
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1,1)      
        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            TD = rewards[i] + params.gamma * values[i+1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD
            policy_loss = policy_loss - log_probs[i] * gae - 0.01 * entropies[i]
            optimizer.zero_grad()
            loss = policy_loss + 0.5*value_loss
            loss.backward(retain_graph = True)
            #print("BACKWARD COMPLETE")
            nn.utils.clip_grad_norm_(shared_model.parameters(), 40)
            ensure_shared_grads(model, shared_model)
            optimizer.step()
            #print("STEP COMPLETE")
        avg_reward = sum(rewards) / len(rewards)
        all_rewards.append(avg_reward)
        current_avg_reward = sum(all_rewards[-100::])/100
        if episode_length <= 100 and episode_length % 5 == 0:
            print("Episode {} complete. Avg reward: {}".format(episode_length, avg_reward))
        elif episode_length > 100 and episode_length % 10 == 0:
            print("Episode {};  Current_avg_reward: {}".format(episode_length, current_avg_reward ))
            f = open("avg_rewards.txt", "a")
            f.write("{},{}\n".format(episode_length,current_avg_reward))
            f.close()
        end = time.time()
        print("Loop time: {}s".format(end - start))
            




